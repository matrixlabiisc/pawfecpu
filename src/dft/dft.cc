// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

// Include header files
#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <dealiiLinearSolver.h>
#include <densityFirstOrderResponseCalculator.h>
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <energyCalculator.h>
#include <fileReaders.h>
#include <force.h>
#include <linalg.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <meshMovementAffineTransform.h>
#include <meshMovementGaussian.h>
#include <poissonSolverProblem.h>
#include <pseudoConverter.h>
#include <pseudoUtils.h>
#include <symmetry.h>
#include <vectorUtilities.h>
#include <MemoryTransfer.h>
#include <QuadDataCompositeWrite.h>
#include <MPIWriteOnFile.h>

#include <algorithm>
#include <cmath>
#include <complex>
//#include <stdafx.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>

#include <spglib.h>
#include <stdafx.h>
#include <sys/stat.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <chrono>
#include <sys/time.h>
#include <ctime>

#ifdef DFTFE_WITH_DEVICE
#  include <linearAlgebraOperationsDevice.h>
#endif

#include <elpa/elpa.h>


namespace dftfe
{
  //
  // dft constructor
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  dftClass<FEOrder, FEOrderElectro, memorySpace>::dftClass(
    const MPI_Comm &   mpi_comm_parent,
    const MPI_Comm &   mpi_comm_domain,
    const MPI_Comm &   _interpoolcomm,
    const MPI_Comm &   _interBandGroupComm,
    const std::string &scratchFolderName,
    dftParameters &    dftParams)
    : FE(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder + 1)), 1)
    ,
#ifdef USE_COMPLEX
    FEEigen(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder + 1)), 2)
    ,
#else
    FEEigen(dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder + 1)), 1)
    ,
#endif
    mpi_communicator(mpi_comm_domain)
    , d_mpiCommParent(mpi_comm_parent)
    , interpoolcomm(_interpoolcomm)
    , interBandGroupComm(_interBandGroupComm)
    , d_dftfeScratchFolderName(scratchFolderName)
    , d_dftParamsPtr(&dftParams)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , numElectrons(0)
    , numLevels(0)
    , d_autoMesh(1)
    , d_mesh(mpi_comm_parent,
             mpi_comm_domain,
             _interpoolcomm,
             _interBandGroupComm,
             FEOrder,
             dftParams)
    , d_affineTransformMesh(mpi_comm_parent, mpi_comm_domain, dftParams)
    , d_gaussianMovePar(mpi_comm_parent, mpi_comm_domain, dftParams)
    , d_vselfBinsManager(mpi_comm_parent,
                         mpi_comm_domain,
                         _interpoolcomm,
                         dftParams)
    , d_dispersionCorr(mpi_comm_parent,
                       mpi_comm_domain,
                       _interpoolcomm,
                       _interBandGroupComm,
                       dftParams)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0) &&
              dftParams.verbosity >= 0)
    , d_kohnShamDFTOperatorsInitialized(false)
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParams.reproducible_output || dftParams.verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::every_call_and_summary,
                      dealii::TimerOutput::wall_times)
    , computingTimerStandard(mpi_comm_domain,
                             pcout,
                             dftParams.reproducible_output ||
                                 dftParams.verbosity < 1 ?
                               dealii::TimerOutput::never :
                               dealii::TimerOutput::every_call_and_summary,
                             dealii::TimerOutput::wall_times)
    , d_subspaceIterationSolver(mpi_comm_parent,
                                mpi_comm_domain,
                                0.0,
                                0.0,
                                0.0,
                                dftParams)
#ifdef DFTFE_WITH_DEVICE
    , d_subspaceIterationSolverDevice(mpi_comm_parent,
                                      mpi_comm_domain,
                                      0.0,
                                      0.0,
                                      0.0,
                                      dftParams)
    , d_phiTotalSolverProblemDevice(mpi_comm_domain)
    , d_phiPrimeSolverProblemDevice(mpi_comm_domain)
#endif
    , d_phiTotalSolverProblem(mpi_comm_domain)
    , d_phiPrimeSolverProblem(mpi_comm_domain)
    , d_mixingScheme(mpi_comm_parent, mpi_comm_domain, dftParams.verbosity)
  {
    d_nOMPThreads = 1;
    if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
      {
        try
          {
            d_nOMPThreads = std::stoi(std::string(penv));
          }
        catch (...)
          {
            AssertThrow(
              false,
              dealii::ExcMessage(
                std::string(
                  "When specifying the <DFTFE_NUM_THREADS> environment "
                  "variable, it needs to be something that can be interpreted "
                  "as an integer. The text you have in the environment "
                  "variable is <") +
                penv + ">"));
          }

        AssertThrow(d_nOMPThreads > 0,
                    dealii::ExcMessage(
                      "When specifying the <DFTFE_NUM_THREADS> environment "
                      "variable, it needs to be a positive number."));
      }
    if (d_dftParamsPtr->verbosity > 0)
      pcout << "Threads per MPI task: " << d_nOMPThreads << std::endl;
    d_elpaScala = new dftfe::elpaScalaManager(mpi_comm_domain);

    forcePtr = new forceClass<FEOrder, FEOrderElectro, memorySpace>(
      this, mpi_comm_parent, mpi_comm_domain, dftParams);
    symmetryPtr = new symmetryClass<FEOrder, FEOrderElectro, memorySpace>(
      this, mpi_comm_parent, mpi_comm_domain, _interpoolcomm);

    d_excManagerPtr                   = std::make_shared<excManager>();
    d_isRestartGroundStateCalcFromChk = false;

#if defined(DFTFE_WITH_DEVICE)
    d_devicecclMpiCommDomainPtr = new utils::DeviceCCLWrapper;
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)

      d_devicecclMpiCommDomainPtr->init(mpi_comm_domain,
                                        d_dftParamsPtr->useDCCL);
#endif
    d_pspCutOff =
      d_dftParamsPtr->reproducible_output ?
        30.0 :
        (std::max(d_dftParamsPtr->pspCutoffImageCharges, d_pspCutOffTrunc));
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  dftClass<FEOrder, FEOrderElectro, memorySpace>::~dftClass()
  {
    finalizeKohnShamDFTOperator();
    delete symmetryPtr;
    matrix_free_data.clear();
    delete forcePtr;
#if defined(DFTFE_WITH_DEVICE)
    delete d_devicecclMpiCommDomainPtr;
#endif

    d_elpaScala->elpaDeallocateHandles(*d_dftParamsPtr);
    delete d_elpaScala;
  }

  namespace internaldft
  {
    void
    convertToCellCenteredCartesianCoordinates(
      std::vector<std::vector<double>> &      atomLocations,
      const std::vector<std::vector<double>> &latticeVectors)
    {
      std::vector<double> cartX(atomLocations.size(), 0.0);
      std::vector<double> cartY(atomLocations.size(), 0.0);
      std::vector<double> cartZ(atomLocations.size(), 0.0);

      //
      // convert fractional atomic coordinates to cartesian coordinates
      //
      for (int i = 0; i < atomLocations.size(); ++i)
        {
          cartX[i] = atomLocations[i][2] * latticeVectors[0][0] +
                     atomLocations[i][3] * latticeVectors[1][0] +
                     atomLocations[i][4] * latticeVectors[2][0];
          cartY[i] = atomLocations[i][2] * latticeVectors[0][1] +
                     atomLocations[i][3] * latticeVectors[1][1] +
                     atomLocations[i][4] * latticeVectors[2][1];
          cartZ[i] = atomLocations[i][2] * latticeVectors[0][2] +
                     atomLocations[i][3] * latticeVectors[1][2] +
                     atomLocations[i][4] * latticeVectors[2][2];
        }

      //
      // define cell centroid (confirm whether it will work for non-orthogonal
      // lattice vectors)
      //
      double cellCentroidX =
        0.5 *
        (latticeVectors[0][0] + latticeVectors[1][0] + latticeVectors[2][0]);
      double cellCentroidY =
        0.5 *
        (latticeVectors[0][1] + latticeVectors[1][1] + latticeVectors[2][1]);
      double cellCentroidZ =
        0.5 *
        (latticeVectors[0][2] + latticeVectors[1][2] + latticeVectors[2][2]);

      for (int i = 0; i < atomLocations.size(); ++i)
        {
          atomLocations[i][2] = cartX[i] - cellCentroidX;
          atomLocations[i][3] = cartY[i] - cellCentroidY;
          atomLocations[i][4] = cartZ[i] - cellCentroidZ;
        }
    }
  } // namespace internaldft

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeVolume(
    const dealii::DoFHandler<3> &_dofHandler)
  {
    double                       domainVolume = 0;
    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(_dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_JxW_values);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = _dofHandler.begin_active(),
      endc = _dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < quadrature.size(); ++q_point)
            domainVolume += fe_values.JxW(q_point);
        }

    domainVolume = dealii::Utilities::MPI::sum(domainVolume, mpi_communicator);
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Volume of the domain (Bohr^3): " << domainVolume << std::endl;
    return domainVolume;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::set()
  {
    computingTimerStandard.enter_subsection("Atomic system initialization");
    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Entered call to set");

    d_numEigenValues = d_dftParamsPtr->numberEigenValues;

    //
    // read coordinates
    //
    unsigned int numberColumnsCoordinatesFile =
      d_dftParamsPtr->useMeshSizesFromAtomsFile ? 7 : 5;

    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        //
        // read fractionalCoordinates of atoms in periodic case
        //
        dftUtils::readFile(numberColumnsCoordinatesFile,
                           atomLocations,
                           d_dftParamsPtr->coordinatesFile);
        AssertThrow(
          d_dftParamsPtr->natoms == atomLocations.size(),
          dealii::ExcMessage(
            "DFT-FE Error: The number atoms"
            "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
            "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
            "blank row at the end can cause this issue too."));
        pcout << "number of atoms: " << atomLocations.size() << "\n";
        atomLocationsFractional.resize(atomLocations.size());
        //
        // find unique atom types
        //
        for (std::vector<std::vector<double>>::iterator it =
               atomLocations.begin();
             it < atomLocations.end();
             it++)
          {
            atomTypes.insert((unsigned int)((*it)[0]));
            d_atomTypeAtributes[(unsigned int)((*it)[0])] =
              (unsigned int)((*it)[1]);

            if (!d_dftParamsPtr->isPseudopotential)
              AssertThrow(
                (*it)[0] <= 50,
                dealii::ExcMessage(
                  "DFT-FE Error: One of the atomic numbers exceeds 50."
                  "Currently, for all-electron calculations we have single atom wavefunction and electron-density"
                  "initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
                  "added in the next release. In the mean time, you could also contact the developers of DFT-FE, who can provide"
                  "you the data for the single atom wavefunction and electron-density data for"
                  "atomic numbers beyond 50."));
          }

        //
        // print fractional coordinates
        //
        for (int i = 0; i < atomLocations.size(); ++i)
          {
            atomLocationsFractional[i] = atomLocations[i];
          }
      }
    else
      {
        dftUtils::readFile(numberColumnsCoordinatesFile,
                           atomLocations,
                           d_dftParamsPtr->coordinatesFile);

        AssertThrow(
          d_dftParamsPtr->natoms == atomLocations.size(),
          dealii::ExcMessage(
            "DFT-FE Error: The number atoms"
            "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
            "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
            "blank row at the end can cause this issue too."));
        pcout << "number of atoms: " << atomLocations.size() << "\n";

        //
        // find unique atom types
        //
        for (std::vector<std::vector<double>>::iterator it =
               atomLocations.begin();
             it < atomLocations.end();
             it++)
          {
            atomTypes.insert((unsigned int)((*it)[0]));
            d_atomTypeAtributes[(unsigned int)((*it)[0])] =
              (unsigned int)((*it)[1]);

            if (!d_dftParamsPtr->isPseudopotential)
              AssertThrow(
                (*it)[0] <= 50,
                dealii::ExcMessage(
                  "DFT-FE Error: One of the atomic numbers exceeds 50."
                  "Currently, for all-electron calculations we have single atom wavefunction and electron-density"
                  "initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
                  "added in the next release. You could also contact the developers of DFT-FE, who can provide"
                  "you with the code to generate the single atom wavefunction and electron-density data for"
                  "atomic numbers beyond 50."));
          }
      }

    //
    // read Gaussian atomic displacements
    //
    std::vector<std::vector<double>> atomsDisplacementsGaussian;
    d_atomsDisplacementsGaussianRead.resize(atomLocations.size(),
                                            dealii::Tensor<1, 3, double>());
    d_gaussianMovementAtomsNetDisplacements.resize(
      atomLocations.size(), dealii::Tensor<1, 3, double>());
    if (d_dftParamsPtr->coordinatesGaussianDispFile != "")
      {
        dftUtils::readFile(3,
                           atomsDisplacementsGaussian,
                           d_dftParamsPtr->coordinatesGaussianDispFile);

        for (int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
          for (int j = 0; j < 3; ++j)
            d_atomsDisplacementsGaussianRead[i][j] =
              atomsDisplacementsGaussian[i][j];

        d_isAtomsGaussianDisplacementsReadFromFile = true;
      }

    //
    // read domain bounding Vectors
    //
    unsigned int numberColumnsLatticeVectorsFile = 3;
    dftUtils::readFile(numberColumnsLatticeVectorsFile,
                       d_domainBoundingVectors,
                       d_dftParamsPtr->domainBoundingVectorsFile);

    AssertThrow(
      d_domainBoundingVectors.size() == 3,
      dealii::ExcMessage(
        "DFT-FE Error: The number of domain bounding"
        "vectors read from input file (input through DOMAIN VECTORS FILE) should be 3. Please check"
        "your domain vectors file. Sometimes an extra blank row at the end can cause this issue too."));

    //
    // evaluate cross product of
    //
    std::vector<double> cross;
    dftUtils::cross_product(d_domainBoundingVectors[0],
                            d_domainBoundingVectors[1],
                            cross);

    double scalarConst = d_domainBoundingVectors[2][0] * cross[0] +
                         d_domainBoundingVectors[2][1] * cross[1] +
                         d_domainBoundingVectors[2][2] * cross[2];
    AssertThrow(
      scalarConst > 0,
      dealii::ExcMessage(
        "DFT-FE Error: Domain bounding vectors or lattice vectors read from"
        "input file (input through DOMAIN VECTORS FILE) should form a right-handed coordinate system."
        "Please check your domain vectors file. This is usually fixed by changing the order of the"
        "vectors in the domain vectors file."));

    pcout << "number of atoms types: " << atomTypes.size() << "\n";


    //
    // determine number of electrons
    //
    for (unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        const unsigned int Z        = atomLocations[iAtom][0];
        const unsigned int valenceZ = atomLocations[iAtom][1];

        if (d_dftParamsPtr->isPseudopotential)
          numElectrons += valenceZ;
        else
          numElectrons += Z;
      }

    numElectrons = numElectrons + d_dftParamsPtr->netCharge;
    if (d_dftParamsPtr->verbosity >= 1 and
        std::abs(d_dftParamsPtr->netCharge) > 1e-12)
      pcout << "Setting netcharge " << d_dftParamsPtr->netCharge << std::endl;

    if (d_dftParamsPtr->solverMode == "NSCF" &&
        d_dftParamsPtr->numberEigenValues == 0 &&
        d_dftParamsPtr->highestStateOfInterestForChebFiltering != 0)
      {
        d_numEigenValues =
          std::max(d_dftParamsPtr->highestStateOfInterestForChebFiltering * 1.1,
                   d_dftParamsPtr->highestStateOfInterestForChebFiltering +
                     10.0);

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Setting the number of Kohn-Sham wave functions to be 10 percent more than the HIGHEST STATE OF INTEREST FOR CHEBYSHEV FILTERING "
              << d_numEigenValues << std::endl;
          }
      }
    else if (d_dftParamsPtr->numberEigenValues <= numElectrons / 2.0 ||
             d_dftParamsPtr->numberEigenValues == 0)
      {
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Warning: User has requested the number of Kohn-Sham wavefunctions to be less than or"
                 "equal to half the number of electrons in the system. Setting the Kohn-Sham wavefunctions"
                 "to half the number of electrons with a 20 percent buffer to avoid convergence issues in"
                 "SCF iterations"
              << std::endl;
          }
        d_numEigenValues =
          (numElectrons / 2.0) +
          std::max((d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND" ?
                      0.22 :
                      0.2) *
                     (numElectrons / 2.0),
                   20.0);

        // start with 17-20% buffer to leave room for additional modifications
        // due to block size restrictions
#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice && d_dftParamsPtr->autoDeviceBlockSizes)
          d_numEigenValues =
            (numElectrons / 2.0) + std::max((d_dftParamsPtr->mixingMethod ==
                                                 "LOW_RANK_DIELECM_PRECOND" ?
                                               0.2 :
                                               0.17) *
                                              (numElectrons / 2.0),
                                            20.0);
#endif

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << " Setting the number of Kohn-Sham wave functions to be "
                  << d_numEigenValues << std::endl;
          }
      }

    if (d_dftParamsPtr->algoType == "FAST")
      {
        if (d_dftParamsPtr->TVal < 1000)
          {
            d_dftParamsPtr->numCoreWfcRR = 0.8 * numElectrons / 2.0;
            pcout << " Setting SPECTRUM SPLIT CORE EIGENSTATES to be "
                  << d_dftParamsPtr->numCoreWfcRR << std::endl;
          }
      }


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice && d_dftParamsPtr->autoDeviceBlockSizes)
      {
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


        d_numEigenValues =
          std::ceil(d_numEigenValues / (numberBandGroups * 1.0)) *
          numberBandGroups;

        AssertThrow(
          (d_numEigenValues % numberBandGroups == 0 ||
           d_numEigenValues / numberBandGroups == 0),
          dealii::ExcMessage(
            "DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for Device run."));

        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, d_numEigenValues, bandGroupLowHighPlusOneIndices);

        const unsigned int eigenvaluesInBandGroup =
          bandGroupLowHighPlusOneIndices[1];

        if (eigenvaluesInBandGroup <= 100)
          {
            d_dftParamsPtr->chebyWfcBlockSize = eigenvaluesInBandGroup;
            d_dftParamsPtr->wfcBlockSize      = eigenvaluesInBandGroup;
          }
        else if (eigenvaluesInBandGroup <= 600)
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 90.0) * 90.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 100.0) * 100.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 110.0) * 110.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 120.0) * 120.0 *
                       numberBandGroups;

            temp2[0] = 90;
            temp2[1] = 100;
            temp2[2] = 110;
            temp2[3] = 120;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex];
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else if (eigenvaluesInBandGroup <= 1000)
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 150.0) * 150.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 160.0) * 160.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 170.0) * 170.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 180.0) * 180.0 *
                       numberBandGroups;

            temp2[0] = 150;
            temp2[1] = 160;
            temp2[2] = 170;
            temp2[3] = 180;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else if (eigenvaluesInBandGroup <= 2000)
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 200.0) * 200.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 220.0) * 220.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 240.0) * 240.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 260.0) * 260.0 *
                       numberBandGroups;

            temp2[0] = 200;
            temp2[1] = 220;
            temp2[2] = 240;
            temp2[3] = 260;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 360.0) * 360.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 380.0) * 380.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 400.0) * 400.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 440.0) * 440.0 *
                       numberBandGroups;

            temp2[0] = 360;
            temp2[1] = 380;
            temp2[2] = 400;
            temp2[3] = 440;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }

        if (d_dftParamsPtr->algoType == "FAST")
          d_dftParamsPtr->numCoreWfcRR =
            std::floor(d_dftParamsPtr->numCoreWfcRR /
                       d_dftParamsPtr->wfcBlockSize) *
            d_dftParamsPtr->wfcBlockSize;

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Setting the number of Kohn-Sham wave functions for Device run to be: "
              << d_numEigenValues << std::endl;
            pcout << " Setting CHEBY WFC BLOCK SIZE for Device run to be "
                  << d_dftParamsPtr->chebyWfcBlockSize << std::endl;
            pcout << " Setting WFC BLOCK SIZE for Device run to be "
                  << d_dftParamsPtr->wfcBlockSize << std::endl;
            if (d_dftParamsPtr->algoType == "FAST")
              pcout
                << " Setting SPECTRUM SPLIT CORE EIGENSTATES for Device run to be "
                << d_dftParamsPtr->numCoreWfcRR << std::endl;
          }
      }
#endif

    if (d_dftParamsPtr->constraintMagnetization)
      {
        numElectronsUp   = std::ceil(static_cast<double>(numElectrons) / 2.0);
        numElectronsDown = numElectrons - numElectronsUp;
        //
        int netMagnetization =
          std::round(2.0 * static_cast<double>(numElectrons) *
                     d_dftParamsPtr->start_magnetization);
        //
        while ((numElectronsUp - numElectronsDown) < std::abs(netMagnetization))
          {
            numElectronsDown -= 1;
            numElectronsUp += 1;
          }
        //
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << " Number of spin up electrons " << numElectronsUp
                  << std::endl;
            pcout << " Number of spin down electrons " << numElectronsDown
                  << std::endl;
          }
      }
    // convert pseudopotential files in upf format to dftfe format
    if (d_dftParamsPtr->verbosity >= 1)
      {
        pcout
          << std::endl
          << "Reading Pseudo-potential data for each atom from the list given in : "
          << d_dftParamsPtr->pseudoPotentialFile << std::endl;
      }

    int              nlccFlag = 0;
    int              pawFlag  = 0;
    std::vector<int> pspFlags(2, 0);
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0 &&
        d_dftParamsPtr->isPseudopotential == true)
      pspFlags = pseudoUtils::convert(d_dftParamsPtr->pseudoPotentialFile,
                                      d_dftfeScratchFolderName,
                                      d_dftParamsPtr->verbosity,
                                      d_dftParamsPtr->natomTypes,
                                      d_dftParamsPtr->pseudoTestsFlag);

    nlccFlag = pspFlags[0];
    pawFlag  = pspFlags[1];
    nlccFlag = dealii::Utilities::MPI::sum(nlccFlag, d_mpiCommParent);
    pawFlag  = dealii::Utilities::MPI::sum(pawFlag, d_mpiCommParent);
    if (nlccFlag > 0 && d_dftParamsPtr->isPseudopotential == true)
      d_dftParamsPtr->nonLinearCoreCorrection = true;
    if (pawFlag > 0 && d_dftParamsPtr->isPseudopotential == true)
      d_dftParamsPtr->pawPseudoPotential = true;
    // estimate total number of wave functions from atomic orbital filling
    if (d_dftParamsPtr->startingWFCType == "ATOMIC")
      determineOrbitalFilling();

    AssertThrow(
      d_dftParamsPtr->numCoreWfcRR <= d_numEigenValues,
      dealii::ExcMessage(
        "DFT-FE Error: Incorrect input value used- SPECTRUM SPLIT CORE EIGENSTATES should be less than the total number of wavefunctions."));
    d_numEigenValuesRR = d_numEigenValues - d_dftParamsPtr->numCoreWfcRR;


#ifdef USE_COMPLEX
    if (d_dftParamsPtr->solverMode == "NSCF")
      {
        if (!(d_dftParamsPtr->kPointDataFile == ""))
          {
            readkPointData();
          }
        else
          {
            generateMPGrid();
          }
      }
    else
      generateMPGrid();
#else
    d_kPointCoordinates.resize(3, 0.0);
    d_kPointWeights.resize(1, 1.0);
#endif

    // set size of eigenvalues and eigenvectors data structures
    eigenValues.resize(d_kPointWeights.size());
    eigenValuesRRSplit.resize(d_kPointWeights.size());

    if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      d_densityMatDerFermiEnergy.resize((d_dftParamsPtr->spinPolarized + 1) *
                                        d_kPointWeights.size());

    a0.clear();
    bLow.clear();

    a0.resize((d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(),
              0.0);
    bLow.resize((d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(),
                0.0);

    d_upperBoundUnwantedSpectrumValues.clear();
    d_upperBoundUnwantedSpectrumValues.resize(
      (d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(), 0.0);


    for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        eigenValues[kPoint].resize((d_dftParamsPtr->spinPolarized + 1) *
                                   d_numEigenValues);
        eigenValuesRRSplit[kPoint].resize((d_dftParamsPtr->spinPolarized + 1) *
                                          d_numEigenValuesRR);
      }



    if (d_dftParamsPtr->isPseudopotential == true &&
        d_dftParamsPtr->pawPseudoPotential == false)
      {
        pcout << "Creating ONCV class pointer: " << std::endl;
        d_oncvClassPtr =
          std::make_shared<dftfe::oncvClass<dataTypes::number, memorySpace>>(
            mpi_communicator, // domain decomposition communicator
            d_dftfeScratchFolderName,
            atomTypes,
            d_dftParamsPtr->floatingNuclearCharges,
            d_nOMPThreads,
            d_atomTypeAtributes,
            d_dftParamsPtr->reproducible_output,
            d_dftParamsPtr->verbosity,
            d_dftParamsPtr->useDevice);
      }
    else if (d_dftParamsPtr->isPseudopotential == true &&
             d_dftParamsPtr->pawPseudoPotential == true)
      {
        pcout << "Creating PAW class pointer: " << std::endl;
        d_pawClassPtr =
          std::make_shared<dftfe::pawClass<dataTypes::number, memorySpace>>(
            mpi_communicator, // domain decomposition communicator
            d_dftfeScratchFolderName,
            d_dftParamsPtr,
            atomTypes,
            d_dftParamsPtr->floatingNuclearCharges,
            d_nOMPThreads,
            d_atomTypeAtributes,
            d_dftParamsPtr->reproducible_output,
            d_dftParamsPtr->verbosity,
            d_dftParamsPtr->useDevice);
      }

    if (d_dftParamsPtr->verbosity >= 1)
      if (d_dftParamsPtr->nonLinearCoreCorrection == true)
        pcout
          << "Atleast one atom has pseudopotential with nonlinear core correction"
          << std::endl;

    d_elpaScala->processGridELPASetup(d_numEigenValues,
                                      d_numEigenValuesRR,
                                      *d_dftParamsPtr);

    MPI_Barrier(d_mpiCommParent);
    computingTimerStandard.leave_subsection("Atomic system initialization");
  }

  // dft pseudopotential init
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initPseudoPotentialAll(
    const bool updateNonlocalSparsity)
  {
    if (d_dftParamsPtr->isPseudopotential)
      {
        dealii::TimerOutput::Scope scope(computing_timer, "psp init");
        pcout << std::endl << "Pseudopotential initalization...." << std::endl;

        double init_core;
        MPI_Barrier(d_mpiCommParent);
        init_core = MPI_Wtime();

        if (d_dftParamsPtr->nonLinearCoreCorrection == true)
          initCoreRho();

        MPI_Barrier(d_mpiCommParent);
        init_core = MPI_Wtime() - init_core;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << "initPseudoPotentialAll: Time taken for initializing core density for non-linear core correction: "
            << init_core << std::endl;
        determineAtomsOfInterstPseudopotential(atomLocations);
        MPI_Barrier(d_mpiCommParent);
        if (d_dftParamsPtr->isPseudopotential == true &&
            d_dftParamsPtr->pawPseudoPotential == false)
          {
            d_oncvClassPtr->initialiseNonLocalContribution(
              d_atomLocationsInterestPseudopotential,
              d_imageIdsTrunc,
              d_imagePositionsTrunc,
              d_kPointWeights,     // accounts for interpool
              d_kPointCoordinates, // accounts for interpool
              updateNonlocalSparsity);
          }
        else if (d_dftParamsPtr->isPseudopotential == true &&
                 d_dftParamsPtr->pawPseudoPotential == true)
          {
            d_pawClassPtr->initialiseNonLocalContribution(
              d_atomLocationsInterestPseudopotential,
              d_imageIdsTrunc,
              d_imagePositionsTrunc,
              d_kPointWeights,     // accounts for interpool
              d_kPointCoordinates, // accounts for interpool
              updateNonlocalSparsity,
              d_phiTotDofHandlerIndexElectro);
          }
      }
  }


  // generate image charges and update k point cartesian coordinates based on
  // current lattice vectors
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initImageChargesUpdateKPoints(
    bool flag)
  {
    dealii::TimerOutput::Scope scope(computing_timer,
                                     "image charges and k point generation");
    pcout
      << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
      << std::endl;
    for (int i = 0; i < d_domainBoundingVectors.size(); ++i)
      {
        pcout << "v" << i + 1 << " : " << d_domainBoundingVectors[i][0] << " "
              << d_domainBoundingVectors[i][1] << " "
              << d_domainBoundingVectors[i][2] << std::endl;
      }
    pcout
      << "-----------------------------------------------------------------------------------------"
      << std::endl;

    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        pcout << "-----Fractional coordinates of atoms------ " << std::endl;
        for (unsigned int i = 0; i < atomLocations.size(); ++i)
          {
            atomLocations[i] = atomLocationsFractional[i];
            pcout << "AtomId " << i << ":  " << atomLocationsFractional[i][2]
                  << " " << atomLocationsFractional[i][3] << " "
                  << atomLocationsFractional[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
        // sanity check on fractional coordinates
        std::vector<bool> periodicBc(3, false);
        periodicBc[0]    = d_dftParamsPtr->periodicX;
        periodicBc[1]    = d_dftParamsPtr->periodicY;
        periodicBc[2]    = d_dftParamsPtr->periodicZ;
        const double tol = 1e-6;

        if (flag)
          {
            for (unsigned int i = 0; i < atomLocationsFractional.size(); ++i)
              {
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    if (periodicBc[idim])
                      AssertThrow(
                        atomLocationsFractional[i][2 + idim] > -tol &&
                          atomLocationsFractional[i][2 + idim] < 1.0 + tol,
                        dealii::ExcMessage(
                          "DFT-FE Error: periodic direction fractional coordinates doesn't lie in [0,1]. Please check input"
                          "fractional coordinates, or if this is an ionic relaxation step, please check the corresponding"
                          "algorithm."));
                    if (!periodicBc[idim])
                      AssertThrow(
                        atomLocationsFractional[i][2 + idim] > tol &&
                          atomLocationsFractional[i][2 + idim] < 1.0 - tol,
                        dealii::ExcMessage(
                          "DFT-FE Error: non-periodic direction fractional coordinates doesn't lie in (0,1). Please check"
                          "input fractional coordinates, or if this is an ionic relaxation step, please check the"
                          "corresponding algorithm."));
                  }
              }
          }

        generateImageCharges(d_pspCutOff,
                             d_imageIds,
                             d_imageCharges,
                             d_imagePositions);

        generateImageCharges(d_pspCutOffTrunc,
                             d_imageIdsTrunc,
                             d_imageChargesTrunc,
                             d_imagePositionsTrunc);

        if ((d_dftParamsPtr->verbosity >= 4 ||
             d_dftParamsPtr->reproducible_output))
          pcout << "Number Image Charges  " << d_imageIds.size() << std::endl;

        internaldft::convertToCellCenteredCartesianCoordinates(
          atomLocations, d_domainBoundingVectors);
#ifdef USE_COMPLEX
        recomputeKPointCoordinates();
#endif
        if (d_dftParamsPtr->verbosity >= 4)
          {
            // FIXME: Print all k points across all pools
            pcout
              << "-------------------k points cartesian coordinates and weights-----------------------------"
              << std::endl;
            for (unsigned int i = 0; i < d_kPointWeights.size(); ++i)
              {
                pcout << " [" << d_kPointCoordinates[3 * i + 0] << ", "
                      << d_kPointCoordinates[3 * i + 1] << ", "
                      << d_kPointCoordinates[3 * i + 2] << "] "
                      << d_kPointWeights[i] << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------------"
              << std::endl;
          }
      }
    else
      {
        //
        // print cartesian coordinates
        //
        pcout
          << "------------Cartesian coordinates of atoms (origin at center of domain)------------------"
          << std::endl;
        for (unsigned int i = 0; i < atomLocations.size(); ++i)
          {
            pcout << "AtomId " << i << ":  " << atomLocations[i][2] << " "
                  << atomLocations[i][3] << " " << atomLocations[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;

        //
        // redundant call (check later)
        //
        generateImageCharges(d_pspCutOff,
                             d_imageIds,
                             d_imageCharges,
                             d_imagePositions);

        generateImageCharges(d_pspCutOffTrunc,
                             d_imageIdsTrunc,
                             d_imageChargesTrunc,
                             d_imagePositionsTrunc);
      }
  }

  // dft init
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::init()
  {
    computingTimerStandard.enter_subsection("KSDFT problem initialization");

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator, "Entering init");

    d_BLASWrapperPtrHost = std::make_shared<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>();
    d_basisOperationsPtrHost = std::make_shared<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>(
      d_BLASWrapperPtrHost);
    d_basisOperationsPtrElectroHost = std::make_shared<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>(
      d_BLASWrapperPtrHost);
#if defined(DFTFE_WITH_DEVICE)
    if (d_dftParamsPtr->useDevice)
      {
        d_BLASWrapperPtr = std::make_shared<dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>>();
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        d_BLASWrapperPtr->setMathMode(dftfe::utils::DEVICEBLAS_DEFAULT_MATH);
#  endif
        d_basisOperationsPtrDevice = std::make_shared<
          dftfe::basis::FEBasisOperations<dataTypes::number,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>(
          d_BLASWrapperPtr);
        d_basisOperationsPtrElectroDevice = std::make_shared<
          dftfe::basis::FEBasisOperations<double,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>(
          d_BLASWrapperPtr);
      }
#endif
    initImageChargesUpdateKPoints();

    calculateNearestAtomDistances();

    computing_timer.enter_subsection("mesh generation");
    //
    // generate mesh (both parallel and serial)
    // while parallel meshes are always generated, serial meshes are only
    // generated for following three cases: symmetrization is on, ionic
    // optimization is on as well as reuse wfcs and density from previous ionic
    // step is on, or if serial constraints generation is on.
    //
    if (d_dftParamsPtr->loadRhoData)
      {
        d_mesh.generateCoarseMeshesForRestart(
          atomLocations,
          d_imagePositionsTrunc,
          d_imageIdsTrunc,
          d_nearestAtomDistances,
          d_domainBoundingVectors,
          d_dftParamsPtr->useSymm ||
            d_dftParamsPtr->createConstraintsFromSerialDofhandler);

        loadTriaInfoAndRhoNodalData();
      }
    else
      {
        d_mesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
          atomLocations,
          d_imagePositionsTrunc,
          d_imageIdsTrunc,
          d_nearestAtomDistances,
          d_domainBoundingVectors,
          d_dftParamsPtr->useSymm ||
            d_dftParamsPtr->createConstraintsFromSerialDofhandler);
      }
    computing_timer.leave_subsection("mesh generation");

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Mesh generation completed");
    //
    // get access to triangulation objects from meshGenerator class
    //
    dealii::parallel::distributed::Triangulation<3> &triangulationPar =
      d_mesh.getParallelMeshMoved();

    //
    // initialize dofHandlers and hanging-node constraints and periodic
    // constraints on the unmoved Mesh
    //
    initUnmovedTriangulation(triangulationPar);

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initUnmovedTriangulation completed");
#ifdef USE_COMPLEX
    if (d_dftParamsPtr->useSymm)
      symmetryPtr->initSymmetry();
#endif



    //
    // move triangulation to have atoms on triangulation vertices
    //
    if (!d_dftParamsPtr->floatingNuclearCharges)
      moveMeshToAtoms(triangulationPar, d_mesh.getSerialMeshUnmoved());


    if (d_dftParamsPtr->smearedNuclearCharges)
      calculateSmearedChargeWidths();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "moveMeshToAtoms completed");
    //
    // initialize dirichlet BCs for total potential and vSelf poisson solutions
    //
    initBoundaryConditions();
    d_smearedChargeMomentsComputed = false;

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initBoundaryConditions completed");

    //
    // initialize pseudopotential data for both local and nonlocal part
    //
    if (d_dftParamsPtr->isPseudopotential == true &&
        d_dftParamsPtr->pawPseudoPotential == false)
      {
        d_oncvClassPtr->initialise(d_basisOperationsPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                                   d_basisOperationsPtrDevice,
#endif
                                   d_BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                                   d_BLASWrapperPtr,
#endif
                                   d_densityQuadratureId,
                                   d_lpspQuadratureId,
                                   d_sparsityPatternQuadratureId,
                                   d_nlpspQuadratureId,
                                   d_densityQuadratureIdElectro,
                                   d_excManagerPtr,
                                   atomLocations,
                                   d_numEigenValues,
                                   d_dftParamsPtr->useSinglePrecCheby);
      }
    else if (d_dftParamsPtr->isPseudopotential == true &&
             d_dftParamsPtr->pawPseudoPotential == true)
      {
        d_pawClassPtr->initialise(d_basisOperationsPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                                  d_basisOperationsPtrDevice,
#endif
                                  d_basisOperationsPtrElectroHost,
#if defined(DFTFE_WITH_DEVICE)
                                  d_basisOperationsPtrElectroDevice,
#endif
                                  d_BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
                                  d_BLASWrapperPtr,
#endif
                                  d_densityQuadratureId,
                                  d_lpspQuadratureId,
                                  d_sparsityPatternQuadratureId,
                                  d_nlpspQuadratureId,
                                  d_densityQuadratureIdElectro,
                                  d_excManagerPtr,
                                  atomLocations,
                                  d_numEigenValues,
                                  d_smearedChargeQuadratureIdElectro,
                                  d_bQuadValuesAllAtoms,
                                  d_dftParamsPtr->useSinglePrecCheby);
      }


    //
    // initialize guesses for electron-density and wavefunctions
    //
    initElectronicFields();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initElectronicFields completed");


    initPseudoPotentialAll();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initPseudopotential completed");

    //
    // Apply Gaussian displacments to atoms and mesh if input gaussian
    // displacments are read from file. When restarting a relaxation, this must
    // be done only once at the begining- this is why the flag is to false after
    // the Gaussian movement. The last flag to updateAtomPositionsAndMoveMesh is
    // set to true to force use of single atom solutions.
    //
    if (d_isAtomsGaussianDisplacementsReadFromFile)
      {
        updateAtomPositionsAndMoveMesh(d_atomsDisplacementsGaussianRead,
                                       1e+4,
                                       true);
        d_isAtomsGaussianDisplacementsReadFromFile = false;
      }

    if (d_dftParamsPtr->loadRhoData)
      {
        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "Overwriting input density data to SCF solve with data read from restart file.."
            << std::endl;

        // Note: d_rhoInNodalValuesRead is not compatible with
        // d_matrixFreeDataPRefined
        for (unsigned int i = 0; i < d_densityInNodalValues[0].local_size();
             i++)
          d_densityInNodalValues[0].local_element(i) =
            d_rhoInNodalValuesRead.local_element(i);

        interpolateDensityNodalDataToQuadratureDataGeneral(
          d_basisOperationsPtrElectroHost,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_densityInNodalValues[0],
          d_densityInQuadValues[0],
          d_gradDensityInQuadValues[0],
          d_gradDensityInQuadValues[0],
          d_excManagerPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA);

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            d_densityInNodalValues[1] = 0;
            for (unsigned int i = 0; i < d_densityInNodalValues[1].local_size();
                 i++)
              {
                d_densityInNodalValues[1].local_element(i) =
                  d_magInNodalValuesRead.local_element(i);
              }

            interpolateDensityNodalDataToQuadratureDataGeneral(
              d_basisOperationsPtrElectroHost,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_densityInNodalValues[1],
              d_densityInQuadValues[1],
              d_gradDensityInQuadValues[1],
              d_gradDensityInQuadValues[1],
              d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);
          }
        if ((d_dftParamsPtr->solverMode == "GEOOPT"))
          {
            d_densityOutNodalValues = d_densityInNodalValues;
            for (unsigned int iComp = 0; iComp < d_densityOutNodalValues.size();
                 ++iComp)
              d_densityOutNodalValues[iComp].update_ghost_values();

            d_densityOutQuadValues = d_densityInQuadValues;

            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              d_gradDensityOutQuadValues = d_gradDensityInQuadValues;
          }

        d_isRestartGroundStateCalcFromChk = true;
      }
    if (d_dftParamsPtr->pawPseudoPotential)
      {
        if (d_dftParamsPtr->loadDijData)
          {
            d_pawClassPtr->loadDijEntriesFromFile();
          }
        else
          d_pawClassPtr->computeDijFromPSIinitialGuess(
            &d_eigenVectorsFlattenedHost,
            numElectrons,
            d_numEigenValues,
            d_densityQuadratureId,
            d_kPointWeights,
            interpoolcomm,
            interBandGroupComm);
        double scaleFactor = d_pawClassPtr->densityScalingFactor(atomLocations);
        const double charge =
          totalCharge(d_dofHandlerRhoNodal, d_densityInQuadValues[0]);
        scaleRhoInQuadValues(scaleFactor / charge);
        if (d_dftParamsPtr->memoryOptCompCharge)
          d_pawClassPtr->computeCompensationChargeMemoryOpt(TypeOfField::In);
        else
          d_pawClassPtr->computeCompensationCharge(TypeOfField::In);
        d_pawClassPtr->chargeNeutrality(totalCharge(d_dofHandlerRhoNodal,
                                                    d_densityInQuadValues[0]),
                                        TypeOfField::In,
                                        false);
        computeRhoNodalInverseMassVector();
        computeTotalDensityNodalVector(d_bQuadValuesAllAtoms,
                                       d_densityInNodalValues[0],
                                       d_totalChargeDensityInNodalValues[0]);
      }

    d_isFirstFilteringCall.clear();
    d_isFirstFilteringCall.resize((d_dftParamsPtr->spinPolarized + 1) *
                                    d_kPointWeights.size(),
                                  true);

    initializeKohnShamDFTOperator();

    d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.clear();
    d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.resize(
      atomLocations.size() * 3, 0.0);

    computingTimerStandard.leave_subsection("KSDFT problem initialization");
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initNoRemesh(
    const bool updateImagesAndKPointsAndVselfBins,
    const bool checkSmearedChargeWidthsForOverlap,
    const bool useSingleAtomSolutionOverride,
    const bool isMeshDeformed)
  {
    computingTimerStandard.enter_subsection("KSDFT problem initialization");
    if (updateImagesAndKPointsAndVselfBins)
      {
        initImageChargesUpdateKPoints();
      }

    if (checkSmearedChargeWidthsForOverlap)
      {
        calculateNearestAtomDistances();

        if (d_dftParamsPtr->smearedNuclearCharges)
          calculateSmearedChargeWidths();

        d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.clear();
        d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.resize(
          atomLocations.size() * 3, 0.0);
      }

    //
    // reinitialize dirichlet BCs for total potential and vSelf poisson
    // solutions
    //
    double init_bc;
    MPI_Barrier(d_mpiCommParent);
    init_bc = MPI_Wtime();


    // false option reinitializes vself bins from scratch wheras true option
    // only updates the boundary conditions
    const bool updateOnlyBinsBc = !updateImagesAndKPointsAndVselfBins;
    initBoundaryConditions(isMeshDeformed || d_dftParamsPtr->isCellStress,
                           updateOnlyBinsBc);
    d_smearedChargeMomentsComputed = false;
    MPI_Barrier(d_mpiCommParent);
    init_bc = MPI_Wtime() - init_bc;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "
        << init_bc << std::endl;

    double init_rho;
    MPI_Barrier(d_mpiCommParent);
    init_rho = MPI_Wtime();

    if (useSingleAtomSolutionOverride)
      {
        readPSI();
        initRho();
        if (d_dftParamsPtr->pawPseudoPotential)
          {
            if (d_dftParamsPtr->loadDijData)
              {
                d_pawClassPtr->loadDijEntriesFromFile();
              }
            else
              d_pawClassPtr->computeDijFromPSIinitialGuess(
                &d_eigenVectorsFlattenedHost,
                numElectrons,
                d_numEigenValues,
                d_densityQuadratureId,
                d_kPointWeights,
                interpoolcomm,
                interBandGroupComm);
            double scaleFactor =
              d_pawClassPtr->densityScalingFactor(atomLocations);
            const double charge =
              totalCharge(d_dofHandlerRhoNodal, d_densityInQuadValues[0]);
            scaleRhoInQuadValues(scaleFactor / charge);
          }
      }
    else
      {
        if (!d_dftParamsPtr->reuseWfcGeoOpt)
          readPSI();

        noRemeshRhoDataInit();

        if (d_dftParamsPtr->reuseDensityGeoOpt >= 1 &&
            d_dftParamsPtr->solverMode == "GEOOPT")
          {
            if (d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
                d_dftParamsPtr->spinPolarized != 1)
              {
                d_rhoOutNodalValuesSplit.add(
                  -totalCharge(d_matrixFreeDataPRefined,
                               d_rhoOutNodalValuesSplit) /
                  d_domainVolume);

                initAtomicRho();

                interpolateDensityNodalDataToQuadratureDataGeneral(
                  d_basisOperationsPtrElectroHost,
                  d_densityDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_rhoOutNodalValuesSplit,
                  d_densityInQuadValues[0],
                  d_gradDensityInQuadValues[0],
                  d_gradDensityInQuadValues[0],
                  d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA);

                addAtomicRhoQuadValuesGradients(
                  d_densityInQuadValues[0],
                  d_gradDensityInQuadValues[0],
                  d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA);

                normalizeRhoInQuadValues();

                l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                        d_constraintsRhoNodal,
                                        d_densityDofHandlerIndexElectro,
                                        d_densityQuadratureIdElectro,
                                        d_densityInQuadValues[0],
                                        d_densityInNodalValues[0]);
              }
          }

        else if (d_dftParamsPtr->extrapolateDensity == 1 &&
                 d_dftParamsPtr->spinPolarized != 1 &&
                 d_dftParamsPtr->solverMode == "MD")
          {
            interpolateDensityNodalDataToQuadratureDataGeneral(
              d_basisOperationsPtrElectroHost,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_densityOutNodalValues[0],
              d_densityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_densityInQuadValues[0],
                                    d_densityInNodalValues[0]);
          }
        else if (d_dftParamsPtr->extrapolateDensity == 2 &&
                 d_dftParamsPtr->spinPolarized != 1 &&
                 d_dftParamsPtr->solverMode == "MD")
          {
            initAtomicRho();
            interpolateDensityNodalDataToQuadratureDataGeneral(
              d_basisOperationsPtrElectroHost,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_rhoOutNodalValuesSplit,
              d_densityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);

            addAtomicRhoQuadValuesGradients(
              d_densityInQuadValues[0],
              d_gradDensityInQuadValues[0],
              d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_densityInQuadValues[0],
                                    d_densityInNodalValues[0]);
          }
        else
          {
            initRho();
            if (d_dftParamsPtr->pawPseudoPotential)
              {
                if (d_dftParamsPtr->loadDijData)
                  {
                    d_pawClassPtr->loadDijEntriesFromFile();
                  }
                else
                  d_pawClassPtr->computeDijFromPSIinitialGuess(
                    &d_eigenVectorsFlattenedHost,
                    numElectrons,
                    d_numEigenValues,
                    d_densityQuadratureId,
                    d_kPointWeights,
                    interpoolcomm,
                    interBandGroupComm);
                double scaleFactor =
                  d_pawClassPtr->densityScalingFactor(atomLocations);
                const double charge =
                  totalCharge(d_dofHandlerRhoNodal, d_densityInQuadValues[0]);
                scaleRhoInQuadValues(scaleFactor / charge);
              }
          }
      }

    MPI_Barrier(d_mpiCommParent);
    init_rho = MPI_Wtime() - init_rho;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "updateAtomPositionsAndMoveMesh: Time taken for initRho: "
            << init_rho << std::endl;

    //
    // reinitialize pseudopotential related data structures
    //
    double init_pseudo;
    MPI_Barrier(d_mpiCommParent);
    init_pseudo = MPI_Wtime();

    initPseudoPotentialAll();

    MPI_Barrier(d_mpiCommParent);
    init_pseudo = MPI_Wtime() - init_pseudo;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Time taken for initPseudoPotentialAll: " << init_pseudo
            << std::endl;

    d_isFirstFilteringCall.clear();
    d_isFirstFilteringCall.resize((d_dftParamsPtr->spinPolarized + 1) *
                                    d_kPointWeights.size(),
                                  true);

    double init_ksoperator;
    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime();

    if (isMeshDeformed)
      initializeKohnShamDFTOperator();
    else
      reInitializeKohnShamDFTOperator();

    init_ksoperator = MPI_Wtime() - init_ksoperator;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Time taken for kohnShamDFTOperator class reinitialization: "
            << init_ksoperator << std::endl;

    computingTimerStandard.leave_subsection("KSDFT problem initialization");
  }

  //
  // deform domain and call appropriate reinits
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::deformDomain(
    const dealii::Tensor<2, 3, double> &deformationGradient,
    const bool                          vselfPerturbationUpdateForStress,
    const bool                          useSingleAtomSolutionsOverride,
    const bool                          print)
  {
    d_affineTransformMesh.initMoved(d_domainBoundingVectors);
    d_affineTransformMesh.transform(deformationGradient);

    dftUtils::transformDomainBoundingVectors(d_domainBoundingVectors,
                                             deformationGradient);

    if (print)
      {
        pcout
          << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
          << std::endl;
        for (int i = 0; i < d_domainBoundingVectors.size(); ++i)
          {
            pcout << "v" << i + 1 << " : " << d_domainBoundingVectors[i][0]
                  << " " << d_domainBoundingVectors[i][1] << " "
                  << d_domainBoundingVectors[i][2] << std::endl;
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

#ifdef USE_COMPLEX
    if (!vselfPerturbationUpdateForStress)
      recomputeKPointCoordinates();
#endif

    // update atomic and image positions without any wrapping across periodic
    // boundary
    std::vector<dealii::Tensor<1, 3, double>> imageDisplacements(
      d_imagePositions.size());
    std::vector<dealii::Tensor<1, 3, double>> imageDisplacementsTrunc(
      d_imagePositionsTrunc.size());

    for (int iImage = 0; iImage < d_imagePositions.size(); ++iImage)
      {
        dealii::Point<3>   imageCoor;
        const unsigned int imageChargeId = d_imageIds[iImage];
        imageCoor[0]                     = d_imagePositions[iImage][0];
        imageCoor[1]                     = d_imagePositions[iImage][1];
        imageCoor[2]                     = d_imagePositions[iImage][2];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacements[iImage] = imageCoor - atomCoor;
      }

    for (int iImage = 0; iImage < d_imagePositionsTrunc.size(); ++iImage)
      {
        dealii::Point<3>   imageCoor;
        const unsigned int imageChargeId = d_imageIdsTrunc[iImage];
        imageCoor[0]                     = d_imagePositionsTrunc[iImage][0];
        imageCoor[1]                     = d_imagePositionsTrunc[iImage][1];
        imageCoor[2]                     = d_imagePositionsTrunc[iImage][2];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacementsTrunc[iImage] = imageCoor - atomCoor;
      }

    for (unsigned int i = 0; i < atomLocations.size(); ++i)
      atomLocations[i] = atomLocationsFractional[i];

    if (print)
      {
        pcout << "-----Fractional coordinates of atoms------ " << std::endl;
        for (unsigned int i = 0; i < atomLocations.size(); ++i)
          {
            pcout << "AtomId " << i << ":  " << atomLocationsFractional[i][2]
                  << " " << atomLocationsFractional[i][3] << " "
                  << atomLocationsFractional[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

    internaldft::convertToCellCenteredCartesianCoordinates(
      atomLocations, d_domainBoundingVectors);


    for (int iImage = 0; iImage < d_imagePositions.size(); ++iImage)
      {
        const unsigned int imageChargeId = d_imageIds[iImage];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacements[iImage] =
          deformationGradient * imageDisplacements[iImage];

        d_imagePositions[iImage][0] =
          atomCoor[0] + imageDisplacements[iImage][0];
        d_imagePositions[iImage][1] =
          atomCoor[1] + imageDisplacements[iImage][1];
        d_imagePositions[iImage][2] =
          atomCoor[2] + imageDisplacements[iImage][2];
      }

    for (int iImage = 0; iImage < d_imagePositionsTrunc.size(); ++iImage)
      {
        const unsigned int imageChargeId = d_imageIdsTrunc[iImage];

        dealii::Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacementsTrunc[iImage] =
          deformationGradient * imageDisplacementsTrunc[iImage];

        d_imagePositionsTrunc[iImage][0] =
          atomCoor[0] + imageDisplacementsTrunc[iImage][0];
        d_imagePositionsTrunc[iImage][1] =
          atomCoor[1] + imageDisplacementsTrunc[iImage][1];
        d_imagePositionsTrunc[iImage][2] =
          atomCoor[2] + imageDisplacementsTrunc[iImage][2];
      }

    if (vselfPerturbationUpdateForStress)
      {
        //
        // reinitialize dirichlet BCs for total potential and vSelf poisson
        // solutions
        //
        double init_bc;
        MPI_Barrier(d_mpiCommParent);
        init_bc = MPI_Wtime();


        // first true option only updates the boundary conditions
        // second true option signals update is only for vself perturbation
        initBoundaryConditions(true, true, true);
        d_smearedChargeMomentsComputed = false;
        MPI_Barrier(d_mpiCommParent);
        init_bc = MPI_Wtime() - init_bc;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << "updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "
            << init_bc << std::endl;
      }
    else
      {
        initNoRemesh(false, true, useSingleAtomSolutionsOverride, true);
      }
  }


  //
  // generate a-posteriori mesh
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::aposterioriMeshGenerate()
  {
    //
    // get access to triangulation objects from meshGenerator class
    //
    dealii::parallel::distributed::Triangulation<3> &triangulationPar =
      d_mesh.getParallelMeshMoved();
    unsigned int numberLevelRefinements = d_dftParamsPtr->numLevels;
    unsigned int numberWaveFunctionsErrorEstimate =
      d_dftParamsPtr->numberWaveFunctionsForEstimate;
    bool         refineFlag = true;
    unsigned int countLevel = 0;
    double       traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
    double       traceXtKXPrev = traceXtKX;

    while (refineFlag)
      {
        if (numberLevelRefinements > 0)
          {
            distributedCPUVec<double> tempVec;
            matrix_free_data.initialize_dof_vector(tempVec);

            std::vector<distributedCPUVec<double>> eigenVectorsArray(
              numberWaveFunctionsErrorEstimate);

            for (unsigned int i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
              eigenVectorsArray[i].reinit(tempVec);


            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedHost.data(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              std::make_pair(0, numberWaveFunctionsErrorEstimate),
              eigenVectorsArray);


            for (unsigned int i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
              {
                constraintsNone.distribute(eigenVectorsArray[i]);
                eigenVectorsArray[i].update_ghost_values();
              }


            d_mesh.generateAutomaticMeshApriori(dofHandler,
                                                triangulationPar,
                                                eigenVectorsArray,
                                                FEOrder);
          }


        //
        // initialize dofHandlers of refined mesh and move triangulation
        //
        initUnmovedTriangulation(triangulationPar);
        moveMeshToAtoms(triangulationPar, d_mesh.getSerialMeshUnmoved());
        initBoundaryConditions();
        d_smearedChargeMomentsComputed = false;
        initElectronicFields();
        initPseudoPotentialAll();

        //
        // compute Tr(XtKX) for each level of mesh
        //
        traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
        if (d_dftParamsPtr->verbosity > 0)
          pcout << " Tr(XtKX) value for Level: " << countLevel << " "
                << traceXtKX << std::endl;

        // compute change in traceXtKX
        double deltaKinetic =
          std::abs(traceXtKX - traceXtKXPrev) / atomLocations.size();

        // reset traceXtkXPrev to traceXtKX
        traceXtKXPrev = traceXtKX;

        //
        // set refineFlag
        //
        countLevel += 1;
        if (countLevel >= numberLevelRefinements ||
            deltaKinetic <= d_dftParamsPtr->toleranceKinetic)
          refineFlag = false;
      }
  }


  //
  // dft run
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::run()
  {
    if (d_dftParamsPtr->meshAdaption)
      aposterioriMeshGenerate();

    if (d_dftParamsPtr->restartFolder != "." && d_dftParamsPtr->saveRhoData &&
        dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        mkdir(d_dftParamsPtr->restartFolder.c_str(), ACCESSPERMS);
      }

    if (d_dftParamsPtr->solverMode == "GS")
      {
        solve(true, true, d_isRestartGroundStateCalcFromChk);
        if (d_dftParamsPtr->writeBandsFile)
          writeBands();
      }
    else if (d_dftParamsPtr->solverMode == "NSCF")
      {
        solveNoSCF();
        if (d_dftParamsPtr->writeBandsFile)
          writeBands();
      }

    if (d_dftParamsPtr->writeStructreEnergyForcesFileForPostProcess)
      writeStructureEnergyForcesDataPostProcess(
        "structureEnergyForcesGSData.txt");

    if (d_dftParamsPtr->writeWfcSolutionFields)
      outputWfc();

    if (d_dftParamsPtr->writeDensitySolutionFields)
      outputDensity();

    if (d_dftParamsPtr->writeDensityQuadData)
      writeGSElectronDensity("densityQuadData.txt");

    if (d_dftParamsPtr->writeDosFile)
      compute_tdos(eigenValues,
                   d_dftParamsPtr->highestStateOfInterestForChebFiltering,
                   "dosData.out");

    if (d_dftParamsPtr->writeLdosFile)
      compute_ldos(eigenValues, "ldosData.out");

    if (d_dftParamsPtr->writePdosFile)
      compute_pdos(eigenValues, "pdosData");

    if (d_dftParamsPtr->writeLocalizationLengths)
      compute_localizationLength("localizationLengths.out");

    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << std::endl
        << "------------------DFT-FE ground-state solve completed---------------------------"
        << std::endl;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::trivialSolveForStress()
  {
    initBoundaryConditions();
    noRemeshRhoDataInit();
    solve(false, true);
  }


  //
  // initialize
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initializeKohnShamDFTOperator(
    const bool initializeCublas)
  {
    dealii::TimerOutput::Scope scope(computing_timer,
                                     "kohnShamDFTOperator init");
    double                     init_ksoperator;
    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime();

    if (d_kohnShamDFTOperatorsInitialized)
      finalizeKohnShamDFTOperator();

#ifdef DFTFE_WITH_DEVICE
    if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
      d_kohnShamDFTOperatorPtr = new KohnShamHamiltonianOperator<memorySpace>(
        d_BLASWrapperPtr,
        d_basisOperationsPtrDevice,
        d_basisOperationsPtrHost,
        d_oncvClassPtr,
        d_pawClassPtr,
        d_excManagerPtr,
        d_dftParamsPtr,
        d_densityQuadratureId,
        d_lpspQuadratureId,
        d_feOrderPlusOneQuadratureId,
        d_mpiCommParent,
        mpi_communicator);
    else
#endif
      d_kohnShamDFTOperatorPtr = new KohnShamHamiltonianOperator<memorySpace>(
        d_BLASWrapperPtrHost,
        d_basisOperationsPtrHost,
        d_basisOperationsPtrHost,
        d_oncvClassPtr,
        d_pawClassPtr,
        d_excManagerPtr,
        d_dftParamsPtr,
        d_densityQuadratureId,
        d_lpspQuadratureId,
        d_feOrderPlusOneQuadratureId,
        d_mpiCommParent,
        mpi_communicator);



    KohnShamHamiltonianOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    kohnShamDFTEigenOperator.init(d_kPointCoordinates, d_kPointWeights);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        AssertThrow(
          (d_numEigenValues % d_dftParamsPtr->chebyWfcBlockSize == 0 ||
           d_numEigenValues / d_dftParamsPtr->chebyWfcBlockSize == 0),
          dealii::ExcMessage(
            "DFT-FE Error: total number wavefunctions must be exactly divisible by cheby wfc block size for Device run."));


        AssertThrow(
          (d_numEigenValues % d_dftParamsPtr->wfcBlockSize == 0 ||
           d_numEigenValues / d_dftParamsPtr->wfcBlockSize == 0),
          dealii::ExcMessage(
            "DFT-FE Error: total number wavefunctions must be exactly divisible by wfc block size for Device run."));

        AssertThrow(
          (d_dftParamsPtr->wfcBlockSize % d_dftParamsPtr->chebyWfcBlockSize ==
             0 &&
           d_dftParamsPtr->wfcBlockSize / d_dftParamsPtr->chebyWfcBlockSize >=
             0),
          dealii::ExcMessage(
            "DFT-FE Error: wfc block size must be exactly divisible by cheby wfc block size and also larger for Device run."));

        if (d_numEigenValuesRR != d_numEigenValues)
          AssertThrow(
            (d_numEigenValuesRR % d_dftParamsPtr->wfcBlockSize == 0 ||
             d_numEigenValuesRR / d_dftParamsPtr->wfcBlockSize == 0),
            dealii::ExcMessage(
              "DFT-FE Error: total number RR wavefunctions must be exactly divisible by wfc block size for Device run."));

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

        AssertThrow(
          (d_numEigenValues % numberBandGroups == 0 ||
           d_numEigenValues / numberBandGroups == 0),
          dealii::ExcMessage(
            "DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for Device run."));

        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, d_numEigenValues, bandGroupLowHighPlusOneIndices);

        AssertThrow(
          (bandGroupLowHighPlusOneIndices[1] %
             d_dftParamsPtr->chebyWfcBlockSize ==
           0),
          dealii::ExcMessage(
            "DFT-FE Error: band parallelization group size must be exactly divisible by CHEBY WFC BLOCK SIZE for Device run."));

        AssertThrow(
          (bandGroupLowHighPlusOneIndices[1] % d_dftParamsPtr->wfcBlockSize ==
           0),
          dealii::ExcMessage(
            "DFT-FE Error: band parallelization group size must be exactly divisible by WFC BLOCK SIZE for Device run."));
      }
#endif


    d_kohnShamDFTOperatorsInitialized = true;

    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime() - init_ksoperator;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "init: Time taken for kohnShamDFTOperator class initialization: "
            << init_ksoperator << std::endl;
  }


  //
  // re-initialize (significantly cheaper than initialize)
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    reInitializeKohnShamDFTOperator()
  {
    d_kohnShamDFTOperatorPtr->resetExtPotHamFlag();
  }

  //
  // finalize
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::finalizeKohnShamDFTOperator()
  {
    if (d_kohnShamDFTOperatorsInitialized)
      {
        delete d_kohnShamDFTOperatorPtr;
      }
  }

  //
  // dft solve
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  std::tuple<bool, double>
  dftClass<FEOrder, FEOrderElectro, memorySpace>::solve(
    const bool computeForces,
    const bool computestress,
    const bool isRestartGroundStateCalcFromChk)
  {
    KohnShamHamiltonianOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);

    // computingTimerStandard.enter_subsection("Total scf solve");
    energyCalculator energyCalc(d_mpiCommParent,
                                mpi_communicator,
                                interpoolcomm,
                                interBandGroupComm,
                                *d_dftParamsPtr);


    // set up linear solver
    dealiiLinearSolver CGSolver(d_mpiCommParent,
                                mpi_communicator,
                                dealiiLinearSolver::CG);

    // set up linear solver Device
#ifdef DFTFE_WITH_DEVICE
    linearSolverCGDevice CGSolverDevice(d_mpiCommParent,
                                        mpi_communicator,
                                        linearSolverCGDevice::CG,
                                        d_BLASWrapperPtr);
#endif

    //
    // set up solver functions for Helmholtz to be used only when Kerker mixing
    // is on use higher polynomial order dofHandler
    //
    kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
      kerkerPreconditionedResidualSolverProblem(d_mpiCommParent,
                                                mpi_communicator);

    // set up solver functions for Helmholtz Device
#ifdef DFTFE_WITH_DEVICE
    kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
      kerkerPreconditionedResidualSolverProblemDevice(d_mpiCommParent,
                                                      mpi_communicator);
#endif

    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
      {
        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges)
          {
#ifdef DFTFE_WITH_DEVICE
            kerkerPreconditionedResidualSolverProblemDevice.init(
              d_basisOperationsPtrElectroDevice,
              d_constraintsRhoNodal,
              d_preCondTotalDensityResidualVector,
              d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ?
                d_dftParamsPtr->kerkerParameter :
                (d_dftParamsPtr->restaFermiWavevector / 4.0 / M_PI / 4.0 /
                 M_PI),
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro);
#endif
          }
        else
          kerkerPreconditionedResidualSolverProblem.init(
            d_basisOperationsPtrElectroHost,
            d_constraintsRhoNodal,
            d_preCondTotalDensityResidualVector,
            d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ?
              d_dftParamsPtr->kerkerParameter :
              (d_dftParamsPtr->restaFermiWavevector / 4.0 / M_PI / 4.0 / M_PI),
            d_densityDofHandlerIndexElectro,
            d_densityQuadratureIdElectro);
      }

    // FIXME: Check if this call can be removed
    d_phiTotalSolverProblem.clear();

    //
    // solve vself in bins
    //
    if (d_dftParamsPtr->isPseudopotential &&
        !d_dftParamsPtr->pawPseudoPotential)
      {
        computing_timer.enter_subsection("Nuclear self-potential solve");
        computingTimerStandard.enter_subsection("Nuclear self-potential solve");
#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->vselfGPU)
          d_vselfBinsManager.solveVselfInBinsDevice(
            d_basisOperationsPtrElectroHost,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
            FEOrder == FEOrderElectro ?
              d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
              d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
            d_BLASWrapperPtr,
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_localVselfs,
            d_bQuadValuesAllAtoms,
            d_bQuadAtomIdsAllAtoms,
            d_bQuadAtomIdsAllAtomsImages,
            d_bCellNonTrivialAtomIds,
            d_bCellNonTrivialAtomIdsBins,
            d_bCellNonTrivialAtomImageIds,
            d_bCellNonTrivialAtomImageIdsBins,
            d_smearedChargeWidths,
            d_smearedChargeScaling,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);
        else
          d_vselfBinsManager.solveVselfInBins(
            d_basisOperationsPtrElectroHost,
            d_binsStartDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_localVselfs,
            d_bQuadValuesAllAtoms,
            d_bQuadAtomIdsAllAtoms,
            d_bQuadAtomIdsAllAtomsImages,
            d_bCellNonTrivialAtomIds,
            d_bCellNonTrivialAtomIdsBins,
            d_bCellNonTrivialAtomImageIds,
            d_bCellNonTrivialAtomImageIdsBins,
            d_smearedChargeWidths,
            d_smearedChargeScaling,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);
#else
        d_vselfBinsManager.solveVselfInBins(
          d_basisOperationsPtrElectroHost,
          d_binsStartDofHandlerIndexElectro,
          d_phiTotAXQuadratureIdElectro,
          d_constraintsPRefined,
          d_imagePositionsTrunc,
          d_imageIdsTrunc,
          d_imageChargesTrunc,
          d_localVselfs,
          d_bQuadValuesAllAtoms,
          d_bQuadAtomIdsAllAtoms,
          d_bQuadAtomIdsAllAtomsImages,
          d_bCellNonTrivialAtomIds,
          d_bCellNonTrivialAtomIdsBins,
          d_bCellNonTrivialAtomImageIds,
          d_bCellNonTrivialAtomImageIdsBins,
          d_smearedChargeWidths,
          d_smearedChargeScaling,
          d_smearedChargeQuadratureIdElectro,
          d_dftParamsPtr->smearedNuclearCharges);
#endif
        computingTimerStandard.leave_subsection("Nuclear self-potential solve");
        computing_timer.leave_subsection("Nuclear self-potential solve");

        if ((d_dftParamsPtr->isPseudopotential &&
             !d_dftParamsPtr->pawPseudoPotential) ||
            (!d_dftParamsPtr->isPseudopotential &&
             d_dftParamsPtr->smearedNuclearCharges))
          {
            computingTimerStandard.enter_subsection("Init local PSP");
            initLocalPseudoPotential(d_dofHandlerPRefined,
                                     d_lpspQuadratureIdElectro,
                                     d_matrixFreeDataPRefined,
                                     d_phiExtDofHandlerIndexElectro,
                                     d_constraintsPRefinedOnlyHanging,
                                     d_supportPointsPRefined,
                                     d_vselfBinsManager,
                                     d_phiExt,
                                     d_pseudoVLoc,
                                     d_pseudoVLocAtoms);
            kohnShamDFTEigenOperator.computeVEffExternalPotCorr(d_pseudoVLoc);
            computingTimerStandard.leave_subsection("Init local PSP");
          }
      }
    else if (d_dftParamsPtr->isPseudopotential &&
             d_dftParamsPtr->pawPseudoPotential)
      {
        computingTimerStandard.enter_subsection("Init Zero Potential PAW");
        initZeroPotential();
        kohnShamDFTEigenOperator.computeVEffExternalPotCorr(d_zeroPotential);
        computingTimerStandard.leave_subsection("Init Zero Potential PAW");
      }

    computingTimerStandard.enter_subsection("Total scf solve");

    //
    // solve
    //
    computing_timer.enter_subsection("scf solve");
    double firstScfChebyTol =
      d_dftParamsPtr->restrictToOnePass ?
        1e+4 :
        (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
             d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA" ?
           1e-2 :
           2e-2);


    if (d_dftParamsPtr->solverMode == "MD")
      firstScfChebyTol = d_dftParamsPtr->chebyshevTolerance > 1e-4 ?
                           1e-4 :
                           d_dftParamsPtr->chebyshevTolerance;
    else if (d_dftParamsPtr->solverMode == "GEOOPT")
      firstScfChebyTol = d_dftParamsPtr->chebyshevTolerance > 1e-3 ?
                           1e-3 :
                           d_dftParamsPtr->chebyshevTolerance;

    // call the mixing scheme with the mixing variables
    // Have to be called once for each variable
    // initialise the variables in the mixing scheme
    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
        d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
      {
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          rhoNodalMassVec;
        computeRhoNodalMassVector(rhoNodalMassVec);
        d_mixingScheme.addMixingVariable(
          mixingVariable::rho,
          rhoNodalMassVec,
          true, // call MPI REDUCE while computing dot products
          d_dftParamsPtr->mixingParameter,
          d_dftParamsPtr->adaptAndersonMixingParameter);
        if (d_dftParamsPtr->spinPolarized == 1)
          d_mixingScheme.addMixingVariable(
            mixingVariable::magZ,
            rhoNodalMassVec,
            true, // call MPI REDUCE while computing dot products
            d_dftParamsPtr->mixingParameter *
              d_dftParamsPtr->spinMixingEnhancementFactor,
            d_dftParamsPtr->adaptAndersonMixingParameter);
        if (d_dftParamsPtr->pawPseudoPotential)
          {
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                Dij_MassVector;
            std::vector<double> Weights = d_pawClassPtr->getDijWeights();
            Dij_MassVector.resize(Weights.size());
            Dij_MassVector.copyFrom(Weights);
            d_mixingScheme.addMixingVariable(
              mixingVariable::DijMatrix,
              Dij_MassVector,
              true, // call MPI REDUCE while computing dot products
              d_dftParamsPtr->mixingParameter,
              d_dftParamsPtr->adaptAndersonMixingParameter);
          }
      }
    else if (d_dftParamsPtr->mixingMethod == "ANDERSON")
      {
        d_basisOperationsPtrElectroHost->reinit(0,
                                                0,
                                                d_densityQuadratureIdElectro,
                                                false);
        // d_mixingScheme.addMixingVariable(
        //   mixingVariable::rho,
        //   d_basisOperationsPtrElectroHost->JxWBasisData(),
        //   true, // call MPI REDUCE while computing dot products
        //   d_dftParamsPtr->mixingParameter,
        //   d_dftParamsPtr->adaptAndersonMixingParameter);
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          rhoNodalMassVec;
        computeRhoNodalMassVector(rhoNodalMassVec);
        d_mixingScheme.addMixingVariable(
          mixingVariable::rho,
          rhoNodalMassVec,
          true, // call MPI REDUCE while computing dot products
          d_dftParamsPtr->mixingParameter,
          d_dftParamsPtr->adaptAndersonMixingParameter);

        if (d_dftParamsPtr->spinPolarized == 1)
          d_mixingScheme.addMixingVariable(
            mixingVariable::magZ,
            d_basisOperationsPtrElectroHost->JxWBasisData(),
            true, // call MPI REDUCE while computing dot products
            d_dftParamsPtr->mixingParameter *
              d_dftParamsPtr->spinMixingEnhancementFactor,
            d_dftParamsPtr->adaptAndersonMixingParameter);
        if (d_dftParamsPtr->pawPseudoPotential)
          {
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                                Dij_MassVector;
            std::vector<double> Weights = d_pawClassPtr->getDijWeights();
            Dij_MassVector.resize(Weights.size());
            Dij_MassVector.copyFrom(Weights);
            d_mixingScheme.addMixingVariable(
              mixingVariable::DijMatrix,
              Dij_MassVector,
              true, // call MPI REDUCE while computing dot products
              d_dftParamsPtr->mixingParameter,
              d_dftParamsPtr->adaptAndersonMixingParameter);
            d_mixingScheme.addMixingVariable(
              mixingVariable::totalChargeDensity,
              rhoNodalMassVec,
              true, // call MPI REDUCE while computing dot products
              d_dftParamsPtr->mixingParameter,
              d_dftParamsPtr->adaptAndersonMixingParameter);
          }
        if (d_dftParamsPtr->useGradPhiMixing)
          {
            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              gradPhiWeights;
            gradPhiWeights.resize(
              d_basisOperationsPtrElectroHost->JxWBasisData().size() * 3);
            for (unsigned int i = 0; i < gradPhiWeights.size(); ++i)
              gradPhiWeights[i] =
                d_basisOperationsPtrElectroHost->JxWBasisData()[i / 3];
            d_mixingScheme.addMixingVariable(
              mixingVariable::gradPhi,
              gradPhiWeights,
              true, // call MPI REDUCE while computing dot products
              d_dftParamsPtr->mixingParameter,
              d_dftParamsPtr->adaptAndersonMixingParameter);
          }
      }
    //
    // Begin SCF iteration
    //
    unsigned int scfIter                  = 0;
    double       norm                     = 1.0;
    d_rankCurrentLRD                      = 0;
    d_relativeErrorJacInvApproxPrevScfLRD = 100.0;
    // CAUTION: Choosing a looser tolerance might lead to failed tests
    const double adaptiveChebysevFilterPassesTol =
      d_dftParamsPtr->chebyshevTolerance;
    bool scfConverged = false;
    pcout << std::endl;
    if (d_dftParamsPtr->verbosity == 0)
      pcout << "Starting SCF iterations...." << std::endl;

    while ((norm > d_dftParamsPtr->selfConsistentSolverTolerance) &&
           (scfIter < d_dftParamsPtr->numSCFIterations))
      {
        dealii::Timer local_timer(d_mpiCommParent, true);
        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************"
            << std::endl;
        //
        // Mixing scheme
        //
        computing_timer.enter_subsection("density mixing");
        if (scfIter > 0)
          {
            if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  norm =
                    lowrankApproxScfDielectricMatrixInvSpinPolarized(scfIter);
                else
                  norm = lowrankApproxScfDielectricMatrixInv(scfIter);
                if (d_dftParamsPtr->verbosity >= 1)
                  pcout << d_dftParamsPtr->mixingMethod
                        << " mixing, L2 norm of electron-density difference: "
                        << norm << std::endl;
              }
            else if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ||
                     d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
              {
                // Fill in New Kerker framework here
                std::vector<double> norms(
                  d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
                if (scfIter == 1)
                  d_densityResidualNodalValues.resize(
                    d_densityOutNodalValues.size());
                for (unsigned int iComp = 0;
                     iComp < d_densityOutNodalValues.size();
                     ++iComp)
                  {
                    norms[iComp] = computeResidualNodalData(
                      d_densityOutNodalValues[iComp],
                      d_densityInNodalValues[iComp],
                      d_densityResidualNodalValues[iComp]);
                  }
                applyKerkerPreconditionerToTotalDensityResidual(
#ifdef DFTFE_WITH_DEVICE
                  kerkerPreconditionedResidualSolverProblemDevice,
                  CGSolverDevice,
#endif
                  kerkerPreconditionedResidualSolverProblem,
                  CGSolver,
                  d_densityResidualNodalValues[0],
                  d_preCondTotalDensityResidualVector);
                d_mixingScheme.addVariableToInHist(
                  mixingVariable::rho,
                  d_densityInNodalValues[0].begin(),
                  d_densityInNodalValues[0].locally_owned_size());

                d_mixingScheme.addVariableToResidualHist(
                  mixingVariable::rho,
                  d_preCondTotalDensityResidualVector.begin(),
                  d_preCondTotalDensityResidualVector.locally_owned_size());
                if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    d_mixingScheme.addVariableToInHist(
                      mixingVariable::magZ,
                      d_densityInNodalValues[1].begin(),
                      d_densityInNodalValues[1].locally_owned_size());
                    d_mixingScheme.addVariableToResidualHist(
                      mixingVariable::magZ,
                      d_densityResidualNodalValues[1].begin(),
                      d_densityResidualNodalValues[1].locally_owned_size());
                  }
                if (d_dftParamsPtr->pawPseudoPotential)
                  {
                    std::vector<double> Dij_in =
                      d_pawClassPtr->DijVectorForMixing(TypeOfField::In);
                    std::vector<double> Dij_res =
                      d_pawClassPtr->DijVectorForMixing(TypeOfField::Residual);
                    d_mixingScheme.addVariableToInHist(
                      mixingVariable::DijMatrix, Dij_in.data(), Dij_in.size());
                    d_mixingScheme.addVariableToResidualHist(
                      mixingVariable::DijMatrix,
                      Dij_res.data(),
                      Dij_res.size());
                  }
                // Delete old history if it exceeds a pre-described
                // length
                d_mixingScheme.popOldHistory(d_dftParamsPtr->mixingHistory);

                // Compute the mixing coefficients
                if (!d_dftParamsPtr->pawPseudoPotential)
                  d_mixingScheme.computeAndersonMixingCoeff(
                    d_dftParamsPtr->spinPolarized == 1 ?
                      std::vector<mixingVariable>{mixingVariable::rho,
                                                  mixingVariable::magZ} :
                      std::vector<mixingVariable>{mixingVariable::rho});
                else
                  d_mixingScheme.computeAndersonMixingCoeff(
                    d_dftParamsPtr->spinPolarized == 1 ?
                      std::vector<mixingVariable>{mixingVariable::rho,
                                                  mixingVariable::magZ,
                                                  mixingVariable::DijMatrix} :
                      std::vector<mixingVariable>{mixingVariable::rho,
                                                  mixingVariable::DijMatrix});
                for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                  d_mixingScheme.mixVariable(
                    iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                    d_densityInNodalValues[iComp].begin(),
                    d_densityInNodalValues[iComp].locally_owned_size());
                norm = 0.0;
                for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                  norm += norms[iComp] * norms[iComp];
                norm = std::sqrt(norm / ((double)norms.size()));
                // interpolate nodal data to quadrature data
                if (d_dftParamsPtr->verbosity >= 1)
                  for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                    pcout << d_dftParamsPtr->mixingMethod
                          << " mixing, L2 norm of "
                          << (iComp == 0 ? "electron" : "magnetization")
                          << "-density difference: " << norms[iComp]
                          << std::endl;
                for (unsigned int iComp = 0;
                     iComp < d_densityInNodalValues.size();
                     ++iComp)
                  {
                    interpolateDensityNodalDataToQuadratureDataGeneral(
                      d_basisOperationsPtrElectroHost,
                      d_densityDofHandlerIndexElectro,
                      d_densityQuadratureIdElectro,
                      d_densityInNodalValues[iComp],
                      d_densityInQuadValues[iComp],
                      d_gradDensityInQuadValues[iComp],
                      d_gradDensityInQuadValues[iComp],
                      d_excManagerPtr->getDensityBasedFamilyType() ==
                        densityFamilyType::GGA);
                  }
                if (d_dftParamsPtr->pawPseudoPotential)
                  {
                    std::vector<double> Dij_in =
                      d_pawClassPtr->DijVectorForMixing(TypeOfField::In);
                    d_mixingScheme.mixVariable(mixingVariable::DijMatrix,
                                               Dij_in.data(),
                                               Dij_in.size());
                    d_pawClassPtr->fillDijMatrix(TypeOfField::In,
                                                 Dij_in,
                                                 interpoolcomm,
                                                 interBandGroupComm);
                  }
              }
            else if (d_dftParamsPtr->mixingMethod == "ANDERSON")
              {
                std::vector<double> norms(
                  d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
                double              normGradPhi = 0.0;
                double              normDij     = 0.0;
                std::vector<double> normsTotalCharge(1);
                if (scfIter == 1)
                  d_densityResidualNodalValues.resize(
                    d_densityOutNodalValues.size());
                for (unsigned int iComp = 0;
                     iComp < d_densityOutNodalValues.size();
                     ++iComp)
                  {
                    norms[iComp] = computeResidualNodalData(
                      d_densityOutNodalValues[iComp],
                      d_densityInNodalValues[iComp],
                      d_densityResidualNodalValues[iComp]);
                  }
                d_mixingScheme.addVariableToInHist(
                  mixingVariable::rho,
                  d_densityInNodalValues[0].begin(),
                  d_densityInNodalValues[0].locally_owned_size());
                d_mixingScheme.addVariableToResidualHist(
                  mixingVariable::rho,
                  d_densityResidualNodalValues[0].begin(),
                  d_densityResidualNodalValues[0].locally_owned_size());
                if (d_dftParamsPtr->pawPseudoPotential)
                  {
                    if (scfIter == 1)
                      d_totalChargeDensityResidualNodalValues.resize(
                        d_totalChargeDensityInNodalValues.size());
                    for (unsigned int iComp = 0;
                         iComp < d_totalChargeDensityInNodalValues.size();
                         ++iComp)
                      {
                        normsTotalCharge[iComp] = computeResidualNodalData(
                          d_totalChargeDensityOutNodalValues[iComp],
                          d_totalChargeDensityInNodalValues[iComp],
                          d_totalChargeDensityResidualNodalValues[iComp]);
                      }
                    d_mixingScheme.addVariableToInHist(
                      mixingVariable::totalChargeDensity,
                      d_totalChargeDensityInNodalValues[0].begin(),
                      d_totalChargeDensityInNodalValues[0]
                        .locally_owned_size());
                    d_mixingScheme.addVariableToResidualHist(
                      mixingVariable::totalChargeDensity,
                      d_totalChargeDensityResidualNodalValues[0].begin(),
                      d_totalChargeDensityResidualNodalValues[0]
                        .locally_owned_size());
                    if (d_dftParamsPtr->useGradPhiMixing)
                      {
                        if (scfIter == 1)
                          d_gradPhiResQuadValues.resize(
                            d_gradPhiInQuadValues.size());
                        d_basisOperationsPtrElectroHost->reinit(
                          0, 0, d_densityQuadratureIdElectro, false);
                        dftfe::utils::
                          MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                            gradPhiWeights;
                        gradPhiWeights.resize(
                          d_basisOperationsPtrElectroHost->JxWBasisData()
                            .size() *
                          3);
                        for (unsigned int i = 0; i < gradPhiWeights.size(); ++i)
                          gradPhiWeights[i] = d_basisOperationsPtrElectroHost
                                                ->JxWBasisData()[i / 3];
                        normGradPhi =
                          computeResidualQuadData(d_gradPhiOutQuadValues,
                                                  d_gradPhiInQuadValues,
                                                  d_gradPhiResQuadValues,
                                                  gradPhiWeights,
                                                  true);

                        d_mixingScheme.addVariableToInHist(
                          mixingVariable::gradPhi,
                          d_gradPhiInQuadValues.data(),
                          d_gradPhiInQuadValues.size());
                        d_mixingScheme.addVariableToResidualHist(
                          mixingVariable::gradPhi,
                          d_gradPhiResQuadValues.data(),
                          d_gradPhiResQuadValues.size());
                      }
                    std::vector<double> Dij_in =
                      d_pawClassPtr->DijVectorForMixing(TypeOfField::In);
                    std::vector<double> Dij_res =
                      d_pawClassPtr->DijVectorForMixing(TypeOfField::Residual);
                    normDij = d_pawClassPtr->computeNormDij(Dij_res);
                    d_mixingScheme.addVariableToInHist(
                      mixingVariable::DijMatrix, Dij_in.data(), Dij_in.size());
                    d_mixingScheme.addVariableToResidualHist(
                      mixingVariable::DijMatrix,
                      Dij_res.data(),
                      Dij_res.size());
                  }
                // Delete old history if it exceeds a pre-described
                // length
                d_mixingScheme.popOldHistory(d_dftParamsPtr->mixingHistory);

                // Compute the mixing coefficients
                if (!d_dftParamsPtr->pawPseudoPotential)
                  {
                    d_mixingScheme.computeAndersonMixingCoeff(
                      d_dftParamsPtr->spinPolarized == 1 ?
                        std::vector<mixingVariable>{mixingVariable::rho,
                                                    mixingVariable::magZ} :
                        std::vector<mixingVariable>{mixingVariable::rho});
                  }
                else
                  {
                    if (d_dftParamsPtr->useGradPhiMixing)
                      d_mixingScheme.computeAndersonMixingCoeff(
                        std::vector<mixingVariable>{
                          mixingVariable::gradPhi,
                          mixingVariable::totalChargeDensity});
                    else
                      d_mixingScheme.computeAndersonMixingCoeff(
                        d_dftParamsPtr->spinPolarized == 1 ?
                          std::vector<mixingVariable>{
                            mixingVariable::totalChargeDensity,
                            mixingVariable::magZ} :
                          std::vector<mixingVariable>{
                            mixingVariable::totalChargeDensity});
                  }

                // update the mixing variables
                // for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                //   d_mixingScheme.mixVariable(
                //     iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                //     d_densityInQuadValues[iComp].data(),
                //     d_densityInQuadValues[iComp].size());
                for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                  d_mixingScheme.mixVariable(
                    iComp == 0 ? mixingVariable::rho : mixingVariable::magZ,
                    d_densityInNodalValues[iComp].begin(),
                    d_densityInNodalValues[iComp].locally_owned_size());
                norm = 0.0;
                for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                  norm += norms[iComp] * norms[iComp];


                norm = std::sqrt(norm / ((double)norms.size()));
                // if (d_excManagerPtr->getDensityBasedFamilyType() ==
                //     densityFamilyType::GGA)
                //   {
                //     for (unsigned int iComp = 0; iComp < norms.size();
                //     ++iComp)
                //       d_mixingScheme.mixVariable(
                //         iComp == 0 ? mixingVariable::gradRho :
                //                      mixingVariable::gradMagZ,
                //         d_gradDensityInQuadValues[iComp].data(),
                //         d_gradDensityInQuadValues[iComp].size());
                //   }
                if (d_dftParamsPtr->pawPseudoPotential)
                  {
                    std::vector<double> Dij_in =
                      d_pawClassPtr->DijVectorForMixing(TypeOfField::In);
                    d_mixingScheme.mixVariable(mixingVariable::DijMatrix,
                                               Dij_in.data(),
                                               Dij_in.size());
                    d_pawClassPtr->fillDijMatrix(TypeOfField::In,
                                                 Dij_in,
                                                 interpoolcomm,
                                                 interBandGroupComm);
                    d_mixingScheme.mixVariable(
                      mixingVariable::totalChargeDensity,
                      d_totalChargeDensityInNodalValues[0].begin(),
                      d_totalChargeDensityInNodalValues[0]
                        .locally_owned_size());
                  }
                if (d_dftParamsPtr->verbosity >= 1)
                  {
                    for (unsigned int iComp = 0; iComp < norms.size(); ++iComp)
                      pcout << d_dftParamsPtr->mixingMethod
                            << " mixing, L2 norm of "
                            << (iComp == 0 ? "electron" : "magnetization")
                            << "-density difference: " << norms[iComp]
                            << std::endl;
                    if (d_dftParamsPtr->pawPseudoPotential)
                      {
                        pcout << d_dftParamsPtr->mixingMethod
                              << " mixing, L2 norm of "
                              << "Dij matrix"
                              << "difference: " << normDij << std::endl;
                        pcout << d_dftParamsPtr->mixingMethod
                              << " mixing, L2 norm of "
                              << "total Charge Density"
                              << "difference: " << normsTotalCharge[0]
                              << std::endl;
                      }
                    if (d_dftParamsPtr->useGradPhiMixing)
                      pcout << d_dftParamsPtr->mixingMethod
                            << " mixing, L2 norm of "
                            << "gradPhi"
                            << "difference: " << (normGradPhi) << std::endl;
                  }
                for (unsigned int iComp = 0;
                     iComp < d_densityInNodalValues.size();
                     ++iComp)
                  {
                    interpolateDensityNodalDataToQuadratureDataGeneral(
                      d_basisOperationsPtrElectroHost,
                      d_densityDofHandlerIndexElectro,
                      d_densityQuadratureIdElectro,
                      d_densityInNodalValues[iComp],
                      d_densityInQuadValues[iComp],
                      d_gradDensityInQuadValues[iComp],
                      d_gradDensityInQuadValues[iComp],
                      d_excManagerPtr->getDensityBasedFamilyType() ==
                        densityFamilyType::GGA);
                  }
              }

            if (d_dftParamsPtr->verbosity >= 1 &&
                d_dftParamsPtr->spinPolarized == 1)
              pcout << d_dftParamsPtr->mixingMethod
                    << " mixing, L2 norm of total density difference: " << norm
                    << std::endl;
          }

        if (d_dftParamsPtr->computeEnergyEverySCF &&
            d_numEigenValuesRR == d_numEigenValues)
          d_phiTotRhoIn = d_phiTotRhoOut;
        computing_timer.leave_subsection("density mixing");

        if (!(norm > d_dftParamsPtr->selfConsistentSolverTolerance))
          scfConverged = true;

        if (d_dftParamsPtr->multipoleBoundaryConditions)
          {
            computing_timer.enter_subsection("Update inhomogenous BC");
            computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                    d_densityQuadratureIdElectro,
                                    d_densityInQuadValues[0],
                                    &d_bQuadValuesAllAtoms);
            updatePRefinedConstraints();
            computing_timer.leave_subsection("Update inhomogenous BC");
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          densityInQuadValuesCopy = d_densityInQuadValues[0];
        if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
            (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
             d_dftParamsPtr->periodicZ))
          {
            double *tempvec = densityInQuadValuesCopy.data();
            for (unsigned int iquad = 0; iquad < densityInQuadValuesCopy.size();
                 iquad++)
              tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
          }
        //
        // phiTot with rhoIn
        //
        if (d_dftParamsPtr->isPseudopotential &&
            d_dftParamsPtr->pawPseudoPotential)
          {
            if (d_dftParamsPtr->memoryOptCompCharge)
              d_pawClassPtr->computeCompensationChargeMemoryOpt(
                TypeOfField::In);
            else
              d_pawClassPtr->computeCompensationCharge(TypeOfField::In);
            if (d_dftParamsPtr->verbosity >= 4 || scfIter == 0)
              d_pawClassPtr->chargeNeutrality(
                totalCharge(d_dofHandlerRhoNodal, d_densityInQuadValues[0]),
                TypeOfField::In,
                false);
          }
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << std::endl
            << "Poisson solve for total electrostatic potential (rhoIn+b): ";
        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
          {
#ifdef DFTFE_WITH_DEVICE
            if (scfIter > 0)
              d_phiTotalSolverProblemDevice.reinit(
                d_basisOperationsPtrElectroHost,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                densityInQuadValuesCopy,
                d_BLASWrapperPtr,
                false,
                false,
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                false,
                true,
                d_dftParamsPtr->multipoleBoundaryConditions);
            else
              {
                d_phiTotalSolverProblemDevice.reinit(
                  d_basisOperationsPtrElectroHost, // basisOperationsElectro
                  d_phiTotRhoIn,                   // Phi(x)
                  *d_constraintsVectorElectro
                    [d_phiTotDofHandlerIndexElectro], // Constraint Matrix
                  d_phiTotDofHandlerIndexElectro,     // DofHandler
                  d_densityQuadratureIdElectro,       // DensityQuadrature Rule
                  d_phiTotAXQuadratureIdElectro,      // AX quadrature Rule
                  d_atomNodeIdToChargeMap,            //...
                  d_bQuadValuesAllAtoms, // Compensation charge Quad
                  d_smearedChargeQuadratureIdElectro, // Compensation charge
                                                      // Quad ID
                  densityInQuadValuesCopy,            // RhoIn
                  d_BLASWrapperPtr,
                  true,
                  d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                    d_dftParamsPtr->periodicZ &&
                    !d_dftParamsPtr->pinnedNodeForPBC,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  true,
                  false,
                  d_dftParamsPtr->multipoleBoundaryConditions);
              }
#endif
          }
        else
          {
            if (d_dftParamsPtr->pawPseudoPotential)
              {
                if (scfIter > 0)
                  d_phiTotalSolverProblem.reinit(
                    d_basisOperationsPtrElectroHost, // basisOperatorHost
                    d_phiTotRhoIn,                   // phi(x)
                    *d_constraintsVectorElectro
                      [d_phiTotDofHandlerIndexElectro], // constraintMatrix
                    d_phiTotDofHandlerIndexElectro,     // DofHandlerId
                    d_densityQuadratureIdElectro,       // DensityQuadID
                    d_phiTotAXQuadratureIdElectro,      // AXQuadID
                    d_atomNodeIdToChargeMap,            // atoms
                    d_bQuadValuesAllAtoms, // COmpensationChargeQuad
                    d_smearedChargeQuadratureIdElectro, // CompensationCharge ID
                    densityInQuadValuesCopy,            // Rho(x)
                    d_dftParamsPtr->nonLinearCoreCorrection, //
                    d_rhoCore,                               //
                    false,                                   // computeDiagonalA
                    false, // MeanValueContraint
                    d_dftParamsPtr
                      ->smearedNuclearCharges, // smearedNuclearCharges
                    true,                      // isRhoValues
                    false,                     // isGradSmeraedChargeRhs
                    0,                         // smeardChargeGradinetCompnoetId
                    false,                     // storesmeared
                    false,                     // reuseSmearedChargeRhs
                    d_dftParamsPtr->multipoleBoundaryConditions);
                else
                  d_phiTotalSolverProblem.reinit(
                    d_basisOperationsPtrElectroHost,
                    d_phiTotRhoIn,
                    *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                    d_phiTotDofHandlerIndexElectro,
                    d_densityQuadratureIdElectro,
                    d_phiTotAXQuadratureIdElectro,
                    d_atomNodeIdToChargeMap,
                    d_bQuadValuesAllAtoms,
                    d_smearedChargeQuadratureIdElectro,
                    densityInQuadValuesCopy,
                    d_dftParamsPtr->nonLinearCoreCorrection, //
                    d_rhoCore,                               //
                    true,
                    d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                      d_dftParamsPtr->periodicZ &&
                      !d_dftParamsPtr->pinnedNodeForPBC,
                    d_dftParamsPtr->smearedNuclearCharges,
                    true,
                    false,
                    0,
                    false,
                    false,
                    d_dftParamsPtr->multipoleBoundaryConditions);
              }
            else
              {
                if (scfIter > 0)
                  d_phiTotalSolverProblem.reinit(
                    d_basisOperationsPtrElectroHost, // basisOperatorHost
                    d_phiTotRhoIn,                   // phi(x)
                    *d_constraintsVectorElectro
                      [d_phiTotDofHandlerIndexElectro], // constraintMatrix
                    d_phiTotDofHandlerIndexElectro,     // DofHandlerId
                    d_densityQuadratureIdElectro,       // DensityQuadID
                    d_phiTotAXQuadratureIdElectro,      // AXQuadID
                    d_atomNodeIdToChargeMap,            // atoms
                    d_bQuadValuesAllAtoms, // COmpensationChargeQuad
                    d_smearedChargeQuadratureIdElectro, // CompensationCharge ID
                    densityInQuadValuesCopy,            // Rho(x)
                    false,
                    false,
                    d_dftParamsPtr->smearedNuclearCharges,
                    true,
                    false,
                    0,
                    false,
                    true,
                    d_dftParamsPtr->multipoleBoundaryConditions);
                else
                  d_phiTotalSolverProblem.reinit(
                    d_basisOperationsPtrElectroHost,
                    d_phiTotRhoIn,
                    *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                    d_phiTotDofHandlerIndexElectro,
                    d_densityQuadratureIdElectro,
                    d_phiTotAXQuadratureIdElectro,
                    d_atomNodeIdToChargeMap,
                    d_bQuadValuesAllAtoms,
                    d_smearedChargeQuadratureIdElectro,
                    densityInQuadValuesCopy,
                    true,
                    d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                      d_dftParamsPtr->periodicZ &&
                      !d_dftParamsPtr->pinnedNodeForPBC,
                    d_dftParamsPtr->smearedNuclearCharges,
                    true,
                    false,
                    0,
                    true,
                    false,
                    d_dftParamsPtr->multipoleBoundaryConditions);
              }
          }

        computing_timer.enter_subsection("phiTot solve");

        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
          {
#ifdef DFTFE_WITH_DEVICE
            CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                                 d_dftParamsPtr->absLinearSolverTolerance,
                                 d_dftParamsPtr->maxLinearSolverIterations,
                                 d_dftParamsPtr->verbosity);
#endif
          }
        else
          {
            CGSolver.solve(d_phiTotalSolverProblem,
                           d_dftParamsPtr->absLinearSolverTolerance,
                           d_dftParamsPtr->maxLinearSolverIterations,
                           d_dftParamsPtr->verbosity);
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          dummy;
        if (!d_dftParamsPtr->pawPseudoPotential)
          interpolateElectroNodalDataToQuadratureDataGeneral(
            d_basisOperationsPtrElectroHost,
            d_phiTotDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_phiTotRhoIn,
            d_phiInQuadValues,
            dummy);
        else
          interpolateElectroNodalDataToQuadratureDataGeneral(
            d_basisOperationsPtrElectroHost,
            d_phiTotDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_phiTotRhoIn,
            d_phiInQuadValues,
            d_gradPhiInQuadValues,
            true);
        // interpolateElectroNodalDataToQuadratureDataGeneral(
        //   d_basisOperationsPtrElectroHost,
        //   d_phiTotDofHandlerIndexElectro,
        //   d_smearedChargeQuadratureIdElectro,
        //   d_phiTotRhoIn,
        //   d_phitTotQuadPointsCompensation,
        //   dummy);


        computing_timer.leave_subsection("phiTot solve");

        unsigned int numberChebyshevSolvePasses = 0;
        //
        // eigen solve
        //
        if (d_dftParamsPtr->spinPolarized == 1)
          {
            std::vector<std::vector<std::vector<double>>> eigenValuesSpins(
              2,
              std::vector<std::vector<double>>(
                d_kPointWeights.size(),
                std::vector<double>(
                  (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                   scfConverged) ?
                    d_numEigenValues :
                    d_numEigenValuesRR)));

            std::vector<std::vector<std::vector<double>>>
              residualNormWaveFunctionsAllkPointsSpins(
                2,
                std::vector<std::vector<double>>(
                  d_kPointWeights.size(),
                  std::vector<double>(
                    (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                     scfConverged) ?
                      d_numEigenValues :
                      d_numEigenValuesRR)));

            for (unsigned int s = 0; s < 2; ++s)
              {
                computing_timer.enter_subsection("VEff Computation");
                kohnShamDFTEigenOperator.computeVEff(d_densityInQuadValues,
                                                     d_gradDensityInQuadValues,
                                                     d_phiInQuadValues,
                                                     d_rhoCore,
                                                     d_gradRhoCore,
                                                     s);
                computing_timer.leave_subsection("VEff Computation");

                if (d_dftParamsPtr->isPseudopotential &&
                    d_dftParamsPtr->pawPseudoPotential)
                  {
                    computing_timer.enter_subsection(
                      "Computing Non-Local Hamiltonian Entries");
                    computing_timer.enter_subsection(
                      "Computing Non-Local XC Term Entries");
                    d_pawClassPtr
                      ->initialiseExchangeCorrelationEnergyCorrection(s);
                    computing_timer.leave_subsection(
                      "Computing Non-Local XC Term Entries");
                    computing_timer.enter_subsection(
                      "Computing Non-Local Electrostatics Term Entries");
                    if (s == 0)
                      d_pawClassPtr
                        ->evaluateNonLocalHamiltonianElectrostaticsValue(
                          d_phiTotRhoIn, d_phiTotDofHandlerIndexElectro);
                    // if (s == 0)
                    // d_pawClassPtr
                    //   ->evaluateNonLocalHamiltonianElectrostaticsValue(
                    //     d_phitTotQuadPointsCompensation,
                    //     d_phiTotDofHandlerIndexElectro);
                    computing_timer.leave_subsection(
                      "Computing Non-Local Electrostatics Term Entries");
                    d_pawClassPtr->computeNonlocalPseudoPotentialConstants(
                      CouplingType::HamiltonianEntries, s);
                    computing_timer.leave_subsection(
                      "Computing Non-Local Hamiltonian Entries");
                  }


                for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                     ++kPoint)
                  {
                    kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);



                    computing_timer.enter_subsection(
                      "Hamiltonian Matrix Computation");
                    kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
                    computing_timer.leave_subsection(
                      "Hamiltonian Matrix Computation");


                    for (unsigned int j = 0; j < 1; ++j)
                      {
                        if (d_dftParamsPtr->verbosity >= 2)
                          {
                            pcout << "Beginning Chebyshev filter pass " << j + 1
                                  << " for spin " << s + 1 << std::endl;
                          }
                        if (d_dftParamsPtr->verbosity >= 5)
                          computeTraceXtHX(d_numEigenValues, kPoint);

#ifdef DFTFE_WITH_DEVICE
                        if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                      memorySpace)
                          kohnShamEigenSpaceCompute(
                            s,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolverDevice,
                            residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                            (scfIter == 0 ||
                             d_dftParamsPtr
                               ->allowMultipleFilteringPassesAfterFirstScf) ?
                              true :
                              false,
                            0,
                            (scfIter <
                               d_dftParamsPtr->spectrumSplitStartingScfIter ||
                             scfConverged) ?
                              false :
                              true,
                            scfConverged ? false : true,
                            scfIter == 0);
#endif
                        if constexpr (dftfe::utils::MemorySpace::HOST ==
                                      memorySpace)
                          kohnShamEigenSpaceCompute(
                            s,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolver,
                            residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                            (scfIter == 0 ||
                             d_dftParamsPtr
                               ->allowMultipleFilteringPassesAfterFirstScf) ?
                              true :
                              false,
                            (scfIter <
                               d_dftParamsPtr->spectrumSplitStartingScfIter ||
                             scfConverged) ?
                              false :
                              true,
                            scfConverged ? false : true,
                            scfIter == 0);
                      }
                  }
              }


            for (unsigned int s = 0; s < 2; ++s)
              for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                   ++kPoint)
                {
                  if (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                      scfConverged)
                    for (unsigned int i = 0; i < d_numEigenValues; ++i)
                      eigenValuesSpins[s][kPoint][i] =
                        eigenValues[kPoint][d_numEigenValues * s + i];
                  else
                    for (unsigned int i = 0; i < d_numEigenValuesRR; ++i)
                      eigenValuesSpins[s][kPoint][i] =
                        eigenValuesRRSplit[kPoint][d_numEigenValuesRR * s + i];
                }
            //
            // fermi energy
            //
            if (d_dftParamsPtr->constraintMagnetization)
              compute_fermienergy_constraintMagnetization(eigenValues);
            else
              compute_fermienergy(eigenValues, numElectrons);

            unsigned int count = 1;

            if (!scfConverged &&
                (scfIter == 0 ||
                 d_dftParamsPtr->allowMultipleFilteringPassesAfterFirstScf))
              {
                // maximum of the residual norm of the state closest to and
                // below the Fermi level among all k points, and also the
                // maximum between the two spins
                double maxRes =
                  std::max(computeMaximumHighestOccupiedStateResidualNorm(
                             residualNormWaveFunctionsAllkPointsSpins[0],
                             eigenValuesSpins[0],
                             fermiEnergy),
                           computeMaximumHighestOccupiedStateResidualNorm(
                             residualNormWaveFunctionsAllkPointsSpins[1],
                             eigenValuesSpins[1],
                             fermiEnergy));

                if (d_dftParamsPtr->verbosity >= 2)
                  {
                    pcout
                      << "Maximum residual norm among all states with occupation number greater than 1e-3: "
                      << maxRes << std::endl;
                  }

                // if the residual norm is greater than
                // adaptiveChebysevFilterPassesTol (a heuristic value)
                // do more passes of chebysev filter till the check passes.
                // This improves the scf convergence performance.

                const double filterPassTol =
                  (scfIter == 0 && isRestartGroundStateCalcFromChk) ?
                    1.0e-8 :
                    ((scfIter == 0 &&
                      adaptiveChebysevFilterPassesTol > firstScfChebyTol) ?
                       firstScfChebyTol :
                       adaptiveChebysevFilterPassesTol);
                while (maxRes > filterPassTol && count < 100)
                  {
                    for (unsigned int s = 0; s < 2; ++s)
                      {
                        if (d_dftParamsPtr->memOptMode)
                          {
                            computing_timer.enter_subsection(
                              "VEff Computation");
                            kohnShamDFTEigenOperator.computeVEff(
                              d_densityInQuadValues,
                              d_gradDensityInQuadValues,
                              d_phiInQuadValues,
                              d_rhoCore,
                              d_gradRhoCore,
                              s);
                            computing_timer.leave_subsection(
                              "VEff Computation");
                            if (d_dftParamsPtr->isPseudopotential &&
                                d_dftParamsPtr->pawPseudoPotential)
                              {
                                computing_timer.enter_subsection(
                                  "Computing Non-Local Hamiltonian Entries");
                                computing_timer.enter_subsection(
                                  "Computing Non-Local XC Term Entries");
                                d_pawClassPtr
                                  ->initialiseExchangeCorrelationEnergyCorrection(
                                    s);
                                computing_timer.leave_subsection(
                                  "Computing Non-Local XC Term Entries");
                                computing_timer.enter_subsection(
                                  "Computing Non-Local Electrostatics Term Entries");
                                // if (s == 0)
                                d_pawClassPtr
                                  ->evaluateNonLocalHamiltonianElectrostaticsValue(
                                    d_phiTotRhoIn,
                                    d_phiTotDofHandlerIndexElectro);
                                // d_pawClassPtr
                                //   ->evaluateNonLocalHamiltonianElectrostaticsValue(
                                //     d_phitTotQuadPointsCompensation,
                                //     d_phiTotDofHandlerIndexElectro);
                                computing_timer.leave_subsection(
                                  "Computing Non-Local Electrostatics Term Entries");
                                d_pawClassPtr
                                  ->computeNonlocalPseudoPotentialConstants(
                                    CouplingType::HamiltonianEntries, s);
                                computing_timer.leave_subsection(
                                  "Computing Non-Local Hamiltonian Entries");
                              }
                          }
                        for (unsigned int kPoint = 0;
                             kPoint < d_kPointWeights.size();
                             ++kPoint)
                          {
                            if (d_dftParamsPtr->verbosity >= 2)
                              pcout << "Beginning Chebyshev filter pass "
                                    << 1 + count << " for spin " << s + 1
                                    << std::endl;

                            kohnShamDFTEigenOperator.reinitkPointSpinIndex(
                              kPoint, s);
                            if (d_dftParamsPtr->memOptMode)
                              {
                                computing_timer.enter_subsection(
                                  "Hamiltonian Matrix Computation");
                                kohnShamDFTEigenOperator
                                  .computeCellHamiltonianMatrix();
                                computing_timer.leave_subsection(
                                  "Hamiltonian Matrix Computation");
                              }
                            if (d_dftParamsPtr->verbosity >= 5)
                              computeTraceXtHX(d_numEigenValues, kPoint);

#ifdef DFTFE_WITH_DEVICE
                            if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                          memorySpace)
                              kohnShamEigenSpaceCompute(
                                s,
                                kPoint,
                                kohnShamDFTEigenOperator,
                                *d_elpaScala,
                                d_subspaceIterationSolverDevice,
                                residualNormWaveFunctionsAllkPointsSpins
                                  [s][kPoint],
                                true,
                                0,
                                (scfIter <
                                 d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                                  false :
                                  true,
                                true,
                                scfIter == 0);
#endif
                            if constexpr (dftfe::utils::MemorySpace::HOST ==
                                          memorySpace)
                              kohnShamEigenSpaceCompute(
                                s,
                                kPoint,
                                kohnShamDFTEigenOperator,
                                *d_elpaScala,
                                d_subspaceIterationSolver,
                                residualNormWaveFunctionsAllkPointsSpins
                                  [s][kPoint],
                                true,
                                (scfIter <
                                 d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                                  false :
                                  true,
                                true,
                                scfIter == 0);
                          }
                      }

                    for (unsigned int s = 0; s < 2; ++s)
                      for (unsigned int kPoint = 0;
                           kPoint < d_kPointWeights.size();
                           ++kPoint)
                        {
                          if (scfIter <
                                d_dftParamsPtr->spectrumSplitStartingScfIter ||
                              scfConverged)
                            for (unsigned int i = 0; i < d_numEigenValues; ++i)
                              eigenValuesSpins[s][kPoint][i] =
                                eigenValues[kPoint][d_numEigenValues * s + i];
                          else
                            for (unsigned int i = 0; i < d_numEigenValuesRR;
                                 ++i)
                              eigenValuesSpins[s][kPoint][i] =
                                eigenValuesRRSplit[kPoint]
                                                  [d_numEigenValuesRR * s + i];
                        }
                    //
                    if (d_dftParamsPtr->constraintMagnetization)
                      compute_fermienergy_constraintMagnetization(eigenValues);
                    else
                      compute_fermienergy(eigenValues, numElectrons);
                    //
                    maxRes =
                      std::max(computeMaximumHighestOccupiedStateResidualNorm(
                                 residualNormWaveFunctionsAllkPointsSpins[0],
                                 eigenValuesSpins[0],
                                 fermiEnergy),
                               computeMaximumHighestOccupiedStateResidualNorm(
                                 residualNormWaveFunctionsAllkPointsSpins[1],
                                 eigenValuesSpins[1],
                                 fermiEnergy));
                    if (d_dftParamsPtr->verbosity >= 2)
                      pcout
                        << "Maximum residual norm among all states with occupation number greater than 1e-3: "
                        << maxRes << std::endl;
                    count++;
                  }
              }

            if (d_dftParamsPtr->verbosity >= 1)
              {
                pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
              }

            numberChebyshevSolvePasses = count;
          }
        else
          {
            std::vector<std::vector<double>>
              residualNormWaveFunctionsAllkPoints;
            residualNormWaveFunctionsAllkPoints.resize(d_kPointWeights.size());
            for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                 ++kPoint)
              residualNormWaveFunctionsAllkPoints[kPoint].resize(
                (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                 scfConverged) ?
                  d_numEigenValues :
                  d_numEigenValuesRR);

            computing_timer.enter_subsection("VEff Computation");
            kohnShamDFTEigenOperator.computeVEff(d_densityInQuadValues,
                                                 d_gradDensityInQuadValues,
                                                 d_phiInQuadValues,
                                                 d_rhoCore,
                                                 d_gradRhoCore);
            computing_timer.leave_subsection("VEff Computation");
            if (d_dftParamsPtr->isPseudopotential &&
                d_dftParamsPtr->pawPseudoPotential)
              {
                computing_timer.enter_subsection(
                  "Computing Non-Local Hamiltonian Entries");
                computing_timer.enter_subsection(
                  "Computing Non-Local XC Term Entries");
                d_pawClassPtr->initialiseExchangeCorrelationEnergyCorrection(0);
                computing_timer.leave_subsection(
                  "Computing Non-Local XC Term Entries");
                computing_timer.enter_subsection(
                  "Computing Non-Local Electrostatics Term Entries");
                d_pawClassPtr->evaluateNonLocalHamiltonianElectrostaticsValue(
                  d_phiTotRhoIn, d_phiTotDofHandlerIndexElectro);
                // d_pawClassPtr->evaluateNonLocalHamiltonianElectrostaticsValue(
                //   d_phitTotQuadPointsCompensation,
                //   d_phiTotDofHandlerIndexElectro);
                computing_timer.leave_subsection(
                  "Computing Non-Local Electrostatics Term Entries");
                d_pawClassPtr->computeNonlocalPseudoPotentialConstants(
                  CouplingType::HamiltonianEntries, 0);
                computing_timer.leave_subsection(
                  "Computing Non-Local Hamiltonian Entries");
              }

            for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                 ++kPoint)
              {
                kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, 0);


                computing_timer.enter_subsection(
                  "Hamiltonian Matrix Computation");
                kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
                computing_timer.leave_subsection(
                  "Hamiltonian Matrix Computation");


                for (unsigned int j = 0; j < 1; ++j)
                  {
                    if (d_dftParamsPtr->verbosity >= 2)
                      {
                        pcout << "Beginning Chebyshev filter pass " << j + 1
                              << std::endl;
                      }
                    if (d_dftParamsPtr->verbosity >= 5)
                      computeTraceXtHX(d_numEigenValues, kPoint);

#ifdef DFTFE_WITH_DEVICE
                    if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                  memorySpace)
                      kohnShamEigenSpaceCompute(
                        0,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolverDevice,
                        residualNormWaveFunctionsAllkPoints[kPoint],
                        (scfIter == 0 ||
                         d_dftParamsPtr
                           ->allowMultipleFilteringPassesAfterFirstScf) ?
                          true :
                          false,
                        0,
                        (scfIter <
                           d_dftParamsPtr->spectrumSplitStartingScfIter ||
                         scfConverged) ?
                          false :
                          true,
                        scfConverged ? false : true,
                        scfIter == 0);
#endif
                    if constexpr (dftfe::utils::MemorySpace::HOST ==
                                  memorySpace)
                      kohnShamEigenSpaceCompute(
                        0,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolver,
                        residualNormWaveFunctionsAllkPoints[kPoint],
                        (scfIter == 0 ||
                         d_dftParamsPtr
                           ->allowMultipleFilteringPassesAfterFirstScf) ?
                          true :
                          false,
                        (scfIter <
                           d_dftParamsPtr->spectrumSplitStartingScfIter ||
                         scfConverged) ?
                          false :
                          true,
                        scfConverged ? false : true,
                        scfIter == 0);
                  }
              }


            //
            // fermi energy
            //
            if (d_dftParamsPtr->constraintMagnetization)
              compute_fermienergy_constraintMagnetization(eigenValues);
            else
              compute_fermienergy(eigenValues, numElectrons);

            unsigned int count = 1;

            if (!scfConverged &&
                (scfIter == 0 ||
                 d_dftParamsPtr->allowMultipleFilteringPassesAfterFirstScf))
              {
                //
                // maximum of the residual norm of the state closest to and
                // below the Fermi level among all k points
                //
                double maxRes = computeMaximumHighestOccupiedStateResidualNorm(
                  residualNormWaveFunctionsAllkPoints,
                  (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                    eigenValues :
                    eigenValuesRRSplit,
                  fermiEnergy);
                if (d_dftParamsPtr->verbosity >= 2)
                  pcout
                    << "Maximum residual norm among all states with occupation number greater than 1e-3: "
                    << maxRes << std::endl;

                // if the residual norm is greater than
                // adaptiveChebysevFilterPassesTol (a heuristic value)
                // do more passes of chebysev filter till the check passes.
                // This improves the scf convergence performance.

                const double filterPassTol =
                  (scfIter == 0 && isRestartGroundStateCalcFromChk) ?
                    1.0e-8 :
                    ((scfIter == 0 &&
                      adaptiveChebysevFilterPassesTol > firstScfChebyTol) ?
                       firstScfChebyTol :
                       adaptiveChebysevFilterPassesTol);
                while (maxRes > filterPassTol && count < 100)
                  {
                    for (unsigned int kPoint = 0;
                         kPoint < d_kPointWeights.size();
                         ++kPoint)
                      {
                        if (d_dftParamsPtr->verbosity >= 2)
                          pcout << "Beginning Chebyshev filter pass "
                                << 1 + count << std::endl;

                        kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,
                                                                       0);
                        if (d_dftParamsPtr->memOptMode &&
                            d_kPointWeights.size() > 0)
                          {
                            computing_timer.enter_subsection(
                              "Hamiltonian Matrix Computation");
                            kohnShamDFTEigenOperator
                              .computeCellHamiltonianMatrix();
                            computing_timer.leave_subsection(
                              "Hamiltonian Matrix Computation");
                          }
                        if (d_dftParamsPtr->verbosity >= 5)
                          computeTraceXtHX(d_numEigenValues, kPoint);
#ifdef DFTFE_WITH_DEVICE
                        if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                      memorySpace)
                          kohnShamEigenSpaceCompute(
                            0,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolverDevice,
                            residualNormWaveFunctionsAllkPoints[kPoint],
                            true,
                            0,
                            (scfIter <
                             d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                              false :
                              true,
                            true,
                            scfIter == 0);

#endif
                        if constexpr (dftfe::utils::MemorySpace::HOST ==
                                      memorySpace)
                          kohnShamEigenSpaceCompute(
                            0,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolver,
                            residualNormWaveFunctionsAllkPoints[kPoint],
                            true,
                            (scfIter <
                             d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                              false :
                              true,
                            true,
                            scfIter == 0);
                      }

                    //
                    if (d_dftParamsPtr->constraintMagnetization)
                      compute_fermienergy_constraintMagnetization(eigenValues);
                    else
                      compute_fermienergy(eigenValues, numElectrons);
                    //
                    maxRes = computeMaximumHighestOccupiedStateResidualNorm(
                      residualNormWaveFunctionsAllkPoints,
                      (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                       scfConverged) ?
                        eigenValues :
                        eigenValuesRRSplit,
                      fermiEnergy);
                    if (d_dftParamsPtr->verbosity >= 2)
                      pcout
                        << "Maximum residual norm among all states with occupation number greater than 1e-3: "
                        << maxRes << std::endl;

                    count++;
                  }
              }

            numberChebyshevSolvePasses = count;

            if (d_dftParamsPtr->verbosity >= 1)
              {
                pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
              }
          }
        computing_timer.enter_subsection("compute rho");
        if (d_dftParamsPtr->useSymm)
          {
#ifdef USE_COMPLEX
            symmetryPtr->computeLocalrhoOut();
            symmetryPtr->computeAndSymmetrize_rhoOut();

            l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    d_densityOutQuadValues[0],
                                    d_densityOutNodalValues[0]);

            interpolateDensityNodalDataToQuadratureDataLpsp(
              d_basisOperationsPtrElectroHost,
              d_densityDofHandlerIndexElectro,
              d_lpspQuadratureIdElectro,
              d_densityOutNodalValues[0],
              d_densityTotalOutValuesLpspQuad,
              d_gradDensityTotalOutValuesLpspQuad,
              true);
#endif
          }
        else
          {
            compute_rhoOut(
              (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
               scfConverged) ?
                false :
                true,
              scfConverged ||
                (scfIter == (d_dftParamsPtr->numSCFIterations - 1)));
          }
        computing_timer.leave_subsection("compute rho");

        //
        // compute integral rhoOut
        //
        const double integralRhoValue =
          totalCharge(d_dofHandlerPRefined, d_densityOutQuadValues[0]);
        if (d_dftParamsPtr->pawPseudoPotential)
          {
            if (d_dftParamsPtr->memoryOptCompCharge)
              d_pawClassPtr->computeCompensationChargeMemoryOpt(
                TypeOfField::Out);
            else
              d_pawClassPtr->computeCompensationCharge(TypeOfField::Out);
            if (d_dftParamsPtr->verbosity >= 4)
              d_pawClassPtr->chargeNeutrality(integralRhoValue,
                                              TypeOfField::Out,
                                              false);
            computeTotalDensityNodalVector(
              d_bQuadValuesAllAtoms,
              d_densityOutNodalValues[0],
              d_totalChargeDensityOutNodalValues[0]);
          }
        if (d_dftParamsPtr->verbosity >= 2)
          {
            pcout << std::endl
                  << "number of electrons: " << integralRhoValue << std::endl;
          }

        if (d_dftParamsPtr->verbosity >= 1 &&
            d_dftParamsPtr->spinPolarized == 1)
          pcout << std::endl
                << "net magnetization: "
                << totalMagnetization(d_densityOutQuadValues[1]) << std::endl;

        //
        // phiTot with rhoOut
        //
        if ((d_dftParamsPtr->computeEnergyEverySCF &&
             d_numEigenValuesRR == d_numEigenValues) ||
            d_dftParamsPtr->useGradPhiMixing)
          {
            if (d_dftParamsPtr->verbosity >= 2)
              pcout
                << std::endl
                << "Poisson solve for total electrostatic potential (rhoOut+b): ";

            computing_timer.enter_subsection("phiTot solve");
            if (d_dftParamsPtr->isPseudopotential &&
                d_dftParamsPtr->pawPseudoPotential)
              {
                if (d_dftParamsPtr->verbosity >= 4)
                  d_pawClassPtr->chargeNeutrality(
                    totalCharge(d_dofHandlerRhoNodal,
                                d_densityOutQuadValues[0]),
                    TypeOfField::Out,
                    false);
              }

            if (d_dftParamsPtr->multipoleBoundaryConditions)
              {
                computing_timer.enter_subsection("Update inhomogenous BC");
                computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                        d_densityQuadratureIdElectro,
                                        d_densityOutQuadValues[0],
                                        &d_bQuadValuesAllAtoms);
                updatePRefinedConstraints();
                computing_timer.leave_subsection("Update inhomogenous BC");
              }

            dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              densityOutQuadValuesCopy = d_densityOutQuadValues[0];
            if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
                (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
                 d_dftParamsPtr->periodicZ))
              {
                double *tempvec = densityOutQuadValuesCopy.data();
                for (unsigned int iquad = 0;
                     iquad < densityOutQuadValuesCopy.size();
                     iquad++)
                  tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
              }

            if ((d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
                 d_dftParamsPtr->floatingNuclearCharges and
                 not d_dftParamsPtr->pinnedNodeForPBC))
              {
#ifdef DFTFE_WITH_DEVICE
                d_phiTotalSolverProblemDevice.reinit(
                  d_basisOperationsPtrElectroHost,
                  d_phiTotRhoOut,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  densityOutQuadValuesCopy,
                  d_BLASWrapperPtr,
                  false,
                  false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  false,
                  true,
                  d_dftParamsPtr->multipoleBoundaryConditions);

                CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                                     d_dftParamsPtr->absLinearSolverTolerance,
                                     d_dftParamsPtr->maxLinearSolverIterations,
                                     d_dftParamsPtr->verbosity);
#endif
              }
            else
              {
                if (d_dftParamsPtr->pawPseudoPotential)
                  d_phiTotalSolverProblem.reinit(
                    d_basisOperationsPtrElectroHost,
                    d_phiTotRhoOut,
                    *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                    d_phiTotDofHandlerIndexElectro,
                    d_densityQuadratureIdElectro,
                    d_phiTotAXQuadratureIdElectro,
                    d_atomNodeIdToChargeMap,
                    d_bQuadValuesAllAtoms,
                    d_smearedChargeQuadratureIdElectro,
                    densityOutQuadValuesCopy,
                    d_dftParamsPtr->nonLinearCoreCorrection, //
                    d_rhoCore,                               //
                    false,
                    d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                      d_dftParamsPtr->periodicZ &&
                      !d_dftParamsPtr->pinnedNodeForPBC,
                    d_dftParamsPtr->smearedNuclearCharges,
                    true,
                    false,
                    0,
                    false,
                    false,
                    d_dftParamsPtr->multipoleBoundaryConditions);

                else
                  d_phiTotalSolverProblem.reinit(
                    d_basisOperationsPtrElectroHost,
                    d_phiTotRhoOut,
                    *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                    d_phiTotDofHandlerIndexElectro,
                    d_densityQuadratureIdElectro,
                    d_phiTotAXQuadratureIdElectro,
                    d_atomNodeIdToChargeMap,
                    d_bQuadValuesAllAtoms,
                    d_smearedChargeQuadratureIdElectro,
                    densityOutQuadValuesCopy,
                    false,
                    false,
                    d_dftParamsPtr->smearedNuclearCharges,
                    true,
                    false,
                    0,
                    false,
                    true,
                    d_dftParamsPtr->multipoleBoundaryConditions);

                CGSolver.solve(d_phiTotalSolverProblem,
                               d_dftParamsPtr->absLinearSolverTolerance,
                               d_dftParamsPtr->maxLinearSolverIterations,
                               d_dftParamsPtr->verbosity);
              }
            if (!d_dftParamsPtr->useGradPhiMixing)
              interpolateElectroNodalDataToQuadratureDataGeneral(
                d_basisOperationsPtrElectroHost,
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotRhoOut,
                d_phiOutQuadValues,
                dummy);
            else
              interpolateElectroNodalDataToQuadratureDataGeneral(
                d_basisOperationsPtrElectroHost,
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotRhoOut,
                d_phiOutQuadValues,
                d_gradPhiOutQuadValues,
                true);


            computing_timer.leave_subsection("phiTot solve");
            if (d_dftParamsPtr->computeEnergyEverySCF &&
                d_numEigenValuesRR == d_numEigenValues)
              {
                const dealii::Quadrature<3> &quadrature =
                  matrix_free_data.get_quadrature(d_densityQuadratureId);
                d_dispersionCorr.computeDispresionCorrection(
                  atomLocations, d_domainBoundingVectors);
                const double totalEnergy = energyCalc.computeEnergy(
                  d_basisOperationsPtrHost,
                  d_basisOperationsPtrElectroHost,
                  d_densityQuadratureId,
                  d_densityQuadratureIdElectro,
                  d_smearedChargeQuadratureIdElectro,
                  d_lpspQuadratureIdElectro,
                  eigenValues,
                  d_kPointWeights,
                  fermiEnergy,
                  d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy :
                                                       fermiEnergyUp,
                  d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy :
                                                       fermiEnergyDown,
                  d_excManagerPtr,
                  d_dispersionCorr,
                  d_phiInQuadValues,
                  d_phiOutQuadValues,
                  d_phiTotRhoOut,
                  d_densityInQuadValues,
                  d_densityOutQuadValues,
                  d_gradDensityInQuadValues,
                  d_gradDensityOutQuadValues,
                  d_densityTotalOutValuesLpspQuad,
                  d_rhoCore,
                  d_gradRhoCore,
                  d_bQuadValuesAllAtoms,
                  d_bCellNonTrivialAtomIds,
                  d_localVselfs,
                  d_dftParamsPtr->pawPseudoPotential ? d_zeroPotential :
                                                       d_pseudoVLoc,
                  d_atomNodeIdToChargeMap,
                  atomLocations.size(),
                  lowerBoundKindex,
                  0,
                  d_dftParamsPtr->verbosity >= 0 ? true : false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  d_dftParamsPtr->pawPseudoPotential,
                  d_dftParamsPtr->pawPseudoPotential ?
                    d_pawClassPtr->getDeltaEnergy() :
                    std::vector<double>());
                if (d_dftParamsPtr->verbosity == 1)
                  pcout << "Total energy  : " << totalEnergy << std::endl;
              }
          }
        else
          {
            if (d_numEigenValuesRR != d_numEigenValues &&
                d_dftParamsPtr->computeEnergyEverySCF &&
                d_dftParamsPtr->verbosity >= 1)
              pcout
                << "DFT-FE Message: energy computation is not performed at the end of each scf iteration step\n"
                << "if SPECTRUM SPLIT CORE EIGENSTATES is set to a non-zero value."
                << std::endl;
          }

        if (d_dftParamsPtr->verbosity >= 1)
          pcout << "***********************Self-Consistent-Field Iteration: "
                << std::setw(2) << scfIter + 1
                << " complete**********************" << std::endl;

        local_timer.stop();
        if (d_dftParamsPtr->verbosity >= 1)
          pcout << "Wall time for the above scf iteration: "
                << local_timer.wall_time() << " seconds\n"
                << "Number of Chebyshev filtered subspace iterations: "
                << numberChebyshevSolvePasses << std::endl
                << std::endl;
        //
        scfIter++;

        if (d_dftParamsPtr->saveRhoData && scfIter % 10 == 0 &&
            d_dftParamsPtr->solverMode == "GS")
          saveTriaInfoAndRhoNodalData();

        if (d_dftParamsPtr->saveDijData && scfIter % 10 == 0 &&
            d_dftParamsPtr->solverMode == "GS" &&
            d_dftParamsPtr->pawPseudoPotential)
          d_pawClassPtr->saveDijEntriesToFile(d_mpiCommParent);
      }

    if (d_dftParamsPtr->saveRhoData &&
        !(d_dftParamsPtr->solverMode == "GS" && scfIter % 10 == 0))
      saveTriaInfoAndRhoNodalData();
    if (d_dftParamsPtr->saveDijData &&
        !(d_dftParamsPtr->solverMode == "GS" && scfIter % 10 == 0) &&
        d_dftParamsPtr->pawPseudoPotential)
      d_pawClassPtr->saveDijEntriesToFile(d_mpiCommParent);

    if (scfIter == d_dftParamsPtr->numSCFIterations)
      {
        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          std::cout
            << "DFT-FE Warning: SCF iterations did not converge to the specified tolerance after: "
            << scfIter << " iterations." << std::endl;
      }
    else
      {
        pcout << "SCF iterations converged to the specified tolerance after: "
              << scfIter << " iterations." << std::endl;

        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          {
            if (d_dftParamsPtr->solverMode == "GS" &&
                d_dftParamsPtr->saveRhoData)
              {
                FILE *fermiFile;
                fermiFile = fopen("fermiEnergy.out", "w");
                if (d_dftParamsPtr->constraintMagnetization)
                  {
                    fprintf(fermiFile,
                            "%.14g\n%.14g\n%.14g\n ",
                            fermiEnergy,
                            fermiEnergyUp,
                            fermiEnergyDown);
                  }
                else
                  {
                    fprintf(fermiFile, "%.14g\n", fermiEnergy);
                  }
                fclose(fermiFile);
              }
          }
      }

    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

    const unsigned int localVectorSize =
      matrix_free_data.get_vector_partitioner()->locally_owned_size();
    if (numberBandGroups > 1 && !d_dftParamsPtr->useDevice)
      {
        MPI_Barrier(interBandGroupComm);
        const unsigned int blockSize =
          d_dftParamsPtr->mpiAllReduceMessageBlockSizeMB * 1e+6 /
          sizeof(dataTypes::number);
        for (unsigned int kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          for (unsigned int i = 0; i < d_numEigenValues * localVectorSize;
               i += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, d_numEigenValues * localVectorSize - i);
              MPI_Allreduce(
                MPI_IN_PLACE,
                &d_eigenVectorsFlattenedHost[kPoint * d_numEigenValues *
                                             localVectorSize] +
                  i,
                currentBlockSize,
                dataTypes::mpi_type_id(
                  &d_eigenVectorsFlattenedHost[kPoint * d_numEigenValues *
                                               localVectorSize]),
                MPI_SUM,
                interBandGroupComm);
            }
      }

    if ((!d_dftParamsPtr->computeEnergyEverySCF ||
         d_numEigenValuesRR != d_numEigenValues))
      {
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << std::endl
            << "Poisson solve for total electrostatic potential (rhoOut+b): ";

        computing_timer.enter_subsection("phiTot solve");
        if (d_dftParamsPtr->isPseudopotential &&
            d_dftParamsPtr->pawPseudoPotential)
          {
            if (d_dftParamsPtr->memoryOptCompCharge)
              d_pawClassPtr->computeCompensationChargeMemoryOpt(
                TypeOfField::Out);
            else
              d_pawClassPtr->computeCompensationCharge(TypeOfField::Out);
            if (d_dftParamsPtr->verbosity >= 4)
              d_pawClassPtr->chargeNeutrality(
                totalCharge(d_dofHandlerRhoNodal, d_densityOutQuadValues[0]),
                TypeOfField::Out,
                false);
            computeTotalDensityNodalVector(
              d_bQuadValuesAllAtoms,
              d_densityOutNodalValues[0],
              d_totalChargeDensityOutNodalValues[0]);
          }

        if (d_dftParamsPtr->multipoleBoundaryConditions)
          {
            computing_timer.enter_subsection("Update inhomogenous BC");
            computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                    d_densityQuadratureIdElectro,
                                    d_densityOutQuadValues[0],
                                    &d_bQuadValuesAllAtoms);
            updatePRefinedConstraints();
            computing_timer.leave_subsection("Update inhomogenous BC");
          }

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          densityOutQuadValuesCopy = d_densityOutQuadValues[0];
        if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
            (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
             d_dftParamsPtr->periodicZ))
          {
            double *tempvec = densityOutQuadValuesCopy.data();
            for (unsigned int iquad = 0;
                 iquad < densityOutQuadValuesCopy.size();
                 iquad++)
              tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
          }

        if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
          {
#ifdef DFTFE_WITH_DEVICE
            d_phiTotalSolverProblemDevice.reinit(
              d_basisOperationsPtrElectroHost,
              d_phiTotRhoOut,
              *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
              d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
              d_atomNodeIdToChargeMap,
              d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
              densityOutQuadValuesCopy,
              d_BLASWrapperPtr,
              false,
              false,
              d_dftParamsPtr->smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true,
              d_dftParamsPtr->multipoleBoundaryConditions);

            CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                                 d_dftParamsPtr->absLinearSolverTolerance,
                                 d_dftParamsPtr->maxLinearSolverIterations,
                                 d_dftParamsPtr->verbosity);
#endif
          }
        else
          {
            if (d_dftParamsPtr->pawPseudoPotential)
              {
                d_phiTotalSolverProblem.reinit(
                  d_basisOperationsPtrElectroHost, // basisOperatorHost
                  d_phiTotRhoOut,                  // phi(x)
                  *d_constraintsVectorElectro
                    [d_phiTotDofHandlerIndexElectro], // constraintMatrix
                  d_phiTotDofHandlerIndexElectro,     // DofHandlerId
                  d_densityQuadratureIdElectro,       // DensityQuadID
                  d_phiTotAXQuadratureIdElectro,      // AXQuadID
                  d_atomNodeIdToChargeMap,            // atoms
                  d_bQuadValuesAllAtoms,              // COmpensationChargeQuad
                  d_smearedChargeQuadratureIdElectro, // CompensationCharge ID
                  densityOutQuadValuesCopy,           // Rho(x)
                  d_dftParamsPtr->nonLinearCoreCorrection, //
                  d_rhoCore,                               //
                  false,                                   // computeDiagonalA
                  false,                                   // MeanValueContraint
                  d_dftParamsPtr
                    ->smearedNuclearCharges, // smearedNuclearCharges
                  true,                      // isRhoValues
                  false,                     // isGradSmeraedChargeRhs
                  0,                         // smeardChargeGradinetCompnoetId
                  false,                     // storesmeared
                  false,                     // reuseSmearedChargeRhs
                  d_dftParamsPtr->multipoleBoundaryConditions);
              }
            else
              {
                d_phiTotalSolverProblem.reinit(
                  d_basisOperationsPtrElectroHost,
                  d_phiTotRhoOut,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  densityOutQuadValuesCopy,
                  false,
                  false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  false,
                  true,
                  d_dftParamsPtr->multipoleBoundaryConditions);
              }

            CGSolver.solve(d_phiTotalSolverProblem,
                           d_dftParamsPtr->absLinearSolverTolerance,
                           d_dftParamsPtr->maxLinearSolverIterations,
                           d_dftParamsPtr->verbosity);
          }

        computing_timer.leave_subsection("phiTot solve");
      }
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;

    interpolateElectroNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_phiTotDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_phiTotRhoOut,
      d_phiOutQuadValues,
      dummy);


    //
    // compute and print ground state energy or energy after max scf
    // iterations
    //
    d_dispersionCorr.computeDispresionCorrection(atomLocations,
                                                 d_domainBoundingVectors);
    const double totalEnergy = energyCalc.computeEnergy(
      d_basisOperationsPtrHost,
      d_basisOperationsPtrElectroHost,
      d_densityQuadratureId,
      d_densityQuadratureIdElectro,
      d_smearedChargeQuadratureIdElectro,
      d_lpspQuadratureIdElectro,
      eigenValues,
      d_kPointWeights,
      fermiEnergy,
      d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyUp,
      d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyDown,
      d_excManagerPtr,
      d_dispersionCorr,
      d_phiInQuadValues,
      d_phiOutQuadValues,
      d_phiTotRhoOut,
      d_densityInQuadValues,
      d_densityOutQuadValues,
      d_gradDensityInQuadValues,
      d_gradDensityOutQuadValues,
      d_densityTotalOutValuesLpspQuad,
      d_rhoCore,
      d_gradRhoCore,
      d_bQuadValuesAllAtoms,
      d_bCellNonTrivialAtomIds,
      d_localVselfs,
      d_dftParamsPtr->pawPseudoPotential ? d_zeroPotential : d_pseudoVLoc,
      d_atomNodeIdToChargeMap,
      atomLocations.size(),
      lowerBoundKindex,
      1,
      d_dftParamsPtr->verbosity >= 0 ? true : false,
      d_dftParamsPtr->smearedNuclearCharges,
      d_dftParamsPtr->pawPseudoPotential,
      d_dftParamsPtr->pawPseudoPotential ? d_pawClassPtr->getDeltaEnergy() :
                                           std::vector<double>());

    d_groundStateEnergy = totalEnergy;

    MPI_Barrier(interpoolcomm);

    d_entropicEnergy =
      energyCalc.computeEntropicEnergy(eigenValues,
                                       d_kPointWeights,
                                       fermiEnergy,
                                       fermiEnergyUp,
                                       fermiEnergyDown,
                                       d_dftParamsPtr->spinPolarized == 1,
                                       d_dftParamsPtr->constraintMagnetization,
                                       d_dftParamsPtr->TVal);

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total entropic energy: " << d_entropicEnergy << std::endl;


    d_freeEnergy = d_groundStateEnergy - d_entropicEnergy;

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total free energy: " << d_freeEnergy << std::endl;

    // This step is required for interpolating rho from current mesh to the
    // new mesh in case of atomic relaxation
    // computeNodalRhoFromQuadData();

    computing_timer.leave_subsection("scf solve");
    computingTimerStandard.leave_subsection("Total scf solve");


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice &&
        (d_dftParamsPtr->writeWfcSolutionFields ||
         d_dftParamsPtr->writeLdosFile || d_dftParamsPtr->writePdosFile))
      d_eigenVectorsFlattenedDevice.copyTo(d_eigenVectorsFlattenedHost);
#endif


    if (d_dftParamsPtr->isIonForce)
      {
        if (d_dftParamsPtr->selfConsistentSolverTolerance > 1e-4 &&
            d_dftParamsPtr->verbosity >= 1)
          pcout
            << "DFT-FE Warning: Ion force accuracy may be affected for the given scf iteration solve tolerance: "
            << d_dftParamsPtr->selfConsistentSolverTolerance
            << ", recommended to use TOLERANCE below 1e-4." << std::endl;

        if (computeForces)
          {
            computing_timer.enter_subsection("Ion force computation");
            computingTimerStandard.enter_subsection("Ion force computation");
            forcePtr->computeAtomsForces(matrix_free_data,
                                         d_dispersionCorr,
                                         d_eigenDofHandlerIndex,
                                         d_smearedChargeQuadratureIdElectro,
                                         d_lpspQuadratureIdElectro,
                                         d_matrixFreeDataPRefined,
                                         d_phiTotDofHandlerIndexElectro,
                                         d_phiTotRhoOut,
                                         d_densityOutQuadValues,
                                         d_gradDensityOutQuadValues,
                                         d_densityTotalOutValuesLpspQuad,
                                         d_gradDensityTotalOutValuesLpspQuad,
                                         d_rhoCore,
                                         d_gradRhoCore,
                                         d_hessianRhoCore,
                                         d_gradRhoCoreAtoms,
                                         d_hessianRhoCoreAtoms,
                                         d_pseudoVLoc,
                                         d_pseudoVLocAtoms,
                                         d_constraintsPRefined,
                                         d_vselfBinsManager);
            if (d_dftParamsPtr->verbosity >= 0)
              forcePtr->printAtomsForces();
            computingTimerStandard.leave_subsection("Ion force computation");
            computing_timer.leave_subsection("Ion force computation");
          }
      }

    if (d_dftParamsPtr->isCellStress)
      {
        if (d_dftParamsPtr->selfConsistentSolverTolerance > 1e-4 &&
            d_dftParamsPtr->verbosity >= 1)
          pcout
            << "DFT-FE Warning: Cell stress accuracy may be affected for the given scf iteration solve tolerance: "
            << d_dftParamsPtr->selfConsistentSolverTolerance
            << ", recommended to use TOLERANCE below 1e-4." << std::endl;

        if (computestress)
          {
            computing_timer.enter_subsection("Cell stress computation");
            computingTimerStandard.enter_subsection("Cell stress computation");
            computeStress();
            computingTimerStandard.leave_subsection("Cell stress computation");
            computing_timer.leave_subsection("Cell stress computation");
          }
      }
    return std::make_tuple(scfConverged, norm);
  } // namespace dftfe


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeStress()
  {
    KohnShamHamiltonianOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    if (d_dftParamsPtr->isPseudopotential ||
        d_dftParamsPtr->smearedNuclearCharges)
      {
        computeVselfFieldGateauxDerFD();
      }

    forcePtr->computeStress(matrix_free_data,
                            d_dispersionCorr,
                            d_eigenDofHandlerIndex,
                            d_smearedChargeQuadratureIdElectro,
                            d_lpspQuadratureIdElectro,
                            d_matrixFreeDataPRefined,
                            d_phiTotDofHandlerIndexElectro,
                            d_phiTotRhoOut,
                            d_densityOutQuadValues,
                            d_gradDensityOutQuadValues,
                            d_densityTotalOutValuesLpspQuad,
                            d_gradDensityTotalOutValuesLpspQuad,
                            d_pseudoVLoc,
                            d_pseudoVLocAtoms,
                            d_rhoCore,
                            d_gradRhoCore,
                            d_hessianRhoCore,
                            d_gradRhoCoreAtoms,
                            d_hessianRhoCoreAtoms,
                            d_constraintsPRefined,
                            d_vselfBinsManager);
    if (d_dftParamsPtr->verbosity >= 0)
      forcePtr->printStress();
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeVselfFieldGateauxDerFD()
  {
    d_vselfFieldGateauxDerStrainFDBins.clear();
    d_vselfFieldGateauxDerStrainFDBins.resize(
      (d_vselfBinsManager.getVselfFieldBins()).size() * 6);

    dealii::Tensor<2, 3, double> identityTensor;
    dealii::Tensor<2, 3, double> deformationGradientPerturb1;
    dealii::Tensor<2, 3, double> deformationGradientPerturb2;

    // initialize to indentity tensors
    for (unsigned int idim = 0; idim < 3; idim++)
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        {
          if (idim == jdim)
            {
              identityTensor[idim][jdim]              = 1.0;
              deformationGradientPerturb1[idim][jdim] = 1.0;
              deformationGradientPerturb2[idim][jdim] = 1.0;
            }
          else
            {
              identityTensor[idim][jdim]              = 0.0;
              deformationGradientPerturb1[idim][jdim] = 0.0;
              deformationGradientPerturb2[idim][jdim] = 0.0;
            }
        }

    const double fdparam          = 1e-5;
    unsigned int flattenedIdCount = 0;
    for (unsigned int idim = 0; idim < 3; ++idim)
      for (unsigned int jdim = 0; jdim <= idim; jdim++)
        {
          deformationGradientPerturb1 = identityTensor;
          if (idim == jdim)
            {
              deformationGradientPerturb1[idim][jdim] = 1.0 + fdparam;
            }
          else
            {
              deformationGradientPerturb1[idim][jdim] = fdparam;
              deformationGradientPerturb1[jdim][idim] = fdparam;
            }

          deformDomain(deformationGradientPerturb1 *
                         dealii::invert(deformationGradientPerturb2),
                       true,
                       false,
                       d_dftParamsPtr->verbosity >= 4 ? true : false);

          computing_timer.enter_subsection(
            "Nuclear self-potential perturbation solve");

          d_vselfBinsManager.solveVselfInBinsPerturbedDomain(
            d_basisOperationsPtrElectroHost,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
#ifdef DFTFE_WITH_DEVICE
            FEOrder == FEOrderElectro ?
              d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
              d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
            d_BLASWrapperPtr,
#endif
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_smearedChargeWidths,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);

          computing_timer.leave_subsection(
            "Nuclear self-potential perturbation solve");

          for (unsigned int ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] =
              (d_vselfBinsManager.getPerturbedVselfFieldBins())[ibin];

          deformationGradientPerturb2 = identityTensor;
          if (idim == jdim)
            {
              deformationGradientPerturb2[idim][jdim] = 1.0 - fdparam;
            }
          else
            {
              deformationGradientPerturb2[idim][jdim] = -fdparam;
              deformationGradientPerturb2[jdim][idim] = -fdparam;
            }

          deformDomain(deformationGradientPerturb2 *
                         dealii::invert(deformationGradientPerturb1),
                       true,
                       false,
                       d_dftParamsPtr->verbosity >= 4 ? true : false);

          computing_timer.enter_subsection(
            "Nuclear self-potential perturbation solve");

          d_vselfBinsManager.solveVselfInBinsPerturbedDomain(
            d_basisOperationsPtrElectroHost,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
#ifdef DFTFE_WITH_DEVICE
            FEOrder == FEOrderElectro ?
              d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
              d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
            d_BLASWrapperPtr,
#endif
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_smearedChargeWidths,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);

          computing_timer.leave_subsection(
            "Nuclear self-potential perturbation solve");

          for (unsigned int ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] -=
              (d_vselfBinsManager.getPerturbedVselfFieldBins())[ibin];

          const double fac =
            (idim == jdim) ? (1.0 / 2.0 / fdparam) : (1.0 / 4.0 / fdparam);
          for (unsigned int ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] *=
              fac;

          flattenedIdCount++;
        }

    // reset
    deformDomain(dealii::invert(deformationGradientPerturb2),
                 true,
                 false,
                 d_dftParamsPtr->verbosity >= 4 ? true : false);
  }

  // Output wfc
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::outputWfc()
  {
    //
    // identify the index which is close to Fermi Energy
    //
    int indexFermiEnergy = -1.0;
    for (int spinType = 0; spinType < 1 + d_dftParamsPtr->spinPolarized;
         ++spinType)
      {
        for (int i = 0; i < d_numEigenValues; ++i)
          {
            if (eigenValues[0][spinType * d_numEigenValues + i] >= fermiEnergy)
              {
                if (i > indexFermiEnergy)
                  {
                    indexFermiEnergy = i;
                    break;
                  }
              }
          }
      }

    //
    // create a range of wavefunctions to output the wavefunction files
    //
    int startingRange = 0;
    int endingRange   = d_numEigenValues;

    /*
    int startingRange = indexFermiEnergy - 4;
    int endingRange   = indexFermiEnergy + 4;

    int startingRangeSpin = startingRange;

    for (int spinType = 0; spinType < 1 + d_dftParamsPtr->spinPolarized;
         ++spinType)
      {
        for (int i = indexFermiEnergy - 5; i > 0; --i)
          {
            if (std::abs(eigenValues[0][spinType * d_numEigenValues +
                                        (indexFermiEnergy - 4)] -
                         eigenValues[0][spinType * d_numEigenValues + i]) <=
                5e-04)
              {
                if (spinType == 0)
                  startingRange -= 1;
                else
                  startingRangeSpin -= 1;
              }
          }
      }


    if (startingRangeSpin < startingRange)
      startingRange = startingRangeSpin;
    */
    int numStatesOutput = (endingRange - startingRange) + 1;


    dealii::DataOut<3> data_outEigen;
    data_outEigen.attach_dof_handler(dofHandlerEigen);

    std::vector<distributedCPUVec<double>> tempVec(1);
    tempVec[0].reinit(d_tempEigenVec);

    std::vector<distributedCPUVec<double>> visualizeWaveFunctions(
      d_kPointWeights.size() * (1 + d_dftParamsPtr->spinPolarized) *
      numStatesOutput);

    unsigned int count = 0;
    for (unsigned int s = 0; s < 1 + d_dftParamsPtr->spinPolarized; ++s)
      for (unsigned int k = 0; k < d_kPointWeights.size(); ++k)
        for (unsigned int i = startingRange; i < endingRange; ++i)
          {
#ifdef USE_COMPLEX
            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedHost.data() +
                (k * (1 + d_dftParamsPtr->spinPolarized) + s) *
                  d_numEigenValues *
                  matrix_free_data.get_vector_partitioner()
                    ->locally_owned_size(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              std::make_pair(i, i + 1),
              localProc_dof_indicesReal,
              localProc_dof_indicesImag,
              tempVec);
#else
            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedHost.data() +
                (k * (1 + d_dftParamsPtr->spinPolarized) + s) *
                  d_numEigenValues *
                  matrix_free_data.get_vector_partitioner()
                    ->locally_owned_size(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              std::make_pair(i, i + 1),
              tempVec);
#endif
            tempVec[0].update_ghost_values();
            constraintsNoneEigenDataInfo.distribute(tempVec[0]);
            visualizeWaveFunctions[count] = tempVec[0];

            if (d_dftParamsPtr->spinPolarized == 1)
              data_outEigen.add_data_vector(visualizeWaveFunctions[count],
                                            "wfc_spin" + std::to_string(s) +
                                              "_kpoint" + std::to_string(k) +
                                              "_" + std::to_string(i));
            else
              data_outEigen.add_data_vector(visualizeWaveFunctions[count],
                                            "wfc_kpoint" + std::to_string(k) +
                                              "_" + std::to_string(i));

            count += 1;
          }

    data_outEigen.set_flags(dealii::DataOutBase::VtkFlags(
      std::numeric_limits<double>::min(),
      std::numeric_limits<unsigned int>::min(),
      true,
      dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::
        best_speed, // This flag is version dependent for dealII 9.5.0 it is
                    // dealii::DataOutBase::CompressionLevel::best_speed
      true));       // higher order cells set to true
    data_outEigen.build_patches(FEOrder);

    std::string tempFolder = "waveFunctionOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(dofHandlerEigen,
                                               data_outEigen,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "wfcOutput");
    //"wfcOutput_"+std::to_string(k)+"_"+std::to_string(i));
  }


  // Output density
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::outputDensity()
  {
    //
    // compute nodal electron-density from quad data
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                            d_constraintsRhoNodal,
                            d_densityDofHandlerIndexElectro,
                            d_densityQuadratureIdElectro,
                            d_densityOutQuadValues[0],
                            rhoNodalField);

    distributedCPUVec<double> magNodalField;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        magNodalField.reinit(rhoNodalField);
        magNodalField = 0;
        l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                                d_constraintsRhoNodal,
                                d_densityDofHandlerIndexElectro,
                                d_densityQuadratureIdElectro,
                                d_densityOutQuadValues[1],
                                magNodalField);
      }

    //
    // only generate output for electron-density
    //
    dealii::DataOut<3> dataOutRho;
    dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
    dataOutRho.add_data_vector(rhoNodalField, std::string("chargeDensity"));
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        dataOutRho.add_data_vector(magNodalField, std::string("magDensity"));
      }
    dataOutRho.set_flags(dealii::DataOutBase::VtkFlags(
      std::numeric_limits<double>::min(),
      std::numeric_limits<unsigned int>::min(),
      true,
      dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::
        best_speed, // This flag is version dependent for dealII 9.5.0 it is
                    // dealii::DataOutBase::CompressionLevel::best_speed
      true));       // higher order cells set to true
    dataOutRho.build_patches(FEOrder);

    std::string tempFolder = "densityOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
                                               dataOutRho,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "densityOutput");
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::writeBands()
  {
    int numkPoints =
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
    std::vector<double> eigenValuesFlattened;
    //
    for (unsigned int kPoint = 0; kPoint < numkPoints; ++kPoint)
      for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
        eigenValuesFlattened.push_back(eigenValues[kPoint][iWave]);
    //
    //
    //
    int totkPoints = dealii::Utilities::MPI::sum(numkPoints, interpoolcomm);
    std::vector<int> numkPointsArray(d_dftParamsPtr->npool),
      mpi_offsets(d_dftParamsPtr->npool, 0);
    std::vector<double> eigenValuesFlattenedGlobal(totkPoints *
                                                     d_numEigenValues,
                                                   0.0);
    //
    MPI_Gather(&numkPoints,
               1,
               MPI_INT,
               &(numkPointsArray[0]),
               1,
               MPI_INT,
               0,
               interpoolcomm);
    //
    numkPointsArray[0] = d_numEigenValues * numkPointsArray[0];
    for (unsigned int ipool = 1; ipool < d_dftParamsPtr->npool; ++ipool)
      {
        numkPointsArray[ipool] = d_numEigenValues * numkPointsArray[ipool];
        mpi_offsets[ipool] =
          mpi_offsets[ipool - 1] + numkPointsArray[ipool - 1];
      }
    //
    MPI_Gatherv(&(eigenValuesFlattened[0]),
                numkPoints * d_numEigenValues,
                MPI_DOUBLE,
                &(eigenValuesFlattenedGlobal[0]),
                &(numkPointsArray[0]),
                &(mpi_offsets[0]),
                MPI_DOUBLE,
                0,
                interpoolcomm);
    //
    if (d_dftParamsPtr->reproducible_output && d_dftParamsPtr->verbosity == 0)
      {
        pcout << "Writing Bands File..." << std::endl;
        pcout << "K-Point   WaveNo.  ";
        if (d_dftParamsPtr->spinPolarized)
          pcout << "SpinUpEigenValue          SpinDownEigenValue" << std::endl;
        else
          pcout << "EigenValue" << std::endl;
      }

    double FE = d_dftParamsPtr->spinPolarized ?
                  std::max(fermiEnergyDown, fermiEnergyUp) :
                  fermiEnergy;
    pcout << "Fermi Energy: " << FE << std::endl;
    unsigned int        maxeigenIndex = d_numEigenValues;
    std::vector<double> occupationVector(totkPoints, 0.0);

    for (int iWave = 1; iWave < d_numEigenValues; iWave++)
      {
        double maxOcc = -1.0;
        for (unsigned int kPoint = 0;
             kPoint < totkPoints / (1 + d_dftParamsPtr->spinPolarized);
             ++kPoint)
          {
            if (d_dftParamsPtr->spinPolarized)
              {
                occupationVector[2 * kPoint] = dftUtils::getPartialOccupancy(
                  eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                             iWave],
                  FE,
                  C_kb,
                  d_dftParamsPtr->TVal);
                occupationVector[2 * kPoint + 1] =
                  dftUtils::getPartialOccupancy(
                    eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                 d_numEigenValues +
                                               iWave],
                    FE,
                    C_kb,
                    d_dftParamsPtr->TVal);
                maxOcc = std::max(maxOcc,
                                  std::max(occupationVector[2 * kPoint + 1],
                                           occupationVector[2 * kPoint]));
              }
            else
              {
                occupationVector[kPoint] = dftUtils::getPartialOccupancy(
                  eigenValuesFlattenedGlobal[kPoint * d_numEigenValues + iWave],
                  FE,
                  C_kb,
                  d_dftParamsPtr->TVal);
                maxOcc = std::max(maxOcc, occupationVector[kPoint]);
              }
          }

        if (maxOcc < 1E-5)
          {
            maxeigenIndex = iWave;
            break;
          }
      }

    unsigned int numberEigenValues =
      d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0 ?
        std::min(d_numEigenValues, maxeigenIndex + 10) :
        d_dftParamsPtr->highestStateOfInterestForChebFiltering;
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        FILE *pFile;
        pFile = fopen("bands.out", "w");
        fprintf(pFile, "%d %d \n", totkPoints, numberEigenValues);
        for (unsigned int kPoint = 0;
             kPoint < totkPoints / (1 + d_dftParamsPtr->spinPolarized);
             ++kPoint)
          {
            for (unsigned int iWave = 0; iWave < numberEigenValues; ++iWave)
              {
                if (d_dftParamsPtr->spinPolarized)
                  {
                    double occupancyUp = dftUtils::getPartialOccupancy(
                      eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                                 iWave],
                      FE,
                      C_kb,
                      d_dftParamsPtr->TVal);

                    double occupancyDown = dftUtils::getPartialOccupancy(
                      eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                   d_numEigenValues +
                                                 iWave],
                      FE,
                      C_kb,
                      d_dftParamsPtr->TVal);

                    fprintf(
                      pFile,
                      "%d  %d   %.14g   %.14g   %.14g   %.14g\n",
                      kPoint,
                      iWave,
                      eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                                 iWave],
                      eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                   d_numEigenValues +
                                                 iWave],
                      occupancyUp,
                      occupancyDown);
                    if (d_dftParamsPtr->reproducible_output &&
                        d_dftParamsPtr->verbosity == 0)
                      {
                        double eigenUpTrunc =
                          std::floor(
                            1000000000 *
                            (eigenValuesFlattenedGlobal
                               [2 * kPoint * d_numEigenValues + iWave])) /
                          1000000000.0;
                        double eigenDownTrunc =
                          std::floor(
                            1000000000 *
                            (eigenValuesFlattenedGlobal
                               [(2 * kPoint + 1) * d_numEigenValues + iWave])) /
                          1000000000.0;
                        double occupancyUpTrunc =
                          std::floor(1000000000 * (occupancyUp)) / 1000000000.0;
                        double occupancyDownTrunc =
                          std::floor(1000000000 * (occupancyDown)) /
                          1000000000.0;
                        pcout << kPoint << "  " << iWave << "  " << std::fixed
                              << std::setprecision(8) << eigenUpTrunc << "  "
                              << eigenDownTrunc << "  " << occupancyUpTrunc
                              << "  " << occupancyDownTrunc << std::endl;
                      }
                  }
                else
                  {
                    double occupancy = dftUtils::getPartialOccupancy(
                      eigenValuesFlattenedGlobal[kPoint * d_numEigenValues +
                                                 iWave],
                      FE,
                      C_kb,
                      d_dftParamsPtr->TVal);
                    fprintf(
                      pFile,
                      "%d  %d %.14g %.14g\n",
                      kPoint,
                      iWave,
                      eigenValuesFlattenedGlobal[kPoint * d_numEigenValues +
                                                 iWave],
                      occupancy);
                    if (d_dftParamsPtr->reproducible_output &&
                        d_dftParamsPtr->verbosity == 0)
                      {
                        double eigenTrunc =
                          std::floor(1000000000 *
                                     (eigenValuesFlattenedGlobal
                                        [kPoint * d_numEigenValues + iWave])) /
                          1000000000.0;
                        double occupancyTrunc =
                          std::floor(1000000000 * (occupancy)) / 1000000000.0;
                        pcout << kPoint << "  " << iWave << "  " << std::fixed
                              << std::setprecision(8) << eigenTrunc << " "
                              << occupancyTrunc << std::endl;
                      }
                  }
              }
          }
      }
    MPI_Barrier(d_mpiCommParent);
    //
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getAtomLocationsCart() const
  {
    return atomLocations;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getImageAtomLocationsCart()
    const
  {
    return d_imagePositionsTrunc;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::vector<int> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getImageAtomIDs() const
  {
    return d_imageIdsTrunc;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getAtomLocationsFrac() const
  {
    return atomLocationsFractional;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::vector<std::vector<double>> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getCell() const
  {
    return d_domainBoundingVectors;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getCellVolume() const
  {
    return d_domainVolume;
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::set<unsigned int> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getAtomTypes() const
  {
    return atomTypes;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const std::vector<double> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getForceonAtoms() const
  {
    return (forcePtr->getAtomsForces());
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const dealii::Tensor<2, 3, double> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getCellStress() const
  {
    return (forcePtr->getStress());
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  dftParameters &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getParametersObject() const
  {
    return (*d_dftParamsPtr);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getInternalEnergy() const
  {
    return d_groundStateEnergy;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getEntropicEnergy() const
  {
    return d_entropicEnergy;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getFreeEnergy() const
  {
    return d_freeEnergy;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const distributedCPUVec<double> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getRhoNodalOut() const
  {
    return d_densityOutNodalValues[0];
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  const distributedCPUVec<double> &
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getRhoNodalSplitOut() const
  {
    return d_rhoOutNodalValuesSplit;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::getTotalChargeforRhoSplit()
  {
    double temp =
      (-totalCharge(d_matrixFreeDataPRefined, d_rhoOutNodalValuesSplit) /
       d_domainVolume);
    return (temp);
  }



  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::resetRhoNodalIn(
    distributedCPUVec<double> &OutDensity)
  {
    d_densityOutNodalValues[0] = OutDensity;
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::resetRhoNodalSplitIn(
    distributedCPUVec<double> &OutDensity)
  {
    d_rhoOutNodalValuesSplit = OutDensity;
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::writeGSElectronDensity(
    const std::string Path) const
  {
    const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);

    if (poolId == 0 && bandGroupId == 0)
      {
        std::vector<std::shared_ptr<dftUtils::CompositeData>> data(0);

        const dealii::Quadrature<3> &quadrature_formula =
          matrix_free_data.get_quadrature(d_densityQuadratureId);
        dealii::FEValues<3> fe_values(FE,
                                      quadrature_formula,
                                      dealii::update_quadrature_points |
                                        dealii::update_JxW_values);
        const unsigned int  n_q_points = quadrature_formula.size();

        // loop over elements
        typename dealii::DoFHandler<3>::active_cell_iterator
          cell = dofHandler.begin_active(),
          endc = dofHandler.end();
        for (; cell != endc; ++cell)
          {
            if (cell->is_locally_owned())
              {
                fe_values.reinit(cell);
                const unsigned int cellIndex =
                  d_basisOperationsPtrHost->cellIndex(cell->id());
                const double *rhoValues =
                  d_densityOutQuadValues[0].data() + cellIndex * n_q_points;
                const double *magValues =
                  d_dftParamsPtr->spinPolarized == 1 ?
                    d_densityOutQuadValues[1].data() + cellIndex * n_q_points :
                    NULL;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  {
                    std::vector<double> quadVals(0);

                    const dealii::Point<3> &quadPoint =
                      fe_values.quadrature_point(q_point);
                    const double jxw = fe_values.JxW(q_point);

                    quadVals.push_back(quadPoint[0]);
                    quadVals.push_back(quadPoint[1]);
                    quadVals.push_back(quadPoint[2]);
                    quadVals.push_back(jxw);

                    if (d_dftParamsPtr->spinPolarized == 1)
                      {
                        quadVals.push_back(rhoValues[q_point]);
                        quadVals.push_back(magValues[q_point]);
                      }
                    else
                      {
                        quadVals.push_back(rhoValues[q_point]);
                      }

                    data.push_back(
                      std::make_shared<dftUtils::QuadDataCompositeWrite>(
                        quadVals));
                  }
              }
          }

        std::vector<dftUtils::CompositeData *> dataRawPtrs(data.size());
        for (unsigned int i = 0; i < data.size(); ++i)
          dataRawPtrs[i] = data[i].get();
        dftUtils::MPIWriteOnFile().writeData(dataRawPtrs,
                                             Path,
                                             mpi_communicator);
      }
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::writeMesh()
  {
    //
    // compute nodal electron-density from quad data
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    l2ProjectionQuadToNodal(d_basisOperationsPtrElectroHost,
                            d_constraintsRhoNodal,
                            d_densityDofHandlerIndexElectro,
                            d_densityQuadratureIdElectro,
                            d_densityInQuadValues[0],
                            rhoNodalField);

    //
    // only generate output for electron-density
    //
    dealii::DataOut<3> dataOutRho;
    dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
    dataOutRho.add_data_vector(rhoNodalField, std::string("density"));
    dataOutRho.set_flags(dealii::DataOutBase::VtkFlags(
      std::numeric_limits<double>::min(),
      std::numeric_limits<unsigned int>::min(),
      true,
      dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::
        best_speed, // This flag is version dependent for dealII 9.5.0 it is
                    // dealii::DataOutBase::CompressionLevel::best_speed
      true));       // higher order cells set to true
    dataOutRho.build_patches(FEOrder);

    std::string tempFolder = "meshOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
                                               dataOutRho,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "intialDensityOutput");



    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << std::endl
        << "------------------DFT-FE mesh file creation completed---------------------------"
        << std::endl;
  }


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeResidualQuadData(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &outValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &inValues,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &residualValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &        JxW,
    const bool computeNorm)
  {
    std::transform(outValues.begin(),
                   outValues.end(),
                   inValues.begin(),
                   residualValues.begin(),
                   std::minus<>{});
    double normValue = 0.0;
    if (computeNorm)
      {
        for (unsigned int iQuad = 0; iQuad < residualValues.size(); ++iQuad)
          normValue +=
            residualValues[iQuad] * residualValues[iQuad] * JxW[iQuad];
        MPI_Allreduce(
          MPI_IN_PLACE, &normValue, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      }
    return std::sqrt(normValue);
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeResidualNodalData(
    const distributedCPUVec<double> &outValues,
    const distributedCPUVec<double> &inValues,
    distributedCPUVec<double> &      residualValues)
  {
    residualValues.reinit(inValues);

    residualValues = 0.0;

    // compute residual = rhoOut - rhoIn
    residualValues.add(1.0, outValues, -1.0, inValues);

    // compute l2 norm of the field residual
    double normValue = rhofieldl2Norm(d_matrixFreeDataPRefined,
                                      residualValues,
                                      d_densityDofHandlerIndexElectro,
                                      d_densityQuadratureIdElectro);
    return normValue;
  }
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    determineAtomsOfInterstPseudopotential(
      const std::vector<std::vector<double>> &atomCoordinates)
  {
    d_atomLocationsInterestPseudopotential.clear();
    d_atomIdPseudopotentialInterestToGlobalId.clear();
    unsigned atomIdPseudo = 0;
    // pcout<<"Atoms of interest: "<<std::endl;
    for (unsigned int iAtom = 0; iAtom < atomCoordinates.size(); iAtom++)
      {
        if (true)
          {
            d_atomLocationsInterestPseudopotential.push_back(
              atomCoordinates[iAtom]);
            d_atomIdPseudopotentialInterestToGlobalId[atomIdPseudo] = iAtom;
            // pcout<<iAtom<<" "<<atomIdPseudo<<" ";
            // for(int i = 0; i <
            // d_atomLocationsInterestPseudopotential[atomIdPseudo].size(); i++)
            //   pcout<<d_atomLocationsInterestPseudopotential[atomIdPseudo][i]<<"
            //   ";
            // pcout<<std::endl;
            atomIdPseudo++;
          }
      }
  }



#include "dft.inst.cc"
} // namespace dftfe
