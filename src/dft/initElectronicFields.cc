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
// @author  Phani Motamarri, Sambit Das
//
#include <dft.h>
#include <dftUtils.h>

namespace dftfe
{
  // init
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initElectronicFields()
  {
    dealii::TimerOutput::Scope scope(computing_timer, "init electronic fields");


    // initialize electrostatics fields
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_phiTotRhoIn, d_phiTotDofHandlerIndexElectro);
    d_phiTotRhoOut.reinit(d_phiTotRhoIn);
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_phiPrime, d_phiPrimeDofHandlerIndexElectro);
    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_phiExt, d_phiExtDofHandlerIndexElectro);

    d_densityInNodalValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    d_densityOutNodalValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);

    d_totalChargeDensityInNodalValues.resize(1);
    d_totalChargeDensityOutNodalValues.resize(1);

    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_densityInNodalValues[0], d_densityDofHandlerIndexElectro);
    for (unsigned int iComp = 1; iComp < d_densityInNodalValues.size(); ++iComp)
      d_densityInNodalValues[iComp].reinit(d_densityInNodalValues[0]);
    for (unsigned int iComp = 0; iComp < d_densityOutNodalValues.size();
         ++iComp)
      d_densityOutNodalValues[iComp].reinit(d_densityInNodalValues[0]);

    for (unsigned int iComp = 0; iComp < d_densityInNodalValues.size(); ++iComp)
      d_densityInNodalValues[iComp] = 0;
    for (unsigned int iComp = 0; iComp < d_densityOutNodalValues.size();
         ++iComp)
      d_densityOutNodalValues[iComp] = 0;

    d_matrixFreeDataPRefined.initialize_dof_vector(
      d_totalChargeDensityInNodalValues[0], d_densityDofHandlerIndexElectro);
    for (unsigned int iComp = 1;
         iComp < d_totalChargeDensityInNodalValues.size();
         ++iComp)
      d_totalChargeDensityInNodalValues[iComp].reinit(
        d_totalChargeDensityInNodalValues[0]);
    for (unsigned int iComp = 0;
         iComp < d_totalChargeDensityInNodalValues.size();
         ++iComp)
      d_totalChargeDensityOutNodalValues[iComp].reinit(
        d_totalChargeDensityInNodalValues[0]);

    for (unsigned int iComp = 0;
         iComp < d_totalChargeDensityInNodalValues.size();
         ++iComp)
      d_totalChargeDensityInNodalValues[iComp] = 0;
    for (unsigned int iComp = 0;
         iComp < d_totalChargeDensityInNodalValues.size();
         ++iComp)
      d_totalChargeDensityOutNodalValues[iComp] = 0;


    if ((d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
         d_dftParamsPtr->solverMode == "GEOOPT") ||
        (d_dftParamsPtr->extrapolateDensity == 2 &&
         d_dftParamsPtr->solverMode == "MD"))
      {
        initAtomicRho();
      }

    //
    // initialize eigen vectors
    //
    matrix_free_data.initialize_dof_vector(d_tempEigenVec,
                                           d_eigenDofHandlerIndex);

    //
    // store constraintEigen Matrix entries into STL vector
    //
    constraintsNoneEigenDataInfo.initialize(d_tempEigenVec.get_partitioner(),
                                            constraintsNoneEigen);

    constraintsNoneDataInfo.initialize(
      matrix_free_data.get_vector_partitioner(), constraintsNone);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      d_constraintsNoneDataInfoDevice.initialize(
        matrix_free_data.get_vector_partitioner(), constraintsNone);
#endif

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(
        mpi_communicator, "Overloaded constraint matrices initialized");

    //
    // initialize PSI and density
    //

    MPI_Barrier(d_mpiCommParent);
    if (matrix_free_data.get_vector_partitioner()->local_size() > 0)
      AssertThrow(
        (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size() *
            d_numEigenValues <
          INT_MAX / matrix_free_data.get_vector_partitioner()->local_size(),
        dealii::ExcMessage(
          "DFT-FE error: size of local wavefunctions storage exceeds integer bounds. Please increase number of MPI tasks"));

    d_eigenVectorsFlattenedHost.resize(
      (d_numEigenValues *
       matrix_free_data.get_vector_partitioner()->local_size()) *
        (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size(),
      dataTypes::number(0.0));
    if (d_numEigenValuesRR != d_numEigenValues)
      {
        d_eigenVectorsRotFracDensityFlattenedHost.resize(
          d_numEigenValuesRR *
            matrix_free_data.get_vector_partitioner()->local_size() *
            (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size(),
          dataTypes::number(0.0));
      }
    MPI_Barrier(d_mpiCommParent);
    pcout << std::endl
          << "Setting initial guess for wavefunctions...." << std::endl;

    if (d_dftParamsPtr->verbosity >= 2)
      dftUtils::printCurrentMemoryUsage(
        mpi_communicator,
        "Created flattened array eigenvectors before update ghost values");
    readPSI();
    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Created flattened array eigenvectors");

    // if(!(d_dftParamsPtr->chkType==2 && d_dftParamsPtr->restartFromChk))
    //{
    initRho();
    // d_rhoOutNodalValues.reinit(d_rhoInNodalValues);
    //}

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator, "initRho called");

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        d_eigenVectorsFlattenedDevice.resize(
          d_eigenVectorsFlattenedHost.size());

        if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
          d_eigenVectorsDensityMatrixPrimeFlattenedDevice.resize(
            d_eigenVectorsFlattenedHost.size());

        if (d_numEigenValuesRR != d_numEigenValues)
          d_eigenVectorsRotFracFlattenedDevice.resize(
            d_eigenVectorsRotFracDensityFlattenedHost.size());
        else
          d_eigenVectorsRotFracFlattenedDevice.resize(1);

        d_eigenVectorsFlattenedDevice.copyFrom(d_eigenVectorsFlattenedHost);
      }
#endif

    if (!d_dftParamsPtr->useDevice &&
        d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      {
        d_eigenVectorsDensityMatrixPrimeHost = d_eigenVectorsFlattenedHost;
      }

    if (d_dftParamsPtr->verbosity >= 2 && d_dftParamsPtr->spinPolarized == 1)
      pcout << std::endl
            << "net magnetization: "
            << totalMagnetization(d_densityInQuadValues[1]) << std::endl;
  }
#include "dft.inst.cc"
} // namespace dftfe
