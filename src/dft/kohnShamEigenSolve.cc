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
#include <complex>
#include <vector>
#include <dft.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  namespace internal
  {
    //     void
    //     pointWiseScaleWithDiagonal(const double *     diagonal,
    //                                const unsigned int numberFields,
    //                                const unsigned int numberDofs,
    //                                dataTypes::number *fieldsArrayFlattened)
    //     {
    //       const unsigned int inc = 1;

    //       for (unsigned int i = 0; i < numberDofs; ++i)
    //         {
    // #ifdef USE_COMPLEX
    //           double scalingCoeff = diagonal[i];
    //           zdscal_(&numberFields,
    //                   &scalingCoeff,
    //                   &fieldsArrayFlattened[i * numberFields],
    //                   &inc);
    // #else
    //           double scalingCoeff = diagonal[i];
    //           dscal_(&numberFields,
    //                  &scalingCoeff,
    //                  &fieldsArrayFlattened[i * numberFields],
    //                  &inc);
    // #endif
    //         }
    //     }
  } // namespace internal

  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeTraceXtHX(
    unsigned int       numberWaveFunctions,
    const unsigned int kpoint)
  {
    if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
      {
        std::vector<dataTypes::number> ProjHam, ProjS, projPawHam, ProjI,
          ProjSinv, projPawHamSinglePrec, projHamSinglePrec;
        std::vector<dataTypes::number> projSSinv, projSinvS;
        // linearAlgebraOperations::XtHX(, ProjHam);
        // linearAlgebraOperations::XtOX(, ProjS);
        // linearAlgebraOperations::XtpawHX(, ProjPawHam);
        unsigned int kPointIndex = kpoint;
        unsigned int spinType    = 0;
        MPI_Barrier(d_mpiCommParent);
        pcout << "XtpawHX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtpawHX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          projPawHam);
        MPI_Barrier(d_mpiCommParent);
        pcout << "XtOX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtOX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          ProjS);
        MPI_Barrier(d_mpiCommParent);
        pcout << "XtHX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtHX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          ProjHam);
        MPI_Barrier(d_mpiCommParent);
        pcout << "XtIX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtIX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          ProjI);
        MPI_Barrier(d_mpiCommParent);
        pcout << "XtSinvX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtSinvX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          ProjSinv);
        pcout << "XtSinvSX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtSinvSX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          projSinvS);
        pcout << "XtSSinvX called: " << std::endl;
        MPI_Barrier(d_mpiCommParent);
        linearAlgebraOperations::XtSinvSX(
          *d_kohnShamDFTOperatorPtr,
          d_eigenVectorsFlattenedHost.data() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          mpi_communicator,
          interBandGroupComm,
          *d_dftParamsPtr,
          projSSinv);
        MPI_Barrier(d_mpiCommParent);
        if (d_dftParamsPtr->useSinglePrecCheby)
          {
            pcout << "XtpawHXSinglePrec called: " << std::endl;
            MPI_Barrier(d_mpiCommParent);
            linearAlgebraOperations::XtpawHXsinglePrecision(
              *d_kohnShamDFTOperatorPtr,
              d_eigenVectorsFlattenedHost.data() +
                ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
                  d_numEigenValues *
                  matrix_free_data.get_vector_partitioner()
                    ->locally_owned_size(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              mpi_communicator,
              interBandGroupComm,
              *d_dftParamsPtr,
              projPawHamSinglePrec);
            MPI_Barrier(d_mpiCommParent);
            pcout << "XtpawHXSinglePrec called: " << std::endl;
            MPI_Barrier(d_mpiCommParent);
            linearAlgebraOperations::XtHXsinglePrecision(
              *d_kohnShamDFTOperatorPtr,
              d_eigenVectorsFlattenedHost.data() +
                ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
                  d_numEigenValues *
                  matrix_free_data.get_vector_partitioner()
                    ->locally_owned_size(),
              d_numEigenValues,
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
              mpi_communicator,
              interBandGroupComm,
              *d_dftParamsPtr,
              projHamSinglePrec);
          }
        MPI_Barrier(d_mpiCommParent);
        // d_BLASWrapperHostPtr->xnrm2();
        pcout << "XtIX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << ProjI[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        pcout << "XtHX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << ProjHam[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        pcout << "XtSX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << ProjS[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        pcout << "XtpawHX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << projPawHam[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        pcout << "XtSinvX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << ProjSinv[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        pcout << "XtSinvSX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << projSinvS[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        pcout << "XtSSinvX: ";
        for (int i = 0; i < d_numEigenValues; i++)
          pcout << projSSinv[i * d_numEigenValues + i] << " ";
        pcout << std::endl;
        if (d_dftParamsPtr->useSinglePrecCheby)
          {
            pcout << "XtpawHXSinglePrec: ";
            for (int i = 0; i < d_numEigenValues; i++)
              pcout << projPawHamSinglePrec[i * d_numEigenValues + i] << " ";
            pcout << std::endl;
            pcout << "XtHXSinglePrec: ";
            for (int i = 0; i < d_numEigenValues; i++)
              pcout << projHamSinglePrec[i * d_numEigenValues + i] << " ";
            pcout << std::endl;
          }
        // std::exit(0);
      }
    else
      {}
    // //
    // // set up poisson solver
    // //
    // dealiiLinearSolver                            CGSolver(d_mpiCommParent,
    //                             mpi_communicator,
    //                             dealiiLinearSolver::CG);
    // poissonSolverProblem<FEOrder, FEOrderElectro> phiTotalSolverProblem(
    //   mpi_communicator);

    // //
    // // solve for vself and compute Tr(XtHX)
    // //
    // d_vselfBinsManager.solveVselfInBins(d_basisOperationsPtrElectroHost,
    //                                     d_binsStartDofHandlerIndexElectro,
    //                                     d_phiTotAXQuadratureIdElectro,
    //                                     d_constraintsPRefined,
    //                                     d_imagePositionsTrunc,
    //                                     d_imageIdsTrunc,
    //                                     d_imageChargesTrunc,
    //                                     d_localVselfs,
    //                                     d_bQuadValuesAllAtoms,
    //                                     d_bQuadAtomIdsAllAtomsImages,
    //                                     d_bQuadAtomIdsAllAtoms,
    //                                     d_bCellNonTrivialAtomIds,
    //                                     d_bCellNonTrivialAtomIdsBins,
    //                                     d_bCellNonTrivialAtomImageIds,
    //                                     d_bCellNonTrivialAtomImageIdsBins,
    //                                     d_smearedChargeWidths,
    //                                     d_smearedChargeScaling,
    //                                     d_smearedChargeQuadratureIdElectro);

    // //
    // // solve for potential corresponding to initial electron-density
    // //
    // phiTotalSolverProblem.reinit(
    //   d_basisOperationsPtrElectroHost,
    //   d_phiTotRhoIn,
    //   *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
    //   d_phiTotDofHandlerIndexElectro,
    //   d_phiTotAXQuadratureIdElectro,
    //   d_densityQuadratureIdElectro,
    //   d_atomNodeIdToChargeMap,
    //   d_bQuadValuesAllAtoms,
    //   d_smearedChargeQuadratureIdElectro,
    //   d_densityInQuadValues[0],
    //   true,
    //   d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
    //     d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC,
    //   d_dftParamsPtr->smearedNuclearCharges);

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    //   phiInValues;

    // CGSolver.solve(phiTotalSolverProblem,
    //                d_dftParamsPtr->absLinearSolverTolerance,
    //                d_dftParamsPtr->maxLinearSolverIterations,
    //                d_dftParamsPtr->verbosity);

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    // dummy; interpolateElectroNodalDataToQuadratureDataGeneral(
    //   d_basisOperationsPtrElectroHost,
    //   d_phiTotDofHandlerIndexElectro,
    //   d_densityQuadratureIdElectro,
    //   d_phiTotRhoIn,
    //   phiInValues,
    //   dummy);

    // //
    // // create kohnShamDFTOperatorClass object
    // //
    // kohnShamDFTOperatorClass<FEOrder, FEOrderElectro, memorySpace>
    //   kohnShamDFTEigenOperator(this, d_mpiCommParent, mpi_communicator);
    // kohnShamDFTEigenOperator.init();

    // //
    // // precompute shapeFunctions and shapeFunctionGradients and
    // // shapeFunctionGradientIntegrals
    // //
    // kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
    //   d_lpspQuadratureId);

    // //
    // // compute Veff
    // //
    // if (d_excManagerPtr->getDensityBasedFamilyType() ==
    // densityFamilyType::LDA)
    //   {
    //     kohnShamDFTEigenOperator.computeVEff(d_densityInQuadValues,
    //                                          phiInValues,
    //                                          d_pseudoVLoc,
    //                                          d_rhoCore,
    //                                          d_lpspQuadratureId);
    //   }
    // else if (d_excManagerPtr->getDensityBasedFamilyType() ==
    //          densityFamilyType::GGA)
    //   {
    //     kohnShamDFTEigenOperator.computeVEff(d_densityInQuadValues,
    //                                          d_gradDensityInQuadValues,
    //                                          phiInValues,
    //                                          d_pseudoVLoc,
    //                                          d_rhoCore,
    //                                          d_gradRhoCore,
    //                                          d_lpspQuadratureId);
    //   }

    // //
    // // compute Hamiltonian matrix
    // //
    // kohnShamDFTEigenOperator.computeHamiltonianMatrix(0, 0);

    // //
    // // scale the eigenVectors (initial guess of single atom wavefunctions or
    // // previous guess) to convert into Lowden Orthonormalized FE basis
    // multiply
    // // by M^{1/2}
    // internal::pointWiseScaleWithDiagonal(
    //   kohnShamDFTEigenOperator.d_sqrtMassVector,
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   d_eigenVectorsFlattenedHost.data());


    // //
    // // compute projected Hamiltonian
    // //
    // std::vector<dataTypes::number> ProjHam;

    // dftfe::linearAlgebraOperations::XtHX(
    //   kohnShamDFTEigenOperator,
    //   d_eigenVectorsFlattenedHost.data(),
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   mpi_communicator,
    //   interBandGroupComm,
    //   *d_dftParamsPtr,
    //   ProjHam);

    // //
    // // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // // the usual FE basis
    // //
    // internal::pointWiseScaleWithDiagonal(
    //   kohnShamDFTEigenOperator.getInverseSqrtMassVector().data(),
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   d_eigenVectorsFlattenedHost.data());


    // dataTypes::number trXtHX = 0.0;
    // for (unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
    //   {
    //     trXtHX += ProjHam[d_numEigenValues * i + i];
    //   }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeTraceXtKX(
    unsigned int numberWaveFunctionsEstimate)
  {
    //     //
    //     // create kohnShamDFTOperatorClass object
    //     //
    //     kohnShamDFTOperatorClass<FEOrder, FEOrderElectro, memorySpace>
    //       kohnShamDFTEigenOperator(this, d_mpiCommParent, mpi_communicator);
    //     kohnShamDFTEigenOperator.init();

    //     //
    //     // precompute shapeFunctions and shapeFunctionGradients and
    //     // shapeFunctionGradientIntegrals
    //     //
    //     kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
    //       d_lpspQuadratureId);


    //     //
    //     // compute Hamiltonian matrix
    //     //
    //     kohnShamDFTEigenOperator.computeKineticMatrix();

    //     //
    //     // scale the eigenVectors (initial guess of single atom wavefunctions
    //     or
    //     // previous guess) to convert into Lowden Orthonormalized FE basis
    //     multiply
    //     // by M^{1/2}
    //     internal::pointWiseScaleWithDiagonal(
    //       kohnShamDFTEigenOperator.d_sqrtMassVector,
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       d_eigenVectorsFlattenedHost.data());


    //     //
    //     // orthogonalize the vectors
    //     //
    //     linearAlgebraOperations::gramSchmidtOrthogonalization(
    //       d_eigenVectorsFlattenedHost.data(),
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       mpi_communicator);

    //     //
    //     // compute projected Hamiltonian
    //     //
    //     std::vector<dataTypes::number> ProjHam;

    //     dftfe::linearAlgebraOperations::XtHX(
    //       kohnShamDFTEigenOperator,
    //       d_eigenVectorsFlattenedHost.data(),
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       mpi_communicator,
    //       interBandGroupComm,
    //       *d_dftParamsPtr,
    //       ProjHam);

    //     //
    //     // scale the eigenVectors with M^{-1/2} to represent the
    //     wavefunctions in
    //     // the usual FE basis
    //     //
    //     internal::pointWiseScaleWithDiagonal(
    //       kohnShamDFTEigenOperator.getInverseSqrtMassVector().data(),
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       d_eigenVectorsFlattenedHost.data());

    //     double trXtKX = 0.0;
    // #ifdef USE_COMPLEX
    //     trXtKX = 0.0;
    // #else
    //     for (unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
    //       {
    //         trXtKX += ProjHam[d_numEigenValues * i + i];
    //       }
    // #endif

    //     return trXtKX;
  }



  // chebyshev solver
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::kohnShamEigenSpaceCompute(
    const unsigned int spinType,
    const unsigned int kPointIndex,
    KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>
      &                                             kohnShamDFTEigenOperator,
    elpaScalaManager &                              elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
    std::vector<double> &                           residualNormWaveFunctions,
    const bool                                      computeResidual,
    const bool                                      isSpectrumSplit,
    const bool                                      useMixedPrec,
    const bool                                      isFirstScf)
  {
    computing_timer.enter_subsection("Chebyshev solve");


    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }

    std::vector<double> eigenValuesTemp(isSpectrumSplit ? d_numEigenValuesRR :
                                                          d_numEigenValues,
                                        0.0);
    if (d_dftParamsPtr->useSinglePrecCheby)
      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          eigenValuesTemp[i] =
            eigenValues[kPointIndex][spinType * d_numEigenValues + i];
        }

    if (d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) *
                                 kPointIndex +
                               spinType])
      {
        computing_timer.enter_subsection("Lanczos k-step Upper Bound");

        std::pair<double, double> bounds = linearAlgebraOperations::
          generalisedLanczosLowerUpperBoundEigenSpectrum(
            d_BLASWrapperPtrHost,
            kohnShamDFTEigenOperator,
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 1),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 2),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 3),
            *d_dftParamsPtr);

        const double upperBoundUnwantedSpectrum = bounds.second;
        const double lowerBoundWantedSpectrum   = bounds.first;

        a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
          lowerBoundWantedSpectrum;
        computing_timer.leave_subsection("Lanczos k-step Upper Bound");

        d_upperBoundUnwantedSpectrumValues[(1 + d_dftParamsPtr->spinPolarized) *
                                             kPointIndex +
                                           spinType] =
          upperBoundUnwantedSpectrum;

        subspaceIterationSolver.reinitSpectrumBounds(
          lowerBoundWantedSpectrum,
          lowerBoundWantedSpectrum +
            (upperBoundUnwantedSpectrum - lowerBoundWantedSpectrum) /
              kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0)
                .globalSize() *
              d_numEigenValues *
              (d_dftParamsPtr->reproducible_output ? 10.0 : 200.0),
          upperBoundUnwantedSpectrum);
      }
    else
      {
        if (!d_dftParamsPtr->reuseLanczosUpperBoundFromFirstCall)
          {
            computing_timer.enter_subsection("Lanczos k-step Upper Bound");

            std::pair<double, double> bounds = linearAlgebraOperations::
              generalisedLanczosLowerUpperBoundEigenSpectrum(
                d_BLASWrapperPtrHost,
                kohnShamDFTEigenOperator,
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0),
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 1),
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 2),
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 3),
                *d_dftParamsPtr);
            d_upperBoundUnwantedSpectrumValues
              [(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
                bounds.second;
            computing_timer.leave_subsection("Lanczos k-step Upper Bound");
          }

        subspaceIterationSolver.reinitSpectrumBounds(
          a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
          bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
          d_upperBoundUnwantedSpectrumValues
            [(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType]);
      }

    subspaceIterationSolver.solve(
      kohnShamDFTEigenOperator,
      d_BLASWrapperPtrHost,
      elpaScala,
      d_eigenVectorsFlattenedHost.data() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_eigenVectorsRotFracDensityFlattenedHost.data() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValuesRR *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues,
      matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      eigenValuesTemp,
      residualNormWaveFunctions,
      interBandGroupComm,
      mpi_communicator,
      d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                             spinType],
      computeResidual,
      useMixedPrec,
      isFirstScf);

    //
    // copy the eigenValues and corresponding residual norms back to data
    // members
    //
    if (isSpectrumSplit)
      {
        for (unsigned int i = 0; i < d_numEigenValuesRR; i++)
          {
            if (d_dftParamsPtr->verbosity >= 4 &&
                d_numEigenValues == d_numEigenValuesRR)
              pcout << "eigen value " << std::setw(3) << i << ": "
                    << eigenValuesTemp[i] << std::endl;
            else if (d_dftParamsPtr->verbosity >= 4 &&
                     d_numEigenValues != d_numEigenValuesRR)
              pcout << "valence eigen value " << std::setw(3) << i << ": "
                    << eigenValuesTemp[i] << std::endl;

            eigenValuesRRSplit[kPointIndex][spinType * d_numEigenValuesRR + i] =
              eigenValuesTemp[i];
          }

        for (unsigned int i = 0; i < d_numEigenValues; i++)
          {
            if (i >= (d_numEigenValues - d_numEigenValuesRR))
              eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                eigenValuesTemp[i - (d_numEigenValues - d_numEigenValuesRR)];
            else
              eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                -100.0;
          }
      }
    else
      {
        for (unsigned int i = 0; i < d_numEigenValues; i++)
          {
            if (d_dftParamsPtr->verbosity >= 4)
              pcout << "eigen value " << std::setw(3) << i << ": "
                    << eigenValuesTemp[i] << std::endl;

            eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
              eigenValuesTemp[i];
          }
      }

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << std::endl;


    // set a0 and bLow
    /* a0[(1+d_dftParamsPtr->spinPolarized)*kPointIndex+spinType]=isSpectrumSplit?
       d_dftParamsPtr->lowerEndWantedSpectrum
       :eigenValuesTemp[0];*/


    bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp.back();
    d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                           spinType] = false;

    if (!isSpectrumSplit)
      {
        a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
          eigenValuesTemp[0];
      }

    computing_timer.leave_subsection("Chebyshev solve");
  }

#ifdef DFTFE_WITH_DEVICE
  // chebyshev solver
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::kohnShamEigenSpaceCompute(
    const unsigned int spinType,
    const unsigned int kPointIndex,
    KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>
      &               kohnShamDFTEigenOperator,
    elpaScalaManager &elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolverDevice
      &                  subspaceIterationSolverDevice,
    std::vector<double> &residualNormWaveFunctions,
    const bool           computeResidual,
    const unsigned int   numberRayleighRitzAvoidancePasses,
    const bool           isSpectrumSplit,
    const bool           useMixedPrec,
    const bool           isFirstScf)
  {
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }

    std::vector<double> eigenValuesTemp(isSpectrumSplit ? d_numEigenValuesRR :
                                                          d_numEigenValues,
                                        0.0);
    std::vector<double> eigenValuesDummy(isSpectrumSplit ? d_numEigenValuesRR :
                                                           d_numEigenValues,
                                         0.0);
    if (d_dftParamsPtr->useSinglePrecCheby)
      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          eigenValuesTemp[i] =
            eigenValues[kPointIndex][spinType * d_numEigenValues + i];
        }

    subspaceIterationSolverDevice.reinitSpectrumBounds(
      a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
      bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
      d_upperBoundUnwantedSpectrumValues[(1 + d_dftParamsPtr->spinPolarized) *
                                           kPointIndex +
                                         spinType]);

    if (numberRayleighRitzAvoidancePasses > 0)
      {
        subspaceIterationSolverDevice.solveNoRR(
          kohnShamDFTEigenOperator,
          d_BLASWrapperPtr,
          elpaScala,
          d_eigenVectorsFlattenedDevice.begin() +
            ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
              d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues *
            matrix_free_data.get_vector_partitioner()->locally_owned_size(),
          d_numEigenValues,
          eigenValuesDummy,
          *d_devicecclMpiCommDomainPtr,
          interBandGroupComm,
          numberRayleighRitzAvoidancePasses,
          useMixedPrec);
      }
    else
      {
        d_upperBoundUnwantedSpectrumValues[(1 + d_dftParamsPtr->spinPolarized) *
                                             kPointIndex +
                                           spinType] =
          subspaceIterationSolverDevice.solve(
            kohnShamDFTEigenOperator,
            d_BLASWrapperPtr,
            elpaScala,
            d_eigenVectorsFlattenedDevice.begin() +
              ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
                d_numEigenValues *
                matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            d_eigenVectorsRotFracFlattenedDevice.begin() +
              ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
                d_numEigenValuesRR *
                matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            d_numEigenValues *
              matrix_free_data.get_vector_partitioner()->locally_owned_size(),
            d_numEigenValues,
            eigenValuesTemp,
            residualNormWaveFunctions,
            *d_devicecclMpiCommDomainPtr,
            interBandGroupComm,
            d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) *
                                     kPointIndex +
                                   spinType],
            computeResidual,
            useMixedPrec,
            isFirstScf);



        //
        // copy the eigenValues and corresponding residual norms back to data
        // members
        //
        if (isSpectrumSplit)
          {
            for (unsigned int i = 0; i < d_numEigenValuesRR; i++)
              {
                if (d_dftParamsPtr->verbosity >= 5 &&
                    d_numEigenValues == d_numEigenValuesRR)
                  pcout << "eigen value " << std::setw(3) << i << ": "
                        << eigenValuesTemp[i] << std::endl;
                else if (d_dftParamsPtr->verbosity >= 5 &&
                         d_numEigenValues != d_numEigenValuesRR)
                  pcout << "valence eigen value " << std::setw(3) << i << ": "
                        << eigenValuesTemp[i] << std::endl;

                eigenValuesRRSplit[kPointIndex][spinType * d_numEigenValuesRR +
                                                i] = eigenValuesTemp[i];
              }

            for (unsigned int i = 0; i < d_numEigenValues; i++)
              {
                if (i >= (d_numEigenValues - d_numEigenValuesRR))
                  eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                    eigenValuesTemp[i -
                                    (d_numEigenValues - d_numEigenValuesRR)];
                else
                  eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                    -100.0;
              }
          }
        else
          {
            for (unsigned int i = 0; i < d_numEigenValues; i++)
              {
                if (d_dftParamsPtr->verbosity >= 5)
                  pcout << "eigen value " << std::setw(3) << i << ": "
                        << eigenValuesTemp[i] << std::endl;

                eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                  eigenValuesTemp[i];
              }
          }

        if (d_dftParamsPtr->verbosity >= 4)
          pcout << std::endl;


        bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
          eigenValuesTemp.back();
        d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) *
                                 kPointIndex +
                               spinType] = false;
        if (!isSpectrumSplit)
          {
            a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
              eigenValuesTemp[0];
          }
      }
  }
#endif


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    kohnShamEigenSpaceFirstOrderDensityMatResponse(
      const unsigned int spinType,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala)
  {
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }



    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);
    for (unsigned int i = 0; i < d_numEigenValues; i++)
      {
        eigenValuesTemp[i] =
          eigenValues[kPointIndex][spinType * d_numEigenValues + i];
      }


    linearAlgebraOperations::densityMatrixEigenBasisFirstOrderResponse(
      kohnShamDFTEigenOperator,
      d_eigenVectorsDensityMatrixPrimeHost.data() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues,
      matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_mpiCommParent,
      kohnShamDFTEigenOperator.getMPICommunicatorDomain(),
      interBandGroupComm,
      eigenValuesTemp,
      fermiEnergy,
      d_densityMatDerFermiEnergy[(1 + d_dftParamsPtr->spinPolarized) *
                                   kPointIndex +
                                 spinType],
      elpaScala,
      *d_dftParamsPtr);
  }

#ifdef DFTFE_WITH_DEVICE
  // chebyshev solver
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    kohnShamEigenSpaceFirstOrderDensityMatResponse(
      const unsigned int spinType,
      const unsigned int kPointIndex,
      KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::DEVICE>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolverDevice
        &subspaceIterationSolverDevice)
  {
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }

    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);
    for (unsigned int i = 0; i < d_numEigenValues; i++)
      {
        eigenValuesTemp[i] =
          eigenValues[kPointIndex][spinType * d_numEigenValues + i];
      }

    subspaceIterationSolverDevice.densityMatrixEigenBasisFirstOrderResponse(
      kohnShamDFTEigenOperator,
      d_BLASWrapperPtr,
      d_eigenVectorsDensityMatrixPrimeFlattenedDevice.begin() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues *
        matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues,
      eigenValuesTemp,
      fermiEnergy,
      d_densityMatDerFermiEnergy[(1 + d_dftParamsPtr->spinPolarized) *
                                   kPointIndex +
                                 spinType],
      *d_devicecclMpiCommDomainPtr,
      interBandGroupComm,
      elpaScala);
  }
#endif

  // chebyshev solver
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::kohnShamEigenSpaceComputeNSCF(
    const unsigned int spinType,
    const unsigned int kPointIndex,
    KohnShamHamiltonianOperator<dftfe::utils::MemorySpace::HOST>
      &                                             kohnShamDFTEigenOperator,
    chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
    std::vector<double> &                           residualNormWaveFunctions,
    unsigned int                                    ipass)
  {
    computing_timer.enter_subsection("Chebyshev solve");

    if (d_dftParamsPtr->verbosity == 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        pcout << "spin: " << spinType + 1 << std::endl;
      }



    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);
    if (d_dftParamsPtr->useSinglePrecCheby)
      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          eigenValuesTemp[i] =
            eigenValues[kPointIndex][spinType * d_numEigenValues + i];
        }


    if (d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) *
                                 kPointIndex +
                               spinType])
      {
        computing_timer.enter_subsection("Lanczos k-step Upper Bound");

        std::pair<double, double> bounds = linearAlgebraOperations::
          generalisedLanczosLowerUpperBoundEigenSpectrum(
            d_BLASWrapperPtrHost,
            kohnShamDFTEigenOperator,
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 1),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 2),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 3),
            *d_dftParamsPtr);
        const double upperBoundUnwantedSpectrum = bounds.second;
        const double lowerBoundWantedSpectrum   = bounds.first;
        a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
          lowerBoundWantedSpectrum;
        computing_timer.leave_subsection("Lanczos k-step Upper Bound");

        subspaceIterationSolver.reinitSpectrumBounds(
          lowerBoundWantedSpectrum,
          lowerBoundWantedSpectrum +
            (upperBoundUnwantedSpectrum - lowerBoundWantedSpectrum) /
              kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0)
                .globalSize() *
              d_numEigenValues *
              (d_dftParamsPtr->reproducible_output ? 10.0 : 200.0),
          upperBoundUnwantedSpectrum);
      }
    else
      {
        computing_timer.enter_subsection("Lanczos k-step Upper Bound");

        std::pair<double, double> bounds = linearAlgebraOperations::
          generalisedLanczosLowerUpperBoundEigenSpectrum(
            d_BLASWrapperPtrHost,
            kohnShamDFTEigenOperator,
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 1),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 2),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 3),

            *d_dftParamsPtr);
        const double upperBoundUnwantedSpectrum = bounds.second;
        computing_timer.leave_subsection("Lanczos k-step Upper Bound");

        subspaceIterationSolver.reinitSpectrumBounds(
          a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
          bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
          upperBoundUnwantedSpectrum);
      }


    subspaceIterationSolver.solve(
      kohnShamDFTEigenOperator,
      d_BLASWrapperPtrHost,
      *d_elpaScala,
      d_eigenVectorsFlattenedHost.data() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_eigenVectorsFlattenedHost.data() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues,
      matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      eigenValuesTemp,
      residualNormWaveFunctions,
      interBandGroupComm,
      mpi_communicator,
      d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                             spinType],
      true,
      false);


    if (d_dftParamsPtr->verbosity >= 5)
      {
#ifdef USE_PETSC
        PetscLogDouble bytes;
        PetscMemoryGetCurrentUsage(&bytes);
        FILE *       dummy;
        unsigned int this_mpi_process =
          dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
        PetscSynchronizedPrintf(
          mpi_communicator,
          "[%d] Memory after recreating STL vector and exiting from subspaceIteration solver  %e\n",
          this_mpi_process,
          bytes);
        PetscSynchronizedFlush(mpi_communicator, dummy);
#endif
      }



    //
    // copy the eigenValues and corresponding residual norms back to data
    // members
    //
    for (unsigned int i = 0; i < d_numEigenValues; i++)
      {
        // if(d_dftParamsPtr->verbosity==2)
        //    pcout<<"eigen value "<< std::setw(3) <<i <<":
        //    "<<eigenValuesTemp[i]
        //    <<std::endl;

        eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
          eigenValuesTemp[i];
      }

    // if (d_dftParamsPtr->verbosity==2)
    //   pcout <<std::endl;


    // set a0 and bLow
    a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp[0];
    bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp.back();
    d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                           spinType] = false;
    //


    computing_timer.leave_subsection("Chebyshev solve");
  }



  // compute the maximum of the residual norm of the highest state of interest
  // across all K points
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeMaximumHighestOccupiedStateResidualNorm(
      const std::vector<std::vector<double>>
        &residualNormWaveFunctionsAllkPoints,
      const std::vector<std::vector<double>> &eigenValuesAllkPoints,
      const unsigned int                      highestState)
  {
    double maxHighestOccupiedStateResNorm = -1e+6;
    for (int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
      {
        if (residualNormWaveFunctionsAllkPoints[kPoint][highestState] >
            maxHighestOccupiedStateResNorm)
          {
            maxHighestOccupiedStateResNorm =
              residualNormWaveFunctionsAllkPoints[kPoint][highestState];
          }
      }
    maxHighestOccupiedStateResNorm =
      dealii::Utilities::MPI::max(maxHighestOccupiedStateResNorm,
                                  interpoolcomm);
    return maxHighestOccupiedStateResNorm;
  }
  // compute the maximum of the residual norm of the highest occupied state
  // among all k points
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::
    computeMaximumHighestOccupiedStateResidualNorm(
      const std::vector<std::vector<double>>
        &residualNormWaveFunctionsAllkPoints,
      const std::vector<std::vector<double>> &eigenValuesAllkPoints,
      const double                            fermiEnergy)
  {
    double maxHighestOccupiedStateResNorm = -1e+6;
    if (d_dftParamsPtr->reproducible_output)
      {
        for (int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
          {
            unsigned int highestOccupiedState = 0;

            for (unsigned int i = 0; i < eigenValuesAllkPoints[kPoint].size();
                 i++)
              {
                const double factor =
                  (eigenValuesAllkPoints[kPoint][i] - fermiEnergy) /
                  (C_kb * d_dftParamsPtr->TVal);
                if (factor < 0)
                  highestOccupiedState = i;
              }

            if (residualNormWaveFunctionsAllkPoints[kPoint]
                                                   [highestOccupiedState] >
                maxHighestOccupiedStateResNorm)
              {
                maxHighestOccupiedStateResNorm =
                  residualNormWaveFunctionsAllkPoints[kPoint]
                                                     [highestOccupiedState];
              }
          }
      }
    else
      {
        for (int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
          {
            unsigned int highestOccupiedState = 0;

            for (unsigned int i = 0; i < eigenValuesAllkPoints[kPoint].size();
                 i++)
              {
                const double factor =
                  (eigenValuesAllkPoints[kPoint][i] - fermiEnergy) /
                  (C_kb * d_dftParamsPtr->TVal);
                double functionValue;
                if (factor <= 0.0)
                  {
                    double temp2 = 1.0 / (1.0 + exp(factor));
                    functionValue =
                      (2.0 - d_dftParamsPtr->spinPolarized) * temp2;
                  }
                else
                  {
                    double temp2  = 1.0 / (1.0 + exp(-factor));
                    functionValue = (2.0 - d_dftParamsPtr->spinPolarized) *
                                    exp(-factor) * temp2;
                  }
                if (functionValue > 1e-3)
                  highestOccupiedState = i;
              }
            for (unsigned int i = 0; i <= highestOccupiedState; i++)
              {
                if (residualNormWaveFunctionsAllkPoints[kPoint][i] >
                    maxHighestOccupiedStateResNorm)
                  {
                    maxHighestOccupiedStateResNorm =
                      residualNormWaveFunctionsAllkPoints[kPoint][i];
                  }
              }
          }
      }
    maxHighestOccupiedStateResNorm =
      dealii::Utilities::MPI::max(maxHighestOccupiedStateResNorm,
                                  interpoolcomm);
    return maxHighestOccupiedStateResNorm;
  }
#include "dft.inst.cc"
} // namespace dftfe
