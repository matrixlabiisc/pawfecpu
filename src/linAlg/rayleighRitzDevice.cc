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
// @author Sambit Das


#include "dftUtils.h"
#include "linearAlgebraOperationsDevice.h"
#include "linearAlgebraOperationsInternal.h"
#include "constants.h"
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
    namespace
    {
      __global__ void
      setZeroKernel(const unsigned int BVec,
                    const unsigned int M,
                    const unsigned int N,
                    double *           yVec,
                    const unsigned int startingXVecId)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const unsigned int gangBlockId = blockIdx.x / numGangsPerBVec;
        const unsigned int localThreadId =
          globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

        if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
            localThreadId < BVec)
          {
            *(yVec + gangBlockId * N + startingXVecId + localThreadId) = 0.0;
          }
      }


      __global__ void
      setZeroKernel(const unsigned int                 BVec,
                    const unsigned int                 M,
                    const unsigned int                 N,
                    dftfe::utils::deviceDoubleComplex *yVec,
                    const unsigned int                 startingXVecId)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const unsigned int gangBlockId = blockIdx.x / numGangsPerBVec;
        const unsigned int localThreadId =
          globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

        if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
            localThreadId < BVec)
          {
            *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
              dftfe::utils::makeComplex(0.0, 0.0);
          }
      }
    } // namespace

    void
    rayleighRitz(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      elpaScalaManager &                                   elpaScala,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &         devicecclMpiCommDomain,
      const MPI_Comm &                  interBandGroupComm,
      std::vector<double> &             eigenValues,
      dftfe::utils::deviceBlasHandle_t &handle,
      const dftParameters &             dftParams,
      const bool                        useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(N,
                                                           processGrid,
                                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  dataTypes::number(0.0));


      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecXTHXSpectrumSplit && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR step");
          else
            computing_timer.enter_subsection(
              "HConjProj=X^{T}*HConj*XConj, RR step");
        }


      if (useMixedPrecOverall && dftParams.useMixedPrecXTHXSpectrumSplit)
        {
          if (dftParams.useMixedPrecCommunOnlyXTHXCGSO)
            XtHXMixedPrecCommunOverlapComputeCommun(operatorMatrix,
                                                    X,
                                                    Xb,
                                                    HXb,
                                                    M,
                                                    N,
                                                    dftParams.numCoreWfcXtHX,
                                                    handle,
                                                    processGrid,
                                                    projHamPar,
                                                    devicecclMpiCommDomain,
                                                    mpiCommDomain,
                                                    interBandGroupComm,
                                                    dftParams);
          else
            XtHXMixedPrecOverlapComputeCommun(operatorMatrix,
                                              X,
                                              Xb,
                                              HXb,
                                              M,
                                              N,
                                              dftParams.numCoreWfcXtHX,
                                              handle,
                                              processGrid,
                                              projHamPar,
                                              devicecclMpiCommDomain,
                                              mpiCommDomain,
                                              interBandGroupComm,
                                              dftParams);
        }
      else
        {
          if (dftParams.overlapComputeCommunOrthoRR)
            XtHXOverlapComputeCommun(operatorMatrix,
                                     X,
                                     Xb,
                                     HXb,
                                     M,
                                     N,
                                     handle,
                                     processGrid,
                                     projHamPar,
                                     devicecclMpiCommDomain,
                                     mpiCommDomain,
                                     interBandGroupComm,
                                     dftParams);
          else
            XtHX(operatorMatrix,
                 X,
                 Xb,
                 HXb,
                 M,
                 N,
                 handle,
                 processGrid,
                 projHamPar,
                 devicecclMpiCommDomain,
                 mpiCommDomain,
                 interBandGroupComm,
                 dftParams);
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecXTHXSpectrumSplit && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR step");
          else
            computing_timer.leave_subsection(
              "HConjProj=X^{T}*HConj*XConj, RR step");
        }

      //
      // compute eigendecomposition of ProjHam HConjProj= QConj*D*QConj^{C} (C
      // denotes conjugate transpose LAPACK notation)
      //
      const unsigned int numberEigenValues = N;
      eigenValues.resize(numberEigenValues);
      if (dftParams.useELPA)
        {
          if (dftParams.deviceFineGrainedTimings)
            {
              dftfe::utils::deviceSynchronize();
              computing_timer.enter_subsection("ELPA eigen decomp, RR step");
            }
          dftfe::ScaLAPACKMatrix<dataTypes::number> eigenVectors(N,
                                                                 processGrid,
                                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      dataTypes::number(0.0));

          // For ELPA eigendecomposition the full matrix is required unlike
          // ScaLAPACK which can work with only the lower triangular part
          dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParConjTrans(
            N, processGrid, rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&projHamParConjTrans.local_el(0, 0),
                      &projHamParConjTrans.local_el(0, 0) +
                        projHamParConjTrans.local_m() *
                          projHamParConjTrans.local_n(),
                      dataTypes::number(0.0));


          projHamParConjTrans.copy_conjugate_transposed(projHamPar);
          projHamPar.add(projHamParConjTrans,
                         dataTypes::number(1.0),
                         dataTypes::number(1.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
              {
                const unsigned int glob_i = projHamPar.global_column(i);
                for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
                  {
                    const unsigned int glob_j = projHamPar.global_row(j);
                    if (glob_i == glob_j)
                      projHamPar.local_el(j, i) *= dataTypes::number(0.5);
                  }
              }

          if (processGrid->is_process_active())
            {
              int error;
              elpa_eigenvectors(elpaScala.getElpaHandle(),
                                &projHamPar.local_el(0, 0),
                                &eigenValues[0],
                                &eigenVectors.local_el(0, 0),
                                &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiCommDomain);


          eigenVectors.copy_to(projHamPar);

          if (dftParams.deviceFineGrainedTimings)
            {
              dftfe::utils::deviceSynchronize();
              computing_timer.leave_subsection("ELPA eigen decomp, RR step");
            }
        }
      else
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, N - 1), true);
          dftfe::utils::deviceSynchronize();
          computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }

      linearAlgebraOperations::internal::broadcastAcrossInterCommScaLAPACKMat(
        processGrid, projHamPar, interBandGroupComm, 0);

      //
      // rotate the basis in the subspace X = X*Q, implemented as
      // X^{T}=Qc^{C}*X^{T} with X^{T} stored in the column major format
      //
      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrecOverall))
            computing_timer.enter_subsection(
              "X^{T}={QConj}^{C}*X^{T}, RR step");
          else
            computing_timer.enter_subsection(
              "X^{T}={QConj}^{C}*X^{T} mixed prec, RR step");
        }
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParCopy(N,
                                                               processGrid,
                                                               rowsBlockSize);
      projHamParCopy.copy_conjugate_transposed(projHamPar);

      if (useMixedPrecOverall && dftParams.useMixedPrecSubspaceRotRR)
        subspaceRotationRRMixedPrecScalapack(X,
                                             M,
                                             N,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             devicecclMpiCommDomain,
                                             interBandGroupComm,
                                             projHamParCopy,
                                             dftParams,
                                             false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  devicecclMpiCommDomain,
                                  interBandGroupComm,
                                  projHamParCopy,
                                  dftParams,
                                  false);
      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrecOverall))
            computing_timer.leave_subsection(
              "X^{T}={QConj}^{C}*X^{T}, RR step");
          else
            computing_timer.leave_subsection(
              "X^{T}={QConj}^{C}*X^{T} mixed prec, RR step");
        }
    }

    void
    rayleighRitzGEP(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      elpaScalaManager &                                   elpaScala,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &         devicecclMpiCommDomain,
      const MPI_Comm &                  interBandGroupComm,
      std::vector<double> &             eigenValues,
      dftfe::utils::deviceBlasHandle_t &handle,
      const dftParameters &             dftParams,
      const bool                        useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      //
      // SConj=X^{T}*XConj.
      //

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (useMixedPrecOverall)
            computing_timer.enter_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection("SConj=X^{T}XConj, RR GEP step");
        }

      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatPar(N,
                                                              processGrid,
                                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  dataTypes::number(0.0));

      if (useMixedPrecOverall && dftParams.useMixedPrecXTHXSpectrumSplit)
        {
          if (dftParams.useMixedPrecCommunOnlyXTHXCGSO)
            XtOXMixedPrecCommunOverlapComputeCommun(operatorMatrix,
                                                    X,
                                                    Xb,
                                                    HXb,
                                                    M,
                                                    N,
                                                    dftParams.numCoreWfcXtHX,
                                                    handle,
                                                    processGrid,
                                                    overlapMatPar,
                                                    devicecclMpiCommDomain,
                                                    mpiCommDomain,
                                                    interBandGroupComm,
                                                    dftParams);
          else
            XtOXMixedPrecOverlapComputeCommun(operatorMatrix,
                                              X,
                                              Xb,
                                              HXb,
                                              M,
                                              N,
                                              dftParams.numCoreWfcXtHX,
                                              handle,
                                              processGrid,
                                              overlapMatPar,
                                              devicecclMpiCommDomain,
                                              mpiCommDomain,
                                              interBandGroupComm,
                                              dftParams);
        }
      else
        {
          if (dftParams.overlapComputeCommunOrthoRR)
            XtOXOverlapComputeCommun(operatorMatrix,
                                     X,
                                     Xb,
                                     HXb,
                                     M,
                                     N,
                                     handle,
                                     processGrid,
                                     overlapMatPar,
                                     devicecclMpiCommDomain,
                                     mpiCommDomain,
                                     interBandGroupComm,
                                     dftParams);
          else
            XtOX(operatorMatrix,
                 X,
                 Xb,
                 HXb,
                 M,
                 N,
                 handle,
                 processGrid,
                 overlapMatPar,
                 devicecclMpiCommDomain,
                 mpiCommDomain,
                 interBandGroupComm,
                 dftParams);
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (useMixedPrecOverall)
            computing_timer.leave_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection("SConj=X^{T}XConj, RR GEP step");
        }

      //
      // SConj=LConj*L^{T}
      //
      if (dftParams.deviceFineGrainedTimings)
        computing_timer.enter_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");


      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParams.useELPA)
        {
          // For ELPA cholesky only the upper triangular part of the hermitian
          // matrix is required
          dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatParConjTrans(
            N, processGrid, rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      dataTypes::number(0.0));

          overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

          if (processGrid->is_process_active())
            {
              int error;

              if (dftParams.useELPADeviceKernel)
                {
#ifdef DFTFE_WITH_DEVICE_NVIDIA
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "nvidia-gpu",
                                   0,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#elif DFTFE_WITH_DEVICE_AMD
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "amd-gpu",
                                   0,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
                }

              elpa_cholesky(elpaScala.getElpaHandle(),
                            &overlapMatParConjTrans.local_el(0, 0),
                            &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));

              if (dftParams.useELPADeviceKernel)
                {
#ifdef DFTFE_WITH_DEVICE_NVIDIA
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "nvidia-gpu",
                                   1,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#elif DFTFE_WITH_DEVICE_AMD
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "amd-gpu",
                                   1,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
                }
            }
          overlapMatPar.copy_conjugate_transposed(overlapMatParConjTrans);
          overlapMatPropertyPostCholesky =
            dftfe::LAPACKSupport::Property::lower_triangular;
        }
      else
        {
          overlapMatPar.compute_cholesky_factorization();

          overlapMatPropertyPostCholesky = overlapMatPar.get_property();
        }

      AssertThrow(
        overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


      // extract LConj
      dftfe::ScaLAPACKMatrix<dataTypes::number> LMatPar(
        N,
        processGrid,
        rowsBlockSize,
        dftfe::LAPACKSupport::Property::lower_triangular);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = LMatPar.global_column(i);
            for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = LMatPar.global_row(j);
                if (glob_j < glob_i)
                  LMatPar.local_el(j, i) = dataTypes::number(0);
                else
                  LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
              }
          }

      // compute LConj^{-1}
      LMatPar.invert();

      if (dftParams.deviceFineGrainedTimings)
        computing_timer.leave_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");


      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.enter_subsection(
            "HConjProj= X^{T}*HConj*XConj, RR GEP step");
        }


      //
      // compute projected Hamiltonian conjugate HConjProj= X^{T}*HConj*XConj
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(N,
                                                           processGrid,
                                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  dataTypes::number(0.0));

      if (dftParams.overlapComputeCommunOrthoRR)
        XtHXOverlapComputeCommun(operatorMatrix,
                                 X,
                                 Xb,
                                 HXb,
                                 M,
                                 N,
                                 handle,
                                 processGrid,
                                 projHamPar,
                                 devicecclMpiCommDomain,
                                 mpiCommDomain,
                                 interBandGroupComm,
                                 dftParams);
      else
        XtHX(operatorMatrix,
             X,
             Xb,
             HXb,
             M,
             N,
             handle,
             processGrid,
             projHamPar,
             devicecclMpiCommDomain,
             mpiCommDomain,
             interBandGroupComm,
             dftParams);

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.leave_subsection(
            "HConjProj= X^{T}*HConj*XConj, RR GEP step");
        }

      if (dftParams.deviceFineGrainedTimings)
        computing_timer.enter_subsection(
          "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParConjTrans(
        N, processGrid, rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  dataTypes::number(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      projHamPar.add(projHamParConjTrans,
                     dataTypes::number(1.0),
                     dataTypes::number(1.0));

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= dataTypes::number(0.5);
              }
          }

      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParCopy(N,
                                                               processGrid,
                                                               rowsBlockSize);

      // compute HSConjProj= Lconj^{-1}*HConjProj*(Lconj^{-1})^C  (C denotes
      // conjugate transpose LAPACK notation)
      LMatPar.mmult(projHamParCopy, projHamPar);
      projHamParCopy.zmCmult(projHamPar, LMatPar);

      if (dftParams.deviceFineGrainedTimings)
        computing_timer.leave_subsection(
          "Compute Lconj^{-1}*HConjProj*(Lconj^{-1})^C, RR GEP step");
      //
      // compute standard eigendecomposition HSConjProj: {QConjPrime,D}
      // HSConjProj=QConjPrime*D*QConjPrime^{C} QConj={Lc^{-1}}^{C}*QConjPrime
      const unsigned int numberEigenValues = N;
      eigenValues.resize(numberEigenValues);
      if (dftParams.useELPA)
        {
          if (dftParams.deviceFineGrainedTimings)
            computing_timer.enter_subsection("ELPA eigen decomp, RR GEP step");
          dftfe::ScaLAPACKMatrix<dataTypes::number> eigenVectors(N,
                                                                 processGrid,
                                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      dataTypes::number(0.0));

          if (processGrid->is_process_active())
            {
              int error;
              elpa_eigenvectors(elpaScala.getElpaHandle(),
                                &projHamPar.local_el(0, 0),
                                &eigenValues[0],
                                &eigenVectors.local_el(0, 0),
                                &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_eigenvectors error."));
            }


          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiCommDomain);


          eigenVectors.copy_to(projHamPar);

          if (dftParams.deviceFineGrainedTimings)
            computing_timer.leave_subsection("ELPA eigen decomp, RR GEP step");
        }
      else
        {
          if (dftParams.deviceFineGrainedTimings)
            computing_timer.enter_subsection(
              "ScaLAPACK eigen decomp, RR GEP step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(0, N - 1), true);
          if (dftParams.deviceFineGrainedTimings)
            computing_timer.leave_subsection(
              "ScaLAPACK eigen decomp, RR GEP step");
        }

      linearAlgebraOperations::internal::broadcastAcrossInterCommScaLAPACKMat(
        processGrid, projHamPar, interBandGroupComm, 0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */
      //
      // rotate the basis in the subspace
      // X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, stored in the column major
      // format In the above we use Q^{T}={QConjPrime}^{C}*LConj^{-1}

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrecOverall))
            computing_timer.enter_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR GEP step");
          else
            computing_timer.enter_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR GEP step");
        }

      projHamParCopy.copy_conjugate_transposed(projHamPar);
      projHamParCopy.mmult(projHamPar, LMatPar);

      if (useMixedPrecOverall && dftParams.useMixedPrecSubspaceRotRR)
        subspaceRotationRRMixedPrecScalapack(X,
                                             M,
                                             N,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             devicecclMpiCommDomain,
                                             interBandGroupComm,
                                             projHamPar,
                                             dftParams,
                                             false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  devicecclMpiCommDomain,
                                  interBandGroupComm,
                                  projHamPar,
                                  dftParams,
                                  false);

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (!(dftParams.useMixedPrecSubspaceRotRR && useMixedPrecOverall))
            computing_timer.leave_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T}, RR GEP step");
          else
            computing_timer.leave_subsection(
              "X^{T}={QConjPrime}^{C}*LConj^{-1}*X^{T} mixed prec, RR GEP step");
        }
    }

    void
    rayleighRitzGEPSpectrumSplitDirect(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      elpaScalaManager &                                   elpaScala,
      dataTypes::number *                                  X,
      dataTypes::number *                                  XFrac,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const unsigned int                                   Noc,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &         devicecclMpiCommDomain,
      const MPI_Comm &                  interBandGroupComm,
      std::vector<double> &             eigenValues,
      dftfe::utils::deviceBlasHandle_t &handle,
      const dftParameters &             dftParams,
      const bool                        useMixedPrecOverall)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);

      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      //
      // SConj=X^{T}*XConj
      //
      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (useMixedPrecOverall)
            computing_timer.enter_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection("SConj=X^{T}XConj, RR GEP step");
        }


      //
      // compute overlap matrix
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatPar(N,
                                                              processGrid,
                                                              rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&overlapMatPar.local_el(0, 0),
                  &overlapMatPar.local_el(0, 0) +
                    overlapMatPar.local_m() * overlapMatPar.local_n(),
                  dataTypes::number(0.0));

      if (useMixedPrecOverall && dftParams.useMixedPrecXTHXSpectrumSplit &&
          false)
        {
          if (dftParams.useMixedPrecCommunOnlyXTHXCGSO)
            XtOXMixedPrecCommunOverlapComputeCommun(operatorMatrix,
                                                    X,
                                                    Xb,
                                                    HXb,
                                                    M,
                                                    N,
                                                    dftParams.numCoreWfcXtHX,
                                                    handle,
                                                    processGrid,
                                                    overlapMatPar,
                                                    devicecclMpiCommDomain,
                                                    mpiCommDomain,
                                                    interBandGroupComm,
                                                    dftParams);
          else
            XtOXMixedPrecOverlapComputeCommun(operatorMatrix,
                                              X,
                                              Xb,
                                              HXb,
                                              M,
                                              N,
                                              dftParams.numCoreWfcXtHX,
                                              handle,
                                              processGrid,
                                              overlapMatPar,
                                              devicecclMpiCommDomain,
                                              mpiCommDomain,
                                              interBandGroupComm,
                                              dftParams);
        }
      else
        {
          if (dftParams.overlapComputeCommunOrthoRR)
            XtOXOverlapComputeCommun(operatorMatrix,
                                     X,
                                     Xb,
                                     HXb,
                                     M,
                                     N,
                                     handle,
                                     processGrid,
                                     overlapMatPar,
                                     devicecclMpiCommDomain,
                                     mpiCommDomain,
                                     interBandGroupComm,
                                     dftParams);
          else
            XtOX(operatorMatrix,
                 X,
                 Xb,
                 HXb,
                 M,
                 N,
                 handle,
                 processGrid,
                 overlapMatPar,
                 devicecclMpiCommDomain,
                 mpiCommDomain,
                 interBandGroupComm,
                 dftParams);
        }


      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (useMixedPrecOverall)
            computing_timer.leave_subsection(
              "SConj=X^{T}XConj Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection("SConj=X^{T}XConj, RR GEP step");
        }

      // Sc=Lc*L^{T}
      if (dftParams.deviceFineGrainedTimings)
        computing_timer.enter_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");

      dftfe::LAPACKSupport::Property overlapMatPropertyPostCholesky;
      if (dftParams.useELPA)
        {
          // For ELPA cholesky only the upper triangular part of the hermitian
          // matrix is required
          dftfe::ScaLAPACKMatrix<dataTypes::number> overlapMatParConjTrans(
            N, processGrid, rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&overlapMatParConjTrans.local_el(0, 0),
                      &overlapMatParConjTrans.local_el(0, 0) +
                        overlapMatParConjTrans.local_m() *
                          overlapMatParConjTrans.local_n(),
                      dataTypes::number(0.0));

          overlapMatParConjTrans.copy_conjugate_transposed(overlapMatPar);

          if (processGrid->is_process_active())
            {
              int error;

              if (dftParams.useELPADeviceKernel)
                {
#ifdef DFTFE_WITH_DEVICE_NVIDIA
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "nvidia-gpu",
                                   0,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#elif DFTFE_WITH_DEVICE_AMD
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "amd-gpu",
                                   0,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
                }


              elpa_cholesky(elpaScala.getElpaHandle(),
                            &overlapMatParConjTrans.local_el(0, 0),
                            &error);
              AssertThrow(error == ELPA_OK,
                          dealii::ExcMessage(
                            "DFT-FE Error: elpa_cholesky error."));

              if (dftParams.useELPADeviceKernel)
                {
#ifdef DFTFE_WITH_DEVICE_NVIDIA
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "nvidia-gpu",
                                   1,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#elif DFTFE_WITH_DEVICE_AMD
                  elpa_set_integer(elpaScala.getElpaHandle(),
                                   "amd-gpu",
                                   1,
                                   &error);
                  AssertThrow(error == ELPA_OK,
                              dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
                }
            }
          overlapMatPar.copy_conjugate_transposed(overlapMatParConjTrans);
          overlapMatPropertyPostCholesky =
            dftfe::LAPACKSupport::Property::lower_triangular;
        }
      else
        {
          overlapMatPar.compute_cholesky_factorization();

          overlapMatPropertyPostCholesky = overlapMatPar.get_property();
        }

      AssertThrow(
        overlapMatPropertyPostCholesky ==
          dftfe::LAPACKSupport::Property::lower_triangular,
        dealii::ExcMessage(
          "DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));


      // extract LConj
      dftfe::ScaLAPACKMatrix<dataTypes::number> LMatPar(
        N,
        processGrid,
        rowsBlockSize,
        dftfe::LAPACKSupport::Property::lower_triangular);

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < LMatPar.local_n(); ++i)
          {
            const unsigned int glob_i = LMatPar.global_column(i);
            for (unsigned int j = 0; j < LMatPar.local_m(); ++j)
              {
                const unsigned int glob_j = LMatPar.global_row(j);
                if (glob_j < glob_i)
                  LMatPar.local_el(j, i) = dataTypes::number(0);
                else
                  LMatPar.local_el(j, i) = overlapMatPar.local_el(j, i);
              }
          }

      // compute LConj^{-1}
      LMatPar.invert();
      if (dftParams.deviceFineGrainedTimings)
        computing_timer.leave_subsection(
          "Cholesky and triangular matrix invert, RR GEP step");

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecCGS_SR && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "X^{T}=Lconj^{-1}*X^{T} Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection(
              "X^{T}=Lconj^{-1}*X^{T}, RR GEP step");
        }

      //
      // X^{T}=LConj^{-1}*X^{T}
      //
      if (useMixedPrecOverall && dftParams.useMixedPrecCGS_SR)
        subspaceRotationCGSMixedPrecScalapack(X,
                                              M,
                                              N,
                                              handle,
                                              processGrid,
                                              mpiCommDomain,
                                              devicecclMpiCommDomain,
                                              interBandGroupComm,
                                              LMatPar,
                                              dftParams,
                                              false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  devicecclMpiCommDomain,
                                  interBandGroupComm,
                                  LMatPar,
                                  dftParams,
                                  false,
                                  true);

      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


      if (numberBandGroups > 1)
        {
          // band group parallelization data structures
          const unsigned int bandGroupTaskId =
            dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
          std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
          dftUtils::createBandParallelizationIndices(
            interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

          const unsigned int vectorsBlockSize =
            std::min(dftParams.wfcBlockSize, N);
          for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

              if (!((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                {
                  // set to zero wavefunctions which are not inside a given band
                  // paral group
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                  setZeroKernel<<<(BVec +
                                   (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE * M,
                                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                    BVec,
                    M,
                    N,
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    jvec);
#elif DFTFE_WITH_DEVICE_LANG_HIP
                  hipLaunchKernelGGL(
                    setZeroKernel,
                    (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE * M,
                    dftfe::utils::DEVICE_BLOCK_SIZE,
                    0,
                    0,
                    BVec,
                    M,
                    N,
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    jvec);
#endif
                }
            }



          std::vector<dataTypes::number> eigenVectorsFlattenedHost(
            M * N, dataTypes::number(0.0));

          dftfe::utils::deviceMemcpyD2H(
            dftfe::utils::makeDataTypeDeviceCompatible(
              &eigenVectorsFlattenedHost[0]),
            X,
            M * N * sizeof(dataTypes::number));

          MPI_Barrier(interBandGroupComm);


          MPI_Allreduce(MPI_IN_PLACE,
                        &eigenVectorsFlattenedHost[0],
                        M * N,
                        dataTypes::mpi_type_id(&eigenVectorsFlattenedHost[0]),
                        MPI_SUM,
                        interBandGroupComm);

          MPI_Barrier(interBandGroupComm);

          dftfe::utils::deviceMemcpyH2D(
            X,
            dftfe::utils::makeDataTypeDeviceCompatible(
              &eigenVectorsFlattenedHost[0]),
            M * N * sizeof(dataTypes::number));
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecCGS_SR && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "X^{T}=Lconj^{-1}*X^{T} Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection(
              "X^{T}=Lconj^{-1}*X^{T}, RR GEP step");
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecXTHXSpectrumSplit && useMixedPrecOverall)
            computing_timer.enter_subsection(
              "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR GEP step");
          else
            computing_timer.enter_subsection(
              "HConjProj=X^{T}*HConj*XConj, RR GEP step");
        }

      //
      // compute projected Hamiltonian HConjProj=X^{T}*HConj*XConj
      //
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPar(N,
                                                           processGrid,
                                                           rowsBlockSize);
      if (processGrid->is_process_active())
        std::fill(&projHamPar.local_el(0, 0),
                  &projHamPar.local_el(0, 0) +
                    projHamPar.local_m() * projHamPar.local_n(),
                  dataTypes::number(0.0));

      if (useMixedPrecOverall && dftParams.useMixedPrecXTHXSpectrumSplit)
        {
          XtHXMixedPrecOverlapComputeCommun(operatorMatrix,
                                            X,
                                            Xb,
                                            HXb,
                                            M,
                                            N,
                                            Noc,
                                            handle,
                                            processGrid,
                                            projHamPar,
                                            devicecclMpiCommDomain,
                                            mpiCommDomain,
                                            interBandGroupComm,
                                            dftParams);
        }
      else
        {
          if (dftParams.overlapComputeCommunOrthoRR)
            XtHXOverlapComputeCommun(operatorMatrix,
                                     X,
                                     Xb,
                                     HXb,
                                     M,
                                     N,
                                     handle,
                                     processGrid,
                                     projHamPar,
                                     devicecclMpiCommDomain,
                                     mpiCommDomain,
                                     interBandGroupComm,
                                     dftParams);
          else
            XtHX(operatorMatrix,
                 X,
                 Xb,
                 HXb,
                 M,
                 N,
                 handle,
                 processGrid,
                 projHamPar,
                 devicecclMpiCommDomain,
                 mpiCommDomain,
                 interBandGroupComm,
                 dftParams);
        }

      // Construct the full HConjProj matrix
      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParConjTrans(
        N, processGrid, rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&projHamParConjTrans.local_el(0, 0),
                  &projHamParConjTrans.local_el(0, 0) +
                    projHamParConjTrans.local_m() *
                      projHamParConjTrans.local_n(),
                  dataTypes::number(0.0));


      projHamParConjTrans.copy_conjugate_transposed(projHamPar);
      if (dftParams.useELPA)
        projHamPar.add(projHamParConjTrans,
                       dataTypes::number(-1.0),
                       dataTypes::number(-1.0));
      else
        projHamPar.add(projHamParConjTrans,
                       dataTypes::number(1.0),
                       dataTypes::number(1.0));


      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < projHamPar.local_n(); ++i)
          {
            const unsigned int glob_i = projHamPar.global_column(i);
            for (unsigned int j = 0; j < projHamPar.local_m(); ++j)
              {
                const unsigned int glob_j = projHamPar.global_row(j);
                if (glob_i == glob_j)
                  projHamPar.local_el(j, i) *= dataTypes::number(0.5);
              }
          }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          if (dftParams.useMixedPrecXTHXSpectrumSplit && useMixedPrecOverall)
            computing_timer.leave_subsection(
              "HConjProj=X^{T}*HConj*XConj Mixed Prec, RR GEP step");
          else
            computing_timer.leave_subsection(
              "HConjProj=X^{T}*HConj*XConj, RR GEP step");
        }


      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamParCopy(N,
                                                               processGrid,
                                                               rowsBlockSize);

      //
      // compute standard eigendecomposition HConjProj: {QConj,D}
      // HConjProj=QConj*D*QConj^{C}
      //
      const unsigned int Nfr = N - Noc;
      eigenValues.resize(Nfr);
      if (dftParams.useELPA)
        {
          if (dftParams.deviceFineGrainedTimings)
            computing_timer.enter_subsection("ELPA eigen decomp, RR step");
          std::vector<double>                       allEigenValues(N, 0.0);
          dftfe::ScaLAPACKMatrix<dataTypes::number> eigenVectors(N,
                                                                 processGrid,
                                                                 rowsBlockSize);

          if (processGrid->is_process_active())
            std::fill(&eigenVectors.local_el(0, 0),
                      &eigenVectors.local_el(0, 0) +
                        eigenVectors.local_m() * eigenVectors.local_n(),
                      dataTypes::number(0.0));

          if (processGrid->is_process_active())
            {
              int error;
              elpa_eigenvectors(elpaScala.getElpaHandlePartialEigenVec(),
                                &projHamPar.local_el(0, 0),
                                &allEigenValues[0],
                                &eigenVectors.local_el(0, 0),
                                &error);
              AssertThrow(
                error == ELPA_OK,
                dealii::ExcMessage(
                  "DFT-FE Error: elpa_eigenvectors error in case spectrum splitting."));
            }

          for (unsigned int i = 0; i < Nfr; ++i)
            eigenValues[Nfr - i - 1] = -allEigenValues[i];

          MPI_Bcast(
            &eigenValues[0], eigenValues.size(), MPI_DOUBLE, 0, mpiCommDomain);


          dftfe::ScaLAPACKMatrix<dataTypes::number> permutedIdentityMat(
            N, processGrid, rowsBlockSize);
          if (processGrid->is_process_active())
            std::fill(&permutedIdentityMat.local_el(0, 0),
                      &permutedIdentityMat.local_el(0, 0) +
                        permutedIdentityMat.local_m() *
                          permutedIdentityMat.local_n(),
                      dataTypes::number(0.0));

          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < permutedIdentityMat.local_m(); ++i)
              {
                const unsigned int glob_i = permutedIdentityMat.global_row(i);
                if (glob_i < Nfr)
                  {
                    for (unsigned int j = 0; j < permutedIdentityMat.local_n();
                         ++j)
                      {
                        const unsigned int glob_j =
                          permutedIdentityMat.global_column(j);
                        if (glob_j < Nfr)
                          {
                            const unsigned int rowIndexToSetOne =
                              (Nfr - 1) - glob_j;
                            if (glob_i == rowIndexToSetOne)
                              permutedIdentityMat.local_el(i, j) =
                                dataTypes::number(1.0);
                          }
                      }
                  }
              }

          eigenVectors.mmult(projHamPar, permutedIdentityMat);

          if (dftParams.deviceFineGrainedTimings)
            computing_timer.leave_subsection("ELPA eigen decomp, RR step");
        }
      else
        {
          if (dftParams.deviceFineGrainedTimings)
            computing_timer.enter_subsection("ScaLAPACK eigen decomp, RR step");
          eigenValues = projHamPar.eigenpairs_hermitian_by_index_MRRR(
            std::make_pair(Noc, N - 1), true);
          if (dftParams.deviceFineGrainedTimings)
            computing_timer.leave_subsection("ScaLAPACK eigen decomp, RR step");
        }

      linearAlgebraOperations::internal::broadcastAcrossInterCommScaLAPACKMat(
        processGrid, projHamPar, interBandGroupComm, 0);

      /*
         MPI_Bcast(&eigenValues[0],
         eigenValues.size(),
         MPI_DOUBLE,
         0,
         interBandGroupComm);
       */

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.enter_subsection(
            "Xfr^{T}={QfrConj}^{C}*X^{T}, RR GEP step");
        }

      //
      // rotate the basis in the subspace
      // Xfr^{T}={QfrConj}^{C}*X^{T}
      //
      projHamParCopy.copy_conjugate_transposed(projHamPar);

      subspaceRotationSpectrumSplitScalapack(X,
                                             XFrac,
                                             M,
                                             N,
                                             Nfr,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             devicecclMpiCommDomain,
                                             projHamParCopy,
                                             dftParams,
                                             false);

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.leave_subsection(
            "Xfr^{T}={QfrConj}^{C}*X^{T}, RR GEP step");
        }
    }


    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &         devicecclMpiCommDomain,
      const MPI_Comm &                  interBandGroupComm,
      const std::vector<double> &       eigenValues,
      const double                      fermiEnergy,
      std::vector<double> &             densityMatDerFermiEnergy,
      dftfe::elpaScalaManager &         elpaScala,
      dftfe::utils::deviceBlasHandle_t &handle,
      const dftParameters &             dftParams)
    {
      dealii::ConditionalOStream pcout(
        std::cout,
        (dealii::Utilities::MPI::this_mpi_process(mpiCommParent) == 0));

      dealii::TimerOutput computing_timer(mpiCommDomain,
                                          pcout,
                                          dftParams.reproducible_output ||
                                              dftParams.verbosity < 4 ?
                                            dealii::TimerOutput::never :
                                            dealii::TimerOutput::summary,
                                          dealii::TimerOutput::wall_times);


      const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
      std::shared_ptr<const dftfe::ProcessGrid> processGrid =
        elpaScala.getProcessGridDftfeScalaWrapper();

      dftfe::ScaLAPACKMatrix<dataTypes::number> projHamPrimePar(N,
                                                                processGrid,
                                                                rowsBlockSize);

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.enter_subsection("Blocked XtHX, DMFOR step");
        }

      if (processGrid->is_process_active())
        std::fill(&projHamPrimePar.local_el(0, 0),
                  &projHamPrimePar.local_el(0, 0) +
                    projHamPrimePar.local_m() * projHamPrimePar.local_n(),
                  dataTypes::number(0.0));

      if (dftParams.singlePrecLRD && dftParams.overlapComputeCommunOrthoRR)
        XtHXMixedPrecOverlapComputeCommun(operatorMatrix,
                                          X,
                                          Xb,
                                          HXb,
                                          M,
                                          N,
                                          N,
                                          handle,
                                          processGrid,
                                          projHamPrimePar,
                                          devicecclMpiCommDomain,
                                          mpiCommDomain,
                                          interBandGroupComm,
                                          dftParams,
                                          true);
      else if (dftParams.overlapComputeCommunOrthoRR)
        XtHXOverlapComputeCommun(operatorMatrix,
                                 X,
                                 Xb,
                                 HXb,
                                 M,
                                 N,
                                 handle,
                                 processGrid,
                                 projHamPrimePar,
                                 devicecclMpiCommDomain,
                                 mpiCommDomain,
                                 interBandGroupComm,
                                 dftParams,
                                 true);
      else
        XtHX(operatorMatrix,
             X,
             Xb,
             HXb,
             M,
             N,
             handle,
             processGrid,
             projHamPrimePar,
             devicecclMpiCommDomain,
             mpiCommDomain,
             interBandGroupComm,
             dftParams,
             true);

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.leave_subsection("Blocked XtHX, DMFOR step");
        }

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.enter_subsection(
            "Recursive fermi operator expansion operations, DMFOR step");
        }

      const int    m    = 10;
      const double beta = 1.0 / C_kb / dftParams.TVal;
      const double c    = std::pow(2.0, -2.0 - m) * beta;

      std::vector<double> H0 = eigenValues;
      std::vector<double> X0(N, 0.0);
      for (unsigned int i = 0; i < N; ++i)
        {
          X0[i] = 0.5 - c * (H0[i] - fermiEnergy);
        }

      dftfe::ScaLAPACKMatrix<dataTypes::number> densityMatPrimePar(
        N, processGrid, rowsBlockSize);
      densityMatPrimePar.add(projHamPrimePar,
                             dataTypes::number(0.0),
                             dataTypes::number(-c)); //-c*HPrime

      dftfe::ScaLAPACKMatrix<dataTypes::number> X1Temp(N,
                                                       processGrid,
                                                       rowsBlockSize);
      dftfe::ScaLAPACKMatrix<dataTypes::number> X1Tempb(N,
                                                        processGrid,
                                                        rowsBlockSize);
      dftfe::ScaLAPACKMatrix<dataTypes::number> X1Tempc(N,
                                                        processGrid,
                                                        rowsBlockSize);

      std::vector<double> Y0Temp(N, 0.0);

      for (unsigned int i = 0; i < m; ++i)
        {
          // step1
          X1Temp.add(densityMatPrimePar,
                     dataTypes::number(0.0),
                     dataTypes::number(1.0)); // copy
          X1Tempb.add(densityMatPrimePar,
                      dataTypes::number(0.0),
                      dataTypes::number(1.0)); // copy
          X1Temp.scale_rows_realfactors(X0);
          X1Tempb.scale_columns_realfactors(X0);
          X1Temp.add(X1Tempb, dataTypes::number(1.0), dataTypes::number(1.0));

          // step2 and 3
          for (unsigned int j = 0; j < N; ++j)
            {
              Y0Temp[j] = 1.0 / (2.0 * X0[j] * (X0[j] - 1.0) + 1.0);
              X0[j]     = Y0Temp[j] * X0[j] * X0[j];
            }

          // step4
          X1Tempc.add(X1Temp,
                      dataTypes::number(0.0),
                      dataTypes::number(1.0)); // copy
          X1Temp.scale_rows_realfactors(Y0Temp);
          X1Tempc.scale_columns_realfactors(X0);
          X1Tempc.scale_rows_realfactors(Y0Temp);
          X1Temp.add(X1Tempc, dataTypes::number(1.0), dataTypes::number(-2.0));
          X1Tempb.add(densityMatPrimePar,
                      dataTypes::number(0.0),
                      dataTypes::number(1.0)); // copy
          X1Tempb.scale_columns_realfactors(X0);
          X1Tempb.scale_rows_realfactors(Y0Temp);
          densityMatPrimePar.add(X1Temp,
                                 dataTypes::number(0.0),
                                 dataTypes::number(1.0));
          densityMatPrimePar.add(X1Tempb,
                                 dataTypes::number(1.0),
                                 dataTypes::number(2.0));
        }

      std::vector<double> Pmu0(N, 0.0);
      double              sum = 0.0;
      for (unsigned int i = 0; i < N; ++i)
        {
          Pmu0[i] = beta * X0[i] * (1.0 - X0[i]);
          sum += Pmu0[i];
        }
      densityMatDerFermiEnergy = Pmu0;


      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.leave_subsection(
            "Recursive fermi operator expansion operations, DMFOR step");
        }

      //
      // subspace transformation Y^{T} = DMP^T*X^{T}, implemented as
      // Y^{T}=DMPc^{C}*X^{T} with X^{T} stored in the column major format
      //
      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.enter_subsection(
            "Blocked subspace transformation, DMFOR step");
        }

      // For subspace transformation the full matrix is required
      dftfe::ScaLAPACKMatrix<dataTypes::number> densityMatPrimeParConjTrans(
        N, processGrid, rowsBlockSize);

      if (processGrid->is_process_active())
        std::fill(&densityMatPrimeParConjTrans.local_el(0, 0),
                  &densityMatPrimeParConjTrans.local_el(0, 0) +
                    densityMatPrimeParConjTrans.local_m() *
                      densityMatPrimeParConjTrans.local_n(),
                  dataTypes::number(0.0));


      densityMatPrimeParConjTrans.copy_conjugate_transposed(densityMatPrimePar);
      densityMatPrimePar.add(densityMatPrimeParConjTrans,
                             dataTypes::number(1.0),
                             dataTypes::number(1.0));

      if (processGrid->is_process_active())
        for (unsigned int i = 0; i < densityMatPrimePar.local_n(); ++i)
          {
            const unsigned int glob_i = densityMatPrimePar.global_column(i);
            for (unsigned int j = 0; j < densityMatPrimePar.local_m(); ++j)
              {
                const unsigned int glob_j = densityMatPrimePar.global_row(j);
                if (glob_i == glob_j)
                  densityMatPrimePar.local_el(j, i) *= dataTypes::number(0.5);
              }
          }

      densityMatPrimeParConjTrans.copy_conjugate_transposed(densityMatPrimePar);

      if (dftParams.singlePrecLRD)
        subspaceRotationRRMixedPrecScalapack(X,
                                             M,
                                             N,
                                             handle,
                                             processGrid,
                                             mpiCommDomain,
                                             devicecclMpiCommDomain,
                                             interBandGroupComm,
                                             densityMatPrimeParConjTrans,
                                             dftParams,
                                             false);
      else
        subspaceRotationScalapack(X,
                                  M,
                                  N,
                                  handle,
                                  processGrid,
                                  mpiCommDomain,
                                  devicecclMpiCommDomain,
                                  interBandGroupComm,
                                  densityMatPrimeParConjTrans,
                                  dftParams,
                                  false);

      if (dftParams.deviceFineGrainedTimings)
        {
          dftfe::utils::deviceSynchronize();
          computing_timer.leave_subsection(
            "Blocked subspace transformation, DMFOR step");
        }
    }

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
