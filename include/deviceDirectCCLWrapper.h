// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das, David M. Rogers

#if defined(DFTFE_WITH_DEVICE)
#  ifndef deviceDirectCCLWrapper_h
#    define deviceDirectCCLWrapper_h

#    include <complex>
#    include <mpi.h>
#    include <DeviceTypeConfig.h>

#    if defined(DFTFE_WITH_CUDA_NCCL)
#      include <nccl.h>
#      include <DeviceTypeConfig.h>
#    elif defined(DFTFE_WITH_HIP_RCCL)
#      include <rccl.h>
#      include <DeviceTypeConfig.h>
#    endif

namespace dftfe
{
  namespace utils
  {
#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
#      define NCCLCHECK(cmd)                              \
        do                                                \
          {                                               \
            ncclResult_t r = cmd;                         \
            if (r != ncclSuccess)                         \
              {                                           \
                printf("Failed, NCCL error %s:%d '%s'\n", \
                       __FILE__,                          \
                       __LINE__,                          \
                       ncclGetErrorString(r));            \
                exit(EXIT_FAILURE);                       \
              }                                           \
          }                                               \
        while (0)
#    endif
    /**
     *  @brief Wrapper class for Device Direct collective communications library.
     *  Adapted from
     * https://code.ornl.gov/99R/olcf-cookbook/-/blob/develop/comms/nccl_allreduce.rst
     *
     *  @author Sambit Das, David M. Rogers
     */
    class DeviceCCLWrapper
    {
    public:
      DeviceCCLWrapper();

      void
      init(const MPI_Comm &mpiComm, const bool useDCCL);

      ~DeviceCCLWrapper();

      int
      deviceDirectAllReduceWrapper(const float *   send,
                                   float *         recv,
                                   int             size,
                                   deviceStream_t &stream);


      int
      deviceDirectAllReduceWrapper(const double *  send,
                                   double *        recv,
                                   int             size,
                                   deviceStream_t &stream);


      int
      deviceDirectAllReduceWrapper(const std::complex<double> *send,
                                   std::complex<double> *      recv,
                                   int                         size,
                                   double *                    tempReal,
                                   double *                    tempImag,
                                   deviceStream_t &            stream);

      int
      deviceDirectAllReduceWrapper(const std::complex<float> *send,
                                   std::complex<float> *      recv,
                                   int                        size,
                                   float *                    tempReal,
                                   float *                    tempImag,
                                   deviceStream_t &           stream);


      int
      deviceDirectAllReduceMixedPrecGroupWrapper(const double *  send1,
                                                 const float *   send2,
                                                 double *        recv1,
                                                 float *         recv2,
                                                 int             size1,
                                                 int             size2,
                                                 deviceStream_t &stream);

      int
      deviceDirectAllReduceMixedPrecGroupWrapper(
        const std::complex<double> *send1,
        const std::complex<float> * send2,
        std::complex<double> *      recv1,
        std::complex<float> *       recv2,
        int                         size1,
        int                         size2,
        double *                    tempReal1,
        float *                     tempReal2,
        double *                    tempImag1,
        float *                     tempImag2,
        deviceStream_t &            stream);



      inline void
      deviceDirectAllReduceWrapper(const std::complex<float> *send,
                                   std::complex<float> *      recv,
                                   int                        size,
                                   deviceStream_t &           stream)
      {}


      inline void
      deviceDirectAllReduceWrapper(const std::complex<double> *send,
                                   std::complex<double> *      recv,
                                   int                         size,
                                   deviceStream_t &            stream)
      {}

      inline void
      deviceDirectAllReduceMixedPrecGroupWrapper(
        const std::complex<double> *send1,
        const std::complex<float> * send2,
        std::complex<double> *      recv1,
        std::complex<float> *       recv2,
        int                         size1,
        int                         size2,
        deviceStream_t &            stream)
      {}


      inline void
      deviceDirectAllReduceWrapper(const double *  send,
                                   double *        recv,
                                   int             size,
                                   double *        tempReal,
                                   double *        tempImag,
                                   deviceStream_t &stream)
      {}

      inline void
      deviceDirectAllReduceWrapper(const float *   send,
                                   float *         recv,
                                   int             size,
                                   float *         tempReal,
                                   float *         tempImag,
                                   deviceStream_t &stream)
      {}

      inline void
      deviceDirectAllReduceMixedPrecGroupWrapper(const double *  send1,
                                                 const float *   send2,
                                                 double *        recv1,
                                                 float *         recv2,
                                                 int             size1,
                                                 int             size2,
                                                 double *        tempReal1,
                                                 float *         tempReal2,
                                                 double *        tempImag1,
                                                 float *         tempImag2,
                                                 deviceStream_t &stream)
      {}

#    if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
      inline static ncclUniqueId *ncclIdPtr;
      inline static ncclComm_t *  ncclCommPtr;
#    endif
      inline static bool                         ncclCommInit;
      inline static dftfe::utils::deviceStream_t d_deviceCommStream;
      inline static bool                         commStreamCreated;

    private:
      int      myRank;
      int      totalRanks;
      MPI_Comm d_mpiComm;
    };
  } // namespace utils
} // namespace dftfe

#  endif
#endif
