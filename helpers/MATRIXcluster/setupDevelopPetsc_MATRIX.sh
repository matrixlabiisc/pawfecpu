#!/bin/bash
# script to setup and build DFT-FE.

set -e
set -o pipefail

if [ -s CMakeLists.txt ]; then
    echo "This script must be run from the build directory!"
    exit 1
fi

# Path to project source
SRC=`dirname $0` # location of source directory

########################################################################
#Provide paths below for external libraries, compiler options and flags,
# and optimization flag

#Paths for required external libraries
dealiiPetscRealDir="/shared/dftfesoftwares2021/dealii/installgcc9.2cudaawarempiWithPetscSlpec"
dealiiPetscComplexDir="/shared/dftfesoftwares2021/dealii/installgcc9.2cudaawarempiWithPetscSlpecComplex"
alglibDir="/shared/dftfesoftwares2021/alglib/cpp/src"
libxcDir="/shared/dftfesoftwares2021/libxc/gcc9.2_libxc_5.1.5"
spglibDir="/shared/dftfesoftwares2021/spglib/gcc9.2_spglib"
xmlIncludeDir="/usr/include/libxml2"
xmlLibDir="/usr/lib64"
ELPA_PATH="/shared/dftfesoftwares2021/elpa2022/installgcc9.2elpa2022withcudaawarempi"
numdiffdir="/shared/dftfesoftwares2021/numdiff/build"

#Paths for optional external libraries
# path for NCCL/RCCL libraries
DCCL_PATH=""
mdiPath=""

#Toggle GPU compilation
withGPU=ON
gpuLang="cuda"     # Use "cuda"/"hip"
gpuVendor="nvidia" # Use "nvidia/amd"
withGPUAwareMPI=OFF #Please use this option with care
                   #Only use if the machine supports 
                   #device aware MPI and is profiled
                   #to be fast

#Option to link to DCCL library (Only for GPU compilation)
withDCCL=OFF
withMDI=OFF
withTorch=OFF
withCustomizedDealii=ON

#Compiler options and flags
cxx_compiler=mpic++  #sets DCMAKE_CXX_COMPILER
cxx_flags="-march=native -fPIC" #sets DCMAKE_CXX_FLAGS
cxx_flagsRelease="-O2" #sets DCMAKE_CXX_FLAGS_RELEASE
device_flags="-arch=sm_70" # set DCMAKE_CXX_CUDA_FLAGS 
                           #(only applicable for withGPU=ON)
device_architectures="70" # set DCMAKE_CXX_CUDA_ARCHITECTURES 
                           #(only applicable for withGPU=ON)


#Option to compile with default or higher order quadrature for storing pseudopotential data
#ON is recommended for MD simulations with hard pseudopotentials
withHigherQuadPSP=OFF

# build type: "Release" or "Debug"
build_type=Release

testing=ON
minimal_compile=ON
###########################################################################
#Usually, no changes are needed below this line
#

#if [[ x"$build_type" == x"Release" ]]; then
#  c_flags="$c_flagsRelease"
#  cxx_flags="$c_flagsRelease"
#else
#fi
out=`echo "$build_type" | tr '[:upper:]' '[:lower:]'`

function cmake_configure() {
  if [ "$gpuLang" = "cuda" ]; then
    cmake -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch \
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_CUDA_FLAGS="$device_flags" -DCMAKE_CUDA_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  elif [ "$gpuLang" = "hip" ]; then
    cmake -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch \
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex -DWITH_GPU=$withGPU -DGPU_LANG=$gpuLang -DGPU_VENDOR=$gpuVendor -DWITH_GPU_AWARE_MPI=$withGPUAwareMPI -DCMAKE_HIP_FLAGS="$device_flags" -DCMAKE_HIP_ARCHITECTURES="$device_architectures"\
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1
  else
    cmake -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_COMPILER=$cxx_compiler\
    -DCMAKE_CXX_FLAGS="$cxx_flags"\
    -DCMAKE_CXX_FLAGS_RELEASE="$cxx_flagsRelease" \
    -DCMAKE_BUILD_TYPE=$build_type -DDEAL_II_DIR=$dealiiDir \
    -DALGLIB_DIR=$alglibDir -DLIBXC_DIR=$libxcDir \
    -DSPGLIB_DIR=$spglibDir -DXML_LIB_DIR=$xmlLibDir \
    -DXML_INCLUDE_DIR=$xmlIncludeDir\
    -DWITH_MDI=$withMDI -DMDI_PATH=$mdiPath -DWITH_TORCH=$withTorch \
    -DWITH_CUSTOMIZED_DEALII=$withCustomizedDealii\
    -DWITH_DCCL=$withDCCL -DCMAKE_PREFIX_PATH="$ELPA_PATH;$DCCL_PATH;$numdiffdir"\
    -DWITH_COMPLEX=$withComplex \
    -DWITH_TESTING=$testing -DMINIMAL_COMPILE=$minimal_compile\
    -DHIGHERQUAD_PSP=$withHigherQuadPSP $1    
  fi
}

RCol='\e[0m'
Blu='\e[0;34m';
if [ -d "$out" ]; then # build directory exists
    echo -e "${Blu}$out directory already present${RCol}"
else
    rm -rf "$out"
    echo -e "${Blu}Creating $out ${RCol}"
    mkdir -p "$out"
fi

cd $out

withComplex=OFF
dealiiDir=$dealiiPetscRealDir
echo -e "${Blu}Building Real executable in $build_type mode...${RCol}"
mkdir -p real && cd real
cmake_configure "$SRC" && make -j8
cd ..

withComplex=ON
dealiiDir=$dealiiPetscComplexDir
echo -e "${Blu}Building Complex executable in $build_type mode...${RCol}"
mkdir -p complex && cd complex
cmake_configure "$SRC" && make -j8
cd ..

echo -e "${Blu}Build complete.${RCol}"
