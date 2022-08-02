cmake_minimum_required(VERSION 3.22)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda/bin/nvcc)
project(GPU_PoissonRecon CUDA)

#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_CUDA_STANDARD 14)

#set(CMAKE_CUDA_FLAGS "--expt-extended-lambda")

add_executable(GPU_PoissonRecon main.cu
        CmdLineParser.cuh CmdLineParser.inl CmdLineParser.cu
        Geometry.cuh Geometry.inl Geometry.cu
        plyfile.cu PlyFile.cuh PointStream.cuh PointStream.inl
        Debug.cuh OctNode.cuh Hash.cuh
        BinaryNode.cuh
        Factor.cu Factor.cuh
        Polynomial.cuh Polynomial.inl
        PPolynomial.cuh PPolynomial.inl
        FunctionData.cuh FunctionData.inl
        ConfirmedPPolynomial.cuh)

set_target_properties(GPU_PoissonRecon PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
