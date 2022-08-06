
//  referenced from https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/csrqr/cusolver_csrqr_example1.cu
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include "cusolver_utils.h"

//  all array should be pre-allocated(including x), the rest array should also be assigned
int solveCG_HostToHost(const int &m,const int &nnzA,
                          int *csrRowPtrA,  //m+1
                          int *csrColIndA,  //nzzA
                          double *csrValA,     //nzzA
                          double *b,           //m
                          double *x)           //m
{
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cudaStream_t stream = NULL;

    // GPU does batch QR
    // d_A is CSR format, d_csrValA is of size nnzA*batchSize
    // d_x is a matrix of size batchSize * m
    // d_b is a matrix of size batchSize * m
    int *d_csrRowPtrA = nullptr;
    int *d_csrColIndA = nullptr;
    double *d_csrValA = nullptr;
    double *d_b = nullptr; // batchSize * m
    double *d_x = nullptr; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = nullptr; // working space for numerical factorization

    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverSpSetStream(cusolverH, stream));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));

    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); // base-1

    CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));


    CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_csrValA), sizeof(double) * nnzA));
    CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_csrColIndA), sizeof(int) * nnzA));
    CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_csrRowPtrA), sizeof(int) * (m+1)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * m));

    CUDA_CHECK(cudaMemcpyAsync(d_csrValA, csrValA, sizeof(double) * nnzA,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrColIndA, csrColIndA, sizeof(int) * nnzA,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b, sizeof(double) * m,
                               cudaMemcpyHostToDevice, stream));


    CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                                   d_csrColIndA, info));

    CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                                     d_csrRowPtrA, d_csrColIndA, 1, info,
                                                     &size_internal, &size_qr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("numerical factorization needs internal data %lld bytes\n",
                static_cast<long long>(size_internal));
    std::printf("numerical factorization needs working space %lld bytes\n",
                static_cast<long long>(size_qr));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_qr), size_qr));


    CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                             d_csrColIndA, d_b, d_x, 1, info, buffer_qr));

    CUDA_CHECK(cudaMemcpyAsync(x, d_x, sizeof(double) * m,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));


    CUDA_CHECK(cudaFree(d_csrRowPtrA));
    CUDA_CHECK(cudaFree(d_csrColIndA));
    CUDA_CHECK(cudaFree(d_csrValA));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(buffer_qr));

    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}

//  all array except d_x should be pre-allocated and assigned,
//  d_x should be NULL
int solveCG_DeviceToDevice(const int &m,const int &nnzA,
                         int *d_csrRowPtrA,  //m+1
                         int *d_csrColIndA,  //nzzA
                         double *d_csrValA,     //nzzA
                         double *d_b,           //m
                         double *&d_x)           //m
{
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cudaStream_t stream = NULL;

    // GPU does batch QR
    // d_A is CSR format, d_csrValA is of size nnzA*batchSize
    // d_x is a matrix of size batchSize * m
    // d_b is a matrix of size batchSize * m
    d_x = nullptr; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = nullptr; // working space for numerical factorization

    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverSpSetStream(cusolverH, stream));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));

    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); // base-1

    CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * m));


    CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                                   d_csrColIndA, info));

    CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                                     d_csrRowPtrA, d_csrColIndA, 1, info,
                                                     &size_internal, &size_qr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("numerical factorization needs internal data %lld bytes\n",
                static_cast<long long>(size_internal));
    std::printf("numerical factorization needs working space %lld bytes\n",
                static_cast<long long>(size_qr));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_qr), size_qr));


    CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                             d_csrColIndA, d_b, d_x, 1, info, buffer_qr));

    CUDA_CHECK(cudaStreamSynchronize(stream));


    CUDA_CHECK(cudaFree(buffer_qr));

    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}

//  all array except d_x should be pre-allocated and assigned,
//  d_x should be pre-allocated
int solveCG_DeviceToDeviceAssigned(const int &m,const int &nnzA,
                                   int *d_csrRowPtrA,  //m+1
                                   int *d_csrColIndA,  //nzzA
                                   double *d_csrValA,     //nzzA
                                   double *d_b,           //m
                                   double *d_x)           //m
{
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cudaStream_t stream = NULL;

    // GPU does batch QR
    // d_A is CSR format, d_csrValA is of size nnzA*batchSize
    // d_x is a matrix of size batchSize * m
    // d_b is a matrix of size batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = nullptr; // working space for numerical factorization

    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverSpSetStream(cusolverH, stream));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));

    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); // base-1

    CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));


    CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                                   d_csrColIndA, info));

    CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                                     d_csrRowPtrA, d_csrColIndA, 1, info,
                                                     &size_internal, &size_qr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("numerical factorization needs internal data %lld bytes\n",
                static_cast<long long>(size_internal));
    std::printf("numerical factorization needs working space %lld bytes\n",
                static_cast<long long>(size_qr));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_qr), size_qr));


    CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                             d_csrColIndA, d_b, d_x, 1, info, buffer_qr));

    CUDA_CHECK(cudaStreamSynchronize(stream));


    CUDA_CHECK(cudaFree(buffer_qr));

    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}

//  all array except x should be pre-allocated and assigned,
//  x should be pre-allocated
int solveCG_DeviceToHost(const int &m,const int &nnzA,
                       int *d_csrRowPtrA,  //m+1
                       int *d_csrColIndA,  //nzzA
                       double *d_csrValA,     //nzzA
                       double *d_b,           //m
                       double *x)           //m
{
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cudaStream_t stream = NULL;

    // GPU does batch QR
    // d_A is CSR format, d_csrValA is of size nnzA*batchSize
    // d_x is a matrix of size batchSize * m
    // d_b is a matrix of size batchSize * m
    double *d_x = nullptr; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = nullptr; // working space for numerical factorization

    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverSpSetStream(cusolverH, stream));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));

    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); // base-1

    CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));


    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * m));


    CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                                   d_csrColIndA, info));

    CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                                     d_csrRowPtrA, d_csrColIndA, 1, info,
                                                     &size_internal, &size_qr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("numerical factorization needs internal data %lld bytes\n",
                static_cast<long long>(size_internal));
    std::printf("numerical factorization needs working space %lld bytes\n",
                static_cast<long long>(size_qr));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_qr), size_qr));


    CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                             d_csrColIndA, d_b, d_x, 1, info, buffer_qr));

    CUDA_CHECK(cudaMemcpyAsync(x, d_x, sizeof(double) * m,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));


    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(buffer_qr));

    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}

//  all array except d_x should be pre-allocated and assigned,
//  d_x should be NULL
int solveCG_HostToDevice(const int &m,const int &nnzA,
                       int *csrRowPtrA,  //m+1
                       int *csrColIndA,  //nzzA
                       double *csrValA,     //nzzA
                       double *b,           //m
                       double *&d_x)           //m
{
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cudaStream_t stream = NULL;

    // GPU does batch QR
    // d_A is CSR format, d_csrValA is of size nnzA*batchSize
    // d_x is a matrix of size batchSize * m
    // d_b is a matrix of size batchSize * m
    int *d_csrRowPtrA = nullptr;
    int *d_csrColIndA = nullptr;
    double *d_csrValA = nullptr;
    double *d_b = nullptr; // batchSize * m
    d_x = nullptr; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = nullptr; // working space for numerical factorization

    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverSpSetStream(cusolverH, stream));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));

    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); // base-1

    CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));


    CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_csrValA), sizeof(double) * nnzA));
    CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_csrColIndA), sizeof(int) * nnzA));
    CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_csrRowPtrA), sizeof(int) * (m+1)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * m));

    CUDA_CHECK(cudaMemcpyAsync(d_csrValA, csrValA, sizeof(double) * nnzA,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrColIndA, csrColIndA, sizeof(int) * nnzA,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrRowPtrA, csrRowPtrA, sizeof(int) * (m+1),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, b, sizeof(double) * m,
                               cudaMemcpyHostToDevice, stream));


    CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                                   d_csrColIndA, info));

    CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                                     d_csrRowPtrA, d_csrColIndA, 1, info,
                                                     &size_internal, &size_qr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("numerical factorization needs internal data %lld bytes\n",
                static_cast<long long>(size_internal));
    std::printf("numerical factorization needs working space %lld bytes\n",
                static_cast<long long>(size_qr));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_qr), size_qr));


    CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA, d_csrRowPtrA,
                                             d_csrColIndA, d_b, d_x, 1, info, buffer_qr));


    CUDA_CHECK(cudaStreamSynchronize(stream));


    CUDA_CHECK(cudaFree(d_csrRowPtrA));
    CUDA_CHECK(cudaFree(d_csrColIndA));
    CUDA_CHECK(cudaFree(d_csrValA));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(buffer_qr));

    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceSynchronize());

    return EXIT_SUCCESS;
}
