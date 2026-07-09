// HIP compatibility layer for TIDES tile substrate.
//
// Maps CUDA runtime / cuBLAS API calls to HIP / rocBLAS equivalents.
// When building with HIP, .cu files are compiled as .hip (via hipify-perl
// or the compatibility macros below).
//
// Usage: define TIDES_USE_HIP, then include this header before any CUDA headers.
// The macros below remap CUDA symbols to HIP equivalents.

#pragma once

#ifdef TIDES_USE_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>

// --- CUDA runtime → HIP runtime ------------------------------------------

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaGetDeviceAttribute hipGetDeviceAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxThreadsPerBlock hipDeviceAttributeMaxThreadsPerBlock
#define cudaDevAttrMaxThreadsPerMultiProcessor hipDeviceAttributeMaxThreadsPerMultiProcessor
#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDevAttrMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSharedMemConfig hipSharedMemConfig
#define cudaDeviceSetSharedMemConfig hipDeviceSetSharedMemConfig
#define cudaSharedMemBankSizeEightByte hipSharedMemBankSizeEightByte

// --- cuBLAS → rocBLAS / hipBLAS -------------------------------------------

#define cublasHandle_t hipblasHandle_t
#define cublasStatus_t hipblasStatus_t
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasSetStream hipblasSetStream
#define cublasGetStream hipblasGetStream
#define cublasDgemm hipblasDgemm
#define cublasSgemm hipblasSgemm
#define cublasHgemm hipblasHgemm
#define cublasGemmEx hipblasGemmEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasDgemmStridedBatched hipblasDgemmStridedBatched
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasGemmAlgo_t hipblasGemmAlgo_t
#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define cublasMath_t hipblasMath_t
#define CUBLAS_MATH_DISCARD_REDUNDANT_PRECISION HIPBLAS_MATH_DEFAULT
#define cublasSetMathMode hipblasSetMathMode
#define cublasComputeType_t hipblasComputeType_t
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_64F HIPBLAS_COMPUTE_64F
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F_FAST_16BF HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_TF32 HIPBLAS_COMPUTE_32F

// cuBLASLt → hipBLASLt (if available) or fall back to hipBLAS
#ifdef __has_include
#if __has_include(<hipblaslt/hipblaslt.h>)
#include <hipblaslt/hipblaslt.h>
#define TIDES_HAVE_HIPBLASLT 1
#define cublasLtHandle_t hipblasLtHandle_t
#define cublasLtCreate hipblasLtCreate
#define cublasLtDestroy hipblasLtDestroy
#define cublasLtMatmulDesc_t hipblasLtMatmulDesc_t
#define cublasLtMatmulDescCreate hipblasLtMatmulDescCreate
#define cublasLtMatmulDescDestroy hipblasLtMatmulDescDestroy
#define cublasLtMatmulPreference_t hipblasLtMatmulPreference_t
#define cublasLtMatmulPreferenceCreate hipblasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceDestroy hipblasLtMatmulPreferenceDestroy
#define cublasLtMatmulAlgoGetHeuristic hipblasLtMatmulAlgoGetHeuristic
#define cublasLtMatmul hipblasLtMatmul
#define cublasLtMatmulAlgo_t hipblasLtMatmulAlgo_t
#define cublasLtMatrixLayout_t hipblasLtMatrixLayout_t
#define cublasLtMatrixLayoutCreate hipblasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutDestroy hipblasLtMatrixLayoutDestroy
#define CUBLASLT_MATMUL_DESC_COMPUTE_TYPE HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE
#define CUBLASLT_MATMUL_DESC_TRANSA HIPBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSB HIPBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_OP_N HIPBLASLT_OP_N
#define CUBLASLT_OP_T HIPBLASLT_OP_T
#define CUBLASLT_EPILOGUE_DEFAULT HIPBLASLT_EPILOGUE_DEFAULT
#define CUBLASLT_MATMUL_DESC_EPILOGUE HIPBLASLT_MATMUL_DESC_EPILOGUE
#define CUBLASLT_MATMUL_DESC_POINTER_MODE HIPBLASLT_MATMUL_DESC_POINTER_MODE
#define CUBLASLT_POINTER_MODE_HOST HIPBLASLT_POINTER_MODE_HOST
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUBLASLT_MATMUL_DESC_A_SCALE_POINTER HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_B_SCALE_POINTER HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_C_SCALE_POINTER HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_D_SCALE_POINTER HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_AMAX_D_POINTER HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER
#endif
#endif

// --- cuSOLVER → rocSOLVER -------------------------------------------------
#ifdef __has_include
#if __has_include(<rocsolver/rocsolver.h>)
#include <rocsolver/rocsolver.h>
#define cusolverDnHandle_t rocblas_handle
#define cusolverStatus_t rocblas_status
#define CUSOLVER_STATUS_SUCCESS rocblas_status_success
#define cusolverDnCreate(handle) rocblas_create_handle(handle)
#define cusolverDnDestroy(handle) rocblas_destroy_handle(handle)
#define cusolverDnSetStream rocblas_set_stream
#endif
#endif

// --- cuFFT → rocFFT -------------------------------------------------------
#ifdef __has_include
#if __has_include(<rocfft/rocfft.h>)
#include <rocfft/rocfft.h>
// rocFFT has a different API than cuFFT; mapping is done in poisson_fft.cu
#define TIDES_USE_ROCFFT 1
#endif
#endif

// --- cuSPARSE → hipSPARSE -------------------------------------------------
#define cusparseHandle_t hipsparseHandle_t
#define cusparseStatus_t hipsparseStatus_t
#define CUSPARSE_STATUS_SUCCESS HIPSPARSE_STATUS_SUCCESS
#define cusparseCreate hipsparseCreate
#define cusparseDestroy hipsparseDestroy
#define cusparseSetStream hipsparseSetStream
#define cusparseSpMM hipsparseSpMM
#define cusparseSpMM_bufferSize hipsparseSpMM_bufferSize
#define cusparseSpMatDescr_t hipsparseSpMatDescr_t
#define cusparseDnMatDescr_t hipsparseDnMatDescr_t
#define CUSPARSE_OPERATION_NON_TRANSPOSE HIPSPARSE_OPERATION_NON_TRANSPOSE
#define CUSPARSE_OPERATION_TRANSPOSE HIPSPARSE_OPERATION_TRANSPOSE
#define CUSPARSE_INDEX_32I HIPSPARSE_INDEX_32I
#define CUSPARSE_INDEX_64I HIPSPARSE_INDEX_64I
#define CUDA_R_32F HIP_R_32F
#define CUDA_R_64F HIP_R_64F

// --- CUDA kernel qualifiers → HIP (same syntax) --------------------------
// HIP supports __global__, __device__, __shared__, __host__ identically.
// No remapping needed.

// --- warp-level primitives ------------------------------------------------
#define __shfl_sync(mask, val, src) __shfl(val, src)
#define __shfl_down_sync(mask, val, delta) __shfl_down(val, delta)
#define __shfl_xor_sync(mask, val, laneMask) __shfl_xor(val, laneMask)
#define __ballot_sync(mask, pred) __ballot(pred)

// --- half type ------------------------------------------------------------
// hip_fp16.h provides __half equivalent to cuda_fp16.h's __half.
// No remapping needed.

#endif  // TIDES_USE_HIP
