#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdio.h>
#include <vector>
// #include <iostream>
// #include <complex>
#include <cublas_v2.h>

// // 核函数：计算复数模平方并按 npolar 分组求和
// __global__ void computeModTotalWeight(
//     const cuComplex *__restrict__ complex_result,
//     double *__restrict__ final_result,
//     int nEvents, int npolar)
// {
//     int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (event_idx >= nEvents)
//         return;

//     double sum = 0.0;

//     // 每个线程处理一个 event，累加对应的 npolar 个元素的模平方
//     for (int polar_idx = 0; polar_idx < npolar; polar_idx++)
//     {
//         int global_idx = event_idx * npolar + polar_idx;
//         cuComplex val = complex_result[global_idx];
//         double mod_square = val.x * val.x + val.y * val.y;
//         sum += mod_square;
//     }

//     final_result[event_idx] = sum;
// }

// 核函数：计算复数模平方并按 npolar 分组求和，同时计算总和
__global__ void computeModTotalWeight(
    const cuComplex *__restrict__ complex_result,
    double *__restrict__ final_result,
    double *__restrict__ total_sum,
    int nEvents, int npolar)
{
    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (event_idx >= nEvents)
        return;

    double sum = 0.0;

    // 每个线程处理一个 event，累加对应的 npolar 个元素的模平方
    for (int polar_idx = 0; polar_idx < npolar; polar_idx++)
    {
        int global_idx = event_idx * npolar + polar_idx;
        cuComplex val = complex_result[global_idx];
        double mod_square = val.x * val.x + val.y * val.y;
        sum += mod_square;
    }

    final_result[event_idx] = sum;

    // 使用原子操作累加总和
    atomicAdd(total_sum, sum);
}

// __global__ void computeModPartialWeight(
//     const cuComplex *__restrict__ complex_matrix,
//     const cuComplex *__restrict__ complex_vector,
//     double *__restrict__ final_result,
//     int *d_nSLvectors,
//     int npartials,
//     int nEvents, int npolar)
// {
//     int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (event_idx >= nEvents)
//         return;

//     // double sum = 0.0;

//     int sltotal = 0;
//     for (int p_idx = 0; p_idx < npartials; p_idx++)
//     {
//         int nSL = d_nSLvectors[p_idx];
//         double partial_sum = 0.0;
//         for (int sl_idx = 0; sl_idx < nSL; sl_idx++)
//         {
//             for (int polar_idx = 0; polar_idx < nSL; polar_idx++)
//             {
//                 int global_idx = sltotal * nEvents * npolar + event_idx * npolar + polar_idx;
//                 cuComplex val = complex_matrix[global_idx];
//                 cuComplex vec_val = complex_vector[polar_idx];
//                 // 计算矩阵元素与向量元素的乘积
//                 cuComplex prod = cuCmul(val, vec_val);
//                 double mod_square = prod.x * prod.x + prod.y * prod.y;
//                 partial_sum += mod_square;
//             }
//             sltotal++;
//         }
//         final_result[event_idx * npartials + p_idx] = partial_sum;
//     }

//     // final_result[event_idx] = sum;
// }

__global__ void computeModPartialWeight(
    const cuComplex *__restrict__ complex_matrix,
    const cuComplex *__restrict__ complex_vector,
    double *__restrict__ final_result,
    double *__restrict__ partial_sums,
    int *d_nSLvectors,
    int npartials,
    int nEvents, int npolar)
{
    extern __shared__ double shared_sums[];

    int event_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 初始化共享内存
    for (int i = tid; i < npartials; i += blockDim.x)
    {
        shared_sums[i] = 0.0;
    }
    __syncthreads();

    if (event_idx < nEvents)
    {
        int sltotal = 0;

        // 为每个部分计算权重
        for (int p_idx = 0; p_idx < npartials; p_idx++)
        {
            int nSL = d_nSLvectors[p_idx];
            double partial_sum = 0.0;

            // 计算当前部分在当前事件上的权重
            for (int sl_idx = 0; sl_idx < nSL; sl_idx++)
            {
                for (int polar_idx = 0; polar_idx < npolar; polar_idx++)
                {
                    int global_idx = sltotal * nEvents * npolar + event_idx * npolar + polar_idx;
                    cuComplex val = complex_matrix[global_idx];
                    cuComplex vec_val = complex_vector[p_idx * nSL + sl_idx];
                    cuComplex prod = cuCmulf(val, vec_val);
                    // printf("Event %d, Partial %d, SL %d, Polar %d: Matrix Element = (%f, %f i), Vector Element = (%f, %f i), Product = (%f, %f i)\n", event_idx, p_idx, sl_idx, polar_idx, val.x, val.y, vec_val.x, vec_val.y, prod.x, prod.y);
                    double mod_square = prod.x * prod.x + prod.y * prod.y;
                    partial_sum += mod_square;
                }
                sltotal++;
            }

            // 存储当前事件当前部分的结果
            final_result[event_idx * npartials + p_idx] = partial_sum;

            // printf("Event %d, Partial %d, Partial Sum = %f\n", event_idx, p_idx, partial_sum);

            // 累加到共享内存
            atomicAdd(&shared_sums[p_idx], partial_sum);
        }
    }

    __syncthreads();

    // 将共享内存中的结果累加到全局内存
    for (int i = tid; i < npartials; i += blockDim.x)
    {
        if (shared_sums[i] != 0.0)
        {
            atomicAdd(&partial_sums[i], shared_sums[i]);
        }
    }
}

void computeWeightResult(
    const cuComplex *d_matrix,
    const cuComplex *d_vector,
    double *d_total_result,
    double *d_total_integral,
    double *d_partial_result,
    double *d_partial_sums,
    int *d_nSLvectors,
    int npartials,
    int nEvents, int ngls, int npolar)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配设备内存
    cuComplex *d_complex_result = nullptr;
    cudaMalloc(&d_complex_result, nEvents * npolar * sizeof(cuComplex));

    // cuBLAS 矩阵向量乘法
    const cuComplex alpha = make_cuComplex(1.0, 0.0);
    const cuComplex beta = make_cuComplex(0.0, 0.0);

    cublasCgemv(handle, CUBLAS_OP_N, nEvents * npolar, ngls, &alpha,
                d_matrix, nEvents * npolar, d_vector, 1, &beta, d_complex_result, 1);

    // 检查 cuBLAS 调用
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        printf("cuBLAS error: %s\n", cudaGetErrorString(cuda_error));
    }

    // 计算总权重
    int blockSize = 256;
    int gridSize = (nEvents + blockSize - 1) / blockSize;

    computeModTotalWeight<<<gridSize, blockSize>>>(d_complex_result, d_total_result, d_total_integral, nEvents, npolar);

    // 计算部分权重
    // computeModPartialWeight<<<gridSize, blockSize>>>(d_matrix, d_vector, d_partial_result, d_nSLvectors, npartials, nEvents, npolar);
    size_t shared_mem_size = npartials * sizeof(double);
    computeModPartialWeight<<<gridSize, blockSize, shared_mem_size>>>(d_matrix, d_vector, d_partial_result, d_partial_sums, d_nSLvectors, npartials, nEvents, npolar);

    // 检查核函数执行
    cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess)
    {
        printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
    }

    // 同步确保所有操作完成
    cudaDeviceSynchronize();

    // 清理资源
    cudaFree(d_complex_result);
    cublasDestroy(handle);
}

// void computeWeightResult(
//     const cuComplex *d_matrix,
//     const cuComplex *d_vector,
//     double *d_total_result,
//     double *d_total_integral,
//     double *d_partial_result,
//     int *d_nSLvectors,
//     int npartials,
//     int nEvents, int ngls, int npolar)
// {
//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     // 分配设备内存
//     cuComplex *d_complex_result = nullptr;
//     cudaMalloc(&d_complex_result, nEvents * npolar * sizeof(cuComplex));

//     // cuBLAS 矩阵向量乘法参数
//     const cuComplex alpha = make_cuComplex(1.0, 0.0);
//     const cuComplex beta = make_cuComplex(0.0, 0.0);

//     // 执行矩阵向量乘法: complex_result = matrix * vector
//     // 注意：cuBLAS 使用列主序，所以我们需要适当调整参数
//     // 如果矩阵是行主序的 M x N 矩阵，在 cuBLAS 中相当于列主序的 N x M 矩阵的转置
//     cublasCgemv(handle, CUBLAS_OP_N, nEvents * npolar, ngls, &alpha, d_matrix, nEvents * npolar, d_vector, 1, &beta, d_complex_result, 1);

//     // 检查 cuBLAS 调用是否成功
//     cudaError_t cuda_error = cudaGetLastError();
//     if (cuda_error != cudaSuccess)
//     {
//         printf("cuBLAS error: %s\n", cudaGetErrorString(cuda_error));
//     }

//     // 对于较小的 npolar，使用简单版本
//     int blockSize = 256;
//     int gridSize = (nEvents + blockSize - 1) / blockSize;

//     computeModTotalWeight<<<gridSize, blockSize>>>(d_complex_result, d_total_result, d_total_integral, nEvents, npolar);

//     computeModPartialWeight<<<gridSize, blockSize>>>(d_matrix, d_vector, d_partial_result, d_nSLvectors, npartials, nEvents, npolar);

//     // 检查核函数执行是否成功
//     cuda_error = cudaGetLastError();
//     if (cuda_error != cudaSuccess)
//     {
//         printf("Kernel error: %s\n", cudaGetErrorString(cuda_error));
//     }

//     // 同步确保所有操作完成
//     cudaDeviceSynchronize();

//     // 清理资源
//     cudaFree(d_complex_result);
//     cublasDestroy(handle);
// }
