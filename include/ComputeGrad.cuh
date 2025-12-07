#ifndef COMPUTEGRAD_CUH
#define COMPUTEGRAD_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

// 包装函数：计算梯度
// void computeGradient(
//     cuDoubleComplex *d_grad, // 设备上的梯度向量
//     // const cuDoubleComplex *d_v, // 设备上的向量
//     const cuDoubleComplex *d_A, // 设备上的矩阵
//     const cuDoubleComplex *d_B, // 设备上的中间结果B
//     const double C,             // 中间结果C
//     const cuDoubleComplex *d_T, // 设备上的张量
//     const cuDoubleComplex *d_S, // 设备上的中间结果S
//     const double *d_Q,          // 设备上的中间结果Q
//     int nlength, int ngls, int npolar, int phsp_length);
// void compute_gradient(
//     const cuDoubleComplex *d_D,
//     const cuDoubleComplex *d_P,
//     const cuDoubleComplex *d_S,
//     const double *d_U,
//     const cuDoubleComplex *d_Q,
//     double phsp_factor,
//     int K, int A, int B, int N,
//     cuDoubleComplex *d_grad,
//     cudaStream_t stream = 0);
// void compute_gradient(
//     const cuDoubleComplex *d_D,
//     const cuDoubleComplex *d_P,
//     const cuDoubleComplex *d_S,
//     const double *d_U,
//     const cuDoubleComplex *d_Q,
//     double phsp_factor,
//     int K, int A, int B, int N,
//     cuDoubleComplex *d_grad); //, cublasHandle_t cublas_handle);

void compute_gradient(
    const cuDoubleComplex *d_D, // [K, A, B]
    const cuDoubleComplex *d_P, // [K, N]
    const cuDoubleComplex *d_S, // [A, B]
    const double *d_U,          // [A]
    const cuDoubleComplex *d_Q, // [N]
    double phsp_factor,
    int K, int A, int B, int N,
    cuDoubleComplex *d_grad, // [K]
    cublasHandle_t cublas_handle);

#endif