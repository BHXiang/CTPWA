#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cassert>

// 优化后的M计算核函数
__global__ void compute_M_kernel(
    const cuDoubleComplex *d_D, // [K, A, B]
    const cuDoubleComplex *d_S, // [A, B]
    int K, int A, int B,
    cuDoubleComplex *d_M) // [K, A]
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;

    if (a < A)
    {
        // 每个线程处理所有K值和当前a的B维度
        for (int m = 0; m < K; m++)
        {
            cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

            // 预计算基础索引
            int base_d_index = m * A * B + a * B;
            int base_s_index = a * B;

            // 循环B维度（B很小，可以展开）
            for (int b = 0; b < B; b++)
            {
                // 计算 S[a,b] * conj(D[m,a,b])
                sum = cuCadd(sum,
                             cuCmul(d_S[base_s_index + b],
                                    cuConj(d_D[base_d_index + b])));
            }

            d_M[m * A + a] = sum;
        }
    }
}

// 计算梯度的主函数
void compute_gradient(
    const cuDoubleComplex *d_D, // [K, A, B]
    const cuDoubleComplex *d_P, // [K, N]
    const cuDoubleComplex *d_S, // [A, B]
    const cuDoubleComplex *d_U, // [A]
    const cuDoubleComplex *d_Q, // [N]
    double phsp_factor,
    int K, int A, int B, int N,
    cuDoubleComplex *d_grad, // [K]
    cublasHandle_t cublas_handle)
{

    // 1. 计算中间矩阵 M[A, K] = ∑b (S[a,b] * conj(D[m,a,b]))
    cuDoubleComplex *d_M;
    cudaMalloc(&d_M, A * K * sizeof(cuDoubleComplex));

    // 优化后的线程分配：每个线程处理一个a的所有K和B
    int threadsPerBlock = 256;
    int blocksPerGrid = (A + threadsPerBlock - 1) / threadsPerBlock;

    compute_M_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_D, d_S, K, A, B, d_M);

    // 3. 计算数据项: grad_data = M^T * invU
    // 注意: M是[A, K]矩阵，我们需要计算 M^T * invU 得到[K]向量
    cuDoubleComplex alpha_c = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta_c = make_cuDoubleComplex(0.0, 0.0);

    // 注意：cuBLAS 默认使用列优先存储
    // 如果我们使用行优先存储的 M，需要转置它
    cublasZgemv(cublas_handle,
                CUBLAS_OP_T, // 转置操作，因为 M 是行优先存储
                A, K,        // 转置后的维度是 A × K
                &alpha_c,
                d_M, A, // lda = A (行优先存储的 leading dimension)
                d_U, 1,
                &beta_c,
                d_grad, 1);

    // cuDoubleComplex h_grad[K];
    // cudaMemcpy(h_grad, d_grad, K * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < K; i++)
    // {
    //     std::cout << "grad after data term[" << i << "] = (" << cuCreal(h_grad[i]) << "," << cuCimag(h_grad[i]) << ")\n";
    // }

    // 4. 取负号
    cuDoubleComplex neg_one = make_cuDoubleComplex(-1.0, 0.0);
    cublasZscal(cublas_handle, K, &neg_one, d_grad, 1);

    // 5. 计算相位空间项
    if (phsp_factor != 0.0)
    {
        cuDoubleComplex *d_term2;
        cudaMalloc(&d_term2, K * sizeof(cuDoubleComplex));

        cuDoubleComplex alpha2 = make_cuDoubleComplex(A / phsp_factor, 0.0);
        cublasZgemv(cublas_handle,
                    CUBLAS_OP_C,
                    N, K,
                    &alpha2,
                    d_P, N,
                    d_Q, 1,
                    &beta_c,
                    d_term2, 1);

        // 将term2加到梯度上
        cublasZaxpy(cublas_handle,
                    K,
                    &alpha_c, // 注意这里使用1.0而不是alpha2，因为alpha2已经在gemv中使用了
                    d_term2, 1,
                    d_grad, 1);

        cudaFree(d_term2);
    }

    // 6. 乘以2
    cuDoubleComplex two = make_cuDoubleComplex(2.0, 0.0);
    cublasZscal(cublas_handle, K, &two, d_grad, 1);

    // 释放临时内存
    cudaFree(d_M);
    // cudaFree(d_invU);
}
