#ifndef HELICITY_CUH
#define HELICITY_CUH

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>

// 四矢量结构体
struct LorentzVector
{
    double E, Px, Py, Pz;

    __device__ __host__ LorentzVector() : E(0), Px(0), Py(0), Pz(0) {}
    __device__ __host__ LorentzVector(double e, double px, double py, double pz)
        : E(e), Px(px), Py(py), Pz(pz) {}

    __device__ __host__ double P() const
    {
        return sqrt(Px * Px + Py * Py + Pz * Pz);
    }

    __device__ __host__ double M() const
    {
        return sqrt(E * E - Px * Px - Py * Py - Pz * Pz);
    }

    __device__ __host__ double Dot(const LorentzVector &other) const
    {
        return E * other.E - (Px * other.Px + Py * other.Py + Pz * other.Pz);
    }

    __device__ __host__ LorentzVector operator+(const LorentzVector &other) const
    {
        return LorentzVector(E + other.E, Px + other.Px, Py + other.Py, Pz + other.Pz);
    }
};

// 设备函数声明
// __device__ double gamma_function(double x);
// __device__ double factorial_device(double n);
// __device__ bool is_half_integer(double x);
// __device__ bool is_integer(double x);
// __device__ double pow_neg_one(int k);
// __device__ double max3_device(double a, double b, double c);
// __device__ double min3_device(double a, double b, double c);
// __device__ double abs_device(double x);
// __device__ double ClebschGordan_half_integer(double j1, double m1, double j2, double m2, double J, double M);
__device__ thrust::complex<double> dfunc_device(int j, int m1, int m2, double beta);
__device__ thrust::complex<double> wignerD_element_device(int j, int m1, int m2, double alpha, double beta, double gamma);
__device__ void compute_wignerD_matrix_device(thrust::complex<double> *d_result, int j, double alpha, double beta, double gamma);
__device__ void get_wignerD_matrix(thrust::complex<double> *output, int j, double alpha, double beta, double gamma, int block_dim = 16);
__device__ double associated_legendre_poly(int l, int m, double x);
__device__ void spherical_harmonic_complex(int l, int m, double theta, double phi, double *real_part, double *imag_part);
__device__ void MassiveTrans_device(thrust::complex<double> *trans, LorentzVector p, LorentzVector q, int dj);

// 主函数：计算螺旋度振幅
__device__ void pwahelicity_device(thrust::complex<double> *amp, LorentzVector p1, int dj1, LorentzVector p2, int dj2, int dj, int dS, int dL);

// // 包装函数：用于在主机代码中调用螺旋度振幅计算
// __global__ void compute_helicity_amplitude_kernel(thrust::complex<double> *d_amp,
//                                                   LorentzVector *d_p1, int dj1,
//                                                   LorentzVector *d_p2, int dj2,
//                                                   int dj, int dS, int dL,
//                                                   int num_events);

// // 主机端包装函数
// void compute_helicity_amplitude(thrust::complex<double> *h_amp,
//                                 LorentzVector *h_p1, int dj1,
//                                 LorentzVector *h_p2, int dj2,
//                                 int dj, int dS, int dL,
//                                 int num_events);

#endif // HELICITY_CUH
