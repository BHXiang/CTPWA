#include <helicity.cuh>
#include <cuda_runtime.h>
#include <iostream>

__device__ double gamma_function(double x)
{
    // 前7个整数的阶乘直接返回结果（优化）
    if (x >= 1.0 && x <= 7.0)
    {
        double rounded = round(x);
        if (fabs(x - rounded) < 1e-10)
        {
            int n = static_cast<int>(rounded);
            switch (n)
            {
            case 1:
                return 1.0;
            case 2:
                return 1.0;
            case 3:
                return 2.0;
            case 4:
                return 6.0;
            case 5:
                return 24.0;
            case 6:
                return 120.0;
            case 7:
                return 720.0;
            }
        }
    }

    if (x <= 0.0)
    {
        return M_PI / (sin(M_PI * x) * gamma_function(1.0 - x));
    }

    double rounded = round(2.0 * x) / 2.0;
    if (fabs(x - rounded) < 1e-10)
    {
        if (fabs(x - round(x)) < 1e-10)
        {
            int n = static_cast<int>(round(x));
            if (n <= 0)
                return 1.0 / 0.0;
            if (n <= 20)
            {
                unsigned long long fact = 1;
                for (int i = 1; i < n; i++)
                    fact *= i;
                return static_cast<double>(fact);
            }
            return sqrt(2 * M_PI * (n - 1)) * pow(static_cast<double>(n - 1) / M_E, n - 1);
        }
        else
        {
            double n = x - 0.5;
            if (fabs(n - round(n)) < 1e-10)
            {
                int int_n = static_cast<int>(round(n));
                if (int_n == 0)
                    return sqrt(M_PI);

                double double_fact = 1.0;
                for (int i = 1; i <= 2 * int_n; i++)
                {
                    double_fact *= i;
                }

                double single_fact = 1.0;
                for (int i = 1; i <= int_n; i++)
                {
                    single_fact *= i;
                }

                return double_fact * sqrt(M_PI) / (pow(4.0, int_n) * single_fact);
            }
        }
    }

    const double g = 7.0;
    const double lanczos_coeff[] = {
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

    if (x < 0.5)
    {
        return M_PI / (sin(M_PI * x) * gamma_function(1.0 - x));
    }

    x -= 1.0;
    double a = lanczos_coeff[0];
    double t = x + g + 0.5;

    for (int i = 1; i < 9; i++)
    {
        a += lanczos_coeff[i] / (x + i);
    }

    return sqrt(2 * M_PI) * pow(t, x + 0.5) * exp(-t) * a;
}

// __device__ double factorial_device(double n)
// {
//     if (n < 0)
//         return 0.0;

//     // 整数情况
//     if (fabs(n - round(n)) < 1e-10)
//     {
//         int int_n = (int)round(n);
//         if (int_n <= 20)
//         {
//             static const double fact_table[] = {
//                 1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
//                 3628800.0, 39916800.0};
//             return fact_table[int_n];
//         }
//     }

//     // 半整数情况使用Gamma函数
//     return gamma_function(n + 1.0);
// }
__device__ double factorial_device(double n)
{
    // 处理负数和非法输入
    if (n < 0.0 || isnan(n))
    {
        return 0.0;
    }

    // 小数值查表
    if (n <= 20.0 && fabs(n - round(n)) < 1e-10)
    {
        int int_n = (int)round(n);
        static const double fact_table[] = {
            1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
            3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
            1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0,
            121645100408832000.0, 2432902008176640000.0};
        return fact_table[int_n];
    }

    // 对于大数值或半整数，使用斯特林近似
    if (n > 20.0)
    {
        // 斯特林公式: n! ≈ sqrt(2πn) * (n/e)^n
        return sqrt(2.0 * M_PI * n) * pow(n / M_E, n);
    }

    // 对于中等数值，使用迭代计算
    double result = 1.0;
    for (int i = 2; i <= (int)n; i++)
    {
        result *= i;
    }
    return result;
}

__device__ bool is_half_integer(double x)
{
    double twice = 2.0 * x;
    return fabs(twice - round(twice)) < 1e-10 && fabs(x - round(x)) > 1e-10;
}

__device__ bool is_integer(double x)
{
    return fabs(x - round(x)) < 1e-10;
}

__device__ double pow_neg_one(int k)
{
    return (k % 2 == 0) ? 1.0 : -1.0;
}

__device__ double max3_device(double a, double b, double c)
{
    double max_ab = (a > b) ? a : b;
    return (max_ab > c) ? max_ab : c;
}

__device__ double min3_device(double a, double b, double c)
{
    double min_ab = (a < b) ? a : b;
    return (min_ab < c) ? min_ab : c;
}

__device__ double abs_device(double x)
{
    return (x < 0) ? -x : x;
}

// 主接口函数
__device__ double ClebschGordan_corrected(double j1, double m1, double j2, double m2, double J, double M)
{
    // 基本检查
    if (fabs(M - (m1 + m2)) > 1e-10)
        return 0.0;
    if (J < fabs(j1 - j2) || J > j1 + j2)
        return 0.0;
    if (fabs(m1) > j1 || fabs(m2) > j2 || fabs(M) > J)
        return 0.0;

    // 使用更稳定的数值方法
    double term1 = sqrt((2.0 * J + 1.0) * factorial_device(j1 + j2 - J) * factorial_device(j1 - j2 + J) * factorial_device(-j1 + j2 + J) / factorial_device(j1 + j2 + J + 1.0));

    double term2 = sqrt(factorial_device(J + M) * factorial_device(J - M) * factorial_device(j1 - m1) * factorial_device(j1 + m1) * factorial_device(j2 - m2) * factorial_device(j2 + m2));

    // 求和的界限
    int kmin = (int)ceil(fmax(0.0, fmax(j2 - J - m1, j1 - J + m2)));
    int kmax = (int)floor(fmin(j1 + j2 - J, fmin(j1 - m1, j2 + m2)));

    double sum = 0.0;
    for (int k = kmin; k <= kmax; k++)
    {
        double denominator = factorial_device(k) *
                             factorial_device(j1 + j2 - J - k) *
                             factorial_device(j1 - m1 - k) *
                             factorial_device(j2 + m2 - k) *
                             factorial_device(J - j2 + m1 + k) *
                             factorial_device(J - j1 - m2 + k);

        if (fabs(denominator) > 1e-15)
        {
            double sign = (k % 2 == 0) ? 1.0 : -1.0;
            sum += sign / denominator;
        }
    }

    return term1 * term2 * sum;
}

__device__ thrust::complex<double> dfunc_device(int j, int m1, int m2, double beta)
{
    // 您的dfunc_device实现
    if (abs(m1) > j || abs(m2) > j)
    {
        return thrust::complex<double>(0.0, 0.0);
    }

    thrust::complex<double> sum(0.0, 0.0);
    int k_min = (m1 < m2) ? 0 : m1 - m2;
    int k_max = (m1 > -m2) ? j - m2 : j + m1;

    for (int k = k_min; k <= k_max; ++k)
    {
        double term1 = gamma_function(j + m2 + 1.0);
        double term2 = gamma_function(j - m2 + 1.0);
        double term3 = gamma_function(j + m1 + 1.0);
        double term4 = gamma_function(j - m1 + 1.0);

        double numerator = term1 * term2 * term3 * term4;

        double denom1 = gamma_function(k + 1.0);
        double denom2 = gamma_function(j - m2 - k + 1.0);
        double denom3 = gamma_function(j + m1 - k + 1.0);
        double denom4 = gamma_function(k - m1 + m2 + 1.0);

        double denominator = denom1 * denom2 * denom3 * denom4;

        double comb = sqrt(numerator) / denominator;

        double cos_half = cos(beta / 2.0);
        double sin_half = sin(beta / 2.0);

        double cos_term = pow(cos_half, 2 * j + m1 - m2 - 2 * k);
        double sin_term = pow(sin_half, 2 * k - m1 + m2);

        double term = comb * cos_term * sin_term;
        double sign = (k % 2 == 0) ? 1.0 : -1.0;

        sum += sign * term;
    }

    return sum;
}

__device__ thrust::complex<double> wignerD_element_device(int j, int m1, int m2,
                                                          double alpha, double beta, double gamma)
{
    thrust::complex<double> d_val = dfunc_device(j, m1, m2, beta);
    return thrust::exp(thrust::complex<double>(0.0, -m1 * alpha)) *
           d_val *
           thrust::exp(thrust::complex<double>(0.0, -m2 * gamma));
}

__device__ void compute_wignerD_matrix_device(thrust::complex<double> *d_result,
                                              int j, double alpha, double beta, double gamma)
{
    int dim = 2 * j + 1;
    int idx = threadIdx.x;
    int idy = threadIdx.y;

    if (idx < dim && idy < dim)
    {
        int m1 = idx - j;
        int m2 = idy - j;
        d_result[idx * dim + idy] = wignerD_element_device(j, m1, m2, alpha, beta, gamma);
    }
}

__device__ void get_wignerD_matrix(thrust::complex<double> *output,
                                   int j, double alpha, double beta, double gamma,
                                   int block_dim)
{
    int dim = 2 * j + 1;
    extern __shared__ thrust::complex<double> shared_mem[];
    thrust::complex<double> *temp_matrix = shared_mem;

    compute_wignerD_matrix_device(temp_matrix, j, alpha, beta, gamma);
    __syncthreads();

    int idx = threadIdx.x;
    int idy = threadIdx.y;
    if (idx < dim && idy < dim)
    {
        output[idx * dim + idy] = temp_matrix[idx * dim + idy];
    }
}

__device__ double associated_legendre_poly(int l, int m, double x)
{
    // 您的associated_legendre_poly实现
    if (l < 0 || m < 0 || m > l)
        return 0.0;

    if (m == 0)
    {
        if (l == 0)
            return 1.0;
        if (l == 1)
            return x;
        if (l == 2)
            return 0.5 * (3.0 * x * x - 1.0);
        if (l == 3)
            return 0.5 * (5.0 * x * x * x - 3.0 * x);

        double P_prev2 = 1.0;
        double P_prev1 = x;
        double P_current = 0.0;

        for (int n = 2; n <= l; ++n)
        {
            P_current = ((2.0 * n - 1.0) * x * P_prev1 - (n - 1.0) * P_prev2) / n;
            P_prev2 = P_prev1;
            P_prev1 = P_current;
        }
        return P_current;
    }

    double P_mm = 1.0;
    for (int i = 1; i <= m; ++i)
    {
        P_mm *= (2 * i - 1) * sqrt(1.0 - x * x);
    }
    P_mm *= pow(-1.0, m);

    if (l == m)
        return P_mm;

    double P_mp1_m = x * (2.0 * m + 1.0) * P_mm;
    if (l == m + 1)
        return P_mp1_m;

    double P_prev2 = P_mm;
    double P_prev1 = P_mp1_m;
    double P_current = 0.0;

    for (int n = m + 2; n <= l; ++n)
    {
        P_current = ((2.0 * n - 1.0) * x * P_prev1 - (n + m - 1.0) * P_prev2) / (n - m);
        P_prev2 = P_prev1;
        P_prev1 = P_current;
    }

    return P_current;
}

__device__ void spherical_harmonic_complex(int l, int m, double theta, double phi,
                                           double *real_part, double *imag_part)
{
    // 您的spherical_harmonic_complex实现
    if (l < 0 || abs(m) > l)
    {
        *real_part = 0.0;
        *imag_part = 0.0;
        return;
    }

    if (l == 0 && m == 0)
    {
        *real_part = 0.5 * sqrt(1.0 / M_PI);
        *imag_part = 0.0;
        return;
    }

    double x = cos(theta);
    int abs_m = abs(m);
    double P_lm = associated_legendre_poly(l, abs_m, x);

    double norm_factor = sqrt((2.0 * l + 1.0) * factorial_device(l - abs_m) /
                              (4.0 * M_PI * factorial_device(l + abs_m)));

    double phase = 1.0;
    if (m < 0)
    {
        phase = (abs_m % 2 == 0) ? 1.0 : -1.0;
    }

    double amplitude = phase * norm_factor * P_lm;
    double m_phi = -1 * m * phi;

    *real_part = amplitude * cos(m_phi);
    *imag_part = amplitude * sin(m_phi);
}

__device__ thrust::complex<double> compute_untransformed_element(int sgm, int sgm1, int sgm2,
                                                                 int dj, int dj1, int dj2,
                                                                 int dS, int dL,
                                                                 LorentzVector p1, LorentzVector p2)
{
    // 计算质心系角度（保持不变）
    double m1 = p1.M();
    double m2 = p2.M();
    double p1p2 = p1.Dot(p2);
    double m = sqrt(m1 * m1 + m2 * m2 + 2 * p1p2);
    double lamb = (2 * m * p1.E + m * m + m1 * m1 - m2 * m2) / (2 * m * (m + p1.E + p2.E));
    double qs = sqrt(pow(m, 4) + pow(m1 * m1 - m2 * m2, 2) - 2 * m * m * (m1 * m1 + m2 * m2)) / (2 * m);
    double nsx = (p1.Px - lamb * (p1.Px + p2.Px)) / qs;
    double nsy = (p1.Py - lamb * (p1.Py + p2.Py)) / qs;
    double nsz = (p1.Pz - lamb * (p1.Pz + p2.Pz)) / qs;
    double theta = acos(nsz);
    double phi = atan2(nsy, nsx);

    // 计算球谐函数 - 修正相位
    double real_part, imag_part;
    spherical_harmonic_complex(dL, sgm - sgm1 - sgm2, theta, phi, &real_part, &imag_part);
    thrust::complex<double> spherical(real_part, imag_part);

    // 使用修正的CG系数
    double cg1 = ClebschGordan_corrected(dj1, sgm1, dj2, sgm2, dS, sgm1 + sgm2);
    double cg2 = ClebschGordan_corrected(dS, sgm1 + sgm2, dL, sgm - sgm1 - sgm2, dj, sgm);

    // 注意：这里可能需要额外的相位因子 (-1)^(j1-j2+...)
    double phase = 1.0;
    if (((int)(dj1 - dj2 + dL)) % 2 != 0)
    {
        phase = -1.0;
    }

    return phase * (pow(qs, dL) / sqrt(2 * dj + 1)) * cg1 * cg2 * spherical;
}

__device__ thrust::complex<double> compute_single_transformation(LorentzVector P_total, LorentzVector p, int j, int m_new, int m_old)
{
    // 计算boost参数
    double Pp = p.P();
    double PP = P_total.P();

    // 计算法向量
    double hat_x = P_total.Py * p.Pz - P_total.Pz * p.Py;
    double hat_y = P_total.Pz * p.Px - P_total.Px * p.Pz;
    double hat_z = P_total.Px * p.Py - P_total.Py * p.Px;

    // 如果动量平行，返回delta函数
    if (fabs(hat_x) < 1e-10 && fabs(hat_y) < 1e-10 && fabs(hat_z) < 1e-10)
    {
        return (m_new == m_old) ? thrust::complex<double>(1.0, 0.0)
                                : thrust::complex<double>(0.0, 0.0);
    }

    // 计算夹角
    double cos_angle = (p.Px * P_total.Px + p.Py * P_total.Py + p.Pz * P_total.Pz) / (Pp * PP);
    double sin_angle = sqrt(1.0 - cos_angle * cos_angle);

    // 计算单位法向量
    double norm = sqrt(hat_x * hat_x + hat_y * hat_y + hat_z * hat_z);
    double nhat_x = hat_x / norm;
    double nhat_y = hat_y / norm;
    double nhat_z = hat_z / norm;

    // 计算欧拉角
    double theta = acos(nhat_z);
    double phi = atan2(nhat_y, nhat_x);

    // 计算boost角（Wigner旋转角）
    double m_part = p.M();
    double gamma_part = p.E / m_part;
    double m_total = P_total.M();
    double gamma_total = P_total.E / m_total;

    double numerator = sin_angle * sqrt((gamma_total - 1.0) * (gamma_part - 1.0));
    double denominator = sqrt((gamma_total + 1.0) * (gamma_part + 1.0)) +
                         cos_angle * sqrt((gamma_total - 1.0) * (gamma_part - 1.0));

    double psi = 2.0 * atan2(numerator, denominator);

    // 计算Wigner D矩阵元素
    // 变换矩阵是 D(φ,θ,ψ) = D(-φ,-θ,-ψ) * D(0,θ,φ) 的乘积
    thrust::complex<double> result(0.0, 0.0);

    // 对中间量子数求和
    for (int k = -j; k <= j; k++)
    {
        thrust::complex<double> D1 = wignerD_element_device(j, m_new, k, -phi, -theta, -psi);
        thrust::complex<double> D2 = wignerD_element_device(j, k, m_old, 0.0, theta, phi);
        result += D1 * D2;
    }

    return result;
}

__device__ void pwahelicity_device(thrust::complex<double> *amp, LorentzVector p1, int dj1, LorentzVector p2, int dj2, int dj, int dS, int dL)
{
    int dim_j = 2 * dj + 1;
    int dim_j1 = 2 * dj1 + 1;
    int dim_j2 = 2 * dj2 + 1;

    LorentzVector P_total = p1 + p2;

    // 一次性计算所有需要的量
    for (int sgm_idx = 0; sgm_idx < dim_j; sgm_idx++)
    {
        int sgm = dj - sgm_idx;

        for (int sgm1_idx_new = 0; sgm1_idx_new < dim_j1; sgm1_idx_new++)
        {
            int sgm1_new = dj1 - sgm1_idx_new;

            for (int sgm2_idx_new = 0; sgm2_idx_new < dim_j2; sgm2_idx_new++)
            {
                int sgm2_new = dj2 - sgm2_idx_new;

                thrust::complex<double> result(0.0, 0.0);

                // 对旧螺旋度求和
                for (int sgm1_idx_old = 0; sgm1_idx_old < dim_j1; sgm1_idx_old++)
                {
                    int sgm1_old = dj1 - sgm1_idx_old;

                    for (int sgm2_idx_old = 0; sgm2_idx_old < dim_j2; sgm2_idx_old++)
                    {
                        int sgm2_old = dj2 - sgm2_idx_old;

                        // 计算未变换的振幅元素
                        thrust::complex<double> untransformed_amp = compute_untransformed_element(sgm, sgm1_old, sgm2_old, dj, dj1, dj2, dS, dL, p1, p2);

                        // 计算变换矩阵元素
                        thrust::complex<double> trans1 = compute_single_transformation(P_total, p1, dj1, sgm1_new, sgm1_old);
                        thrust::complex<double> trans2 = compute_single_transformation(P_total, p2, dj2, sgm2_new, sgm2_old);

                        result += untransformed_amp * trans1 * trans2;
                    }
                }

                int idx = sgm_idx * (dim_j1 * dim_j2) + sgm1_idx_new * dim_j2 + sgm2_idx_new;
                amp[idx] = result;
            }
        }
    }
}

__device__ void MassiveTrans_shared(
    thrust::complex<double> *trans,
    LorentzVector p, LorentzVector q, int dj,
    thrust::complex<double> *shared_buf)
{
    int dim = 2 * dj + 1;
    int matrix_size = dim * dim;

    // 初始化变换矩阵为单位矩阵
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            trans[i * dim + j] = (i == j) ? thrust::complex<double>(1.0, 0.0)
                                          : thrust::complex<double>(0.0, 0.0);
        }
    }

    double Pp = p.P();
    double Pq = q.P();

    // 修正：使用正确的向量叉积计算法向量
    double hat_x = p.Py * q.Pz - p.Pz * q.Py;
    double hat_y = p.Pz * q.Px - p.Px * q.Pz;
    double hat_z = p.Px * q.Py - p.Py * q.Px;

    if (fabs(hat_x) < 1e-10 && fabs(hat_y) < 1e-10 && fabs(hat_z) < 1e-10)
    {
        return;
    }

    double m1 = p.M();
    double gamma1 = p.E / m1;
    double m2 = q.M();
    double gamma2 = q.E / m2;

    // 修正：正确的cos_angle计算，去掉多余的负号
    double cos_angle = (p.Px * q.Px + p.Py * q.Py + p.Pz * q.Pz) / (Pp * Pq);
    double sin_angle = sqrt(1 - cos_angle * cos_angle);

    double norm = sqrt(hat_x * hat_x + hat_y * hat_y + hat_z * hat_z);
    double nhat_x = hat_x / norm;
    double nhat_y = hat_y / norm;
    double nhat_z = hat_z / norm;

    double theta = acos(nhat_z);
    double phi = atan2(nhat_y, nhat_x);

    // 修正：使用与compute_single_transformation相同的boost角计算
    double numerator = sin_angle * sqrt((gamma1 - 1.0) * (gamma2 - 1.0));
    double denominator = sqrt((gamma1 + 1.0) * (gamma2 + 1.0)) +
                         cos_angle * sqrt((gamma1 - 1.0) * (gamma2 - 1.0));
    double psi = 2.0 * atan2(numerator, denominator);

    // 使用共享内存存储中间矩阵
    thrust::complex<double> *wd1 = shared_buf;
    thrust::complex<double> *wd2 = wd1 + matrix_size;

    // 计算Wigner D矩阵元素（与compute_single_transformation保持一致）
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            int m1_val = i - dj;
            int m2_val = j - dj;
            wd1[i * dim + j] = wignerD_element_device(dj, m1_val, m2_val, -phi, -theta, -psi);
            wd2[i * dim + j] = wignerD_element_device(dj, m1_val, m2_val, 0.0, theta, phi);
        }
    }

    // 矩阵乘法：trans = wd1 * wd2
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            thrust::complex<double> sum(0.0, 0.0);
            for (int k = 0; k < dim; k++)
            {
                sum += wd1[i * dim + k] * wd2[k * dim + j];
            }
            // 修正：存储为 T[旧,新] 格式
            trans[i * dim + j] = sum;
        }
    }
}

__device__ void pwahelicity_shared(
    thrust::complex<double> *amp,
    LorentzVector p1, int dj1,
    LorentzVector p2, int dj2,
    int dj, int dS, int dL,
    thrust::complex<double> *shared_buf)
{
    int dim_j = 2 * dj + 1;
    int dim_j1 = 2 * dj1 + 1;
    int dim_j2 = 2 * dj2 + 1;

    // 重新计算维度
    int total_amp_size = dim_j * dim_j1 * dim_j2;
    int trans1_size = dim_j1 * dim_j1;
    int trans2_size = dim_j2 * dim_j2;
    int kron_dim = dim_j1 * dim_j2;

    // 内存布局修正
    thrust::complex<double> *shared_amp = shared_buf;
    thrust::complex<double> *shared_trans1 = shared_amp + total_amp_size;
    thrust::complex<double> *shared_trans2 = shared_trans1 + trans1_size;
    thrust::complex<double> *shared_temp = shared_trans2 + trans2_size;

    // MassiveTrans需要的共享内存 (每个需要2倍矩阵大小)
    int max_dim = max(dim_j1, dim_j2);
    int massive_shared_size = 2 * max_dim * max_dim;
    thrust::complex<double> *massive_shared = shared_temp + total_amp_size;

    // 1. 初始化振幅到共享内存
    for (int i = 0; i < total_amp_size; i++)
    {
        shared_amp[i] = thrust::complex<double>(0.0, 0.0);
    }

    // 2. 计算原始振幅（与pwahelicity_device保持一致）
    LorentzVector P_total = p1 + p2;

    for (int sgm_idx = 0; sgm_idx < dim_j; sgm_idx++)
    {
        int sgm = dj - sgm_idx;

        for (int sgm1_idx = 0; sgm1_idx < dim_j1; sgm1_idx++)
        {
            int sgm1 = dj1 - sgm1_idx;

            for (int sgm2_idx = 0; sgm2_idx < dim_j2; sgm2_idx++)
            {
                int sgm2 = dj2 - sgm2_idx;

                // 计算未变换的振幅元素（与compute_untransformed_element保持一致）
                thrust::complex<double> untransformed_amp =
                    compute_untransformed_element(sgm, sgm1, sgm2, dj, dj1, dj2, dS, dL, p1, p2);

                int idx = sgm_idx * kron_dim + sgm1_idx * dim_j2 + sgm2_idx;
                shared_amp[idx] = untransformed_amp;
            }
        }
    }

    // 3. 计算变换矩阵（修正boost方向）
    LorentzVector zero_vec;
    zero_vec.Px = 0;
    zero_vec.Py = 0;
    zero_vec.Pz = 0;
    zero_vec.E = 0;

    // 修正：使用正确的boost方向
    MassiveTrans_shared(shared_trans1, P_total, p1, dj1, massive_shared);
    MassiveTrans_shared(shared_trans2, P_total, p2, dj2, massive_shared);

    // 4. 将振幅复制到临时存储
    for (int i = 0; i < total_amp_size; i++)
    {
        shared_temp[i] = shared_amp[i];
    }

    // 5. 正确的矩阵乘法：A'[λ,λ1',λ2'] = Σ_{λ1,λ2} A[λ,λ1,λ2] * T1[λ1',λ1] * T2[λ2',λ2]
    // 注意：变换矩阵应该是 T[新,旧]，但MassiveTrans_shared返回的是 T[旧,新]
    for (int lambda = 0; lambda < dim_j; lambda++)
    {
        for (int lambda1_prime = 0; lambda1_prime < dim_j1; lambda1_prime++)
        {
            for (int lambda2_prime = 0; lambda2_prime < dim_j2; lambda2_prime++)
            {
                thrust::complex<double> sum(0.0, 0.0);

                for (int lambda1 = 0; lambda1 < dim_j1; lambda1++)
                {
                    for (int lambda2 = 0; lambda2 < dim_j2; lambda2++)
                    {
                        int old_idx = lambda * kron_dim + lambda1 * dim_j2 + lambda2;
                        thrust::complex<double> orig_amp = shared_temp[old_idx];

                        // 修正索引：变换矩阵应该是 T[新,旧] 但存储为 T[行,列] = T[旧,新]
                        // 所以我们需要 T1[lambda1_prime, lambda1] 对应 shared_trans1[lambda1 * dim_j1 + lambda1_prime]
                        thrust::complex<double> t1_elem = shared_trans1[lambda1 * dim_j1 + lambda1_prime];
                        thrust::complex<double> t2_elem = shared_trans2[lambda2 * dim_j2 + lambda2_prime];

                        sum += orig_amp * t1_elem * t2_elem;
                    }
                }

                int new_idx = lambda * kron_dim + lambda1_prime * dim_j2 + lambda2_prime;
                shared_amp[new_idx] = sum;
            }
        }
    }

    // 6. 复制回全局内存
    for (int i = 0; i < total_amp_size; i++)
    {
        amp[i] = shared_amp[i];
    }
}

// __device__ void pwahelicity_device(thrust::complex<double> *amp, LorentzVector p1, int dj1, LorentzVector p2, int dj2, int dj, int dS, int dL)
// {
//     // pwahelicity_device实现
//     int dim_j = 2 * dj + 1;
//     int dim_j1 = 2 * dj1 + 1;
//     int dim_j2 = 2 * dj2 + 1;

//     for (int i = 0; i < dim_j * dim_j1 * dim_j2; i++)
//     {
//         amp[i] = thrust::complex<double>(0.0, 0.0);
//     }

//     double m1 = p1.M();
//     double m2 = p2.M();
//     double p1p2 = p1.Dot(p2);
//     double m = sqrt(m1 * m1 + m2 * m2 + 2 * p1p2);
//     double lamb = (2 * m * p1.E + m * m + m1 * m1 - m2 * m2) / (2 * m * (m + p1.E + p2.E));
//     double qs = sqrt(pow(m, 4) + pow(m1 * m1 - m2 * m2, 2) - 2 * m * m * (m1 * m1 + m2 * m2)) / (2 * m);
//     double nsx = (p1.Px - lamb * (p1.Px + p2.Px)) / qs;
//     double nsy = (p1.Py - lamb * (p1.Py + p2.Py)) / qs;
//     double nsz = (p1.Pz - lamb * (p1.Pz + p2.Pz)) / qs;
//     double theta = acos(nsz);
//     double phi = atan2(nsy, nsx);

//     for (int i = 0; i < dim_j1; i++)
//     {
//         int sgm1 = dj1 - i;
//         for (int j = 0; j < dim_j2; j++)
//         {
//             int sgm2 = dj2 - j;
//             for (int k = 0; k < dim_j; k++)
//             {
//                 int sgm = dj - k;

//                 thrust::complex<double> spherical(0.0, 0.0);
//                 double real_part, imag_part;
//                 spherical_harmonic_complex(dL, sgm - sgm1 - sgm2, theta, phi, &real_part, &imag_part);
//                 spherical = thrust::complex<double>(real_part, imag_part);

//                 double cg1 = ClebschGordan_corrected(dS, sgm1 + sgm2, dL, sgm - sgm1 - sgm2, dj, sgm);
//                 double cg2 = ClebschGordan_corrected(dj1, sgm1, dj2, sgm2, dS, sgm1 + sgm2);

//                 thrust::complex<double> tmp = (pow(qs, dL) / sqrt(2 * dj + 1)) * cg1 * cg2 * spherical;

//                 amp[k * (dim_j1 * dim_j2) + i * dim_j2 + j] = tmp;
//             }
//         }
//     }

//     int trans1_dim = 2 * dj1 + 1;
//     int trans2_dim = 2 * dj2 + 1;
//     thrust::complex<double> *trans1 = new thrust::complex<double>[trans1_dim * trans1_dim];
//     thrust::complex<double> *trans2 = new thrust::complex<double>[trans2_dim * trans2_dim];

//     MassiveTrans_device(trans1, p1 + p2, p1, dj1);
//     MassiveTrans_device(trans2, p1 + p2, p2, dj2);

//     int kron_dim = trans1_dim * trans2_dim;
//     thrust::complex<double> *kron = new thrust::complex<double>[kron_dim * trans1_dim * trans2_dim];

//     for (int i = 0; i < trans1_dim; i++)
//     {
//         for (int j = 0; j < trans1_dim; j++)
//         {
//             for (int k = 0; k < trans2_dim; k++)
//             {
//                 for (int l = 0; l < trans2_dim; l++)
//                 {
//                     int row = i * trans2_dim + k;
//                     int col = j * trans2_dim + l;
//                     kron[row * (trans1_dim * trans2_dim) + col] = trans1[i * trans1_dim + j] * trans2[k * trans2_dim + l];
//                 }
//             }
//         }
//     }

//     thrust::complex<double> *temp_amp = new thrust::complex<double>[dim_j * kron_dim];

//     for (int i = 0; i < dim_j; i++)
//     {
//         for (int m = 0; m < kron_dim; m++)
//         {
//             temp_amp[i * kron_dim + m] = amp[i * kron_dim + m];
//         }
//     }

//     for (int i = 0; i < dim_j; i++)
//     {
//         for (int j = 0; j < trans1_dim; j++)
//         {
//             for (int k = 0; k < trans2_dim; k++)
//             {
//                 thrust::complex<double> sum(0.0, 0.0);
//                 for (int m = 0; m < kron_dim; m++)
//                 {
//                     int col_idx = j * trans2_dim + k;
//                     sum += temp_amp[i * kron_dim + m] * kron[m * (trans1_dim * trans2_dim) + col_idx];
//                 }
//                 amp[i * (dim_j1 * dim_j2) + j * dim_j2 + k] = sum;
//             }
//         }
//     }

//     delete[] trans1;
//     delete[] trans2;
//     delete[] kron;
//     delete[] temp_amp;
// }

// // CUDA内核函数
// __global__ void compute_helicity_amplitude_kernel(thrust::complex<double> *d_amp,
//                                                   LorentzVector *d_p1, int dj1,
//                                                   LorentzVector *d_p2, int dj2,
//                                                   int dj, int dS, int dL,
//                                                   int num_events)
// {
//     int event_idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (event_idx < num_events)
//     {
//         int dim_j = 2 * dj + 1;
//         int dim_j1 = 2 * dj1 + 1;
//         int dim_j2 = 2 * dj2 + 1;
//         int amp_size = dim_j * dim_j1 * dim_j2;

//         thrust::complex<double> *event_amp = &d_amp[event_idx * amp_size];
//         LorentzVector p1 = d_p1[event_idx];
//         LorentzVector p2 = d_p2[event_idx];

//         pwahelicity_device(event_amp, p1, dj1, p2, dj2, dj, dS, dL);
//     }
// }

// // 主机端包装函数
// void compute_helicity_amplitude(thrust::complex<double> *h_amp,
//                                 LorentzVector *h_p1, int dj1,
//                                 LorentzVector *h_p2, int dj2,
//                                 int dj, int dS, int dL,
//                                 int num_events)
// {
//     // 计算振幅张量的大小
//     int dim_j = 2 * dj + 1;
//     int dim_j1 = 2 * dj1 + 1;
//     int dim_j2 = 2 * dj2 + 1;
//     int amp_size = dim_j * dim_j1 * dim_j2;
//     int total_amp_size = num_events * amp_size;

//     // 设备内存分配
//     thrust::complex<double> *d_amp;
//     LorentzVector *d_p1, *d_p2;

//     cudaMalloc(&d_amp, total_amp_size * sizeof(thrust::complex<double>));
//     cudaMalloc(&d_p1, num_events * sizeof(LorentzVector));
//     cudaMalloc(&d_p2, num_events * sizeof(LorentzVector));

//     // 拷贝数据到设备
//     cudaMemcpy(d_p1, h_p1, num_events * sizeof(LorentzVector), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_p2, h_p2, num_events * sizeof(LorentzVector), cudaMemcpyHostToDevice);

//     // 配置内核参数
//     int block_size = 256;
//     int grid_size = (num_events + block_size - 1) / block_size;

//     // 启动内核
//     compute_helicity_amplitude_kernel<<<grid_size, block_size>>>(
//         d_amp, d_p1, dj1, d_p2, dj2, dj, dS, dL, num_events);

//     // 检查内核执行错误
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
//     }

//     // 拷贝结果回主机
//     cudaMemcpy(h_amp, d_amp, total_amp_size * sizeof(thrust::complex<double>), cudaMemcpyDeviceToHost);

//     // 清理设备内存
//     cudaFree(d_amp);
//     cudaFree(d_p1);
//     cudaFree(d_p2);
// }
