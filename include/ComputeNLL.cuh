#ifndef COMPUTENLL_CUH
#define COMPUTENLL_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

// void computeWeightResult(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, double *d_final_result, double *d_row_results, int M, int N);
// void computeWeightResult(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, double *d_final_result, double *d_row_results, int nEvents, int ngls, int npolar);
// void computeWeightResult(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, double *d_final_result, int nEvents, int ngls, int npolar);

void computeNll(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, cuDoubleComplex *d_S, double *d_Q, double *d_final_result, int nlength, int ngls, int npolar, double phsp_factor);
void computePHSPfactor(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, cuDoubleComplex *d_B, double *d_final_result, int M, int N);

#endif
