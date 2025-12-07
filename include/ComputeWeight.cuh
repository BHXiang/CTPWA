#ifndef COMPUTEWEIGHT_CUH
#define COMPUTEWEIGHT_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

// void computeWeightResult(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, double *d_total_result, double *d_total_integral, double *d_partial_result, int *d_nSLvectors, int npartials, int nEvents, int ngls, int npolar);
void computeWeightResult(const cuDoubleComplex *d_matrix, const cuDoubleComplex *d_vector, double *d_total_result, double *d_total_integral, double *d_partial_result, double *d_partial_sums, int *d_nSLvectors, int npartials, int nEvents, int ngls, int npolar);

#endif
