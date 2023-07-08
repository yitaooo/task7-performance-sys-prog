#include "matrix.h"

#include <immintrin.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

namespace chrono = std::chrono;

void transpose(float *dst, float *src, int h, int w, int stride) {
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            dst[j * h + i] = src[i * stride + j];
        }
    }
}

float vector_sum_sse(float *a, int n) {
    float sum = 0;
    for (int i = 0; i < n; i += 4) {
        sum += a[i] + a[i + 1] + a[i + 2] + a[i + 3];
    }
    return sum;
}

float vector_mul_sum(float *src1, float *src2, int n) {
    // use SSE2
    __m128 sum128 = _mm_setzero_ps();
    for (int i = 0; i < n / 4 * 4; i += 4) {
        __m128 v1 = _mm_loadu_ps(&src1[i]);
        __m128 v2 = _mm_loadu_ps(&src2[i]);
        __m128 v3 = _mm_mul_ps(v1, v2);
        sum128 = _mm_add_ps(sum128, v3);
    }

    // sum up the 4 floats in sum128
    // this can be optimized to use SSE3
    float tmp[4];
    _mm_storeu_ps(tmp, sum128);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // the rest
    for (int i = n / 4 * 4; i < n; i++) {
        sum += src1[i] * src2[i];
    }
    return sum;
}

// transpose right matrix for better cache performance
void matmul_transb(float *dst, float *left, float *right, int n, int ci, int co,
                   int left_stride, int right_stride) {
    // left: n * ci, right: co * ci
    float *left_tmp = left;
    for (int i = 0; i < n; i++) {
        float *right_tmp = right;
        for (int j = 0; j < co; j++) {
            dst[i * co + j] = vector_mul_sum(left_tmp, right_tmp, ci);
            right_tmp += right_stride;
        }
        left_tmp += left_stride;
    }
}

void matmul(float *a, float *b, float *c, int n, int ci, int co) {
    float *b_trans = new float[co * ci];
    transpose(b_trans, b, ci, co, co);
    // matmul_transb(c, a, b_trans, n, ci, co, ci, co);
    std::vector<std::thread> threads;
    // use 4 threads
    int n_per_thread = n / 4;
    for (int i = 0; i < 4; i++) {
        threads.emplace_back(matmul_transb, c + i * n_per_thread * co,
                             a + i * n_per_thread * ci, b_trans, n_per_thread,
                             ci, co, ci, co);
    }
    for (int i = 0; i < 4; i++) {
        threads[i].join();
    }
    delete[] b_trans;
}

#ifdef __cplusplus
extern "C" {
#endif

void matrix_multiply(float *a, float *b, float *c, int n) {
    matmul(a, b, c, n, n, n);
}

#ifdef __cplusplus
}
#endif