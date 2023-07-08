#include "mandelbrot.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int mandelbrot_calc_base(float x, float y) {
    auto re = x;
    auto im = y;

    for (auto i = 0; i < LOOP; i++) {
        float re2 = re * re;
        float im2 = im * im;

        // verify if f(z) diverges to infinity
        if (re2 + im2 > 4.0f)
            return i;

        im = 2 * re * im + y;
        re = re2 - im2 + x;
    }

    return LOOP;
}

__m128i mandelbrot_sse2(__m128 x, __m128 y) {
    auto re = x;
    auto im = y;
    static const __m128 four = _mm_set1_ps(4.0f);
    static const __m128 two = _mm_set1_ps(2.0f);
    static const __m128i ONE = _mm_set1_epi32(1);

    __m128i ret = _mm_set1_epi32(0);
    for (auto i = 0; i < LOOP; i++) {
        // re2 = re * re
        auto re2 = _mm_mul_ps(re, re);
        // im2 = im * im
        auto im2 = _mm_mul_ps(im, im);

        // re2_plus_im2 = re2 + im2
        auto re2_plus_im2 = _mm_add_ps(re2, im2);

        // ok = (int)(re2_plus_im2 <= 4.0f)
        __m128i ok = _mm_castps_si128(_mm_cmple_ps(re2_plus_im2, four));

        // ret += (ok & 1)
        ret = _mm_add_epi32(ret, _mm_and_si128(ok, ONE));

        // if no one is ok, return
        if (_mm_movemask_ps(_mm_castsi128_ps(ok)) == 0)
            return ret;

        // im = 2 * re * im + y;
        im = _mm_add_ps(_mm_mul_ps(two, _mm_mul_ps(re, im)), y);

        // re = re2 - im2 + x;
        re = _mm_add_ps(_mm_sub_ps(re2, im2), x);
    }
    return ret;
}
void mandelbrot_thread_worker(int height_begin,
                              float dx,
                              float dy,
                              float* xs,
                              int block_height,
                              int width,
                              int* plot) {
    for (int i = height_begin; i < height_begin + block_height; i++) {
        __m128 y = _mm_set1_ps(Y_END - i * dy);
        for (int j = 0; j < width; j += 4) {
            __m128 x = _mm_load_ps(xs + j);
            __m128i result = mandelbrot_sse2(x, y);
            _mm_store_si128((__m128i*)&plot[i * width + j], result);
        }
    }
}

void naive_mandelbrot(int width, int height, int* plot) {
    float dx = (X_END - X_START) / (width - 1);
    float dy = (Y_END - Y_START) / (height - 1);

    // reduce redundant calculations
    float* xs = new float[width];
    for (int j = 0; j < width; j += 1) {
        xs[j] = X_START + j * dx;
    }

    std::vector<std::thread> threads;
#define THREAD_NUM 4
    for (int i = 0; i < THREAD_NUM; i++) {
        int height_begin = i * height / THREAD_NUM;
        int block_height = height / THREAD_NUM;
        threads.emplace_back(mandelbrot_thread_worker, height_begin, dx, dy, xs,
                             block_height, width, plot);
    }
    for (auto& t : threads) {
        t.join();
    }
    delete xs;
}

#ifdef __cplusplus
extern "C" {
#endif

void mandelbrot(int width, int height, int* plot) {
    naive_mandelbrot(width, height, plot);
}

#ifdef __cplusplus
}
#endif
