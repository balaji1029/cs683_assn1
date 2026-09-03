// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        const float* ibase = in + (long)oy * in_stride;
        float*       orow  = out + (long)oy * W;

        int ox = 0;
        for (; ox + 32 <= W; ox += 32) {
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* irow = ibase + (long)ky * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx +  0), w, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx +  8), w, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx + 16), w, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx + 24), w, a3);
                }
            }

            _mm256_storeu_ps(orow + ox +  0, a0);
            _mm256_storeu_ps(orow + ox +  8, a1);
            _mm256_storeu_ps(orow + ox + 16, a2);
            _mm256_storeu_ps(orow + ox + 24, a3);
        }

        for (; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* irow = ibase + (long)ky * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx), w, acc);
                }
            }
            _mm256_storeu_ps(orow + ox, acc);
        }
    }
}
