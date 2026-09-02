#include <immintrin.h>

#include "matmul.h"

float sum(__m256 x) {
  __m128 hi = _mm256_extractf128_ps(x, 1);
  __m128 lo = _mm256_extractf128_ps(x, 0);
  lo = _mm_add_ps(hi, lo);
  hi = _mm_movehl_ps(hi, lo);
  lo = _mm_add_ps(hi, lo);
  hi = _mm_shuffle_ps(lo, lo, 1);
  lo = _mm_add_ss(hi, lo);
  return _mm_cvtss_f32(lo);
}

void matmul_simd(const float *A, const float *B, float *C, int M, int N, int K,
                 int lda, int ldb, int ldc) {
  for (int i = 0; i < M; ++i) {
    const float *ai = A + i * lda;
    for (int j = 0; j < N; ++j) {
      const float *bj = B + j * ldb;

      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();

      int p = 0;
      for (; p + 32 <= K; p += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ai + p), _mm256_loadu_ps(bj + p),
                               acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(ai + p + 8),
                               _mm256_loadu_ps(bj + p + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(ai + p + 16),
                               _mm256_loadu_ps(bj + p + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(ai + p + 24),
                               _mm256_loadu_ps(bj + p + 24), acc3);
      }
      acc0 =
          _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

      for (; p + 8 <= K; p += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(ai + p), _mm256_loadu_ps(bj + p),
                               acc0);

      float s = sum(acc0);

      for (; p < K; ++p)
        s += ai[p] * bj[p];

      C[i * ldc + j] = s;
    }
  }
}
