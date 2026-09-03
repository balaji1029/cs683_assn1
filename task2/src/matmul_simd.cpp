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
  constexpr int kAsz = 4;
  constexpr int kBsz = 3;

  int i = 0;
  for (; i + kAsz <= M; i += kAsz) {
    // we take 4 row elements of A at a time to reuse
    const float *a0 = A + i * lda;
    const float *a1 = A + (i + 1) * lda;
    const float *a2 = A + (i + 2) * lda;
    const float *a3 = A + (i + 3) * lda;

    int j = 0;
    for (; j + kBsz <= N; j += kBsz) {

      // and take 3 row elements of B at a time
      const float *b0 = B + j * ldb;
      const float *b1 = B + (j + 1) * ldb;
      const float *b2 = B + (j + 2) * ldb;

      // registers that will carry the values of C; kAsz x kBsz
      __m256 c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;

      // initializing all of them to 0
      c00 = c01 = c02 = c10 = c11 = c12 = c20 = c21 = c22 = c30 = c31 = c32 =
          _mm256_setzero_ps();

      int p = 0;

      // kAsz + kBsz loads but kAz * kBsz values are computed
      for (; p + 8 <= K; p += 8) {
        __m256 bv0 = _mm256_loadu_ps(b0 + p);
        __m256 bv1 = _mm256_loadu_ps(b1 + p);
        __m256 bv2 = _mm256_loadu_ps(b2 + p);

        __m256 av;

        av = _mm256_loadu_ps(a0 + p);
        c00 = _mm256_fmadd_ps(av, bv0, c00);
        c01 = _mm256_fmadd_ps(av, bv1, c01);
        c02 = _mm256_fmadd_ps(av, bv2, c02);

        av = _mm256_loadu_ps(a1 + p);
        c10 = _mm256_fmadd_ps(av, bv0, c10);
        c11 = _mm256_fmadd_ps(av, bv1, c11);
        c12 = _mm256_fmadd_ps(av, bv2, c12);

        av = _mm256_loadu_ps(a2 + p);
        c20 = _mm256_fmadd_ps(av, bv0, c20);
        c21 = _mm256_fmadd_ps(av, bv1, c21);
        c22 = _mm256_fmadd_ps(av, bv2, c22);

        av = _mm256_loadu_ps(a3 + p);
        c30 = _mm256_fmadd_ps(av, bv0, c30);
        c31 = _mm256_fmadd_ps(av, bv1, c31);
        c32 = _mm256_fmadd_ps(av, bv2, c32);
      }

      float s00 = sum(c00), s01 = sum(c01), s02 = sum(c02);
      float s10 = sum(c10), s11 = sum(c11), s12 = sum(c12);
      float s20 = sum(c20), s21 = sum(c21), s22 = sum(c22);
      float s30 = sum(c30), s31 = sum(c31), s32 = sum(c32);

      // the rest of the hidden dimension
      for (; p < K; ++p) {
        float a0v = a0[p], a1v = a1[p], a2v = a2[p], a3v = a3[p];
        float b0v = b0[p], b1v = b1[p], b2v = b2[p];
        s00 += a0v * b0v;
        s01 += a0v * b1v;
        s02 += a0v * b2v;
        s10 += a1v * b0v;
        s11 += a1v * b1v;
        s12 += a1v * b2v;
        s20 += a2v * b0v;
        s21 += a2v * b1v;
        s22 += a2v * b2v;
        s30 += a3v * b0v;
        s31 += a3v * b1v;
        s32 += a3v * b2v;
      }

      int row0 = i * ldc + j;
      int row1 = (i + 1) * ldc + j;
      int row2 = (i + 2) * ldc + j;
      int row3 = (i + 3) * ldc + j;

      C[row0] = s00;
      C[row0 + 1] = s01;
      C[row0 + 2] = s02;
      C[row1] = s10;
      C[row1 + 1] = s11;
      C[row1 + 2] = s12;
      C[row2] = s20;
      C[row2 + 1] = s21;
      C[row2 + 2] = s22;
      C[row3] = s30;
      C[row3 + 1] = s31;
      C[row3 + 2] = s32;
    }

    // rest of the b rows
    for (; j < N; ++j) {
      const float *bj = B + j * ldb;

      // we only need 4 accumulators because we are dotting 4 rows of A with one
      // row of B
      __m256 ac0, ac1, ac2, ac3;
      ac0 = ac1 = ac2 = ac3 = _mm256_setzero_ps();

      int p = 0;
      for (; p + 8 <= K; p += 8) {

        __m256 bv = _mm256_loadu_ps(bj + p);
        ac0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), bv, ac0);
        ac1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), bv, ac1);
        ac2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), bv, ac2);
        ac3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + p), bv, ac3);
      }

      float s0 = sum(ac0), s1 = sum(ac1);
      float s2 = sum(ac2), s3 = sum(ac3);

      for (; p < K; ++p) {

        float bv = bj[p];
        s0 += a0[p] * bv;
        s1 += a1[p] * bv;
        s2 += a2[p] * bv;
        s3 += a3[p] * bv;
      }

      C[i * ldc + j] = s0;
      C[(i + 1) * ldc + j] = s1;
      C[(i + 2) * ldc + j] = s2;
      C[(i + 3) * ldc + j] = s3;
    }
  }

  // We calculate the rest one C element at a time, no reuse
  for (; i < M; ++i) {
    const float *ai = A + i * lda;

    for (int j = 0; j < N; ++j) {

      const float *bj = B + (j)*ldb;
      __m256 acc = _mm256_setzero_ps();
      int p = 0;

      for (; p + 8 <= K; p += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ai + p), _mm256_loadu_ps(bj + p),
                              acc);
      }
      float s = sum(acc);

      for (; p < K; ++p) {
        s += ai[p] * bj[p];
      }

      C[(i)*ldc + j] = s;
    }
  }
}
