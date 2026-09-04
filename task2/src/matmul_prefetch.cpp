#include <immintrin.h>

#include "matmul.h"

#ifndef PF_DIST
#define PF_DIST 128
#endif
#ifndef PF_HINT
#define PF_HINT _MM_HINT_T1
#endif

void matmul_prefetch(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + i * lda;
            const float* b = B + j * ldb;
            int p = 0;
            for (; p + 16 <= K; p += 16) {
#if PF_DIST > 0
                _mm_prefetch((const char*)(a + p + PF_DIST), PF_HINT);
                _mm_prefetch((const char*)(b + p + PF_DIST), PF_HINT);
#endif
                for (int q = p; q < p + 16; ++q) acc += a[q] * b[q];
            }
            for (; p < K; ++p) acc += a[p] * b[p];
            C[i * ldc + j] = acc;
        }
    }
}
