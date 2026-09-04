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
            for (int p = 0; p < K; ++p) {
                
#if PF_DIST > 0
                if (p + PF_DIST < K) {
                    _mm_prefetch((const char*)(a + p + PF_DIST), PF_HINT);
                    _mm_prefetch((const char*)(b + p + PF_DIST), PF_HINT);
                }
#endif
                acc += a[p] * b[p];
            }
            C[i * ldc + j] = acc;
        }
    }
}
