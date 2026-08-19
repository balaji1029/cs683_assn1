// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
//
// Combines the earlier stages, then keeps only what actually measured faster.
//
// WHAT IS IN, AND WHY
//
// 1. Reorder (Stage 1) -- but NOT its memory-resident accumulator. The whole
//    reason reorder lost to naive is that partial sums lived in `out`, forcing
//    K*K DRAM passes. Here the accumulator is a REGISTER (__m256) held across the
//    entire K*K kernel, so `out` is written exactly once and never re-read. That
//    also means no pre-zeroing pass.
//
// 2. SIMD (Stage 4) -- one _mm256_fmadd_ps per (ky,kx) with the weight broadcast
//    by _mm256_set1_ps, 8 output columns per instruction.
//
// 3. Unroll (Stage 2) -- the piece Stage 4 was missing. conv_simd keeps ONE
//    accumulator, so its K*K FMAs form a single dependency chain: at K=3 that is
//    9 links x ~4 cycle FMA latency = ~36 cycles per 8 outputs, while the two FMA
//    ports could have retired those 9 FMAs in ~4.5. Carrying NV independent
//    accumulators overlaps NV such chains and fills the latency shadow.
//
// 4. Streaming stores -- a normal store to `out` first READS the line it is about
//    to overwrite (read-for-ownership), so writing the 16 MB output costs 16 MB of
//    pointless reads. _mm256_stream_ps bypasses that. Legal here because the
//    buffers are _mm_malloc'd 64-byte aligned and W is a multiple of 8, so every
//    (orow + ox) is 32-byte aligned. Needs an _mm_sfence() before returning since
//    streaming stores are weakly ordered.
//
// WHAT IS OUT
//
//    Explicit cache tiling (Stage 3). It earns its 1.35x in conv_tile only because
//    that loop re-traverses `out` K*K times. Here the accumulator never leaves a
//    register, so there is no `out` reuse left to capture, and the input rows a
//    single output row needs (K x 2048 floats = 24 KB at K=3) already sit in L1
//    without any blocking. Measured: adding tiles changed nothing. Stage 3's
//    lesson is that tiling fixes a traffic problem -- if you remove the traffic
//    problem a different way, tiling has nothing left to fix.
//
// TUNING (2048x2048, best-of-3 ms, naive ~27 ms)
//
//    NV        1      2      3      4      5      6      8     12
//    K=3    3.93   2.92   2.68   2.76   2.54   2.59   2.50   2.55
//    K=5   11.41   6.30   5.33   4.98   4.73   4.85   4.69   4.63
//
//    Steep to NV=3 (dependency chains being filled), then flat -- past ~5 chains
//    the FMA ports are no longer the limit and memory is. NV=8 sits on the plateau.
//    Streaming stores then buy a further ~13% (2.50 -> 2.19 ms at K=3).
//
//    NOTE: an earlier version of this sweep was non-monotonic (NV=4 SLOWER than
//    NV=2) because GCC would not unroll `for (i < NV)`, so a[] was indexed
//    dynamically, spilled to the stack, and defeated the entire optimization.
//    The `#pragma GCC unroll` directives below are load-bearing, not decoration.
#include <immintrin.h>

#include "convolution.h"

#ifndef NV
#define NV 8      // __m256 accumulators kept in flight (NV*8 output columns per block)
#endif

#ifndef NT_STORE
#define NT_STORE 1
#endif
#if NT_STORE
#define STORE(p_, v_) _mm256_stream_ps((p_), (v_))
#else
#define STORE(p_, v_) _mm256_storeu_ps((p_), (v_))
#endif

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int BLK = NV * 8;

    for (int oy = 0; oy < H; ++oy) {
        float*       orow  = out + (long)oy * W;
        const float* ibase = in + (long)oy * in_stride;

        int ox = 0;
        for (; ox + BLK <= W; ox += BLK) {
            __m256 a[NV];
#pragma GCC unroll 16
            for (int i = 0; i < NV; ++i) a[i] = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* irow = ibase + (long)ky * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    // MUST be fully unrolled: otherwise a[] is indexed dynamically,
                    // lands on the stack, and the whole point (registers) is lost.
#pragma GCC unroll 16
                    for (int i = 0; i < NV; ++i)
                        a[i] = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx + i * 8), w, a[i]);
                }
            }
#pragma GCC unroll 16
            for (int i = 0; i < NV; ++i) STORE(orow + ox + i * 8, a[i]);
        }

        // W is a multiple of 8, so the tail is whole vectors, never scalars.
        for (; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* irow = ibase + (long)ky * in_stride + ox;
                for (int kx = 0; kx < K; ++kx)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(irow + kx),
                                          _mm256_set1_ps(ker[ky * K + kx]), acc);
            }
            _mm256_storeu_ps(orow + ox, acc);
        }
    }
#if NT_STORE
    _mm_sfence();  // streaming stores are weakly ordered; publish them before return
#endif
}
