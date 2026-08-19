// conv_tile.cpp  STAGE 3: CACHE TILING
//
// Stage 1 (reorder) gave the inner loop a perfect unit-stride AXPY shape, but it
// paid for that by holding partial sums in `out` instead of a register: it makes
// K*K read-modify-write passes over the ENTIRE output image. At 2048x2048 the
// output is 16 MB and the padded input another ~16 MB -- far past this machine's
// 12 MB L3 -- so every one of those K*K passes re-streams from DRAM and the stage
// lands at or below naive despite the friendlier loop.
//
// Tiling fixes the reuse pattern without changing the arithmetic. Instead of
// sweeping the whole image K*K times, walk the output in TILE_H x TILE_W blocks
// and do all K*K passes on one block before moving on. The block's output
// (TILE_H*TILE_W floats) plus its input halo ((TILE_H+K-1)*(TILE_W+K-1) floats)
// stay resident across the K*K passes, so a value is fetched from DRAM once and
// then hit in cache K*K times. The traffic drops by roughly a factor of K*K
// while the FLOP count is unchanged.
//
// Tile size is the whole game, and it was chosen by sweeping, not by intuition.
// Measured on this machine (i7-1255U, 32 KB L1d / ~2 MB L2 / 12 MB L3) at
// 2048x2048 K=3, best-of-N ms, naive ~27 ms:
//
//     TH\TW      128      256      512     1024     2048
//        4         -     20.8     19.9     19.8     20.4
//        8      23.3     21.2     20.2        -     21.3
//       16      23.7     21.3     20.4        -     21.2
//       64      24.5     21.9     21.1        -     21.9
//     2048      47.4     44.2     38.7        -     27.3   <- 2048x2048 = untiled
//
// Two things to read off that table. The bottom row is the untiled reorder loop
// (27.3 ms), so tiling is worth ~1.35x on top of it. And tall-narrow tiles are a
// trap: TH=2048/TW=512 is 38.7 ms, considerably WORSE than not tiling, because a
// tall strip's working set (4 MB out + 4 MB halo) still misses L2 while the short
// 512-float rows also break the prefetcher's long sequential runs.
//
// The winner is a short, wide strip: TH=4 x TW=1024 keeps the output tile (16 KB)
// plus its K+3 input halo rows (~24 KB) at ~40 KB -- just past L1d, comfortably
// inside L2 -- while the 1024-float rows stay long enough to prefetch well. The
// top three configs are within ~1% of each other, i.e. at the noise floor.

#include "convolution.h"

// Overridable from the compiler command line for tuning sweeps.
#ifndef TILE_H
#define TILE_H 4
#endif
#ifndef TILE_W
#define TILE_W 1024
#endif

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int ty = 0; ty < H; ty += TILE_H) {
        const int y_end = (ty + TILE_H < H) ? (ty + TILE_H) : H;

        for (int tx = 0; tx < W; tx += TILE_W) {
            const int x_end = (tx + TILE_W < W) ? (tx + TILE_W) : W;

            // Partial sums live in `out`, so clear this tile before accumulating.
            // Only the tile is touched, so it is warm in cache for the passes below.
            for (int oy = ty; oy < y_end; ++oy) {
                float* orow = out + (long)oy * W;
                for (int ox = tx; ox < x_end; ++ox) {
                    orow[ox] = 0.0f;
                }
            }

            // All K*K passes now sweep only this cache-resident tile.
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float w = ker[ky * K + kx];  // invariant: held in a register

                    for (int oy = ty; oy < y_end; ++oy) {
                        float*       orow = out + (long)oy * W;
                        const float* irow = in + (long)(oy + ky) * in_stride + kx;

                        for (int ox = tx; ox < x_end; ++ox) {
                            orow[ox] += w * irow[ox];
                        }
                    }
                }
            }
        }
    }
}
