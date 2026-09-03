// conv_tile.cpp  STAGE 3: CACHE TILING
#include "convolution.h"

#include <cstring>

void conv_tile(const float *in, float *out, const float *ker, int H, int W,
               int K) {
  const int p = K / 2;
  const int in_stride = W + 2 * p;

  const int TILE_H = 4;
  const int TILE_W = 1024;

  std::memset(out, 0, H * W * sizeof(float));

  for (int ty = 0; ty < H; ty += TILE_H) {
    const int y_end = (ty + TILE_H < H) ? (ty + TILE_H) : H;
    for (int tx = 0; tx < W; tx += TILE_W) {
      const int x_end = (tx + TILE_W < W) ? (tx + TILE_W) : W;

      for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
          const float w = ker[ky * K + kx];
          for (int oy = ty; oy < y_end; ++oy) {
            float *orow = out + (long)oy * W;
            const float *irow = in + (long)(oy + ky) * in_stride + kx;
            for (int ox = tx; ox < x_end; ++ox)
              orow[ox] += w * irow[ox];
          }
        }
      }
    }
  }
}
