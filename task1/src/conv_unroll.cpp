// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    // conv_naive(in, out, ker, H, W, K);

    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    const int UNROLL_STRIDE = 8;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += UNROLL_STRIDE) {
            float acc[UNROLL_STRIDE] = {0.0f};
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc[0] += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc[1] += in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx];
                    acc[2] += in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx];
                    acc[3] += in[(oy + ky) * in_stride + (ox + kx + 3)] * ker[ky * K + kx];
                    acc[4] += in[(oy + ky) * in_stride + (ox + kx + 4)] * ker[ky * K + kx];
                    acc[5] += in[(oy + ky) * in_stride + (ox + kx + 5)] * ker[ky * K + kx];
                    acc[6] += in[(oy + ky) * in_stride + (ox + kx + 6)] * ker[ky * K + kx];
                    acc[7] += in[(oy + ky) * in_stride + (ox + kx + 7)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc[0];
            out[oy * W + ox + 1] = acc[1];
            out[oy * W + ox + 2] = acc[2];
            out[oy * W + ox + 3] = acc[3];
            out[oy * W + ox + 4] = acc[4];
            out[oy * W + ox + 5] = acc[5];
            out[oy * W + ox + 6] = acc[6];
            out[oy * W + ox + 7] = acc[7];
        }
    }
}
