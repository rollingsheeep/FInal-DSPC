#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "image_utils.hpp"
#include <cuda_runtime.h>

// CUDA kernel declarations
__global__ void gaussianBlurKernel(const Pixel* input, Pixel* output, float* kernel,
                                 int width, int height, int kernelSize);

__global__ void unsharpMaskKernel(const Pixel* input, const Pixel* blurred,
                                Pixel* output, float amount, int width, int height);

__global__ void histogramKernel(const Pixel* input, int* histograms,
                              int width, int height, int tileWidth, int tileHeight,
                              int tiles);

__global__ void claheKernel(const Pixel* input, Pixel* output,
                          const int* cdfs, float clipLimit,
                          int width, int height, int tileWidth, int tileHeight,
                          int tiles);

__global__ void histogramKernelLAB(const PixelLAB* input, int* histograms,
                                 int width, int height, int tileWidth, int tileHeight,
                                 int tiles);

__global__ void claheKernelLAB(const PixelLAB* input, PixelLAB* output,
                             const int* cdfs, float clipLimit,
                             int width, int height, int tileWidth, int tileHeight,
                             int tiles);

#endif // CUDA_KERNELS_CUH 