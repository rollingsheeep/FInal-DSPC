#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>

#define BLOCK_SIZE 16
#define MAX_THREADS 1024

__device__ unsigned char clampCUDA(float value) {
    if (value < 0.0f) return 0;
    if (value > 255.0f) return 255;
    return static_cast<unsigned char>(value);
}

// CUDA kernel for Gaussian blur
__global__ void gaussianBlurKernel(const Pixel* input, Pixel* output, float* kernel,
                                  int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sumR = 0, sumG = 0, sumB = 0;
        int center = kernelSize / 2;
        
        for (int k = -center; k <= center; k++) {
            int px = min(max(x + k, 0), width - 1);
            sumR += input[y * width + px].r * kernel[k + center];
            sumG += input[y * width + px].g * kernel[k + center];
            sumB += input[y * width + px].b * kernel[k + center];
        }
        
        output[y * width + x].r = clampCUDA(sumR);
        output[y * width + x].g = clampCUDA(sumG);
        output[y * width + x].b = clampCUDA(sumB);
    }
}

// CUDA kernel for unsharp mask
__global__ void unsharpMaskKernel(const Pixel* input, const Pixel* blurred,
                                 Pixel* output, float amount, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float r = input[idx].r + amount * (input[idx].r - blurred[idx].r);
        float g = input[idx].g + amount * (input[idx].g - blurred[idx].g);
        float b = input[idx].b + amount * (input[idx].b - blurred[idx].b);
        
        output[idx].r = clampCUDA(r);
        output[idx].g = clampCUDA(g);
        output[idx].b = clampCUDA(b);
    }
}

// CUDA kernel for histogram calculation
__global__ void histogramKernel(const Pixel* input, int* histograms,
                               int width, int height, int tileWidth, int tileHeight,
                               int tiles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int tileX = x / tileWidth;
        int tileY = y / tileHeight;
        int tileIdx = tileY * tiles + tileX;
        
        // Update histograms for each channel
        atomicAdd(&histograms[tileIdx * 256 + input[y * width + x].r], 1);
        atomicAdd(&histograms[(tiles * tiles + tileIdx) * 256 + input[y * width + x].g], 1);
        atomicAdd(&histograms[(2 * tiles * tiles + tileIdx) * 256 + input[y * width + x].b], 1);
    }
}

// CUDA kernel for CLAHE
__global__ void claheKernel(const Pixel* input, Pixel* output,
                           const int* cdfs, float clipLimit,
                           int width, int height, int tileWidth, int tileHeight,
                           int tiles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int tileX = x / tileWidth;
        int tileY = y / tileHeight;
        int tileIdx = tileY * tiles + tileX;
        
        // Process each channel
        for (int channel = 0; channel < 3; channel++) {
            int histogramOffset = channel * tiles * tiles * 256;
            uint8_t value;
            
            switch(channel) {
                case 0: value = input[y * width + x].r; break;
                case 1: value = input[y * width + x].g; break;
                case 2: value = input[y * width + x].b; break;
            }
            
            float scale = 255.0f / cdfs[histogramOffset + tileIdx * 256 + 255];
            float newValue = cdfs[histogramOffset + tileIdx * 256 + value] * scale;
            
            switch(channel) {
                case 0: output[y * width + x].r = clampCUDA(newValue); break;
                case 1: output[y * width + x].g = clampCUDA(newValue); break;
                case 2: output[y * width + x].b = clampCUDA(newValue); break;
            }
        }
    }
}

// CUDA kernel for LAB histogram calculation
__global__ void histogramKernelLAB(const PixelLAB* input, int* histograms,
                                  int width, int height, int tileWidth, int tileHeight,
                                  int tiles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int tileX = x / tileWidth;
        int tileY = y / tileHeight;
        int tileIdx = tileY * tiles + tileX;
        
        // Update histogram for L channel
        int l_value = static_cast<int>(input[y * width + x].l);
        atomicAdd(&histograms[tileIdx * 256 + l_value], 1);
    }
}

// CUDA kernel for LAB CLAHE
__global__ void claheKernelLAB(const PixelLAB* input, PixelLAB* output,
                              const int* cdfs, float clipLimit,
                              int width, int height, int tileWidth, int tileHeight,
                              int tiles) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int tileX = x / tileWidth;
        int tileY = y / tileHeight;
        int tileIdx = tileY * tiles + tileX;
        
        // Process L channel
        uint8_t l = input[y * width + x].l;
        float scale = 255.0f / cdfs[tileIdx * 256 + 255];
        float newL = cdfs[tileIdx * 256 + l] * scale;
        
        output[y * width + x].l = clampCUDA(newL);
        output[y * width + x].a = input[y * width + x].a;
        output[y * width + x].b = input[y * width + x].b;
    }
} 