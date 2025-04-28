#include "cuda_image_utils.hpp"
#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>
#include <sstream>

Image applyUnsharpMaskCUDA(const Image& input, float sigma, float amount) {
    // Get device properties
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties");
    }

    std::cout << "Using CUDA device: " << prop.name << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    Image output = input;
    int width = input.width;
    int height = input.height;
    
    std::cout << "Processing image of size: " << width << "x" << height << std::endl;
    
    // Calculate required memory
    size_t imageSize = width * height * sizeof(Pixel);
    size_t totalRequiredMemory = imageSize * 3; // input, blurred, output
    totalRequiredMemory += static_cast<int>(ceil(sigma * 6)) * sizeof(float); // kernel
    
    std::cout << "Required memory: " << totalRequiredMemory / (1024 * 1024) << " MB" << std::endl;
    
    if (totalRequiredMemory > prop.totalGlobalMem) {
        std::stringstream ss;
        ss << "Not enough GPU memory. Required: " 
           << (totalRequiredMemory / (1024 * 1024)) 
           << " MB, Available: " 
           << (prop.totalGlobalMem / (1024 * 1024)) 
           << " MB";
        throw std::runtime_error(ss.str());
    }
    
    // Allocate device memory
    Pixel *d_input, *d_blurred, *d_output;
    std::cout << "Allocating device memory..." << std::endl;
    
    err = cudaMalloc(&d_input, imageSize);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input");
    }
    std::cout << "Allocated input memory: " << imageSize / (1024 * 1024) << " MB" << std::endl;
    
    err = cudaMalloc(&d_blurred, imageSize);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        throw std::runtime_error("Failed to allocate device memory for blurred");
    }
    std::cout << "Allocated blurred memory: " << imageSize / (1024 * 1024) << " MB" << std::endl;
    
    err = cudaMalloc(&d_output, imageSize);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        throw std::runtime_error("Failed to allocate device memory for output");
    }
    std::cout << "Allocated output memory: " << imageSize / (1024 * 1024) << " MB" << std::endl;
    
    // Copy input to device
    err = cudaMemcpy(d_input, input.pixels.data(), imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        throw std::runtime_error("Failed to copy input to device");
    }
    
    // Create and copy Gaussian kernel
    int kernelSize = static_cast<int>(ceil(sigma * 6));
    if (kernelSize % 2 == 0) kernelSize++;
    std::vector<float> h_kernel(kernelSize);
    float sum = 0.0f;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; i++) {
        float x = i - center;
        h_kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += h_kernel[i];
    }
    
    for (int i = 0; i < kernelSize; i++) {
        h_kernel[i] /= sum;
    }
    
    float *d_kernel;
    err = cudaMalloc(&d_kernel, kernelSize * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        throw std::runtime_error("Failed to allocate device memory for kernel");
    }
    
    err = cudaMemcpy(d_kernel, h_kernel.data(), kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("Failed to copy kernel to device");
    }
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Launch Gaussian blur kernel
    gaussianBlurKernel<<<gridDim, blockDim>>>(d_input, d_blurred, d_kernel, width, height, kernelSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("Failed to launch Gaussian blur kernel");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("Failed to synchronize after Gaussian blur");
    }
    
    // Launch unsharp mask kernel
    unsharpMaskKernel<<<gridDim, blockDim>>>(d_input, d_blurred, d_output, amount, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("Failed to launch unsharp mask kernel");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("Failed to synchronize after unsharp mask");
    }
    
    // Copy result back to host
    err = cudaMemcpy(output.pixels.data(), d_output, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_kernel);
        throw std::runtime_error("Failed to copy result to host");
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_blurred);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    return output;
}

ImageLAB applyCLAHE_LAB_CUDA(const ImageLAB& input, int tiles, float clipLimit) {
    // Validate input parameters
    if (tiles <= 0) {
        throw std::runtime_error("Number of tiles must be positive");
    }
    if (input.width <= 0 || input.height <= 0) {
        throw std::runtime_error("Image dimensions must be positive");
    }
    if (input.width < tiles || input.height < tiles) {
        throw std::runtime_error("Image dimensions must be larger than number of tiles");
    }

    ImageLAB output = input;
    int width = input.width;
    int height = input.height;
    int tileWidth = width / tiles;
    int tileHeight = height / tiles;
    
    // Allocate device memory
    PixelLAB *d_input, *d_output;
    cudaError_t err = cudaMalloc(&d_input, width * height * sizeof(PixelLAB));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for input");
    }
    err = cudaMalloc(&d_output, width * height * sizeof(PixelLAB));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        throw std::runtime_error("Failed to allocate device memory for output");
    }
    
    // Allocate memory for histograms (one per tile)
    int* d_histograms;
    err = cudaMalloc(&d_histograms, tiles * tiles * 256 * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw std::runtime_error("Failed to allocate device memory for histograms");
    }
    cudaMemset(d_histograms, 0, tiles * tiles * 256 * sizeof(int));
    
    // Copy input to device
    err = cudaMemcpy(d_input, input.pixels.data(), width * height * sizeof(PixelLAB),
                    cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        throw std::runtime_error("Failed to copy input to device");
    }
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    // Calculate histograms
    void* args[] = {&d_input, &d_histograms, &width, &height, &tileWidth, &tileHeight, &tiles};
    cudaLaunchKernel((void*)histogramKernelLAB, gridDim, blockDim, args, 0, NULL);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        throw std::runtime_error("Failed to launch histogram kernel");
    }
    
    // Download histograms to host for processing
    std::vector<int> h_histograms(tiles * tiles * 256);
    err = cudaMemcpy(h_histograms.data(), d_histograms,
                    tiles * tiles * 256 * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        throw std::runtime_error("Failed to copy histograms to host");
    }
    
    // Process histograms and create CDFs on host
    std::vector<int> h_cdfs = h_histograms;
    for (int tile = 0; tile < tiles * tiles; tile++) {
        int histogramOffset = tile * 256;
        
        // Apply clip limit
        int clipCount = static_cast<int>(clipLimit * (tileWidth * tileHeight) / 256);
        int redistBatch = 0;
        
        for (int i = 0; i < 256; i++) {
            if (h_cdfs[histogramOffset + i] > clipCount) {
                redistBatch += (h_cdfs[histogramOffset + i] - clipCount);
                h_cdfs[histogramOffset + i] = clipCount;
            }
        }
        
        // Redistribute clipped pixels
        int redistIncrement = redistBatch / 256;
        for (int i = 0; i < 256; i++) {
            h_cdfs[histogramOffset + i] += redistIncrement;
        }
        
        // Create CDF
        for (int i = 1; i < 256; i++) {
            h_cdfs[histogramOffset + i] += h_cdfs[histogramOffset + i - 1];
        }
    }
    
    // Copy CDFs to device
    int* d_cdfs;
    err = cudaMalloc(&d_cdfs, tiles * tiles * 256 * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        throw std::runtime_error("Failed to allocate device memory for CDFs");
    }
    err = cudaMemcpy(d_cdfs, h_cdfs.data(), tiles * tiles * 256 * sizeof(int),
                    cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        cudaFree(d_cdfs);
        throw std::runtime_error("Failed to copy CDFs to device");
    }
    
    // Apply CLAHE
    void* claheArgs[] = {&d_input, &d_output, &d_cdfs, &clipLimit, &width, &height, &tileWidth, &tileHeight, &tiles};
    cudaLaunchKernel((void*)claheKernelLAB, gridDim, blockDim, claheArgs, 0, NULL);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        cudaFree(d_cdfs);
        throw std::runtime_error("Failed to launch CLAHE kernel");
    }
    
    // Copy result back to host
    err = cudaMemcpy(output.pixels.data(), d_output, width * height * sizeof(PixelLAB),
                    cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_histograms);
        cudaFree(d_cdfs);
        throw std::runtime_error("Failed to copy result to host");
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_histograms);
    cudaFree(d_cdfs);
    
    return output;
} 