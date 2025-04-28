#include "omp_image_utils.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

Image applyUnsharpMaskOMP(const Image& input, float sigma, float amount) {
    Image output = input;
    std::vector<float> kernel;
    int kernelSize = static_cast<int>(ceil(sigma * 6));
    if (kernelSize % 2 == 0) kernelSize++;
    
    // Generate Gaussian kernel
    kernel.resize(kernelSize);
    float sum = 0.0f;
    int center = kernelSize / 2;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < kernelSize; i++) {
        float x = i - center;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    #pragma omp parallel for
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Create temporary image for blur
    Image blurred = input;
    
    // Apply Gaussian blur
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sumR = 0, sumG = 0, sumB = 0;
            
            for (int k = -center; k <= center; k++) {
                int px = clamp(x + k, 0, input.width - 1);
                sumR += input.pixels[y * input.width + px].r * kernel[k + center];
                sumG += input.pixels[y * input.width + px].g * kernel[k + center];
                sumB += input.pixels[y * input.width + px].b * kernel[k + center];
            }
            
            blurred.pixels[y * input.width + x].r = static_cast<uint8_t>(clamp(sumR, 0.0f, 255.0f));
            blurred.pixels[y * input.width + x].g = static_cast<uint8_t>(clamp(sumG, 0.0f, 255.0f));
            blurred.pixels[y * input.width + x].b = static_cast<uint8_t>(clamp(sumB, 0.0f, 255.0f));
        }
    }
    
    // Apply unsharp mask
    #pragma omp parallel for
    for (int i = 0; i < input.width * input.height; i++) {
        float r = input.pixels[i].r + amount * (input.pixels[i].r - blurred.pixels[i].r);
        float g = input.pixels[i].g + amount * (input.pixels[i].g - blurred.pixels[i].g);
        float b = input.pixels[i].b + amount * (input.pixels[i].b - blurred.pixels[i].b);
        
        output.pixels[i].r = static_cast<uint8_t>(clamp(r, 0.0f, 255.0f));
        output.pixels[i].g = static_cast<uint8_t>(clamp(g, 0.0f, 255.0f));
        output.pixels[i].b = static_cast<uint8_t>(clamp(b, 0.0f, 255.0f));
    }
    
    return output;
}

Image applyCLAHEOMP(const Image& input, int tiles, float clipLimit) {
    Image output = input;
    int tileWidth = input.width / tiles;
    int tileHeight = input.height / tiles;
    
    // Process each color channel separately
    #pragma omp parallel for collapse(3)
    for (int channel = 0; channel < 3; channel++) {
        for (int ty = 0; ty < tiles; ty++) {
            for (int tx = 0; tx < tiles; tx++) {
                std::vector<int> histogram(256, 0);
                int startY = ty * tileHeight;
                int startX = tx * tileWidth;
                int endY = std::min((ty + 1) * tileHeight, input.height);
                int endX = std::min((tx + 1) * tileWidth, input.width);
                
                // Build histogram
                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        uint8_t value;
                        switch(channel) {
                            case 0: value = input.pixels[y * input.width + x].r; break;
                            case 1: value = input.pixels[y * input.width + x].g; break;
                            case 2: value = input.pixels[y * input.width + x].b; break;
                        }
                        #pragma omp atomic
                        histogram[value]++;
                    }
                }
                
                // Apply clip limit
                int clipCount = static_cast<int>(clipLimit * (tileWidth * tileHeight) / 256);
                int redistBatch = 0;
                
                for (int i = 0; i < 256; i++) {
                    if (histogram[i] > clipCount) {
                        redistBatch += (histogram[i] - clipCount);
                        histogram[i] = clipCount;
                    }
                }
                
                // Redistribute clipped pixels
                int redistIncrement = redistBatch / 256;
                for (int i = 0; i < 256; i++) {
                    histogram[i] += redistIncrement;
                }
                
                // Create cumulative histogram
                std::vector<int> cdf(256, 0);
                cdf[0] = histogram[0];
                for (int i = 1; i < 256; i++) {
                    cdf[i] = cdf[i-1] + histogram[i];
                }
                
                // Normalize CDF
                float scale = 255.0f / cdf[255];
                
                // Apply CLAHE to tile
                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        uint8_t value;
                        switch(channel) {
                            case 0: value = input.pixels[y * input.width + x].r; break;
                            case 1: value = input.pixels[y * input.width + x].g; break;
                            case 2: value = input.pixels[y * input.width + x].b; break;
                        }
                        
                        float newValue = cdf[value] * scale;
                        switch(channel) {
                            case 0: output.pixels[y * input.width + x].r = static_cast<uint8_t>(clamp(newValue, 0.0f, 255.0f)); break;
                            case 1: output.pixels[y * input.width + x].g = static_cast<uint8_t>(clamp(newValue, 0.0f, 255.0f)); break;
                            case 2: output.pixels[y * input.width + x].b = static_cast<uint8_t>(clamp(newValue, 0.0f, 255.0f)); break;
                        }
                    }
                }
            }
        }
    }
    
    return output;
}

ImageLAB applyCLAHE_LAB_OMP(const ImageLAB& input, int tiles, float clipLimit) {
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
    int tileWidth = input.width / tiles;
    int tileHeight = input.height / tiles;
    
    // Process only the L channel
    #pragma omp parallel for collapse(2)
    for (int ty = 0; ty < tiles; ty++) {
        for (int tx = 0; tx < tiles; tx++) {
            // Calculate histogram for current tile
            std::vector<int> histogram(256, 0);
            int startY = ty * tileHeight;
            int startX = tx * tileWidth;
            int endY = std::min((ty + 1) * tileHeight, input.height);
            int endX = std::min((tx + 1) * tileWidth, input.width);
            
            // Build histogram
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    int value = static_cast<int>(input.pixels[y * input.width + x].l * 255.0f / 100.0f);
                    #pragma omp atomic
                    histogram[clamp(value, 0, 255)]++;
                }
            }
            
            // Apply clip limit
            int clipCount = static_cast<int>(clipLimit * static_cast<float>(tileWidth * tileHeight) / 256.0f);
            int redistBatch = 0;
            
            for (int i = 0; i < 256; i++) {
                if (histogram[i] > clipCount) {
                    redistBatch += (histogram[i] - clipCount);
                    histogram[i] = clipCount;
                }
            }
            
            // Redistribute clipped pixels
            int redistIncrement = redistBatch / 256;
            for (int i = 0; i < 256; i++) {
                histogram[i] += redistIncrement;
            }
            
            // Create cumulative histogram
            std::vector<int> cdf(256, 0);
            cdf[0] = histogram[0];
            for (int i = 1; i < 256; i++) {
                cdf[i] = cdf[i-1] + histogram[i];
            }
            
            // Normalize CDF
            float scale = 255.0f / static_cast<float>(cdf[255]);
            
            // Apply CLAHE to tile
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    int value = static_cast<int>(input.pixels[y * input.width + x].l * 255.0f / 100.0f);
                    float newValue = static_cast<float>(cdf[value]) * scale;
                    output.pixels[y * input.width + x].l = newValue * 100.0f / 255.0f;
                }
            }
        }
    }
    
    return output;
} 