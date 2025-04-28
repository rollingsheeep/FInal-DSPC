#include "mpi_image_utils.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mpi.h>

Image applyUnsharpMaskMPI(const Image& input, float sigma, float amount) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only proceed with MPI if image is large enough to benefit from parallelization
    if (input.height < size * 50) {  // Reduced threshold for more aggressive parallelization
        if (rank == 0) {
            return applyUnsharpMask(input, sigma, amount);
        }
        return input;
    }

    // Generate Gaussian kernel (same for all processes)
    std::vector<float> kernel;
    int kernelSize = static_cast<int>(ceil(sigma * 6));
    if (kernelSize % 2 == 0) kernelSize++;
    
    kernel.resize(kernelSize);
    float sum = 0.0f;
    int center = kernelSize / 2;
    
    for (int i = 0; i < kernelSize; i++) {
        float x = i - center;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }

    // Calculate rows per process with larger chunks
    int rowsPerProcess = (input.height + size - 1) / size;  // Ceiling division
    int startRow = rank * rowsPerProcess;
    int endRow = std::min((rank + 1) * rowsPerProcess, input.height);
    
    // Add minimal overlap for kernel processing
    int overlap = kernelSize / 2;
    int actualStartRow = std::max(0, startRow - overlap);
    int actualEndRow = std::min(input.height, endRow + overlap);
    
    // Create local image portion
    Image localInput;
    localInput.width = input.width;
    localInput.height = actualEndRow - actualStartRow;
    localInput.pixels.resize(localInput.width * localInput.height);
    
    // Use MPI_Scatterv for efficient data distribution
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int currentDispl = 0;
    
    for (int i = 0; i < size; i++) {
        int localStart = i * rowsPerProcess;
        int localEnd = std::min((i + 1) * rowsPerProcess, input.height);
        int localOverlap = (i > 0 ? overlap : 0) + (i < size - 1 ? overlap : 0);
        sendcounts[i] = (localEnd - localStart + localOverlap) * input.width * sizeof(Pixel);
        displs[i] = currentDispl;
        currentDispl += sendcounts[i];
    }
    
    // Use non-blocking scatter for better performance
    MPI_Request scatterRequest;
    MPI_Iscatterv(input.pixels.data(), sendcounts.data(), displs.data(), MPI_BYTE,
                 localInput.pixels.data(), localInput.height * input.width * sizeof(Pixel), MPI_BYTE,
                 0, MPI_COMM_WORLD, &scatterRequest);
    
    // Create temporary image for blur
    Image localBlurred = localInput;
    
    // Wait for scatter to complete
    MPI_Wait(&scatterRequest, MPI_STATUS_IGNORE);
    
    // Apply Gaussian blur to local portion
    #pragma omp parallel for collapse(2)  // Add OpenMP parallelization within MPI processes
    for (int y = 0; y < localInput.height; y++) {
        for (int x = 0; x < localInput.width; x++) {
            float sumR = 0, sumG = 0, sumB = 0;
            
            for (int k = -center; k <= center; k++) {
                int px = clamp(x + k, 0, localInput.width - 1);
                sumR += localInput.pixels[y * localInput.width + px].r * kernel[k + center];
                sumG += localInput.pixels[y * localInput.width + px].g * kernel[k + center];
                sumB += localInput.pixels[y * localInput.width + px].b * kernel[k + center];
            }
            
            localBlurred.pixels[y * localInput.width + x].r = static_cast<uint8_t>(clamp(sumR, 0.0f, 255.0f));
            localBlurred.pixels[y * localInput.width + x].g = static_cast<uint8_t>(clamp(sumG, 0.0f, 255.0f));
            localBlurred.pixels[y * localInput.width + x].b = static_cast<uint8_t>(clamp(sumB, 0.0f, 255.0f));
        }
    }
    
    // Apply unsharp mask to local portion
    Image localOutput = localInput;
    #pragma omp parallel for  // Add OpenMP parallelization within MPI processes
    for (int i = 0; i < localInput.width * localInput.height; i++) {
        float r = localInput.pixels[i].r + amount * (localInput.pixels[i].r - localBlurred.pixels[i].r);
        float g = localInput.pixels[i].g + amount * (localInput.pixels[i].g - localBlurred.pixels[i].g);
        float b = localInput.pixels[i].b + amount * (localInput.pixels[i].b - localBlurred.pixels[i].b);
        
        localOutput.pixels[i].r = static_cast<uint8_t>(clamp(r, 0.0f, 255.0f));
        localOutput.pixels[i].g = static_cast<uint8_t>(clamp(g, 0.0f, 255.0f));
        localOutput.pixels[i].b = static_cast<uint8_t>(clamp(b, 0.0f, 255.0f));
    }
    
    // Prepare for gathering results
    Image output;
    if (rank == 0) {
        output = input;
    }
    
    // Calculate gather parameters
    std::vector<int> recvcounts(size);
    std::vector<int> gatherDispls(size);
    currentDispl = 0;
    
    for (int i = 0; i < size; i++) {
        int localStart = i * rowsPerProcess;
        int localEnd = std::min((i + 1) * rowsPerProcess, input.height);
        recvcounts[i] = (localEnd - localStart) * input.width * sizeof(Pixel);
        gatherDispls[i] = currentDispl;
        currentDispl += recvcounts[i];
    }
    
    // Use non-blocking gather for better performance
    MPI_Request gatherRequest;
    int localStart = (rank > 0 ? overlap : 0);
    int localEnd = localInput.height - (rank < size - 1 ? overlap : 0);
    int localSize = localEnd - localStart;
    
    MPI_Igatherv(localOutput.pixels.data() + localStart * localInput.width,
                localSize * localInput.width * sizeof(Pixel), MPI_BYTE,
                output.pixels.data(), recvcounts.data(), gatherDispls.data(), MPI_BYTE,
                0, MPI_COMM_WORLD, &gatherRequest);
    
    // Wait for gather to complete
    MPI_Wait(&gatherRequest, MPI_STATUS_IGNORE);
    
    return output;
}

Image applyCLAHEMPI(const Image& input, int tiles, float clipLimit, int rank, int size) {
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
    if (size <= 0) {
        throw std::runtime_error("Number of processes must be positive");
    }

    Image output = input;
    int tileWidth = input.width / tiles;
    int tileHeight = input.height / tiles;
    
    // Divide tiles among processes
    int totalTiles = tiles * tiles;
    int tilesPerProcess = totalTiles / size;
    int remainingTiles = totalTiles % size;
    
    // Distribute remaining tiles among processes
    int startTile = rank * tilesPerProcess + std::min(rank, remainingTiles);
    int endTile = startTile + tilesPerProcess + (rank < remainingTiles ? 1 : 0);
    
    // Process assigned tiles
    for (int tileIdx = startTile; tileIdx < endTile; tileIdx++) {
        int ty = tileIdx / tiles;
        int tx = tileIdx % tiles;
        
        for (int channel = 0; channel < 3; channel++) {
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
    
    // Gather results from all processes
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int otherStartTile = i * tilesPerProcess;
            int otherEndTile = (i == size - 1) ? (tiles * tiles) : otherStartTile + tilesPerProcess;
            
            for (int tileIdx = otherStartTile; tileIdx < otherEndTile; tileIdx++) {
                int ty = tileIdx / tiles;
                int tx = tileIdx % tiles;
                int startY = ty * tileHeight;
                int startX = tx * tileWidth;
                int endY = std::min((ty + 1) * tileHeight, input.height);
                int endX = std::min((tx + 1) * tileWidth, input.width);
                
                for (int y = startY; y < endY; y++) {
                    MPI_Recv(&output.pixels[y * input.width + startX],
                            (endX - startX) * sizeof(Pixel),
                            MPI_BYTE, i, tileIdx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    } else {
        for (int tileIdx = startTile; tileIdx < endTile; tileIdx++) {
            int ty = tileIdx / tiles;
            int tx = tileIdx % tiles;
            int startY = ty * tileHeight;
            int startX = tx * tileWidth;
            int endY = std::min((ty + 1) * tileHeight, input.height);
            int endX = std::min((tx + 1) * tileWidth, input.width);
            
            for (int y = startY; y < endY; y++) {
                MPI_Send(&output.pixels[y * input.width + startX],
                        (endX - startX) * sizeof(Pixel),
                        MPI_BYTE, 0, tileIdx, MPI_COMM_WORLD);
            }
        }
    }
    
    // Broadcast complete result to all processes
    MPI_Bcast(output.pixels.data(),
             output.pixels.size() * sizeof(Pixel),
             MPI_BYTE, 0, MPI_COMM_WORLD);
    
    return output;
}

ImageLAB applyCLAHE_LAB_MPI(const ImageLAB& input, int tiles, float clipLimit, int rank, int size) {
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
    if (size <= 0) {
        throw std::runtime_error("Number of processes must be positive");
    }

    ImageLAB output = input;
    int tileWidth = input.width / tiles;
    int tileHeight = input.height / tiles;
    
    // Divide tiles among processes
    int totalTiles = tiles * tiles;
    int tilesPerProcess = totalTiles / size;
    int remainingTiles = totalTiles % size;
    
    // Distribute remaining tiles among processes
    int startTile = rank * tilesPerProcess + std::min(rank, remainingTiles);
    int endTile = startTile + tilesPerProcess + (rank < remainingTiles ? 1 : 0);
    
    // Process assigned tiles
    for (int tileIdx = startTile; tileIdx < endTile; tileIdx++) {
        int ty = tileIdx / tiles;
        int tx = tileIdx % tiles;
        
        // Process only the L channel
        std::vector<int> histogram(256, 0);
        int startY = ty * tileHeight;
        int startX = tx * tileWidth;
        int endY = std::min((ty + 1) * tileHeight, input.height);
        int endX = std::min((tx + 1) * tileWidth, input.width);
        
        // Build histogram
        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                int value = static_cast<int>(input.pixels[y * input.width + x].l * 255.0f / 100.0f);
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
    
    // Gather results from all processes
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            int otherStartTile = i * tilesPerProcess;
            int otherEndTile = (i == size - 1) ? (tiles * tiles) : otherStartTile + tilesPerProcess;
            
            for (int tileIdx = otherStartTile; tileIdx < otherEndTile; tileIdx++) {
                int ty = tileIdx / tiles;
                int tx = tileIdx % tiles;
                int startY = ty * tileHeight;
                int startX = tx * tileWidth;
                int endY = std::min((ty + 1) * tileHeight, input.height);
                int endX = std::min((tx + 1) * tileWidth, input.width);
                
                for (int y = startY; y < endY; y++) {
                    MPI_Recv(&output.pixels[y * input.width + startX],
                            (endX - startX) * sizeof(PixelLAB),
                            MPI_BYTE, i, tileIdx, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
    } else {
        for (int tileIdx = startTile; tileIdx < endTile; tileIdx++) {
            int ty = tileIdx / tiles;
            int tx = tileIdx % tiles;
            int startY = ty * tileHeight;
            int startX = tx * tileWidth;
            int endY = std::min((ty + 1) * tileHeight, input.height);
            int endX = std::min((tx + 1) * tileWidth, input.width);
            
            for (int y = startY; y < endY; y++) {
                MPI_Send(&output.pixels[y * input.width + startX],
                        (endX - startX) * sizeof(PixelLAB),
                        MPI_BYTE, 0, tileIdx, MPI_COMM_WORLD);
            }
        }
    }
    
    // Broadcast complete result to all processes
    MPI_Bcast(output.pixels.data(),
             output.pixels.size() * sizeof(PixelLAB),
             MPI_BYTE, 0, MPI_COMM_WORLD);
    
    return output;
} 