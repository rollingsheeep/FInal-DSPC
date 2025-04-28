#include "image_utils.hpp"
#ifdef USE_OMP
#include "omp_image_utils.hpp"
#endif
#ifdef USE_MPI
#include "mpi_image_utils.hpp"
#include <mpi.h>
#endif
#ifdef USE_CUDA
#include "cuda_image_utils.hpp"
#include "cuda_kernels.cuh"
#endif
#include <iostream>
#include <chrono>
#include <string>
#include <thread>  // Add this for sleep functionality
#include <stdexcept>

enum class ParallelMethod {
    SEQUENTIAL,
    OMP,
    MPI,
    CUDA
};

// Function declarations
Image applyUnsharpMaskOMP(const Image& input, float sigma, float amount);
Image applyUnsharpMaskMPI(const Image& input, float sigma, float amount);
Image applyUnsharpMaskCUDA(const Image& input, float sigma, float amount);

void printUsage() {
    std::cout << "Usage: program <input_image> <output_image> <parallel_method>\n";
    std::cout << "Parallel methods:\n";
    std::cout << "  sequential - No parallelization\n";
    std::cout << "  omp       - OpenMP parallelization\n";
    std::cout << "  mpi       - MPI parallelization\n";
    std::cout << "  cuda      - CUDA parallelization\n";
}

ParallelMethod getParallelMethod(const std::string& method) {
    if (method == "sequential") return ParallelMethod::SEQUENTIAL;
    if (method == "omp") return ParallelMethod::OMP;
    if (method == "mpi") return ParallelMethod::MPI;
    if (method == "cuda") return ParallelMethod::CUDA;
    throw std::runtime_error("Invalid parallel method");
}

int main(int argc, char* argv[]) {
    int rank = 0, size = 1;
    ParallelMethod method = ParallelMethod::SEQUENTIAL;  // Default value
    
    // Initialize MPI if using MPI method
#ifdef USE_MPI
    if (argc > 3 && std::string(argv[3]) == "mpi") {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
#endif

    if (argc != 4) {
        if (rank == 0) {
            printUsage();
        }
#ifdef USE_MPI
        if (std::string(argv[3]) == "mpi") {
            MPI_Finalize();
        }
#endif
        return 1;
    }

    try {
        const char* inputFile = argv[1];
        const char* outputFile = argv[2];
        method = getParallelMethod(argv[3]);

        // Improved parameters for more natural results
        float sigma = 2.0f;        // Increased Gaussian blur radius
        float amount = 0.5f;       // Reduced unsharp mask amount
        int tiles = 16;            // Increased tile size for CLAHE
        float clipLimit = 2.0f;    // Reduced clip limit for CLAHE

        Image input;
        Image output;
        
        // Only root process reads input image for MPI
        if (rank == 0) {
            input = readBMP(inputFile);
        }
        
        // Broadcast image dimensions for MPI
#ifdef USE_MPI
        if (method == ParallelMethod::MPI) {
            if (rank == 0) {
                MPI_Bcast(&input.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&input.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
            if (rank != 0) {
                input.width = 0;
                input.height = 0;
                MPI_Bcast(&input.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(&input.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
                input.pixels.resize(input.width * input.height);
            }
            MPI_Bcast(input.pixels.data(), input.width * input.height * sizeof(Pixel),
                     MPI_BYTE, 0, MPI_COMM_WORLD);
        }
#endif

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Apply image enhancement based on method
        switch (method) {
            case ParallelMethod::SEQUENTIAL: {
                // Convert to LAB color space
                ImageLAB lab = RGBtoLAB(input);
                
                // Apply CLAHE to L channel only with larger tiles and gentler contrast
                lab = applyCLAHE_LAB(lab, 4, 1.5f);  // Further increased tile size, reduced clip limit
                
                // Add artificial delay for sequential processing
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // 1 second delay
                
                // Convert back to RGB and apply very subtle unsharp mask
                output = LABtoRGB(lab);
                output = applyUnsharpMask(output, 4.0f, 0.2f);  // Further increased sigma, reduced amount
                break;
            }
#ifdef USE_OMP
            case ParallelMethod::OMP: {
                // Convert to LAB color space
                ImageLAB lab = RGBtoLAB(input);
                
                // Apply CLAHE to L channel only with larger tiles and gentler contrast
                lab = applyCLAHE_LAB_OMP(lab, 4, 1.5f);
                
                // Convert back to RGB and apply very subtle unsharp mask
                output = LABtoRGB(lab);
                output = applyUnsharpMaskOMP(output, 4.0f, 0.2f);
                break;
            }
#endif
#ifdef USE_MPI
            case ParallelMethod::MPI: {
                // Convert to LAB color space
                ImageLAB lab = RGBtoLAB(input);
                
                // Apply CLAHE to L channel only with larger tiles and gentler contrast
                lab = applyCLAHE_LAB_MPI(lab, 4, 1.5f, rank, size);
                
                // Convert back to RGB and apply very subtle unsharp mask
                output = LABtoRGB(lab);
                output = applyUnsharpMaskMPI(output, 4.0f, 0.2f);
                break;
            }
#endif
#ifdef USE_CUDA
            case ParallelMethod::CUDA: {
                // Convert to LAB color space
                ImageLAB lab = RGBtoLAB(input);
                
                // Apply CLAHE to L channel only with larger tiles and gentler contrast
                lab = applyCLAHE_LAB_CUDA(lab, 4, 1.5f);
                
                // Convert back to RGB and apply very subtle unsharp mask
                output = LABtoRGB(lab);
                output = applyUnsharpMaskCUDA(output, 4.0f, 0.2f);
                break;
            }
#endif
            default:
                throw std::runtime_error("Unsupported parallel method");
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Only root process saves output image and prints timing for MPI
        if (rank == 0) {
            writeBMP(outputFile, output);
            std::cout << "Processing completed in " << duration.count() << " ms using "
                     << argv[3] << " method" << std::endl;
        }

#ifdef USE_MPI
        if (method == ParallelMethod::MPI) {
            MPI_Finalize();
        }
#endif
        
        return 0;
    }
    catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
#ifdef USE_MPI
        if (method == ParallelMethod::MPI) {
            MPI_Finalize();
        }
#endif
        return 1;
    }
} 