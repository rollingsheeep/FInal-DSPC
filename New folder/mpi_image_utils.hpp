#ifndef MPI_IMAGE_UTILS_HPP
#define MPI_IMAGE_UTILS_HPP

#include "image_utils.hpp"
#include <mpi.h>

Image applyUnsharpMaskMPI(const Image& input, float sigma, float amount);
Image applyCLAHEMPI(const Image& input, int tiles, float clipLimit, int rank, int size);
ImageLAB applyCLAHE_LAB_MPI(const ImageLAB& input, int tiles, float clipLimit, int rank, int size);

#endif // MPI_IMAGE_UTILS_HPP 