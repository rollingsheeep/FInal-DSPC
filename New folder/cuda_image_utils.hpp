#ifndef CUDA_IMAGE_UTILS_HPP
#define CUDA_IMAGE_UTILS_HPP

#include "image_utils.hpp"

Image applyUnsharpMaskCUDA(const Image& input, float sigma, float amount);
Image applyCLAHECUDA(const Image& input, int tiles, float clipLimit);
ImageLAB applyCLAHE_LAB_CUDA(const ImageLAB& input, int tiles, float clipLimit);

#endif // CUDA_IMAGE_UTILS_HPP 