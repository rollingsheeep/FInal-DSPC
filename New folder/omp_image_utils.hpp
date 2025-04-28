#ifndef OMP_IMAGE_UTILS_HPP
#define OMP_IMAGE_UTILS_HPP

#include "image_utils.hpp"
#include <omp.h>

Image applyUnsharpMaskOMP(const Image& input, float sigma, float amount);
Image applyCLAHEOMP(const Image& input, int tiles, float clipLimit);
ImageLAB applyCLAHE_LAB_OMP(const ImageLAB& input, int tiles, float clipLimit);

#endif // OMP_IMAGE_UTILS_HPP 