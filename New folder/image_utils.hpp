#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

struct Pixel {
    uint8_t r, g, b;
};

struct PixelLAB {
    float l, a, b;
};

struct Image {
    int width;
    int height;
    std::vector<Pixel> pixels;
};

struct ImageLAB {
    int width;
    int height;
    std::vector<PixelLAB> pixels;
};

// Custom clamp function since std::clamp might not be available
template<typename T>
T clamp(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Function declarations
Image readBMP(const char* filename);
void writeBMP(const char* filename, const Image& image);

// Color space conversion
ImageLAB RGBtoLAB(const Image& rgb);
Image LABtoRGB(const ImageLAB& lab);

// Image processing
Image applyUnsharpMask(const Image& input, float sigma, float amount);
Image applyCLAHE(const Image& input, int tiles, float clipLimit);
ImageLAB applyCLAHE_LAB(const ImageLAB& input, int tiles, float clipLimit);

// Helper functions for color space conversion
void RGBtoHSL(uint8_t r, uint8_t g, uint8_t b, float& h, float& s, float& l);
void HSLtoRGB(float h, float s, float l, uint8_t& r, uint8_t& g, uint8_t& b);

// Advanced image processing functions
Image applyBilateralFilter(const Image& input, float sigmaSpace, float sigmaColor);
Image boostSaturation(const Image& input, float boostFactor);
Image adjustGamma(const Image& input, float gamma);
Image enhanceImage(const Image& input);

#endif // IMAGE_UTILS_HPP 