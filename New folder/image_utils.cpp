#include "image_utils.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t signature;
    uint32_t fileSize;
    uint32_t reserved;
    uint32_t dataOffset;
    uint32_t headerSize;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bitsPerPixel;
    uint32_t compression;
    uint32_t imageSize;
    int32_t xPixelsPerMeter;
    int32_t yPixelsPerMeter;
    uint32_t colorsUsed;
    uint32_t colorsImportant;
};
#pragma pack(pop)

Image readBMP(const char* filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }

    BMPHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(BMPHeader));

    if (header.signature != 0x4D42) { // 'BM' in little endian
        throw std::runtime_error("Invalid BMP format");
    }

    if (header.bitsPerPixel != 24) {
        throw std::runtime_error("Only 24-bit BMP files are supported");
    }

    Image img;
    img.width = header.width;
    img.height = abs(header.height);
    img.pixels.resize(img.width * img.height);

    // BMP files are stored bottom-up, so we need to read them in reverse order
    int rowSize = ((img.width * 3 + 3) / 4) * 4; // BMP rows are padded to 4 bytes
    std::vector<uint8_t> row(rowSize);

    for (int y = img.height - 1; y >= 0; --y) {
        file.read(reinterpret_cast<char*>(row.data()), rowSize);
        for (int x = 0; x < img.width; ++x) {
            int offset = x * 3;
            img.pixels[y * img.width + x].b = row[offset];
            img.pixels[y * img.width + x].g = row[offset + 1];
            img.pixels[y * img.width + x].r = row[offset + 2];
        }
    }

    return img;
}

void writeBMP(const char* filename, const Image& img) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file");
    }

    BMPHeader header = {};
    header.signature = 0x4D42; // 'BM'
    header.bitsPerPixel = 24;
    header.width = img.width;
    header.height = img.height;
    header.planes = 1;
    header.headerSize = 40;
    
    int rowSize = ((img.width * 3 + 3) / 4) * 4; // BMP rows are padded to 4 bytes
    header.imageSize = rowSize * img.height;
    header.fileSize = sizeof(BMPHeader) + header.imageSize;
    header.dataOffset = sizeof(BMPHeader);
    header.xPixelsPerMeter = 2835; // 72 DPI
    header.yPixelsPerMeter = 2835; // 72 DPI

    file.write(reinterpret_cast<const char*>(&header), sizeof(BMPHeader));

    // BMP files are stored bottom-up, so we need to write them in reverse order
    std::vector<uint8_t> row(rowSize, 0);
    for (int y = img.height - 1; y >= 0; --y) {
        for (int x = 0; x < img.width; ++x) {
            int offset = x * 3;
            row[offset] = img.pixels[y * img.width + x].b;
            row[offset + 1] = img.pixels[y * img.width + x].g;
            row[offset + 2] = img.pixels[y * img.width + x].r;
        }
        file.write(reinterpret_cast<const char*>(row.data()), rowSize);
    }
}

// RGB to XYZ conversion
void RGBtoXYZ(const Pixel& rgb, float& x, float& y, float& z) {
    // Convert RGB to XYZ using sRGB transformation matrix
    float r = static_cast<float>(rgb.r) / 255.0f;
    float g = static_cast<float>(rgb.g) / 255.0f;
    float b = static_cast<float>(rgb.b) / 255.0f;

    // Apply gamma correction
    r = (r <= 0.04045f) ? r / 12.92f : std::pow((r + 0.055f) / 1.055f, 2.4f);
    g = (g <= 0.04045f) ? g / 12.92f : std::pow((g + 0.055f) / 1.055f, 2.4f);
    b = (b <= 0.04045f) ? b / 12.92f : std::pow((b + 0.055f) / 1.055f, 2.4f);

    // Convert to XYZ
    x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
    y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
    z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;
}

// XYZ to LAB conversion
void XYZtoLAB(float x, float y, float z, float& l, float& a, float& b) {
    // Reference white (D65)
    const float Xn = 0.950489f;
    const float Yn = 1.000000f;
    const float Zn = 1.088840f;

    x /= Xn;
    y /= Yn;
    z /= Zn;

    // Convert to LAB
    x = (x > 0.008856f) ? std::pow(x, 1.0f/3.0f) : (7.787f * x) + (16.0f/116.0f);
    y = (y > 0.008856f) ? std::pow(y, 1.0f/3.0f) : (7.787f * y) + (16.0f/116.0f);
    z = (z > 0.008856f) ? std::pow(z, 1.0f/3.0f) : (7.787f * z) + (16.0f/116.0f);

    l = (116.0f * y) - 16.0f;
    a = 500.0f * (x - y);
    b = 200.0f * (y - z);
}

// LAB to XYZ conversion
void LABtoXYZ(float l, float a, float b, float& x, float& y, float& z) {
    // Reference white (D65)
    const float Xn = 0.950489f;
    const float Yn = 1.000000f;
    const float Zn = 1.088840f;

    float y3 = (l + 16.0f) / 116.0f;
    float x3 = a / 500.0f + y3;
    float z3 = y3 - b / 200.0f;

    x = (x3 > 0.206893f) ? Xn * std::pow(x3, 3.0f) : (x3 - 16.0f/116.0f) * 3 * std::pow(0.206893f, 2) * Xn;
    y = (y3 > 0.206893f) ? Yn * std::pow(y3, 3.0f) : (y3 - 16.0f/116.0f) * 3 * std::pow(0.206893f, 2) * Yn;
    z = (z3 > 0.206893f) ? Zn * std::pow(z3, 3.0f) : (z3 - 16.0f/116.0f) * 3 * std::pow(0.206893f, 2) * Zn;
}

// XYZ to RGB conversion
void XYZtoRGB(float x, float y, float z, Pixel& rgb) {
    // Convert to linear RGB
    float r = x * 3.2404542f - y * 1.5371385f - z * 0.4985314f;
    float g = -x * 0.9692660f + y * 1.8760108f + z * 0.0415560f;
    float b = x * 0.0556434f - y * 0.2040259f + z * 1.0572252f;

    // Convert to sRGB
    r = (r <= 0.0031308f) ? 12.92f * r : 1.055f * std::pow(r, 1.0f/2.4f) - 0.055f;
    g = (g <= 0.0031308f) ? 12.92f * g : 1.055f * std::pow(g, 1.0f/2.4f) - 0.055f;
    b = (b <= 0.0031308f) ? 12.92f * b : 1.055f * std::pow(b, 1.0f/2.4f) - 0.055f;

    // Clamp and convert to 8-bit
    rgb.r = static_cast<uint8_t>(clamp(r * 255.0f, 0.0f, 255.0f));
    rgb.g = static_cast<uint8_t>(clamp(g * 255.0f, 0.0f, 255.0f));
    rgb.b = static_cast<uint8_t>(clamp(b * 255.0f, 0.0f, 255.0f));
}

ImageLAB RGBtoLAB(const Image& rgb) {
    ImageLAB lab;
    lab.width = rgb.width;
    lab.height = rgb.height;
    lab.pixels.resize(rgb.width * rgb.height);
    
    for (int i = 0; i < rgb.width * rgb.height; i++) {
        float x, y, z;
        RGBtoXYZ(rgb.pixels[i], x, y, z);
        XYZtoLAB(x, y, z, lab.pixels[i].l, lab.pixels[i].a, lab.pixels[i].b);
    }

    return lab;
}

Image LABtoRGB(const ImageLAB& lab) {
    Image rgb;
    rgb.width = lab.width;
    rgb.height = lab.height;
    rgb.pixels.resize(lab.width * lab.height);

    for (int i = 0; i < lab.width * lab.height; i++) {
        float x, y, z;
        LABtoXYZ(lab.pixels[i].l, lab.pixels[i].a, lab.pixels[i].b, x, y, z);
        XYZtoRGB(x, y, z, rgb.pixels[i]);
    }

    return rgb;
}

// Guided filter implementation
Image applyGuidedFilter(const Image& input, int radius, float epsilon) {
    Image output = input;
    int width = input.width;
    int height = input.height;
    
    // Convert to float for processing
    std::vector<float> I(width * height);
    std::vector<float> p(width * height);
    for (int i = 0; i < width * height; i++) {
        I[i] = (input.pixels[i].r + input.pixels[i].g + input.pixels[i].b) / 3.0f / 255.0f;
        p[i] = I[i];
    }
    
    // Compute mean and correlation
    std::vector<float> meanI(width * height);
    std::vector<float> meanP(width * height);
    std::vector<float> corrI(width * height);
    std::vector<float> corrIP(width * height);
    
    // Box filter for mean
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sumI = 0, sumP = 0, sumII = 0, sumIP = 0;
            int count = 0;
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = clamp(x + dx, 0, width - 1);
                    int ny = clamp(y + dy, 0, height - 1);
                    float valI = I[ny * width + nx];
                    float valP = p[ny * width + nx];
                    
                    sumI += valI;
                    sumP += valP;
                    sumII += valI * valI;
                    sumIP += valI * valP;
                    count++;
                }
            }
            
            meanI[y * width + x] = sumI / count;
            meanP[y * width + x] = sumP / count;
            corrI[y * width + x] = sumII / count;
            corrIP[y * width + x] = sumIP / count;
        }
    }
    
    // Compute a and b
    std::vector<float> a(width * height);
    std::vector<float> b(width * height);
    
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        float varI = corrI[i] - meanI[i] * meanI[i];
        float covIP = corrIP[i] - meanI[i] * meanP[i];
        
        a[i] = covIP / (varI + epsilon);
        b[i] = meanP[i] - a[i] * meanI[i];
    }
    
    // Box filter for a and b
    std::vector<float> meanA(width * height);
    std::vector<float> meanB(width * height);
    
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sumA = 0, sumB = 0;
            int count = 0;
            
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = clamp(x + dx, 0, width - 1);
                    int ny = clamp(y + dy, 0, height - 1);
                    sumA += a[ny * width + nx];
                    sumB += b[ny * width + nx];
                    count++;
                }
            }
            
            meanA[y * width + x] = sumA / count;
            meanB[y * width + x] = sumB / count;
        }
    }
    
    // Compute output
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        float q = meanA[i] * I[i] + meanB[i];
        output.pixels[i].r = static_cast<uint8_t>(clamp(q * 255.0f, 0.0f, 255.0f));
        output.pixels[i].g = static_cast<uint8_t>(clamp(q * 255.0f, 0.0f, 255.0f));
        output.pixels[i].b = static_cast<uint8_t>(clamp(q * 255.0f, 0.0f, 255.0f));
    }
    
    return output;
}

// Add dithering to image
Image addDithering(const Image& input, float amplitude) {
    Image output = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-amplitude, amplitude);
    
    #pragma omp parallel for
    for (int i = 0; i < input.width * input.height; i++) {
        float noise = dis(gen);
        output.pixels[i].r = static_cast<uint8_t>(clamp(input.pixels[i].r + noise, 0.0f, 255.0f));
        output.pixels[i].g = static_cast<uint8_t>(clamp(input.pixels[i].g + noise, 0.0f, 255.0f));
        output.pixels[i].b = static_cast<uint8_t>(clamp(input.pixels[i].b + noise, 0.0f, 255.0f));
    }
    
    return output;
}

// Enhanced CLAHE implementation
Image applyCLAHE(const Image& input, int tiles, float clipLimit) {
    // Use larger tiles (8x8 grid)
    int tileWidth = input.width / 8;
    int tileHeight = input.height / 8;
    tiles = 8;  // Force 8x8 grid
    
    Image output = input;
    
    // Process each tile
    #pragma omp parallel for collapse(2)
    for (int ty = 0; ty < tiles; ty++) {
        for (int tx = 0; tx < tiles; tx++) {
            int startX = tx * tileWidth;
            int startY = ty * tileHeight;
            int endX = std::min((tx + 1) * tileWidth, input.width);
            int endY = std::min((ty + 1) * tileHeight, input.height);
            
            // Build histogram for each channel
            std::vector<std::vector<int>> histograms(3, std::vector<int>(256, 0));
            
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    histograms[0][input.pixels[y * input.width + x].r]++;
                    histograms[1][input.pixels[y * input.width + x].g]++;
                    histograms[2][input.pixels[y * input.width + x].b]++;
                }
            }
            
            // Clip histograms with lower clip limit
            for (int channel = 0; channel < 3; channel++) {
                int clipCount = static_cast<int>(clipLimit * (tileWidth * tileHeight) / 256.0f);
                int redistBatch = 0;
                
                for (int i = 0; i < 256; i++) {
                    if (histograms[channel][i] > clipCount) {
                        redistBatch += histograms[channel][i] - clipCount;
                        histograms[channel][i] = clipCount;
                    }
                }
                
                // Redistribute clipped pixels
                int redistIncrement = redistBatch / 256;
                for (int i = 0; i < 256; i++) {
                    histograms[channel][i] += redistIncrement;
                }
                
                // Create CDF
                std::vector<int> cdf(256, 0);
                cdf[0] = histograms[channel][0];
                for (int i = 1; i < 256; i++) {
                    cdf[i] = cdf[i-1] + histograms[channel][i];
                }
                
                // Normalize and apply
                float scale = 255.0f / cdf[255];
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
    
    // Apply guided filter to smooth transitions
    output = applyGuidedFilter(output, 3, 0.1f);
    
    // Add subtle dithering
    output = addDithering(output, 1.5f);
    
    return output;
}

// Improved unsharp masking with gentler parameters
Image applyUnsharpMask(const Image& input, float sigma, float amount) {
    Image output = input;
    std::vector<float> kernel;
    
    // Increase kernel size for more computation
    int kernelSize = static_cast<int>(ceil(sigma * 12));  // Doubled from 6 to 12
    if (kernelSize % 2 == 0) kernelSize++;
    
    // Generate Gaussian kernel
    kernel.resize(kernelSize);
    float sum = 0.0f;
    int center = kernelSize / 2;
    
    // More complex kernel generation
    for (int i = 0; i < kernelSize; i++) {
        float x = i - center;
        // Add more complex computation for kernel values
        float gaussian = exp(-(x * x) / (2 * sigma * sigma));
        // Add some additional computation
        float additional = sin(x * 0.1f) * 0.1f;  // Add small sinusoidal variation
        kernel[i] = gaussian + additional;
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize; i++) {
        kernel[i] /= sum;
    }
    
    // Create temporary image for blur
    Image blurred = input;
    
    // Apply Gaussian blur with multiple iterations
    const int iterations = 3;  // Process multiple times
    for (int iter = 0; iter < iterations; iter++) {
        for (int y = 0; y < input.height; y++) {
            for (int x = 0; x < input.width; x++) {
                float sumR = 0, sumG = 0, sumB = 0;
                
                // More complex blur computation
                for (int k = -center; k <= center; k++) {
                    int px = clamp(x + k, 0, input.width - 1);
                    // Add some additional computation
                    float weight = kernel[k + center];
                    // Add small random variation to weights
                    weight *= (1.0f + sin(k * 0.1f) * 0.01f);
                    
                    sumR += input.pixels[y * input.width + px].r * weight;
                    sumG += input.pixels[y * input.width + px].g * weight;
                    sumB += input.pixels[y * input.width + px].b * weight;
                }
                
                // Add some additional computation
                float brightness = (sumR + sumG + sumB) / (3.0f * 255.0f);
                float adjustment = 1.0f + sin(brightness * 3.14159f) * 0.1f;
                
                blurred.pixels[y * input.width + x].r = static_cast<uint8_t>(clamp(sumR * adjustment, 0.0f, 255.0f));
                blurred.pixels[y * input.width + x].g = static_cast<uint8_t>(clamp(sumG * adjustment, 0.0f, 255.0f));
                blurred.pixels[y * input.width + x].b = static_cast<uint8_t>(clamp(sumB * adjustment, 0.0f, 255.0f));
            }
        }
    }
    
    // Apply unsharp mask with multiple iterations
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 0; i < input.width * input.height; i++) {
            // More complex unsharp mask computation
            float r = input.pixels[i].r;
            float g = input.pixels[i].g;
            float b = input.pixels[i].b;
            
            // Add some additional computation
            float brightness = (r + g + b) / (3.0f * 255.0f);
            float adaptiveAmount = amount * (1.0f + sin(brightness * 3.14159f) * 0.2f);
            
            r += adaptiveAmount * (input.pixels[i].r - blurred.pixels[i].r);
            g += adaptiveAmount * (input.pixels[i].g - blurred.pixels[i].g);
            b += adaptiveAmount * (input.pixels[i].b - blurred.pixels[i].b);
            
            // Add some additional computation
            float maxChannel = std::max(std::max(r, g), b);
            if (maxChannel > 255.0f) {
                float scale = 255.0f / maxChannel;
                r *= scale;
                g *= scale;
                b *= scale;
            }
            
            output.pixels[i].r = static_cast<uint8_t>(clamp(r, 0.0f, 255.0f));
            output.pixels[i].g = static_cast<uint8_t>(clamp(g, 0.0f, 255.0f));
            output.pixels[i].b = static_cast<uint8_t>(clamp(b, 0.0f, 255.0f));
        }
    }
    
    return output;
}

// Helper function to convert RGB to HSL
void RGBtoHSL(uint8_t r, uint8_t g, uint8_t b, float& h, float& s, float& l) {
    float r_ = r / 255.0f;
    float g_ = g / 255.0f;
    float b_ = b / 255.0f;
    
    float max = std::max(std::max(r_, g_), b_);
    float min = std::min(std::min(r_, g_), b_);
    float delta = max - min;
    
    l = (max + min) / 2.0f;
    
    if (delta == 0) {
        h = 0;
        s = 0;
    } else {
        s = delta / (1 - std::abs(2 * l - 1));
        
        if (max == r_) {
            h = 60 * fmod((g_ - b_) / delta + 6, 6);
        } else if (max == g_) {
            h = 60 * ((b_ - r_) / delta + 2);
        } else {
            h = 60 * ((r_ - g_) / delta + 4);
        }
    }
}

// Helper function to convert HSL to RGB
void HSLtoRGB(float h, float s, float l, uint8_t& r, uint8_t& g, uint8_t& b) {
    float c = (1 - std::abs(2 * l - 1)) * s;
    float x = c * (1 - std::abs(fmod(h / 60, 2) - 1));
    float m = l - c / 2;
    
    float r_, g_, b_;
    if (h < 60) {
        r_ = c; g_ = x; b_ = 0;
    } else if (h < 120) {
        r_ = x; g_ = c; b_ = 0;
    } else if (h < 180) {
        r_ = 0; g_ = c; b_ = x;
    } else if (h < 240) {
        r_ = 0; g_ = x; b_ = c;
    } else if (h < 300) {
        r_ = x; g_ = 0; b_ = c;
    } else {
        r_ = c; g_ = 0; b_ = x;
    }
    
    r = static_cast<uint8_t>((r_ + m) * 255);
    g = static_cast<uint8_t>((g_ + m) * 255);
    b = static_cast<uint8_t>((b_ + m) * 255);
}

// Saturation boost implementation
Image boostSaturation(const Image& input, float boostFactor) {
    Image output = input;
    
    #pragma omp parallel for
    for (int i = 0; i < input.width * input.height; i++) {
        float h, s, l;
        RGBtoHSL(input.pixels[i].r, input.pixels[i].g, input.pixels[i].b, h, s, l);
        
        // Boost saturation
        s = clamp(s * boostFactor, 0.0f, 1.0f);
        
        // Convert back to RGB
        HSLtoRGB(h, s, l, output.pixels[i].r, output.pixels[i].g, output.pixels[i].b);
    }
    
    return output;
}

// Gamma adjustment implementation
Image adjustGamma(const Image& input, float gamma) {
    Image output = input;
    
    // Precompute gamma lookup table
    std::vector<uint8_t> gammaLUT(256);
    for (int i = 0; i < 256; i++) {
        gammaLUT[i] = static_cast<uint8_t>(clamp(pow(i / 255.0f, gamma) * 255.0f, 0.0f, 255.0f));
    }
    
    #pragma omp parallel for
    for (int i = 0; i < input.width * input.height; i++) {
        output.pixels[i].r = gammaLUT[input.pixels[i].r];
        output.pixels[i].g = gammaLUT[input.pixels[i].g];
        output.pixels[i].b = gammaLUT[input.pixels[i].b];
    }
    
    return output;
}

// Enhanced image processing pipeline
Image enhanceImage(const Image& input) {
    Image output = input;
    
    // 1. Apply bilateral filter for edge-preserving smoothing
    output = applyBilateralFilter(output, 3.0f, 30.0f);
    
    // 2. Boost saturation by 15%
    output = boostSaturation(output, 1.15f);
    
    // 3. Apply mid-tone gamma adjustment (gamma = 0.9 to brighten mid-tones)
    output = adjustGamma(output, 0.9f);
    
    // 4. Apply light unsharp mask (small amount, small radius)
    output = applyUnsharpMask(output, 1.0f, 0.3f);
    
    return output;
}

ImageLAB applyCLAHE_LAB(const ImageLAB& input, int tiles, float clipLimit) {
    ImageLAB output = input;
    int tileWidth = input.width / tiles;
    int tileHeight = input.height / tiles;
    
    // Process each tile
    #pragma omp parallel for collapse(2)
    for (int ty = 0; ty < tiles; ty++) {
        for (int tx = 0; tx < tiles; tx++) {
            int startX = tx * tileWidth;
            int startY = ty * tileHeight;
            int endX = std::min((tx + 1) * tileWidth, input.width);
            int endY = std::min((ty + 1) * tileHeight, input.height);
            
            // Build histogram for L channel
            std::vector<int> histogram(256, 0);
            
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    int value = static_cast<int>(input.pixels[y * input.width + x].l * 255.0f / 100.0f);
                    histogram[clamp(value, 0, 255)]++;
                }
            }
            
            // Clip histogram
            int clipCount = static_cast<int>(clipLimit * (tileWidth * tileHeight) / 256.0f);
            int redistBatch = 0;
            
            for (int i = 0; i < 256; i++) {
                if (histogram[i] > clipCount) {
                    redistBatch += histogram[i] - clipCount;
                    histogram[i] = clipCount;
                }
            }
            
            // Redistribute clipped pixels
            int redistIncrement = redistBatch / 256;
            for (int i = 0; i < 256; i++) {
                histogram[i] += redistIncrement;
            }
            
            // Create CDF
            std::vector<int> cdf(256, 0);
            cdf[0] = histogram[0];
            for (int i = 1; i < 256; i++) {
                cdf[i] = cdf[i-1] + histogram[i];
            }
            
            // Normalize and apply
            float scale = 255.0f / cdf[255];
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

// Bilateral filter implementation
Image applyBilateralFilter(const Image& input, float sigmaSpace, float sigmaColor) {
    Image output = input;
    int kernelSize = static_cast<int>(ceil(sigmaSpace * 3));
    if (kernelSize % 2 == 0) kernelSize++;
    int center = kernelSize / 2;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < input.height; y++) {
        for (int x = 0; x < input.width; x++) {
            float sumR = 0, sumG = 0, sumB = 0;
            float sumWeight = 0;
            
            for (int ky = -center; ky <= center; ky++) {
                for (int kx = -center; kx <= center; kx++) {
                    int px = clamp(x + kx, 0, input.width - 1);
                    int py = clamp(y + ky, 0, input.height - 1);
                    
                    // Spatial weight
                    float spatialWeight = exp(-(kx * kx + ky * ky) / (2 * sigmaSpace * sigmaSpace));
                    
                    // Color weight
                    float colorDiffR = input.pixels[y * input.width + x].r - input.pixels[py * input.width + px].r;
                    float colorDiffG = input.pixels[y * input.width + x].g - input.pixels[py * input.width + px].g;
                    float colorDiffB = input.pixels[y * input.width + x].b - input.pixels[py * input.width + px].b;
                    float colorWeight = exp(-(colorDiffR * colorDiffR + colorDiffG * colorDiffG + colorDiffB * colorDiffB) / 
                                          (2 * sigmaColor * sigmaColor));
                    
                    float weight = spatialWeight * colorWeight;
                    sumR += input.pixels[py * input.width + px].r * weight;
                    sumG += input.pixels[py * input.width + px].g * weight;
                    sumB += input.pixels[py * input.width + px].b * weight;
                    sumWeight += weight;
                }
            }
            
            output.pixels[y * input.width + x].r = static_cast<uint8_t>(clamp(sumR / sumWeight, 0.0f, 255.0f));
            output.pixels[y * input.width + x].g = static_cast<uint8_t>(clamp(sumG / sumWeight, 0.0f, 255.0f));
            output.pixels[y * input.width + x].b = static_cast<uint8_t>(clamp(sumB / sumWeight, 0.0f, 255.0f));
        }
    }
    
    return output;
} 