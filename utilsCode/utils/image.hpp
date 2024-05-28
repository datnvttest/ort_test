#pragma once
#include <opencv2/core.hpp>

class image {
    public:
        // static std::vector<float> loadImage;
        static cv::Mat loadImage;
        static cv::Mat fixedImage;
        static std::vector<std::vector<std::vector<float>>> mat2vec;
        static std::vector<std::vector<std::vector<float>>> transposeImage;
};