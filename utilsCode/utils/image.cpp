#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>

#include "image.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// static std::vector<float> loadImage(const std::string& filename, int sizeX = 256, int sizeY = 256) {
static cv::Mat loadImage(const std::string& filename) {
    std::cout << "Loading image: " << filename << "\n";
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cout << "No image found.";
        return cv::Mat();
    }

    // convert from BGR to RGB
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // normalize image to [0, 1]
    image.convertTo(image, CV_32FC3, 1.0f / 255.0f, 0);
    return image;
}

static cv::Mat fixedImage(cv::Mat image){
    // if index < 0 index = 0 or if index > 1 index = 1
    cv::Mat fixedImage = cv::Mat::zeros(image.size(), image.type());
    cv::Mat mask = (image < 0);
    fixedImage.setTo(0, mask);
    mask = (image > 1);
    fixedImage.setTo(1, mask);
    return fixedImage;
}

static std::vector<std::vector<std::vector<float>>> mat2vec(
    const cv::Mat& mat) {
    std::vector<std::vector<std::vector<float>>> vec(mat.channels(), std::vector<std::vector<float>>(mat.cols, std::vector<float>(mat.rows)));
    for (int c = 0; c < mat.channels(); ++c) {
        for (int i = 0; i < mat.cols; ++i) {
            for (int j = 0; j < mat.rows; ++j) {
                vec[c][i][j] = mat.at<cv::Vec3f>(j, i)[c];  // note the order of indices
            }
        }
    }
    return vec;
}


static std::vector<std::vector<std::vector<float>>> transposeImage(
    std::vector<std::vector<std::vector<float>>> image, std::vector<int> bf){
    
    // transpose image -> bf order
    // check length of bf
    if (bf.size() != 3) {
        std::cout << "Error: bf must have length 3\n";
        return std::vector<std::vector<std::vector<float>>>();
    }
    std::vector<std::vector<std::vector<float>>> transposeImage;
    for (int i = 0; i < image[0].size(); ++i) {
        std::vector<std::vector<float>> channel;
        for (int j = 0; j < image[0][0].size(); ++j) {
            std::vector<float> row;
            for (int k = 0; k < image.size(); ++k) {
                row.emplace_back(image[k][i][j]);
            }
            channel.emplace_back(row);
        }
        transposeImage.emplace_back(channel);
    }
    return transposeImage;
}