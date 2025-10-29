#ifndef RENDER_H
#define RENDER_H

#include "bfm.h"
#include <string>
#include "optimizationParameters.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

void outputMeshToImage(BFM *bfm, const Eigen::Matrix<double, -1, 1> &vertices, const Eigen::VectorXi &triangles, const OptimizationParameters &params, cv::Mat image);
void outputMeshToObj(const Eigen::Matrix<double, -1, 1> &vertexPosition, const Eigen::VectorXi &faceIndices, const std::string &filename, std::vector<int> &landmarkIndices, OptimizationParameters &params, bool show_landmarks = false);
void outputMeshToPly(const Eigen::VectorXi &faceIndices, const std::string &filename, std::vector<int> &landmarkIndices, const cv::Mat &image, OptimizationParameters &params, bool show_landmarks = false);
void bfm_create_obj(BFM bfm, const OptimizationParameters params);
void applyTransformationsAndProjectionToVertices(Eigen::Matrix<double, -1, 1> &vertexPosition, const Eigen::Vector3d &rotation, const Eigen::Vector3d &translation, int image_width, int image_height, double fov);
std::vector<Eigen::Vector2d> readObservedLandmarks(const std::string &filePath);
void outputMeshToLandmarks(const Eigen::Matrix<double, -1, 1>& vertexPosition, const Eigen::VectorXi& faceIndices, const std::string& filename, std::vector<int>& landmarkIndices, bool show_landmarks);

#endif
