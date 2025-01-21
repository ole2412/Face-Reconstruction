#ifndef RENDER_H
#define RENDER_H

#include "bfm.h"
#include <string>
#include "optimizationParameters.h"
#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

void generateAndOutputMesh(BFM* bfm, const std::string& outputFile, const std::string& landmarksFile, cv::Mat image);
void outputMeshToImage(BFM* bfm, const Eigen::Matrix<double, -1, 1>& vertices, const Eigen::VectorXi& triangles, const OptimizationParameters& params, cv::Mat image);
GLFWwindow* setup_initial_rendering_window(int width, int height);

#endif
