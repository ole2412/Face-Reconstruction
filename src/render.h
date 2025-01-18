#ifndef RENDER_H
#define RENDER_H

#include "bfm.h"
#include <string>

void generateAndOutputMesh(BFM* bfm, const std::string& outputFile, const std::string& landmarksFile);
void outputMeshToImage(const Eigen::Matrix<double, -1, 1>& vertices, const Eigen::VectorXi& triangles, const std::string& outputFile, int image_width, int image_height);

#endif
