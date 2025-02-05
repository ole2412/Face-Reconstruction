#include <sstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "src/bfm.h"
#include "src/render.h"
#include "src/optimizationParameters.h"
#include "src/optimizer.h"

int main(int argc, char *argv[])
{
    // idea: later have two modes:
    // 1. "main <image_name>" -> self reeanactment
    // 2. "main <image_name1> <image_name2>" -> expression params from 1 and put on 2
    if (argc < 2)
    {
        std::cout << "Usage: main <image_name>" << std::endl;
        return -1;
    }

    std::string image_name = argv[1];
    std::string image_path = "../datasets/" + image_name + ".jpg";
    std::string landmark_path = "../datasets/" + image_name + "_landmarks.anl";
    std::string output_path = "../output/" + image_name + "_output.jpg";

    BFM bfm = BFM();
    cv::Mat image = cv::imread(image_path);

    // Optimize
    std::vector<int> landmarkIndices = bfm.get_landmark_indices();
    std::vector<Eigen::Vector2d> observed_landmarks = readObservedLandmarks(landmark_path);
    OptimizationParameters params(&bfm, bfm.get_shape_size(), bfm.get_expr_size(), bfm.get_color_size(), image.cols, image.rows);
    // Optimizer optimizer;
    // Eigen::MatrixXd model_colors = bfm.extractVertexColors(params);
    // Eigen::MatrixXd observed_colors = bfm.get_observed_colors(image);
    // optimizer.optimize(params, observed_landmarks, landmarkIndices, params.image_width, params.image_height, observed_colors, model_colors, image);

    // get 3D vertex positions
    Eigen::Matrix<double, -1, 1> vertexPosition;
    bfm.get_vertex_positions(vertexPosition, params.shapeCoefficients, params.expressionCoefficients);

    // store to mesh
    // applyTransformationsAndProjectionToVertices(vertexPosition, params.rotation, params.translation,
    //                                             params.image_width, params.image_height,
    //                                             params.fov);
    Eigen::VectorXi triangles = bfm.get_current_faces();

    // TODO: replace this with cpp version of rendering
    // outputMeshToObj(vertexPosition, triangles, "../output/" + image_name + "_all_vertices.obj", landmarkIndices, false);
    outputMeshToObj(vertexPosition, triangles, "../output/" + image_name + ".obj", landmarkIndices, false);
    outputMeshToPly(triangles, "../output/" + image_name + ".ply", landmarkIndices, image, params, true);
    // store and show the output image
    // cv::imwrite(output_path, image);
    // cv::waitKey(0);

    return 0;
}
