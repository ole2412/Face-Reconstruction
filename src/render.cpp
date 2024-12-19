#include "render.h"
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "bfm.h"

void outputMeshToObj(const Eigen::Matrix<double, -1, 1>& vertexPosition, const Eigen::VectorXi& faceIndices, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    long nVertices = vertexPosition.size() / 3;
    std::cout << "nVertices: " << nVertices << std::endl;
    for (int v = 0; v < nVertices; v++) {
        double x = vertexPosition[3 * v];
        double y = vertexPosition[3 * v + 1];
        double z = vertexPosition[3 * v + 2];
        outFile << "v " << x << " " << y << " " << z << std::endl;
    }

    long nFaces = faceIndices.size() / 3;
    std::cout << "nFaces: " << nFaces << std::endl;
    for (int f = 0; f < nFaces; f++) {
        int v1 = faceIndices[f + 0 * nFaces] + 1;
        int v2 = faceIndices[f + 1 * nFaces] + 1;
        int v3 = faceIndices[f + 2 * nFaces] + 1;
        outFile << "f " << v1 << " " << v2 << " " << v3 << std::endl;
    }

    outFile.close();
    std::cout << "Mesh successfully written to " << filename << std::endl;
}

void generateAndOutputMesh(BFM* bfm, const std::string& outputFile) {
    // TODO 1.1: write a parameters class where params are stored that we optimize for
    // give that param object to this function

    // TODO 1.2: apply fov, translation, rotation to the results

    // TODO 2.1: (independet of previous tasks!): write function to transform geometry to 2D

    // TODO 2.2: add input_image as param and project the 2D geometry to the image
    // store (return and store then) the image
    
    Eigen::VectorXd shape_var = bfm->get_shape_var();
    Eigen::VectorXd expr_var = bfm->get_expr_var();

    Eigen::MatrixXd shape_pca = bfm->get_shape_pca();
    Eigen::MatrixXd expr_pca = bfm->get_expr_pca();

    Eigen::MatrixXd shape_mean = bfm->get_shape_mean();
    Eigen::MatrixXd expr_mean = bfm->get_expr_mean();

    long nVertices = bfm->get_no_vertices();
    int color_shape_size = bfm->get_shape_size();
    int expr_size = bfm->get_expr_size();

    Eigen::Matrix<double, -1, 1> vertexPosition = shape_mean + expr_mean;

    for (int v = 0; v < nVertices; v++) {
        for (int i = 0; i < color_shape_size; i++) {
            double shapeFactor = sqrt(shape_var[i]) * 0;
            vertexPosition[3 * v] += shapeFactor * shape_pca(3 * v, i);
            vertexPosition[3 * v + 1] += shapeFactor * shape_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += shapeFactor * shape_pca(3 * v + 2, i);
        }

        for (int i = 0; i < expr_size; i++) {
            double exprFactor = sqrt(expr_var[i]) * 0;
            vertexPosition[3 * v] += exprFactor * expr_pca(3 * v, i);
            vertexPosition[3 * v + 1] += exprFactor * expr_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += exprFactor * expr_pca(3 * v + 2, i);
        }
    }

    Eigen::VectorXi triangles = bfm->get_current_faces();

    outputMeshToObj(vertexPosition, triangles, outputFile);
}