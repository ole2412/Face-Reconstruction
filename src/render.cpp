#include "render.h"
#include "costFunctions.h"
#include "optimizationParameters.h"
#include "bfm.h"
#include "optimizer.h"

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>


void outputMeshToObj(const Eigen::Matrix<double, -1, 1>& vertexPosition, const Eigen::VectorXi& faceIndices, const std::string& filename, std::vector<int>& landmarkIndices, bool show_landmarks = false) {
    /**
     * @brief Outputs a 3D mesh to an OBJ file.
     * 
     * This function writes the vertex positions and face indices of a 3D mesh to a file in the OBJ format.
     * Optionally, specific landmark indices can be highlighted if `show_landmarks` is enabled.
     * 
     * @param vertexPosition A matrix containing the 3D coordinates of the vertices.
     *                       Dimensions: (n x 3), where n is the number of vertices.
     * @param faceIndices A vector containing the indices defining the faces of the mesh.
     *                    Each triplet in the vector represents a face (vertex indices are 1-based for OBJ format).
     * @param filename The name of the output OBJ file.
     * @param landmarkIndices A vector of vertex indices representing landmark points.
     *                        These indices will be highlighted in the output if `show_landmarks` is true.
     * @param show_landmarks (Optional) A boolean flag to indicate whether to include and highlight landmarks in the output.
     *                       Default is false.
     */

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

        // red color for landmarks
        if (std::find(landmarkIndices.begin(), landmarkIndices.end(), v) != landmarkIndices.end()) {
            outFile << "v " << x << " " << y << " " << z << " 255 0 0" << std::endl;
            continue;
        }

        outFile << "v " << x << " " << y << " " << z << " 255 255 255" << std::endl;
    }

    long nFaces = faceIndices.size() / 3;
    std::cout << "nFaces: " << nFaces << std::endl;
    for (int f = 0; f < nFaces; f++) {
        int v1 = faceIndices[f + 0 * nFaces];
        int v2 = faceIndices[f + 1 * nFaces];
        int v3 = faceIndices[f + 2 * nFaces];

        outFile << "f " << v1 << " " << v2 << " " << v3 << std::endl;
    }

    outFile.close();
    std::cout << "Mesh successfully written to " << filename << std::endl;
}

//TODO put this in another class?
// For reading our own source landmarks
std::vector<Eigen::Vector2d> readObservedLandmarks(const std::string &filePath)
{
    std::vector<Eigen::Vector2d> observed_landmarks; 

    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return observed_landmarks;
    }

    double x, y;
    // Read pairs of X and Y coordinates 
    while (file >> x >> y)
    {
        observed_landmarks.emplace_back(x, y);
    }

    file.close();

    return observed_landmarks;
}

void applyTransformationsToVertices(Eigen::Matrix<double, -1, 1>& vertexPosition, const Eigen::Vector3d& rotation, const Eigen::Vector3d& translation) {
    /**
     * @brief Applies rotation (using AngleAxis) and translation to vertex positions.
     * 
     * @param vertexPosition A matrix containing the 3D coordinates of the vertices (flattened as a 1D array).
     * @param rotation A vector of rotation angles [rx, ry, rz] in radians.
     * @param translation A translation vector [tx, ty, tz].
     */
    long nVertices = vertexPosition.size() / 3;

    // Create an AngleAxis for the rotation
    Eigen::AngleAxisd angleAxisX(rotation[0], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd angleAxisY(rotation[1], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd angleAxisZ(rotation[2], Eigen::Vector3d::UnitZ());

    // Combined rotation
    Eigen::Matrix3d rotationMatrix = (angleAxisZ * angleAxisY * angleAxisX).toRotationMatrix();

    for (int v = 0; v < nVertices; ++v) {
        Eigen::Vector3d vertex(vertexPosition[3 * v], vertexPosition[3 * v + 1], vertexPosition[3 * v + 2]);

        // Apply rotation and translation
        vertex = rotationMatrix * vertex + translation;

        vertexPosition[3 * v] = vertex[0];
        vertexPosition[3 * v + 1] = vertex[1];
        vertexPosition[3 * v + 2] = vertex[2];
    }
}




void generateAndOutputMesh(BFM* bfm, const std::string& outputFile, const std::string& observedLandmarksPath) {
    /**
     * @brief Generates a 3D mesh using the Basel Face Model and writes it to an OBJ file.
     * 
     * This function generates a 3D mesh using the Basel Face Model and writes it to a file in the OBJ format.
     * The mesh is generated by applying the mean shape and expression to the model, and optionally
     * modifying the shape and expression parameters to create variations.
     * 
     * @param bfm A pointer to the Basel Face Model object.
     * @param outputFile The name of the output OBJ file.
     */
    std::vector<int> landmarkIndices = bfm->get_landmark_indices();

    // TODO 2: write a parameters class where params are stored that we optimize for
    // give that param object to this function

    // Initialize vector to store 3D landmark positions
    std::vector<Eigen::Vector3d> model_landmarks;
    model_landmarks.reserve(landmarkIndices.size());

    // TODO fill the observed_landmarks
    std::vector<Eigen::Vector2d> observed_landmarks = readObservedLandmarks(observedLandmarksPath);;


    //TODO define IMAGE_WIDTH and HEIGHT which is needed for calculating camera intrinsics
    constexpr int IMAGE_WIDTH = 3264;  
    constexpr int IMAGE_HEIGHT = 1836;

    OptimizationParameters params(bfm->get_shape_size(), bfm->get_expr_size(), IMAGE_WIDTH, IMAGE_HEIGHT);


    // Get 3D coordinates of a landmarks vertex from the BFM model
    for (int index : landmarkIndices) {
        Eigen::Vector3d vertex = bfm->get_shape_mean().segment<3>(3 * index) + bfm->get_expr_mean().segment<3>(3 * index);

        // Add PCA contributions for the shape coefficients
        for (int i = 0; i < bfm->get_shape_size(); ++i) {
            double shapeFactor = sqrt(bfm->get_shape_var()[i]) * params.shapeCoefficients[i];
            vertex[0] += shapeFactor * bfm->get_shape_pca()(3 * index, i);
            vertex[1] += shapeFactor * bfm->get_shape_pca()(3 * index + 1, i);
            vertex[2] += shapeFactor * bfm->get_shape_pca()(3 * index + 2, i);
        }

        for (int i = 0; i < bfm->get_expr_size(); ++i) {
            double exprFactor = sqrt(bfm->get_expr_var()[i]) * params.expressionCoefficients[i];
            vertex[0] += exprFactor * bfm->get_expr_pca()(3 * index, i);
            vertex[1] += exprFactor * bfm->get_expr_pca()(3 * index + 1, i);
            vertex[2] += exprFactor * bfm->get_expr_pca()(3 * index + 2, i);
        }

        model_landmarks.push_back(vertex);
    }

    // Print initial parameters for debugging
    params.printParameters();
    Optimizer optimizer;

    // Optimize landmarks
    optimizer.optimizeSparse(params, observed_landmarks, model_landmarks, params.image_width, params.image_height);


    // Print parameters after optimization
    std::cout << "Parameters after Sparse Optimization" << std::endl;
    params.printParameters();


    // TODO 3 (independet of previous tasks!): apply color from the model to the mesh 

    // TODO 4 (independet of previous tasks!): write a seperate function to transform vertexPositions to 2D image coordinates

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
            double shapeFactor = sqrt(shape_var[i]) * params.shapeCoefficients[i];
            vertexPosition[3 * v] += shapeFactor * shape_pca(3 * v, i);
            vertexPosition[3 * v + 1] += shapeFactor * shape_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += shapeFactor * shape_pca(3 * v + 2, i);
        }

        for (int i = 0; i < expr_size; i++) {
            double exprFactor = sqrt(expr_var[i]) * params.expressionCoefficients[i];
            vertexPosition[3 * v] += exprFactor * expr_pca(3 * v, i);
            vertexPosition[3 * v + 1] += exprFactor * expr_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += exprFactor * expr_pca(3 * v + 2, i);
        }
    }



    // Apply transformations (rotation and translation)
    applyTransformationsToVertices(vertexPosition, params.rotation, params.translation);


    Eigen::VectorXi triangles = bfm->get_current_faces();


    outputMeshToObj(vertexPosition, triangles, outputFile, landmarkIndices, true);

    outputMeshToImage(vertexPosition, triangles, outputFile + ".jpg", IMAGE_WIDTH, IMAGE_HEIGHT);
}

void outputMeshToImage(const Eigen::Matrix<double, -1, 1>& vertices, const Eigen::VectorXi& triangles, const std::string& outputFile, int width, int height) {
    /*
    * @brief Renders a 3D mesh to an image.
    *
    * This function renders a 3D mesh to an image using the given vertex positions and triangles.
    * The rendered image is saved to a file.
    * 
    * @param vertices A matrix containing the 3D coordinates of the vertices.
    * @param triangles A vector containing the indices defining the faces of the mesh.
    * @param outputFile The name of the output image file.
    * @param image_width The width of the output image.
    * @param image_height The height of the output image.  
    */
    // 3D coordinates are given through vertices, transform them to 2D, plot on image and save

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window;
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Render", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create GLFW window" << std::endl;
        return;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    // Set up the viewport
    glViewport(0, 0, width, height);

    // Set up the modelview matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Clear the color buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Set up the color
    glColor3f(1.0f, 1.0f, 1.0f);

    // Draw the mesh
    glBegin(GL_TRIANGLES);

    long nVertices = vertices.size() / 3;
    for (int v = 0; v < nVertices; v++) {
        double x = vertices[3 * v];
        double y = vertices[3 * v + 1];
        double z = vertices[3 * v + 2];

        glVertex2d(x, y);
        // set color to red
        glColor3f(255, 0, 0);
    }

    glEnd();

    // Read the pixels from the frame buffer
    unsigned char* pixels = new unsigned char[3 * width * height];
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    // Create an OpenCV image
    cv::Mat image(height, width, CV_8UC3, pixels);

    // Flip the image
    cv::flip(image, image, 0);

    // Save the image
    cv::imwrite(outputFile, image);

    // Terminate GLFW
    glfwTerminate();

    delete[] pixels;
    
    return;
}