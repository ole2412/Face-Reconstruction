#ifndef OPTIMIZATION_PARAMETERS_H
#define OPTIMIZATION_PARAMETERS_H

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <ceres/ceres.h>

struct OptimizationParameters {

    // Camera intrinsics
    double fov; 
    int image_width;  // Image width in pixels
    int image_height; // Image height in pixels

    // Transformation parameters
    Eigen::Vector3d translation; // Translation vector [tx, ty, tz]
    Eigen::Vector3d rotation;    // Rotation vector [rx, ry, rz] in radians

    // Model parameters
    Eigen::VectorXd shapeCoefficients;    // Shape PCA coefficients
    Eigen::VectorXd expressionCoefficients; // Expression PCA coefficients

    // Landmark correspondences
    Eigen::VectorXd landmarkProjections; // Projected 2D coordinates of landmarks

    OptimizationParameters(int shapeSize, int exprSize, int width, int height)
        : fov(60.0),                          // Initialize FOV to a default value (60 degrees)
          image_width(width),                 // Initialize image width
          image_height(height),               // Initialize image height
          translation(Eigen::Vector3d::Zero()),
          rotation(Eigen::Vector3d::Zero()),
          shapeCoefficients(Eigen::VectorXd::Zero(shapeSize)),
          expressionCoefficients(Eigen::VectorXd::Zero(exprSize)),
          landmarkProjections(Eigen::VectorXd::Zero(68 * 2)) // 68 landmarks, x and y coordinates
    {}

    /**
     * @brief Resets all parameters to their default values.
     */
    void resetParameters() {
        fov = 60.0;
        image_width = 0;
        image_height = 0;
        translation.setZero();
        rotation.setZero();
        shapeCoefficients.setZero();
        expressionCoefficients.setZero();
        landmarkProjections.setZero();
    }

    /**
     * @brief Prints the current parameter values
     */
    void printParameters() const {
        std::cout << "Field of View: " << fov << std::endl;
        std::cout << "Image Width: " << image_width << std::endl;
        std::cout << "Image Height: " << image_height << std::endl;
        std::cout << "Translation: " << translation.transpose() << std::endl;
        std::cout << "Rotation: " << rotation.transpose() << std::endl;
        std::cout << "Shape Coefficients: " << shapeCoefficients.transpose() << std::endl;
        std::cout << "Expression Coefficients: " << expressionCoefficients.transpose() << std::endl;
        std::cout << "Landmark Projections: " << landmarkProjections.transpose() << std::endl;
    }
};

#endif // OPTIMIZATION_PARAMETERS_H