#ifndef OPTIMIZATION_PARAMETERS_H
#define OPTIMIZATION_PARAMETERS_H

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <ceres/ceres.h>
#include "bfm.h"
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
    Eigen::VectorXd colorCoefficients; // Expression PCA coefficients
    int colorShapeSize;
    int exprSize;

    // Landmark correspondences
    Eigen::VectorXd landmarkProjections; // Projected 2D coordinates of landmarks
    BFM* bfm;
    OptimizationParameters(BFM* bfm, int shapeSize, int exprSize, int colorSize, int width, int height)
        : bfm(bfm),
          fov(45),                          // Initialize FOV to a default value 
          image_width(width),                 // Initialize image width
          image_height(height),               // Initialize image height
          translation(Eigen::Vector3d(0, 0, -400)), //Eigen::Vector3d(0, 0, -400)
          rotation(Eigen::Vector3d::Zero()),
          shapeCoefficients(Eigen::VectorXd::Zero(shapeSize)),
          expressionCoefficients(Eigen::VectorXd::Zero(exprSize)),
          colorCoefficients(Eigen::VectorXd::Zero(colorSize)),
          colorShapeSize(shapeSize),
          exprSize(exprSize),
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
        colorCoefficients.setZero();
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
        std::cout << "Color Coefficients: " << colorCoefficients.transpose() << std::endl;
        std::cout << "Landmark Projections: " << landmarkProjections.transpose() << std::endl;
    }
};

#endif // OPTIMIZATION_PARAMETERS_H