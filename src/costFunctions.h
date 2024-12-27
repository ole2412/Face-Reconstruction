#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "optimizationParameters.h"
#include <ceres/ceres.h>           
#include <Eigen/Dense>  

namespace CostFunctions {

/**
 * @brief Cost function for the sparse term (E_sparse).
 * 
 * Penalizes differences between projected 3D model landmarks and corresponding 2D image landmarks.
 */
struct SparseTermCostFunction {
    SparseTermCostFunction(const Eigen::Vector2d& observed_landmark, const Eigen::Vector3d& model_landmark, int image_width, int image_height)
        : observed_landmark(observed_landmark), model_landmark(model_landmark), image_width(image_width), image_height(image_height) {}

    template <typename T>
    bool operator()(const T* const translation, const T* const rotation, const T* const fov, T* residuals) const {
        // Rotate the model landmark using the rotation vector
        T rotated_point[3];
        //ceres::AngleAxisRotatePoint(rotation, model_landmark.data(), rotated_point);

        // Apply translation
        rotated_point[0] += translation[0];
        rotated_point[1] += translation[1];
        rotated_point[2] += translation[2];

        // Compute focal lengths fx and fy based on the image dimensions and field of view (fov)
        // from https://flightmare.readthedocs.io/en/latest/first_steps/faq.html
        T fx = T(image_width) * T(0.5) / ceres::tan(T(0.5) * *fov * T(M_PI) / T(180.0));
        T fy = T(image_height) * T(0.5) / ceres::tan(T(0.5) * *fov * T(M_PI) / T(180.0));

        // Perform perspective projection (Dehomogenization)
        T projected_x = (rotated_point[0] / rotated_point[2]) * fx + T(image_width / 2);
        T projected_y = (rotated_point[1] / rotated_point[2]) * fy + T(image_height / 2);

        // Compute residuals (error) L2 norm, ceres handles the summation and squaring
        residuals[0] = projected_x - T(observed_landmark[0]);
        residuals[1] = projected_y - T(observed_landmark[1]);

        return true;
    }


    static ceres::CostFunction* Create(const Eigen::Vector2d& observed_landmark, const Eigen::Vector3d& model_landmark, int image_width, int image_height) {
        return new ceres::AutoDiffCostFunction<SparseTermCostFunction, 2, 3, 3, 1>( //2D error, 3D for translation, 3D rotation, 1D FOV
            new SparseTermCostFunction(observed_landmark, model_landmark, image_width, image_height));
    }

    Eigen::Vector2d observed_landmark; // Observed 2D landmark from our source image
    Eigen::Vector3d model_landmark;    // Corresponding 3D model landmark in the BFM model
    int image_width;
    int image_height;
};

/**
 * @brief Placeholder for dense term (E_dense) cost function.
 * 
 * TODO implement
 */
struct DenseTermCostFunction {
    // TODO: Add implementation
};

/**
 * @brief Placeholder for regularization term (E_reg) cost function.
 * 
 */
struct RegularizationTermCostFunction {
    // TODO: Add implementation
};


} // namespace CostFunctions

#endif 