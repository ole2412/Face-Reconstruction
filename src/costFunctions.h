#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "optimizationParameters.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
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

        // T rotated_point[3] = {T(0), T(0), T(0)};

        // std::cout << "rotation: " << rotation << std::endl;

        // // Convert Euler angles (rotation) to a rotation matrix
        // T rotation_matrix[9]; // 3x3 rotation matrix
        // ceres::EulerAnglesToRotationMatrix<T>(rotation, 3, rotation_matrix);

        // // Rotate the model landmark
        // for (int i = 0; i < 3; ++i) {
        //     rotated_point[i] = T(0);
        //     for (int j = 0; j < 3; ++j) {
        //         rotated_point[i] += rotation_matrix[i * 3 + j] * model_landmark_array[j];
        //     }
        // }
    // Convert Euler angles (rotation) to a rotation matrix using Ceres
        T rotation_matrix[9];
        T eulerAngles[3] = {rotation[0], rotation[1], rotation[2]};
        ceres::EulerAnglesToRotationMatrix(eulerAngles, 3, rotation_matrix);

        // Populate Eigen matrix for rotation
        Eigen::Matrix<T, 3, 3> rotationMatrix;
        rotationMatrix << rotation_matrix[0], rotation_matrix[1], rotation_matrix[2],
                        rotation_matrix[3], rotation_matrix[4], rotation_matrix[5],
                        rotation_matrix[6], rotation_matrix[7], rotation_matrix[8];

        // Convert model landmark to Eigen vector
        Eigen::Matrix<T, 3, 1> model_landmark_vec;
        model_landmark_vec << T(model_landmark[0]), T(model_landmark[1]), T(model_landmark[2]);

        // Rotate the model landmark
        Eigen::Matrix<T, 3, 1> rotated_point = rotationMatrix * model_landmark_vec;

        // Apply translation
        rotated_point[0] += translation[0];
        rotated_point[1] += translation[1];
        rotated_point[2] += translation[2];

        // Compute focal lengths fx and fy based on the image dimensions and field of view (fov)
        // from https://flightmare.readthedocs.io/en/latest/first_steps/faq.html
        T fx = T(image_width) * T(0.5) / ceres::tan(T(0.5) * *fov * T(M_PI) / T(180.0));
        T fy = T(image_height) * T(0.5) / ceres::tan(T(0.5) * *fov * T(M_PI) / T(180.0));

        T X_temp = rotated_point[0] * fx + rotated_point[2] * T(image_width / 2);
        T Y_temp = rotated_point[1] * fy + rotated_point[2] * T(image_height / 2);
        // Perform perspective projection (Dehomogenization)
        T projected_x = (X_temp / rotated_point[2]);
        T projected_y = (Y_temp / rotated_point[2]);

        //std::cout << "model landmark 2d: " << projected_x << ","<< projected_y << std::endl;

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