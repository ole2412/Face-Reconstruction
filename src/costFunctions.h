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
    bool operator()(T const* translation, T const* rotation, T const* fov, T* residuals) const {

        T rotated_point[3];
        //ceres::AngleAxisRotatePoint(rotation, model_landmark.data(), rotated_point);

        T landmark_array[3] = {T(model_landmark[0]), T(model_landmark[1]), T(model_landmark[2])};
        ceres::AngleAxisRotatePoint(rotation, landmark_array, rotated_point);

        // Apply translation
        rotated_point[0] += translation[0];
        rotated_point[1] += translation[1];
        rotated_point[2] += translation[2];


        // Compute focal lengths fx and fy based on the image dimensions and field of view (fov)
        // from https://flightmare.readthedocs.io/en/latest/first_steps/faq.html
        //try ceres::tan((fov[0] * T(0.5));

        //fov times pi durch 180
        // tan of rad /2
        //T fx = T(image_width) * T(0.5) / ceres::tan(((fov[0]) * T(M_PI)) / T(180.0) * T(0.5));
        //T fy = T(image_height) * T(0.5) / ceres::tan(((fov[0]) * T(M_PI)) / T(180.0) * T(0.5));
        
        //T fx = (T(image_width) * T(0.5)) / ceres::tan((fov[0] * T(0.5)));
        //T fy = (T(image_height) * T(0.5)) / ceres::tan((fov[0] * T(0.5)));


        // T X_temp = rotated_point[0] * fx + rotated_point[2] * T(image_width / 2);
        // T Y_temp = rotated_point[1] * fy + rotated_point[2] * T(image_height / 2);
        // // Perform perspective projection (Dehomogenization)
        // T projected_x = (X_temp / rotated_point[2]);
        // T projected_y = (Y_temp / rotated_point[2]);

        // Perform perspective projection

        // T projected_x = (rotated_point[0] * fx / rotated_point[2]) + T(image_width / 2.0);
        // T projected_y = (rotated_point[1] * fy / rotated_point[2]) + T(image_height / 2.0);


        // Project the 3D point to normalized image coordinates (camera space)
        T x_normalized = - rotated_point[0] / rotated_point[2];
        T y_normalized = - rotated_point[1] / rotated_point[2];

        // Step 4: Convert normalized coordinates to pixel coordinates
        const T& focal = fov[0];

        T projected_x = (x_normalized * focal);
        T projected_y = (y_normalized * focal);
        
        //std::cout << "model landmark 2d: " << projected_x << ","<< projected_y << std::endl;

        // Compute residuals (error) L2 norm, ceres handles the summation and squaring
        residuals[0] = projected_x - T(observed_landmark[0]);
        residuals[1] = projected_y - T(observed_landmark[1]);

        //std::cout << "residual x: " << residuals[0] << std::endl;

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
struct RegularizationColorShapeCostFunction {
    RegularizationColorShapeCostFunction(const Eigen::VectorXd& shape_coefficients, const Eigen::VectorXd& color_coefficients, int shape_color_size)
        : shape_coefficients(shape_coefficients), color_coefficients(color_coefficients), shape_color_size(shape_color_size) {}

    template <typename T>
    bool operator()(T const* shape_coefficients_ptr, T const* color_coefficients_ptr, T* residuals) const {

        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> shape_coefficients(shape_coefficients_ptr, shape_color_size);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> color_coefficients(color_coefficients_ptr, shape_color_size);


        T shape_mean = shape_coefficients.mean();
        T shape_std = ceres::sqrt((shape_coefficients.array() - shape_mean).square().sum() / T(shape_coefficients.size()));

        T color_mean = color_coefficients.mean();
        T color_std = ceres::sqrt((color_coefficients.array() - color_mean).square().sum() / T(color_coefficients.size()));

        // Avoid division by zero for standard deviation
        shape_std = (shape_std > T(1e-6)) ? shape_std : T(1.0);
        color_std = (color_std > T(1e-6)) ? color_std : T(1.0);

        for (size_t i = 0; i < shape_color_size; i++) {
            residuals[i] = shape_coefficients[i] / shape_std + color_coefficients[i] / color_std;
        } 
        return true;
    }


    static ceres::CostFunction* Create(const Eigen::VectorXd& shape_coefficients, const Eigen::VectorXd& color_coefficients, int shape_color_size) {
        return new ceres::AutoDiffCostFunction<RegularizationColorShapeCostFunction, 199, 199, 199>( 
            new RegularizationColorShapeCostFunction(shape_coefficients, color_coefficients, shape_color_size));
    }


    Eigen::VectorXd shape_coefficients;
    Eigen::VectorXd color_coefficients;
    int shape_color_size;
};


} // namespace CostFunctions

#endif 