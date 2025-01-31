#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "optimizationParameters.h"
#include <ceres/ceres.h>           
#include <Eigen/Dense>  
// include ceres rotation.h
#include <ceres/rotation.h>
#include "bfm.h"

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

        // T rotated_point[3];
        // //ceres::AngleAxisRotatePoint(rotation, model_landmark.data(), rotated_point);

        // T landmark_array[3] = {T(model_landmark[0]), T(model_landmark[1]), T(model_landmark[2])};
        // ceres::AngleAxisRotatePoint(rotation, landmark_array, rotated_point);

        // // Apply translation
        // rotated_point[0] += translation[0];
        // rotated_point[1] += translation[1];
        // rotated_point[2] += translation[2];

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
        //T x_normalized = - rotated_point[0] / rotated_point[2];
        //T y_normalized = - rotated_point[1] / rotated_point[2];

        // Step 4: Convert normalized coordinates to pixel coordinates
        //const T& focal = fov[0];

        //T projected_x = (x_normalized * focal);
        //T projected_y = (y_normalized * focal);
        
        //std::cout << "model landmark 2d: " << projected_x << ","<< projected_y << std::endl;

        // Compute residuals (error) L2 norm, ceres handles the summation and squaring
        //residuals[0] = projected_x - T(observed_landmark[0]);
        //residuals[1] = projected_y - T(observed_landmark[1]);

        residuals[0] = rotated_point[0] - T(observed_landmark[0]);
        residuals[1] = rotated_point[1] - T(observed_landmark[1]);

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
struct RegularizationShapeCostFunction {
    RegularizationShapeCostFunction(const Eigen::VectorXd& shape_coefficients, BFM& bfm)
        : shape_coefficients(shape_coefficients), bfm(bfm) {}

    template <typename T>
    bool operator()(T const* shape_coefficients, T* residuals) const {

        for (size_t i = 0; i < bfm.color_shape_size; i++) {

            T shape_std = ceres::sqrt(T(bfm.shape_var[i]));
            residuals[i] = shape_coefficients[i] / shape_std;
        } 
        return true;
    }


    static ceres::CostFunction* Create(const Eigen::VectorXd& shape_coefficients, BFM& bfm) {
        return new ceres::AutoDiffCostFunction<RegularizationShapeCostFunction, 199, 199>( 
            new RegularizationShapeCostFunction(shape_coefficients, bfm));
    }

    Eigen::VectorXd shape_coefficients;
    BFM bfm;
};

struct RegularizationColorCostFunction {
    RegularizationColorCostFunction(const Eigen::VectorXd& color_coefficients, BFM& bfm)
        : color_coefficients(color_coefficients), bfm(bfm) {}

    template <typename T>
    bool operator()(T const* color_coefficients, T* residuals) const {


        for (size_t i = 0; i < bfm.color_shape_size; i++) {
            T color_std = ceres::sqrt(T(bfm.color_var[i]));
            residuals[i] = color_coefficients[i] / color_std;
        } 
        return true;
    }


    static ceres::CostFunction* Create(const Eigen::VectorXd& color_coefficients, BFM& bfm) {
        return new ceres::AutoDiffCostFunction<RegularizationColorCostFunction, 199, 199>( 
            new RegularizationColorCostFunction(color_coefficients, bfm));
    }

    Eigen::VectorXd color_coefficients;
    BFM bfm;
};

struct RegularizationExprCostFunction {
    RegularizationExprCostFunction(const Eigen::VectorXd& expr_coefficients, BFM& bfm)
        : expr_coefficients(expr_coefficients), bfm(bfm) {}

    template <typename T>
    bool operator()(T const* expr_coefficients, T* residuals) const {


        for (size_t i = 0; i < bfm.expr_size; i++) {
            T expr_std = ceres::sqrt(T(bfm.expr_var[i]));
            residuals[i] = expr_coefficients[i] / expr_std;
        } 
        return true;
    }


    static ceres::CostFunction* Create(const Eigen::VectorXd& expr_coefficients, BFM& bfm) {
        return new ceres::AutoDiffCostFunction<RegularizationColorCostFunction, 100, 100>( 
            new RegularizationColorCostFunction(expr_coefficients, bfm));
    }

    Eigen::VectorXd expr_coefficients;
    BFM bfm;
};



} // namespace CostFunctions

#endif 