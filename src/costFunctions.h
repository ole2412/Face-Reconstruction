#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include "optimizationParameters.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>
// include ceres rotation.h
#include <ceres/rotation.h>
#include "bfm.h"

namespace CostFunctions
{

    /**
     * @brief Cost function for the sparse term (E_sparse).
     *
     * Penalizes differences between projected 3D model landmarks and corresponding 2D image landmarks.
     * The cost function is defined as:
     * E_sparse = ||P(R * (S * v + t)) - u||^2
     * where:
     * - P is the perspective projection matrix
     * - R is the rotation matrix
     * - S is the shape PCA matrix
     * - v is the 3D landmark position
     * - t is the translation vector
     * - u is the observed 2D landmark
     */
    struct SparseTermCostFunction
    {
        SparseTermCostFunction(const Eigen::Vector2d &observed_landmark, int landmark_index, BFM &bfm, int image_width, int image_height)
            : observed_landmark(observed_landmark), landmark_index(landmark_index), bfm(bfm), image_width(image_width), image_height(image_height) {}

        template <typename T>
        bool operator()(const T *translation, const T *rotation, const T *fov, const T *shape_coefficients, const T *expression_coefficients, T *residuals) const
        {
            // Compute the 3D landmark position inside the cost function
            // TODO it gets shape mean but maybe expr mean is also needed.
            Eigen::Matrix<T, 3, 1> vertex;
            vertex << T(bfm.get_shape_mean()(3 * landmark_index)),
                T(bfm.get_shape_mean()(3 * landmark_index + 1)),
                T(bfm.get_shape_mean()(3 * landmark_index + 2));

            // Apply shape PCA
            for (int i = 0; i < bfm.get_shape_size(); ++i)
            {
                T shapeFactor = ceres::sqrt(T(bfm.get_shape_var()[i])) * shape_coefficients[i];
                vertex[0] += shapeFactor * T(bfm.get_shape_pca()(3 * landmark_index, i));
                vertex[1] += shapeFactor * T(bfm.get_shape_pca()(3 * landmark_index + 1, i));
                vertex[2] += shapeFactor * T(bfm.get_shape_pca()(3 * landmark_index + 2, i));
            }

            // Apply expression PCA
            for (int i = 0; i < bfm.get_expr_size(); ++i)
            {
                T exprFactor = ceres::sqrt(T(bfm.get_expr_var()[i])) * expression_coefficients[i];
                vertex[0] += exprFactor * T(bfm.get_expr_pca()(3 * landmark_index, i));
                vertex[1] += exprFactor * T(bfm.get_expr_pca()(3 * landmark_index + 1, i));
                vertex[2] += exprFactor * T(bfm.get_expr_pca()(3 * landmark_index + 2, i));
            }

            T rotation_matrix[9];
            T eulerAngles[3] = {rotation[0], rotation[1], rotation[2]};
            ceres::EulerAnglesToRotationMatrix(eulerAngles, 3, rotation_matrix);

            // Populate Eigen matrix for rotation
            Eigen::Matrix<T, 3, 3> rotationMatrix;
            rotationMatrix << rotation_matrix[0], rotation_matrix[1], rotation_matrix[2],
                rotation_matrix[3], rotation_matrix[4], rotation_matrix[5],
                rotation_matrix[6], rotation_matrix[7], rotation_matrix[8];

            // Rotate the model landmark
            Eigen::Matrix<T, 3, 1> rotated_point = rotationMatrix * vertex;

            // Apply translation
            rotated_point[0] += translation[0];
            rotated_point[1] += translation[1];
            rotated_point[2] += translation[2];

            T aspect_ratio = T(image_width) / T(image_height);
            T near_plane = T(0.1);
            T far_plane = T(10000.0);

            // Perspective projection matrix elements
            T tan_half_fov = ceres::tan(fov[0] / T(2));
            T proj_00 = T(1) / (aspect_ratio * tan_half_fov);
            T proj_11 = T(1) / tan_half_fov;
            T proj_22 = -(far_plane + near_plane) / (far_plane - near_plane);
            T proj_32 = -(T(2) * far_plane * near_plane) / (far_plane - near_plane);

            T w = proj_22 * rotated_point[2] + proj_32;
            T x_ndc = proj_00 * rotated_point[0] / w;
            T y_ndc = proj_11 * rotated_point[1] / w;

            // Convert to image space
            T projected_x = (x_ndc + T(1)) * T(0.5) * T(image_width);
            T projected_y = T(image_height) - ((y_ndc + T(1)) * T(0.5) * T(image_height));

            residuals[0] = projected_x - T(observed_landmark[0]);
            residuals[1] = projected_y - T(observed_landmark[1]);

            // std::cout << "residual x: " << residuals[0] << std::endl;

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector2d &observed_landmark, int landmark_index, BFM &bfm, int image_width, int image_height)
        {
            return new ceres::AutoDiffCostFunction<SparseTermCostFunction, 2, 3, 3, 1, 199, 100>( // 2D error, 3D for translation, 3D rotation, 1D FOV
                new SparseTermCostFunction(observed_landmark, landmark_index, bfm, image_width, image_height));
        }

        Eigen::Vector2d observed_landmark; // Observed 2D landmark from our source image
        int landmark_index;
        BFM &bfm;
        int image_width;
        int image_height;
    };

    /**
     * @brief Placeholder for dense term (E_dense) cost function.
     *
     * TODO implement
     */
    struct DenseTermCostFunction
    {
        // TODO: Add implementation
    };

    struct ColorCostFunction
    {
        ColorCostFunction(const Eigen::Vector3d &observed_color, const Eigen::Vector3d &model_color)
            : observed_color(observed_color), model_color(model_color) {}

        template <typename T>
        bool operator()(const T *const color_coefficients, T *residuals) const
        {
            for (int i = 0; i < 3; ++i)
            {
                residuals[i] = T(model_color[i]) - T(observed_color[i]) * color_coefficients[i];
            }
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &observed_color, const Eigen::Vector3d &model_color)
        {
            return new ceres::AutoDiffCostFunction<ColorCostFunction, 3, 3>(
                new ColorCostFunction(observed_color, model_color));
        }

        Eigen::Vector3d observed_color;
        Eigen::Vector3d model_color;
    };

    /**
     * @brief Cost function for the regularization term of the shape coefficients.
     *
     * Penalizes deviations from the mean shape.
     * The cost function is defined as:
     * E_shape = sum_i (shape_coefficients[i] / shape_std[i])
     */
    struct RegularizationShapeCostFunction
    {
        RegularizationShapeCostFunction(const Eigen::VectorXd &shape_coefficients, BFM &bfm)
            : shape_coefficients(shape_coefficients), bfm(bfm) {}

        template <typename T>
        bool operator()(T const *shape_coefficients, T *residuals) const
        {

            for (size_t i = 0; i < bfm.color_shape_size; i++)
            {

                T shape_std = ceres::sqrt(T(bfm.shape_var[i]));
                residuals[i] = shape_coefficients[i] / shape_std;
            }
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::VectorXd &shape_coefficients, BFM &bfm)
        {
            return new ceres::AutoDiffCostFunction<RegularizationShapeCostFunction, 199, 199>(
                new RegularizationShapeCostFunction(shape_coefficients, bfm));
        }

        Eigen::VectorXd shape_coefficients;
        BFM bfm;
    };

    /**
     * @brief Cost function for the regularization term of the color coefficients.
     *
     * Penalizes deviations from the mean color.
     * The cost function is defined as:
     * E_color = sum_i (color_coefficients[i] / color_std[i])
     */
    struct RegularizationColorCostFunction
    {
        RegularizationColorCostFunction(const Eigen::VectorXd &color_coefficients, BFM &bfm)
            : color_coefficients(color_coefficients), bfm(bfm) {}

        template <typename T>
        bool operator()(T const *color_coefficients, T *residuals) const
        {

            for (size_t i = 0; i < bfm.color_shape_size; i++)
            {
                T color_std = ceres::sqrt(T(bfm.color_var[i]));
                residuals[i] = color_coefficients[i] / color_std;
            }
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::VectorXd &color_coefficients, BFM &bfm)
        {
            return new ceres::AutoDiffCostFunction<RegularizationColorCostFunction, 199, 199>(
                new RegularizationColorCostFunction(color_coefficients, bfm));
        }

        Eigen::VectorXd color_coefficients;
        BFM bfm;
    };

    /**
     * @brief Cost function for the regularization term of the expression coefficients.
     *
     * Penalizes deviations from the mean expression.
     * The cost function is defined as:
     * E_expr = sum_i (expr_coefficients[i] / expr_std[i])
     */
    struct RegularizationExprCostFunction
    {
        RegularizationExprCostFunction(const Eigen::VectorXd &expr_coefficients, BFM &bfm)
            : expr_coefficients(expr_coefficients), bfm(bfm) {}

        template <typename T>
        bool operator()(T const *expr_coefficients, T *residuals) const
        {

            for (size_t i = 0; i < bfm.expr_size; i++)
            {
                T expr_std = ceres::sqrt(T(bfm.expr_var[i]));
                residuals[i] = expr_coefficients[i] / expr_std;
            }
            return true;
        }

        static ceres::CostFunction *Create(const Eigen::VectorXd &expr_coefficients, BFM &bfm)
        {
            return new ceres::AutoDiffCostFunction<RegularizationExprCostFunction, 100, 100>(
                new RegularizationExprCostFunction(expr_coefficients, bfm));
        }

        Eigen::VectorXd expr_coefficients;
        BFM bfm;
    };

} // namespace CostFunctions

#endif