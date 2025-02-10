#include "costFunctions.h"
#include "optimizationParameters.h"
#include "optimizer.h"

void Optimizer::optimize(
    OptimizationParameters &params,
    const std::vector<Eigen::Vector2d> &observed_landmarks,
    const std::vector<int> &landmark_indices,
    int image_width,
    int image_height,
    const Eigen::MatrixXd &observed_colors,
    const Eigen::MatrixXd &model_colors, const cv::Mat &image)
{
    // optimize rotation translation briefly
    std::cout << "Optimizing rotation and translation" << std::endl;
    optimizeSparse(params, observed_landmarks, landmark_indices, image_width, image_height);

    // optimize model parameters together with rotation and translation
    std::cout << "Optimizing model parameters" << std::endl;
    optimizeSparseReg(params, observed_landmarks, landmark_indices, image_width, image_height);
    std::cout << "Parameters after Sparse Optimization:" << std::endl;

    // optimize color
    std::cout << "Optimizing color parameters" << std::endl;
    optimizeDense(params, observed_landmarks, landmark_indices, image_width, image_height, observed_colors, model_colors, image);

    params.printParameters();
}

void Optimizer::optimizeSparse(
    OptimizationParameters &params,
    const std::vector<Eigen::Vector2d> &observed_landmarks,
    const std::vector<int> &landmark_indices,
    int image_width,
    int image_height)
{

    ceres::Problem problem;

    // Add residual blocks for each landmark
    for (size_t i = 0; i < observed_landmarks.size(); ++i)
    {
        ceres::CostFunction *sparse_cost = CostFunctions::SparseTermCostFunction::Create(observed_landmarks[i], landmark_indices[i], *params.bfm, params.image_width, params.image_height);
        problem.AddResidualBlock(
            sparse_cost,
            nullptr,
            params.translation.data(),
            params.rotation.data(),
            &params.fov,
            params.shapeCoefficients.data(),
            params.expressionCoefficients.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;

    // keep the shape and expression constant
    problem.SetParameterBlockConstant(params.shapeCoefficients.data());
    problem.SetParameterBlockConstant(params.expressionCoefficients.data());

    options.function_tolerance = 1e-10;
    options.max_num_iterations = 10;
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    // std::cout << summary.FullReport() << std::endl;
}

void Optimizer::optimizeSparseReg(
    OptimizationParameters &params,
    const std::vector<Eigen::Vector2d> &observed_landmarks,
    const std::vector<int> &landmark_indices,
    int image_width,
    int image_height)
{

    ceres::Problem problem;

    // Add residual blocks for each landmark
    // observed_landmarks.size()
    for (size_t i = 0; i < observed_landmarks.size(); ++i)
    {
        ceres::CostFunction *sparse_cost = CostFunctions::SparseTermCostFunction::Create(observed_landmarks[i], landmark_indices[i], *params.bfm, params.image_width, params.image_height);
        problem.AddResidualBlock(
            sparse_cost,
            nullptr,
            params.translation.data(),
            params.rotation.data(),
            &params.fov,
            params.shapeCoefficients.data(),
            params.expressionCoefficients.data());
    }

    ceres::CostFunction *reg_shape_cost = CostFunctions::RegularizationShapeCostFunction::Create(params.shapeCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_shape_cost,
        nullptr,
        params.shapeCoefficients.data());

    ceres::CostFunction *reg_color_cost = CostFunctions::RegularizationColorCostFunction::Create(params.colorCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_color_cost,
        nullptr,
        params.colorCoefficients.data());

    ceres::CostFunction *reg_expr_cost = CostFunctions::RegularizationExprCostFunction::Create(params.expressionCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_expr_cost,
        nullptr,
        params.expressionCoefficients.data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;

    // keep the shape and expression constant
    // problem.SetParameterBlockConstant(params.shapeCoefficients.data());
    // problem.SetParameterBlockConstant(params.expressionCoefficients.data());

    options.function_tolerance = 1e-10;
    options.max_num_iterations = 10;
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    // std::cout << summary.FullReport() << std::endl;
}

// // Dense optimizer
// void Optimizer::optimizeDense(OptimizationParameters &params,
//                               const std::vector<Eigen::Vector2d> &observed_landmarks,
//                               const std::vector<int> &landmark_indices,
//                               int image_width,
//                               int image_height, const Eigen::MatrixXd &observed_colors,
//                               const Eigen::MatrixXd &model_colors, const cv::Mat &image)
// {
//     BFM bfm = BFM();
//     ceres::Problem problem;
//     // Eigen::MatrixXd model_colors = bfm.extractVertexColors(params);
//     // Eigen::MatrixXd observed_colors = bfm.get_observed_colors(image);
//     std::cout << "Observed colors (first 5 rows):\n"
//               << observed_colors.topRows(5) << std::endl;
//     std::cout << "Model colors (first 5 rows):\n"
//               << model_colors.topRows(5) << std::endl;
//     // for (int v = 0; v < model_colors.rows(); ++v)
//     for (int v = 0; v < 3; ++v)
//     {
//         ceres::CostFunction *color_cost = new ceres::AutoDiffCostFunction<CostFunctions::ColorCostFunction, 3, 3>(
//             new CostFunctions::ColorCostFunction(observed_colors.row(v), model_colors.row(v)));
//         problem.AddResidualBlock(color_cost, nullptr, params.colorCoefficients.data());
//     }

//     // Set up options
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_QR;
//     options.minimizer_progress_to_stdout = true;
//     options.function_tolerance = 1e-20;
//     options.max_num_iterations = 100;
//     options.parameter_tolerance = 1e-20;
//     options.minimizer_progress_to_stdout = true;

//     // Solve the problem
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     // Final results
//     std::cout << summary.FullReport() << std::endl;
// }

// 假设 CostFunctions::ColorCostFunction 已经按照前面讨论的方式定义，形如：
// struct ColorCostFunction {
//     ColorCostFunction(const Eigen::Vector3d &observed_color,
//                       const Eigen::Vector3d &model_color,
//                       const Eigen::Vector3d &color_mean,
//                       const Eigen::Matrix<double, 3, 199> &color_basis)
//         : observed_color(observed_color),
//           model_color(model_color),
//           color_mean(color_mean),
//           color_basis(color_basis) {}
//
//     template <typename T>
//     bool operator()(const T *const color_coefficients, T *residuals) const {
//         Eigen::Map<const Eigen::Matrix<T, 199, 1>> coeff(color_coefficients);
//         // 计算预测颜色： vertex_color_mean + color_basis * color_coefficients
//         Eigen::Matrix<T, 3, 1> predicted_color = color_mean.cast<T>() + color_basis.cast<T>() * coeff;
//         // 残差 = model_color - predicted_color
//         Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residuals);
//         residual_map = model_color.cast<T>() - predicted_color;
//         return true;
//     }
// };
//
// 同时确保 AutoDiffCostFunction 的模板参数使用的是残差维数3，待优化参数维数199。

// 优化器 dense 优化函数
void Optimizer::optimizeDense(OptimizationParameters &params,
                              const std::vector<Eigen::Vector2d> &observed_landmarks,
                              const std::vector<int> &landmark_indices,
                              int image_width,
                              int image_height,
                              const Eigen::MatrixXd &observed_colors, // 尺寸：(no_vertices, 3)
                              const Eigen::MatrixXd &model_colors,    // 尺寸：(no_vertices, 3)
                              const cv::Mat &image)
{
    // 创建 BFM 模型对象
    BFM bfm;

    // 输出部分信息以供调试
    std::cout << "Observed colors (first 5 rows):\n"
              << observed_colors.topRows(5) << std::endl;
    std::cout << "Model colors (first 5 rows):\n"
              << model_colors.topRows(5) << std::endl;

    // 创建 Ceres 优化问题
    ceres::Problem problem;

    // 假设 observed_colors 与 model_colors 的行数与 BFM 中顶点数相同（例如 no_vertices = 28588）
    // 注意：如果全量顶点数太多，也可以只选部分顶点或对关键区域进行优化。
    const int numVertices = static_cast<int>(observed_colors.rows());

    // 假设你已经正确从图像中提取了 observed_colors
    // 并且 BFM 也提供了 color_mean 和 color_basis
    for (int v = 0; v < numVertices; ++v)
    {
        // 这里 observed_colors.row(v) 是图像采样到的颜色
        Eigen::Vector3d observed_color = observed_colors.row(v).transpose();
        // 如果 model_colors 原本是 BFM 计算得到的颜色，可能不适合作为目标
        // 你应当使用 observed_color 作为目标，使优化驱动颜色系数更新

        // 从 BFM 中提取该顶点的均值和 PCA 基底
        Eigen::Vector3d vertex_color_mean = bfm.get_color_mean().segment<3>(v * 3);
        Eigen::Matrix<double, 3, 199> vertex_color_basis =
            bfm.get_color_pca().block(v * 3, 0, 3, bfm.get_color_size());

        // 构造残差块，此处优化的参数维度仍为 199
        ceres::CostFunction *color_cost =
            new ceres::AutoDiffCostFunction<CostFunctions::ColorCostFunction, 3, 199>(
                new CostFunctions::ColorCostFunction(observed_color,
                                                     /* model_color 参数可以传入占位值，
                                                        但建议修改构造函数只保留 observed_color，
                                                        因为目标应该是 observed_color */
                                                     observed_color,
                                                     vertex_color_mean,
                                                     vertex_color_basis));
        problem.AddResidualBlock(color_cost, nullptr, params.colorCoefficients.data());
    }

    // 设置 Ceres 求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-20;
    options.max_num_iterations = 100;
    options.parameter_tolerance = 1e-20;

    // 求解优化问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出完整的求解报告
    std::cout << summary.FullReport() << std::endl;
}
