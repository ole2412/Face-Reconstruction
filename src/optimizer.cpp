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
    //optimizeDense(params, observed_landmarks, landmark_indices, image_width, image_height, observed_colors, model_colors, image);

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
    for (size_t i = 0; i < 68; ++i)
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
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 12;
    options.minimizer_progress_to_stdout = true;

    // keep the shape and expression constant
    problem.SetParameterBlockConstant(params.shapeCoefficients.data());
    problem.SetParameterBlockConstant(params.expressionCoefficients.data());

/*     ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;
    options.function_tolerance = 1e-10;
    options.max_num_iterations = 15;
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true; */

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
    for (size_t i = 0; i < 68; ++i)
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

/*     ceres::CostFunction *reg_color_cost = CostFunctions::RegularizationColorCostFunction::Create(params.colorCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_color_cost,
        nullptr,
        params.colorCoefficients.data()); */

    ceres::CostFunction *reg_expr_cost = CostFunctions::RegularizationExprCostFunction::Create(params.expressionCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_expr_cost,
        nullptr,
        params.expressionCoefficients.data());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 12;
    options.minimizer_progress_to_stdout = true;
    //options.linear_solver_type = ceres::DENSE_QR;
    //options.use_nonmonotonic_steps = true;
    //options.preconditioner_type = ceres::JACOBI;

    // keep the shape and expression constant
    //problem.SetParameterBlockConstant(params.shapeCoefficients.data());
    //problem.SetParameterBlockConstant(params.expressionCoefficients.data());

    //options.function_tolerance = 1e-10;
    //options.max_num_iterations = 10;
    //options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    // std::cout << summary.FullReport() << std::endl;
}

// Dense optimizer
void Optimizer::optimizeDense(OptimizationParameters &params,
                              const std::vector<Eigen::Vector2d> &observed_landmarks,
                              const std::vector<int> &landmark_indices,
                              int image_width,
                              int image_height, const Eigen::MatrixXd &observed_colors,
                              const Eigen::MatrixXd &model_colors, const cv::Mat &image)
{
    BFM bfm = BFM();
    ceres::Problem problem;
    // Eigen::MatrixXd model_colors = bfm.extractVertexColors(params);
    // Eigen::MatrixXd observed_colors = bfm.get_observed_colors(image);
    std::cout << "Observed colors (first 5 rows):\n"
              << observed_colors.topRows(5) << std::endl;
    std::cout << "Model colors (first 5 rows):\n"
              << model_colors.topRows(5) << std::endl;
    for (int v = 0; v < model_colors.rows(); ++v)
    {
        ceres::CostFunction *color_cost = new ceres::AutoDiffCostFunction<CostFunctions::ColorCostFunction, 3, 3>(
            new CostFunctions::ColorCostFunction(observed_colors.row(v), model_colors.row(v)));
        problem.AddResidualBlock(color_cost, nullptr, params.colorCoefficients.data());
    }

    // Set up options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.function_tolerance = 1e-20;
    options.max_num_iterations = 100;
    options.parameter_tolerance = 1e-20;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    std::cout << summary.FullReport() << std::endl;
}