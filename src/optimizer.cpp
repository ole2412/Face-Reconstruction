#include "costFunctions.h" 
#include "optimizationParameters.h"
#include "optimizer.h"



void Optimizer::optimizeSparse(
    OptimizationParameters& params,
    const std::vector<Eigen::Vector2d>& observed_landmarks,
    const std::vector<Eigen::Vector3d>& model_landmarks,
    int image_width,
    int image_height)
    {
        
    ceres::Problem problem;

    // Add residual blocks for each landmark
    //observed_landmarks.size()
    for (size_t i = 0; i < observed_landmarks.size(); ++i) {
        ceres::CostFunction* sparse_cost = CostFunctions::SparseTermCostFunction::Create(observed_landmarks[i], model_landmarks[i], params.image_width, params.image_height);
        problem.AddResidualBlock(
            sparse_cost,
            nullptr,                   
            params.translation.data(),
            params.rotation.data(),    
            &params.fov              
        );
    }

    //assume simplest setup, first check before optimiz where landmarks are in image frame, then start optimizing and print loss,
    //see where landmark is of bfm and ground truth landmark, do only for landmark, just visualize it every step too. project the landmark on the image.
    //meshlab show axis, so we load the mesh and it automatically adjusts to the axis. 
    // Set up options. ahrdcode fx fy to 15mm from focal length and converrt to pixel. so scale the image in orthographic. 
    // the mean is 0. standard deviation of model look up the mean. predefined.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;

    options.function_tolerance = 1e-50; 
    options.max_num_iterations = 2000;    
    options.parameter_tolerance = 1e-50;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    std::cout << summary.FullReport() << std::endl;
}

// TODO add the Dense optimizer and Reg optimizer

void Optimizer::optimizeReg(
    OptimizationParameters& params
) {

    ceres::Problem problem;

    ceres::CostFunction* reg_shape_cost = CostFunctions::RegularizationShapeCostFunction::Create(params.shapeCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_shape_cost,
        nullptr,                   
        params.shapeCoefficients.data()
    );

    ceres::CostFunction* reg_color_cost = CostFunctions::RegularizationColorCostFunction::Create(params.colorCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_color_cost,
        nullptr,                   
        params.colorCoefficients.data()
    );

    ceres::CostFunction* reg_expr_cost = CostFunctions::RegularizationExprCostFunction::Create(params.expressionCoefficients, *params.bfm);
    problem.AddResidualBlock(
        reg_expr_cost,
        nullptr,                   
        params.expressionCoefficients.data()
    );

    // Set up options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;

    options.function_tolerance = 1e-50; 
    options.max_num_iterations = 2000;    
    options.parameter_tolerance = 1e-50;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    std::cout << summary.FullReport() << std::endl;

}