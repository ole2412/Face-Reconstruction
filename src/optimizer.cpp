#include "costFunctions.h" 
#include "optimizationParameters.h"
#include "optimizer.h"



void Optimizer::optimizeSparse(
    OptimizationParameters& params,
    const std::vector<Eigen::Vector2d>& observed_landmarks,
    const std::vector<int>& landmark_indices,
    int image_width,
    int image_height
    )
    {
        
    ceres::Problem problem;

    // Add residual blocks for each landmark
    //observed_landmarks.size()
    for (size_t i = 0; i < observed_landmarks.size(); ++i) {
        ceres::CostFunction* sparse_cost = CostFunctions::SparseTermCostFunction::Create(observed_landmarks[i], landmark_indices[i], *params.bfm, params.image_width, params.image_height);
        problem.AddResidualBlock(
            sparse_cost,
            nullptr,                   
            params.translation.data(),
            params.rotation.data(),    
            &params.fov,
            params.shapeCoefficients.data(),
            params.expressionCoefficients.data()
        );
    }
	
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;

	// keep the shape and expression constant
	problem.SetParameterBlockConstant(params.shapeCoefficients.data());
	problem.SetParameterBlockConstant(params.expressionCoefficients.data());

    options.function_tolerance = 1e-10; 
    options.max_num_iterations = 50;    
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    std::cout << summary.FullReport() << std::endl;
}

void Optimizer::optimizeSparseReg(
    OptimizationParameters& params,
    const std::vector<Eigen::Vector2d>& observed_landmarks,
    const std::vector<int>& landmark_indices,
    int image_width,
    int image_height
    )
    {
        
    ceres::Problem problem;

    // Add residual blocks for each landmark
    //observed_landmarks.size()
    for (size_t i = 0; i < observed_landmarks.size(); ++i) {
        ceres::CostFunction* sparse_cost = CostFunctions::SparseTermCostFunction::Create(observed_landmarks[i], landmark_indices[i], *params.bfm, params.image_width, params.image_height);
        problem.AddResidualBlock(
            sparse_cost,
            nullptr,                   
            params.translation.data(),
            params.rotation.data(),    
            &params.fov,
            params.shapeCoefficients.data(),
            params.expressionCoefficients.data()
        );
    }

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
    
	
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::JACOBI;

	// keep the shape and expression constant
	//problem.SetParameterBlockConstant(params.shapeCoefficients.data());
	//problem.SetParameterBlockConstant(params.expressionCoefficients.data());

    options.function_tolerance = 1e-10; 
    options.max_num_iterations = 30;    
    options.parameter_tolerance = 1e-10;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    std::cout << summary.FullReport() << std::endl;
}

// TODO add the Dense optimizer and Reg optimizer

// void Optimizer::optimizeReg(
//     OptimizationParameters& params
// ) {

//     ceres::Problem problem;

//     ceres::CostFunction* reg_shape_cost = CostFunctions::RegularizationShapeCostFunction::Create(params.shapeCoefficients, *params.bfm);
//     problem.AddResidualBlock(
//         reg_shape_cost,
//         nullptr,                   
//         params.shapeCoefficients.data()
//     );

//     ceres::CostFunction* reg_color_cost = CostFunctions::RegularizationColorCostFunction::Create(params.colorCoefficients, *params.bfm);
//     problem.AddResidualBlock(
//         reg_color_cost,
//         nullptr,                   
//         params.colorCoefficients.data()
//     );

//     ceres::CostFunction* reg_expr_cost = CostFunctions::RegularizationExprCostFunction::Create(params.expressionCoefficients, *params.bfm);
//     problem.AddResidualBlock(
//         reg_expr_cost,
//         nullptr,                   
//         params.expressionCoefficients.data()
//     );

//     // Set up options
//     ceres::Solver::Options options;
//     options.linear_solver_type = ceres::DENSE_QR;
//     options.use_nonmonotonic_steps = true;
//     options.preconditioner_type = ceres::JACOBI;

//     options.function_tolerance = 1e-10; 
//     options.max_num_iterations = 30;    
//     options.parameter_tolerance = 1e-10;
//     options.minimizer_progress_to_stdout = true;

//     // Solve the problem
//     ceres::Solver::Summary summary;
//     ceres::Solve(options, &problem, &summary);

//     // Final results
//     std::cout << summary.FullReport() << std::endl;

// }