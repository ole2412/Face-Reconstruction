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

    // Set up options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // Solve the problem
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Final results
    std::cout << summary.FullReport() << std::endl;
    params.printParameters();
}

// TODO add the Dense optimizer and Reg optimizer