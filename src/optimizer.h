#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ceres/ceres.h> 
#include <Eigen/Dense>   
#include <iostream>   
#include "optimizationParameters.h"

class Optimizer {
public:
    void optimize(
        OptimizationParameters& params,
        const std::vector<Eigen::Vector2d>& observed_landmarks,
        const std::vector<int>& landmark_indices,
        int image_width,
        int image_height
    );

    void optimizeSparse(
        OptimizationParameters& params,
        const std::vector<Eigen::Vector2d>& observed_landmarks,
        const std::vector<int>& landmark_indices,
        int image_width,
        int image_height);
        
    void optimizeSparseReg(
        OptimizationParameters& params,
        const std::vector<Eigen::Vector2d>& observed_landmarks,
        const std::vector<int>& landmark_indices,
        int image_width,
        int image_height);


    //TODO add the parameters
    void optimizeDense();
    // void optimizeReg(OptimizationParameters& params);
};


#endif