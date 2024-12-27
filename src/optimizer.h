#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ceres/ceres.h> 
#include <Eigen/Dense>   
#include <iostream>   
#include "optimizationParameters.h"

class Optimizer {
public:
    void optimizeSparse(
        OptimizationParameters& params,
        const std::vector<Eigen::Vector2d>& observed_landmarks,
        const std::vector<Eigen::Vector3d>& model_landmarks,
        int image_width,
        int image_height);

    //TODO add the parameters
    void optimizeDence();
    void optimizeReg();
};


#endif