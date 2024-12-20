#ifndef bfm_h
#define bfm_h

#include <Eigen/Dense>
#include <hdf5.h>
#include <H5Cpp.h>

#include <iostream>
#include <string>
#include <fstream>
#include <map>
#include <memory>

class BFM
{
public:
    BFM();

    inline Eigen::MatrixXd &get_shape_pca() { return shape_pca; }
    inline Eigen::VectorXd &get_shape_var() { return shape_var; }
    inline Eigen::VectorXd &get_shape_mean() { return shape_mean; }

    inline Eigen::MatrixXd &get_expr_pca() { return expr_pca; }
    inline Eigen::VectorXd &get_expr_var() { return expr_var; }
    inline Eigen::VectorXd &get_expr_mean() { return expr_mean; }

    inline Eigen::MatrixXd &get_color_pca() { return color_pca; }
    inline Eigen::VectorXd &get_color_var() { return color_var; }
    inline Eigen::VectorXd &get_color_mean() { return color_mean; }

    inline Eigen::VectorXi &get_current_faces() { return triangles; }
    inline std::vector<int> &get_landmark_indices() { return landmark_indices; }

    inline unsigned int get_no_vertices() { return no_vertices; }
    inline unsigned int get_expr_size() { return expr_size; }
    inline unsigned int get_color_size() { return color_shape_size; }
    inline unsigned int get_shape_size() { return color_shape_size; }
    inline unsigned int get_no_faces() { return no_faces; }
    inline unsigned int get_landmarks_size() { return landmark_indices.size(); }

private:
    std::string m_model_path = R"(../models/BFM17/model2017-1_face12_nomouth.h5)";
    std::string m_landmark_path = R"(../models/BFM17/Landmarks68_BFM.anl)";
    // H5::H5File h5_file;

    unsigned int no_vertices = 28588;    // coming form Dataset: /representer/cells, Shape: (3, 56572), Type: int32
    unsigned int no_faces = 56572;       // coming from Dataset: /representer/points, Shape: (3, 28588), Type: float32
    unsigned int color_shape_size = 199; // color and shape vector size
    unsigned int expr_size = 100;        // expression vector size

    Eigen::MatrixXd shape_pca;
    Eigen::VectorXd shape_var;
    Eigen::VectorXd shape_mean;

    Eigen::MatrixXd expr_pca;
    Eigen::VectorXd expr_var;
    Eigen::VectorXd expr_mean;

    Eigen::MatrixXd color_pca;
    Eigen::VectorXd color_var;
    Eigen::VectorXd color_mean;

    Eigen::VectorXi triangles;
    std::vector<int> landmark_indices;

    void load_bfm();
};

#endif
