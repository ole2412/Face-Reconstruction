#include "bfm.h"
#include "optimizationParameters.h"
BFM::BFM()
{
    load_bfm();
}

std::vector<int> readLandmarks(const std::string &filePath)
{
    std::vector<int> landmarks;   // read landmarks from file
    std::ifstream file(filePath); // open the file
    if (!file.is_open())
    {
        std::cerr << "could not open the file: " << filePath << std::endl;
        return landmarks;
    }

    int index;
    while (file >> index)
    { // read landmarks from file
        landmarks.push_back(index);
    }

    file.close(); // close the file
    return landmarks;
}
void BFM::load_bfm()
{
    std::cout << "Start loading bfm..." << std::endl;

    try
    {
        // the respective local variables (see bfm.h line 50 ff)
        // Initialize the HDF5 file
        H5::H5File file(m_model_path, H5F_ACC_RDONLY);

        // ===== Load Shape Data =====
        H5::DataSet shape_mean_dataset = file.openDataSet("shape/model/mean");
        hsize_t shape_mean_dims[1];
        shape_mean_dataset.getSpace().getSimpleExtentDims(shape_mean_dims);
        shape_mean.resize(shape_mean_dims[0]);
        shape_mean_dataset.read(shape_mean.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet shape_pca_dataset = file.openDataSet("shape/model/pcaBasis");
        hsize_t shape_pca_dims[2];
        shape_pca_dataset.getSpace().getSimpleExtentDims(shape_pca_dims);
        shape_pca.resize(shape_pca_dims[0], shape_pca_dims[1]);
        shape_pca_dataset.read(shape_pca.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet shape_var_dataset = file.openDataSet("shape/model/pcaVariance");
        hsize_t shape_var_dims[1];
        shape_var_dataset.getSpace().getSimpleExtentDims(shape_var_dims);
        shape_var.resize(shape_var_dims[0]);
        shape_var_dataset.read(shape_var.data(), H5::PredType::NATIVE_DOUBLE);

        // ===== Load Expression Data =====
        H5::DataSet expr_mean_dataset = file.openDataSet("expression/model/mean");
        hsize_t expr_mean_dims[1];
        expr_mean_dataset.getSpace().getSimpleExtentDims(expr_mean_dims);
        expr_mean.resize(expr_mean_dims[0]);
        expr_mean_dataset.read(expr_mean.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet expr_pca_dataset = file.openDataSet("expression/model/pcaBasis");
        hsize_t expr_pca_dims[2];
        expr_pca_dataset.getSpace().getSimpleExtentDims(expr_pca_dims);
        expr_pca.resize(expr_pca_dims[0], expr_pca_dims[1]);
        expr_pca_dataset.read(expr_pca.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet expr_var_dataset = file.openDataSet("expression/model/pcaVariance");
        hsize_t expr_var_dims[1];
        expr_var_dataset.getSpace().getSimpleExtentDims(expr_var_dims);
        expr_var.resize(expr_var_dims[0]);
        expr_var_dataset.read(expr_var.data(), H5::PredType::NATIVE_DOUBLE);

        // ===== Load Color Data =====
        H5::DataSet color_mean_dataset = file.openDataSet("color/model/mean");
        hsize_t color_mean_dims[1];
        color_mean_dataset.getSpace().getSimpleExtentDims(color_mean_dims);
        color_mean.resize(color_mean_dims[0]);
        color_mean_dataset.read(color_mean.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet color_pca_dataset = file.openDataSet("color/model/pcaBasis");
        hsize_t color_pca_dims[2];
        color_pca_dataset.getSpace().getSimpleExtentDims(color_pca_dims);
        color_pca.resize(color_pca_dims[0], color_pca_dims[1]);
        color_pca_dataset.read(color_pca.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet color_var_dataset = file.openDataSet("color/model/pcaVariance");
        hsize_t color_var_dims[1];
        color_var_dataset.getSpace().getSimpleExtentDims(color_var_dims);
        color_var.resize(color_var_dims[0]);
        color_var_dataset.read(color_var.data(), H5::PredType::NATIVE_DOUBLE);

        // Load Triangles Dataset
        H5::DataSet triangles_dataset = file.openDataSet("shape/representer/cells");
        H5::DataSpace triangles_space = triangles_dataset.getSpace();

        // Get dimensions
        hsize_t triangles_dims[2];
        triangles_space.getSimpleExtentDims(triangles_dims);
        // Temporarily skip validation for debugging
        triangles.resize(triangles_dims[0] * triangles_dims[1]);
        triangles_dataset.read(triangles.data(), H5::PredType::NATIVE_INT);

        landmark_indices = readLandmarks(m_landmark_path);
    }
    catch (H5::Exception &e)
    {
        std::cerr << "Error loading BFM model: " << e.getCDetailMsg() << std::endl;
    }

    std::cout << "BFM model loading complete." << std::endl;
}

void BFM::get_vertex_positions(Eigen::Matrix<double, -1, 1> &vertexPosition, Eigen::VectorXd &shapeCoefficients, Eigen::VectorXd &expressionCoefficients)
{
    vertexPosition = shape_mean + expr_mean;

    for (int v = 0; v < no_vertices; v++)
    {
        for (int i = 0; i < color_shape_size; i++)
        {
            double shapeFactor = sqrt(shape_var[i]) * shapeCoefficients[i];
            vertexPosition[3 * v] += shapeFactor * shape_pca(3 * v, i);
            vertexPosition[3 * v + 1] += shapeFactor * shape_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += shapeFactor * shape_pca(3 * v + 2, i);
        }

        for (int i = 0; i < expr_size; i++)
        {
            double exprFactor = sqrt(expr_var[i]) * expressionCoefficients[i];
            vertexPosition[3 * v] += exprFactor * expr_pca(3 * v, i);
            vertexPosition[3 * v + 1] += exprFactor * expr_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += exprFactor * expr_pca(3 * v + 2, i);
        }
    }
}

Eigen::MatrixXd BFM::extractVertexColors(const OptimizationParameters &params) const
{
    Eigen::MatrixXd vertexColors(this->get_no_vertices(), 3);

    for (int index = 0; index < this->get_no_vertices(); ++index)
    {
        Eigen::Vector3d color = this->get_color_mean().segment<3>(3 * index);

        for (int i = 0; i < this->get_color_size(); ++i)
        {
            double colorFactor = sqrt(this->get_color_var()[i]) * params.colorCoefficients[i];
            color[0] += colorFactor * this->get_color_pca()(3 * index, i);
            color[1] += colorFactor * this->get_color_pca()(3 * index + 1, i);
            color[2] += colorFactor * this->get_color_pca()(3 * index + 2, i);
        }

        vertexColors.row(index) = color;
    }

    return vertexColors;
}

Eigen::MatrixXd BFM::get_observed_colors(const cv::Mat &image) const
{
    // 顶点数量
    long nVertices = this->get_no_vertices();
    Eigen::MatrixXd colors(nVertices, 3); // 每行存储 RGB 值

    for (int v = 0; v < nVertices; ++v)
    {
        // 获取当前顶点的 3D 坐标
        Eigen::Vector3d vertex = this->get_vertex(v);

        // 投影到 2D 图像坐标
        auto [u, v_proj] = projectVertexToImage(vertex);

        // 检查投影是否在图像范围内
        if (u >= 0 && u < image.cols && v_proj >= 0 && v_proj < image.rows)
        {
            // 从图像中获取像素颜色
            cv::Vec3b color = image.at<cv::Vec3b>(v_proj, u);

            // 归一化颜色值并存储到矩阵中
            colors(v, 0) = color[2] / 255.0; // Red
            colors(v, 1) = color[1] / 255.0; // Green
            colors(v, 2) = color[0] / 255.0; // Blue
        }
        else
        {
            // 如果投影超出范围，设置默认颜色为黑色
            colors(v, 0) = 0.0; // Red
            colors(v, 1) = 0.0; // Green
            colors(v, 2) = 0.0; // Blue
        }
    }

    return colors; // 返回颜色矩阵
}

Eigen::Vector3d BFM::get_vertex(int index) const
{
    // 确保索引合法
    if (index < 0 || index >= this->get_no_vertices())
    {
        throw std::out_of_range("Index out of range in get_vertex");
    }

    // 从 shape_mean 和 expr_mean 中计算 3D 顶点
    return Eigen::Vector3d(
        shape_mean(3 * index) + expr_mean(3 * index),
        shape_mean(3 * index + 1) + expr_mean(3 * index + 1),
        shape_mean(3 * index + 2) + expr_mean(3 * index + 2));
}

std::pair<int, int> BFM::projectVertexToImage(const Eigen::Vector3d &vertex) const
{
    // 假设相机内参（焦距 fx 和 fy，光心 cx 和 cy）
    double fx = 500.0; // 焦距 fx
    double fy = 500.0; // 焦距 fy
    double cx = 960.0; // 图像中心 x
    double cy = 540.0; // 图像中心 y

    // 使用 3D 点到 2D 点的投影公式
    int u = static_cast<int>(fx * vertex[0] / vertex[2] + cx);
    int v_proj = static_cast<int>(fy * vertex[1] / vertex[2] + cy);

    return {u, v_proj}; // 返回投影后的 2D 坐标
}

// Eigen::MatrixXd BFM::get_observed_colors(const cv::Mat &image) const
// {
//     // 顶点数量
//     long nVertices = this->get_no_vertices();
//     Eigen::MatrixXd colors(nVertices, 3); // 每行存储 RGB 值

//     for (int v = 0; v < nVertices; ++v)
//     {
//         // 获取当前顶点的 3D 坐标
//         Eigen::Vector3d vertex = this->get_vertex(v);

//         // 投影到 2D 图像坐标
//         auto [u, v_proj] = projectVertexToImage(vertex);

//         // 检查投影是否在图像范围内
//         if (u >= 0 && u < image.cols && v_proj >= 0 && v_proj < image.rows)
//         {
//             // 从图像中获取像素颜色
//             cv::Vec3b color = image.at<cv::Vec3b>(v_proj, u);

//             // 归一化颜色值并存储到矩阵中
//             colors(v, 0) = color[2] / 255.0; // Red
//             colors(v, 1) = color[1] / 255.0; // Green
//             colors(v, 2) = color[0] / 255.0; // Blue
//         }
//         else
//         {
//             // 如果投影超出范围，设置默认颜色为黑色
//             colors(v, 0) = 0.0; // Red
//             colors(v, 1) = 0.0; // Green
//             colors(v, 2) = 0.0; // Blue
//         }
//     }

//     return colors; // 返回颜色矩阵
// }