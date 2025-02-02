#include "bfm.h"

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

void BFM::get_vertex_positions(Eigen::Matrix<double, -1, 1> &vertexPosition, Eigen::VectorXd &shapeCoefficients, Eigen::VectorXd &expressionCoefficients){
    vertexPosition = shape_mean + expr_mean;

    for (int v = 0; v < no_vertices; v++) {
        for (int i = 0; i < color_shape_size; i++) {
            double shapeFactor = sqrt(shape_var[i]) * shapeCoefficients[i];
            vertexPosition[3 * v] += shapeFactor * shape_pca(3 * v, i);
            vertexPosition[3 * v + 1] += shapeFactor * shape_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += shapeFactor * shape_pca(3 * v + 2, i);
        }

        for (int i = 0; i < expr_size; i++) {
            double exprFactor = sqrt(expr_var[i]) * expressionCoefficients[i];
            vertexPosition[3 * v] += exprFactor * expr_pca(3 * v, i);
            vertexPosition[3 * v + 1] += exprFactor * expr_pca(3 * v + 1, i);
            vertexPosition[3 * v + 2] += exprFactor * expr_pca(3 * v + 2, i);
        }
    }
}
