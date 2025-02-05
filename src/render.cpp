#include "render.h"
#include "costFunctions.h"
#include "optimizationParameters.h"
#include "optimizer.h"

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>

void outputMeshToPly(const Eigen::VectorXi &faceIndices, const std::string &filename, std::vector<int> &landmarkIndices, const cv::Mat &image, OptimizationParameters &params, bool show_landmarks)
{
    BFM bfm = BFM();
    auto vertices = bfm.get_no_vertices();
    Eigen::VectorXd shape_mean = bfm.get_shape_mean();
    Eigen::VectorXd expr_mean = bfm.get_expr_mean();
    Eigen::MatrixXd vertexColors = bfm.extractVertexColors(params);
    Eigen::Matrix<double, -1, 1> vertexPosition = shape_mean + expr_mean;

    if (vertexColors.rows() != bfm.get_no_vertices())
    {
        std::cerr << "Error: Vertex color count does not match vertex count." << std::endl;
        return;
    }

    long nVertices = bfm.get_no_vertices();
    // applyTransformationsToVertices(vertexPosition, params.rotation, params.translation);

    std::ofstream outFile(filename);
    if (!outFile.is_open())
    {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }
    outFile << "ply\n";
    outFile << "format ascii 1.0\n";
    outFile << "element vertex " << nVertices << "\n";
    outFile << "property float x\n";
    outFile << "property float y\n";
    outFile << "property float z\n";
    outFile << "property uchar red\n";
    outFile << "property uchar green\n";
    outFile << "property uchar blue\n";

    Eigen::VectorXi triangles = bfm.get_current_faces();
    long nFaces = triangles.size() / 3;

    outFile << "element face " << nFaces << "\n";
    outFile << "property list uchar int vertex_index\n";
    outFile << "end_header\n";

    for (int v = 0; v < nVertices; ++v)
    {
        double x = vertexPosition[3 * v];
        double y = vertexPosition[3 * v + 1];
        double z = vertexPosition[3 * v + 2];

        double r = std::clamp(vertexColors(v, 0) * 255.0, 0.0, 255.0);
        double g = std::clamp(vertexColors(v, 1) * 255.0, 0.0, 255.0);
        double b = std::clamp(vertexColors(v, 2) * 255.0, 0.0, 255.0);

        outFile << x << " " << y << " " << z << " "
                << static_cast<int>(r) << " " << static_cast<int>(g) << " " << static_cast<int>(b) << "\n";
    }

    for (int f = 0; f < nFaces; ++f)
    {
        int v1 = triangles[f + 0 * nFaces]; // NOTE on .obj we need + 1 here, dont know whether we need it here as well
        int v2 = triangles[f + 1 * nFaces];
        int v3 = triangles[f + 2 * nFaces];

        outFile << "3 " << v1 << " " << v2 << " " << v3 << "\n";
    }

    outFile.close();
    std::cout << "PLY file successfully written to " << filename << std::endl;
}

void outputMeshToObj(const Eigen::Matrix<double, -1, 1> &vertexPosition, const Eigen::VectorXi &faceIndices, const std::string &filename, std::vector<int> &landmarkIndices, bool show_landmarks)
{
    /**
     * @brief Outputs a 3D mesh to an OBJ file.
     *
     * This function writes the vertex positions and face indices of a 3D mesh to a file in the OBJ format.
     * Optionally, specific landmark indices can be highlighted if `show_landmarks` is enabled.
     *
     * @param vertexPosition A matrix containing the 3D coordinates of the vertices.
     *                       Dimensions: (n x 3), where n is the number of vertices.
     * @param faceIndices A vector containing the indices defining the faces of the mesh.
     *                    Each triplet in the vector represents a face (vertex indices are 1-based for OBJ format).
     * @param filename The name of the output OBJ file.
     * @param landmarkIndices A vector of vertex indices representing landmark points.
     *                        These indices will be highlighted in the output if `show_landmarks` is true.
     * @param show_landmarks (Optional) A boolean flag to indicate whether to include and highlight landmarks in the output.
     *                       Default is false.
     */

    std::ofstream outFile(filename);

    if (!outFile.is_open())
    {
        std::cerr << "Error: Could not open file for writing." << std::endl;
        return;
    }

    long nVertices = vertexPosition.size() / 3;
    for (int v = 0; v < nVertices; v++)
    {
        double x = vertexPosition[3 * v];
        double y = vertexPosition[3 * v + 1];
        double z = vertexPosition[3 * v + 2];

        // red color for landmarks
        if (show_landmarks && std::find(landmarkIndices.begin(), landmarkIndices.end(), v) != landmarkIndices.end())
        {
            outFile << "v " << x << " " << y << " " << z << " 255 0 0" << std::endl; // landmarks red, rest not stored
        }
        else if (!show_landmarks)
        {
            outFile << "v " << x << " " << y << " " << z << " 0 0 255" << std::endl; // all blue
        }
    }

    if (!show_landmarks)
    {
        long nFaces = faceIndices.size() / 3;
        for (int f = 0; f < nFaces; f++)
        {
            int v1 = faceIndices[f + 0 * nFaces] + 1;
            int v2 = faceIndices[f + 1 * nFaces] + 1;
            int v3 = faceIndices[f + 2 * nFaces] + 1;

            outFile << "f " << v1 << " " << v2 << " " << v3 << std::endl;
        }
    }

    outFile.close();
    std::cout << "Mesh successfully written to " << filename << std::endl;
}

// TODO put this in another class?
//  For reading our own source landmarks
std::vector<Eigen::Vector2d> readObservedLandmarks(const std::string &filePath)
{
    std::vector<Eigen::Vector2d> observed_landmarks;

    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return observed_landmarks;
    }

    double x, y;
    // Read pairs of X and Y coordinates
    while (file >> x >> y)
    {
        observed_landmarks.emplace_back(x, y);
    }

    file.close();

    return observed_landmarks;
}

void applyTransformationsAndProjectionToVertices(Eigen::Matrix<double, -1, 1> &vertexPosition, const Eigen::Vector3d &rotation,
                                                 const Eigen::Vector3d &translation, int image_width, int image_height, double fov)
{
    /**
     * @brief Applies rotation (using AngleAxis) and translation to vertex positions.
     *
     * @param vertexPosition A matrix containing the 3D coordinates of the vertices (flattened as a 1D array).
     * @param rotation A vector of rotation angles [rx, ry, rz] in radians.
     * @param translation A translation vector [tx, ty, tz].
     */
    long nVertices = vertexPosition.size() / 3;

    // Create an AngleAxis for the rotation

    // Convert Euler angles (rotation) to a rotation matrix
    // Initialize the rotation matrix (3x3 = 9 elements)
    double rotation_matrix[9];
    double eulerAngles[3] = {rotation[0], rotation[1], rotation[2]};
    // Convert Euler angles to a 3x3 rotation matrix
    ceres::EulerAnglesToRotationMatrix(eulerAngles, 3, rotation_matrix);

    // Eigen::AngleAxisd angleAxisX(rotation[0], Eigen::Vector3d::UnitX());
    // Eigen::AngleAxisd angleAxisY(rotation[1], Eigen::Vector3d::UnitY());
    // Eigen::AngleAxisd angleAxisZ(rotation[2], Eigen::Vector3d::UnitZ());

    // Combined rotation
    // Eigen::Matrix3d rotationMatrix = (angleAxisZ * angleAxisY * angleAxisX).toRotationMatrix();
    Eigen::Matrix3d rotationMatrix;
    rotationMatrix << rotation_matrix[0], rotation_matrix[1], rotation_matrix[2],
        rotation_matrix[3], rotation_matrix[4], rotation_matrix[5],
        rotation_matrix[6], rotation_matrix[7], rotation_matrix[8];

    // Perspective projection parameters
    double aspect_ratio = static_cast<double>(image_width) / image_height;
    double near_plane = 0.1, far_plane = 10000.0;
    double tan_half_fov = std::tan(fov / 2);

    double proj_00 = 1.0 / (aspect_ratio * tan_half_fov);
    double proj_11 = 1.0 / tan_half_fov;
    double proj_22 = -(far_plane + near_plane) / (far_plane - near_plane);
    double proj_32 = -(2.0 * far_plane * near_plane) / (far_plane - near_plane);

    for (int v = 0; v < nVertices; ++v)
    {
        Eigen::Vector3d vertex(vertexPosition[3 * v], vertexPosition[3 * v + 1], vertexPosition[3 * v + 2]);

        // Apply rotation and translation
        vertex = rotationMatrix * vertex + translation;

        // Apply perspective projection
        double w = proj_22 * vertex[2] + proj_32;
        double x_ndc = proj_00 * vertex[0] / w;
        double y_ndc = proj_11 * vertex[1] / w;

        // Convert to image space
        double projected_x = (x_ndc + 1.0) * 0.5 * image_width;
        double projected_y = image_height - ((y_ndc + 1.0) * 0.5 * image_height);

        vertexPosition[3 * v] = projected_x;
        vertexPosition[3 * v + 1] = projected_y;
        vertexPosition[3 * v + 2] = 0; // this is already the projection to 2D, so the 3rd parameter is not existing anymore
    }
}

void outputMeshToImage(BFM *bfm, const Eigen::Matrix<double, -1, 1> &vertices, const Eigen::VectorXi &triangles, const OptimizationParameters &params, cv::Mat image)
{

    return;
}