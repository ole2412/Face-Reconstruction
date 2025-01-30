#include <sstream>
#include <iomanip>
#include <iostream>
#include "src/bfm.h"
#include "src/render.h"
#include <opencv2/opencv.hpp>


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: main <image_name>" << std::endl;
        return -1;
    }

    std::string image_name = argv[1];
    std::string image_path = "../datasets/" + image_name + ".jpg";
    std::string landmark_path = "../datasets/" + image_name + "_landmarks.anl";
    std::string output_path = "../output/" + image_name + "_output.jpg";

    std::cout << "Output path: " << output_path << std::endl;

    BFM bfm = BFM();

    std::cout << bfm.get_shape_pca().rows() << " x " << bfm.get_shape_pca().cols() << std::endl;


    cv::Mat image = cv::imread(image_path);
    generateAndOutputMesh(&bfm, "../output/" + image_name + ".obj", landmark_path, image);
    
    // store and show the output image
    cv::imwrite(output_path, image);
    cv::waitKey(0);

    return 0;
}

