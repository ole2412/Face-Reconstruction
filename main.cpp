#include <sstream>
#include <iomanip>
#include <iostream>
#include "src/bfm.h"
#include "src/render.h"


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

    // TODO: load the face model
    BFM bfm = BFM();

    std::cout << bfm.get_shape_pca().rows() << " x " << bfm.get_shape_pca().cols() << std::endl;

    // TODO: Optimize (as a first step, just setup the class and return the mesh for default parameters setting (eg all model params = 0) and check if the mesh makes sense)

    // TODO: Render results back to image (how can we display the mesh on top of our image?)

    generateAndOutputMesh(&bfm, "../output/" + image_name + ".obj", landmark_path);

    // &bfm gives a pointer to the bfm object

    return 0;
}

