#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

static std::vector<dlib::full_object_detection> DetectLandmarks(
    const std::string& imagePath = "", 
    bool presentLandmarks = false, 
    bool holdImg = true
) {
    std::vector<dlib::full_object_detection> empty;
    
    try {
        cv::Mat temp;
        cv::VideoCapture cap;

        if (!imagePath.empty()) {
            temp = cv::imread(imagePath);
            if (temp.empty()) {
                std::cerr << "Failed to load image: " << imagePath << std::endl;
                return empty;
            }
        } else {
            cap.open(0); // Use webcam
            if (!cap.isOpened()) {
                std::cerr << "Unable to connect to camera\n";
                return empty;
            }
            cap.read(temp);
        }

        // Load face detection and pose estimation models.
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("/Users/Luka/Downloads/shape_predictor_68_face_landmarks.dat") >> pose_model;
        
        dlib::cv_image<dlib::bgr_pixel> cimg(temp);
        std::vector<dlib::rectangle> faces = detector(cimg);

        std::vector<dlib::full_object_detection> shapes;
        for (const auto& face : faces) {
            shapes.push_back(pose_model(cimg, face));
        }

        for (const auto& shape : shapes) {
            for (unsigned long j = 0; j < shape.num_parts(); ++j) {
                std::cout << shape.part(j).x() << ", " << shape.part(j).y() << std::endl;
            }
        }

        if (imagePath.empty()) {  // Webcam mode: loop until the window closes
            while (true) {
                cap.read(temp);
                if (temp.empty()) break;
                cimg = dlib::cv_image<dlib::bgr_pixel>(temp);
                faces = detector(cimg);

                shapes.clear();
                for (const auto& face : faces) {
                    shapes.push_back(pose_model(cimg, face));
                }

                if (shapes.empty()) break; // Exit if no face is detected
            }
        }

        return shapes;
    } catch (const dlib::serialization_error& e) {
        std::cerr << "Missing shape predictor model. Download it from:\n"
                  << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n";
        return empty;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return empty;
    }
}