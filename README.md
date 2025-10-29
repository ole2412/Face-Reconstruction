# Face Reconstruction

This repository contains our implementation for the **3D Scanning & Motion Capture (IN2354)** course at TUM. The main objective is to reconstruct a 2D face model from a single RGB image and enable expression transfer between different images.

## Overview

1. **Facial Landmark Detection**  
   We begin by detecting facial landmarks on the input RGB image. These landmarks guide the alignment between the 2D image and the Basel Face Model (BFM).

2. **Basel Face Model Parameters**  
   The BFM provides predefined landmarks and a statistical model of facial shape, expression, and color. After detecting the image landmarks, we optimize the model parameters—camera pose, expression, shape, and color using:
   - A **sparse term** based on the detected facial landmarks
   - A **dense term** based on the image color
   - A **statistical regularizer** to maintain realistic deformations

3. **Expression Transfer**  
   Once the 2D-to-3D reconstruction is complete, the resulting model can be used to transfer facial expressions from one person’s image to another. The target image then adopts the same facial expression as the source image while preserving its own identity.

## Dependencies

Install the necessary python libraries with: 
```
pip install -r requirements.txt
```

### PIPNet
Download `pipnet_resnet101_10x68x32x256_300w.pth` from https://github.com/DefTruth/torchlm/releases/tag/torchlm-0.1.6-alpha.

## Download BMF Model and 68 Landmarks

Download the Basel Face Model (BFM) and the 68 facial landmarks from the following Google Drive folder:

[Download BMF Model and 68 Landmarks](https://drive.google.com/drive/folders/1Gt5wggAwxiBhJSeqt2tTcLxUhE5Vk45W?usp=drive_link)

#### Source Images
To be able to run everything, the landmarks need to be extracted using python and PIPNet.

For own images, copy your jpg file into the datasets folder and run the following command from the project directory:
````
python3 /tools/landmark_detector.py ./datasets/<ImageName>.jpg
````
This will create a file with the 68 landmarks `<ImageName>_landmarks.anl` and a file with the landmarks displayed on the input image `<ImageName>_with_landmarks.jpg`.

## Folder structure
```
- datasets
- lib
- models
    - PIPNet
        - pipnet_resnet101_10x68x32x256_300w.pth
    - BFM17
        - model2017-1_face12_nomouth.h5
- src
- tools
```


