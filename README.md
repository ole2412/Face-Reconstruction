# Face Reconstruction

## Dependencies

Install the necessary python libraries with: 
```
pip install -r requirements.txt
```

### PIPNet
Download `pipnet_resnet101_10x68x32x256_300w.pth` from https://github.com/DefTruth/torchlm/releases/tag/torchlm-0.1.6-alpha.

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
- src
- tools
```
