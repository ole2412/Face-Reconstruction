import torchlm
import cv2
import sys
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet


def save_landmarks_as_anl(landmarks, filename):
    """
    Save landmarks as .anl file

    Args:
        landmarks (list): Landmarks to save
        filename (str): Path to save the .anl file
    """
    with open(filename, 'w') as file:
        for landmark in landmarks:
            line = f"{landmark[0]} {landmark[1]}\n"
            file.write(line)


def setup_landmark_model(device, checkpoint):
    """
    Setup the landmark model

    Args:
        device (str): Device to use
        checkpoint (str): Path to the checkpoint
    """

    torchlm.runtime.bind(faceboxesv2(device=device))
    torchlm.runtime.bind(
        pipnet(
            backbone="resnet101",
            pretrained=True,
            num_nb=10,
            num_lms=68,
            net_stride=32,
            input_size=256,
            meanface_type="300w",
            backbone_pretrained=False,
            map_location=device,
            checkpoint=checkpoint
        )
    )


def draw_results(image, landmarks, bboxes):
    image = torchlm.utils.draw_bboxes(image, bboxes=bboxes)
    image = torchlm.utils.draw_landmarks(image, landmarks=landmarks, circle=5)
    return image


def main():
    if len(sys.argv) != 2:
        print("Usage: python landmark_detector.py <image_path>")
        sys.exit(1)
    folder_path = sys.argv[1].rsplit("/", 1)[0]
    file_name = sys.argv[1].split("/")[-1].split(".")[0]

    img_path = f"{folder_path}/{file_name}.jpg"
    save_path = f"{folder_path}/{file_name}_with_landmarks.jpg"
    anl_path = f"{folder_path}/{file_name}_landmarks.anl"
    checkpoint = "./models/PIPNet/pipnet_resnet101_10x68x32x256_300w.pth"
    device = "cpu"

    image = cv2.imread(img_path)
    setup_landmark_model(device, checkpoint)
    landmarks, bboxes = torchlm.runtime.forward(image)

    save_landmarks_as_anl(landmarks[0], anl_path)
    annotated_image = draw_results(image, landmarks, bboxes)
    cv2.imwrite(save_path, annotated_image)


if __name__ == "__main__":
    main()
