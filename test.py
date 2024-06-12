import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

import bodypix
import lasr_vision_feature_extraction
from lasr_vision_feature_extraction.categories_and_attributes import CelebAMaskHQCategoriesAndAttributes, DeepFashion2GeneralizedCategoriesAndAttributes


def capture_rgb_image() -> np.ndarray or None:
    # Initialize the camera capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None

    try:
        # Capture a single frame
        ret, frame = cap.read()

        if ret:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.array(rgb_frame)
            return rgb_frame
        else:
            print("Error: Failed to capture image")
            return None
    finally:
        # Release the camera
        cap.release()


# Usage example:
if __name__ == "__main__":
    rgb_image = capture_rgb_image()
    if rgb_image is not None:
        plt.imshow(rgb_image)
        plt.show()

    head_model = lasr_vision_feature_extraction.load_face_classifier_model()
    head_predictor = lasr_vision_feature_extraction.Predictor(head_model, torch.device('cpu'),
                                                              CelebAMaskHQCategoriesAndAttributes)
    cloth_model = lasr_vision_feature_extraction.load_cloth_classifier_model()
    cloth_model.return_bbox = False  # unify returns
    cloth_predictor = lasr_vision_feature_extraction.Predictor(cloth_model, torch.device('cpu'),
                                                               DeepFashion2GeneralizedCategoriesAndAttributes)
    request_masks = ["torso_front", "torso_back", "left_face", "right_face"]
    request_poses = []
    masks, poses = bodypix.detect(rgb_image, request_masks, request_poses)
    print(masks, poses)

    rst = lasr_vision_feature_extraction.predict_frame(
            rgb_image,
            head_predictor=head_predictor,
            cloth_predictor=cloth_predictor,
        )

    print(rst)
