import cv2
import numpy as np
import matplotlib.pyplot as plt


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
