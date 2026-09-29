import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.inference.face_detector import FaceDetector


IMAGE_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\FER2013_CLEAN\angry\row_00000.png"
)


def main():
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    detector = FaceDetector()

    try:
        face = detector.detect_and_crop(image)

        print("\n========== FACE DETECTION TEST ==========")

        if face is None:
            print("No face detected.")
        else:
            print("Face detected successfully.")
            print(f"Original image size: {image.shape[1]} x {image.shape[0]}")
            print(f"Detected face size: {face.shape[1]} x {face.shape[0]}")

        print("=========================================\n")

    finally:
        detector.close()


if __name__ == "__main__":
    main()