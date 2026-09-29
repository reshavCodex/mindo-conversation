import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config_loader import load_config
from src.data.transforms import get_eval_transforms
from src.inference.face_detector import FaceDetector
from src.inference.predict import load_trained_model
from src.model.efficientnet_b0 import build_model


MODEL_PATH = Path(__file__).parent / "saved_models" / "emotion_model_best.pth"

IMAGE_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\FER2013_CLEAN\angry\row_00000.png"
)


def main():
    cfg = load_config()
    device = torch.device("cpu")

    # Load model
    model, classes = load_trained_model(
        cfg,
        str(MODEL_PATH),
        device
    )

    transform = get_eval_transforms(cfg)

    # Load image
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    # Detect face
    detector = FaceDetector()

    try:
        face = detector.detect_and_crop(image)

        if face is None:
            print("No face detected.")
            return

        print("\n========== FULL FER PIPELINE ==========")
        print("Face detected successfully.")

        # Convert OpenCV BGR → RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Save temporary face crop
        face_image_path = Path(__file__).parent / "test_face_crop.jpg"
        cv2.imwrite(str(face_image_path), face)

        # Run model using the existing inference logic
        from PIL import Image

        pil_face = Image.fromarray(face_rgb)

        tensor = transform(pil_face).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        probabilities = probabilities.cpu().numpy()

        predicted_index = probabilities.argmax()
        predicted_emotion = classes[predicted_index]

        print(f"Original image : {IMAGE_PATH.name}")
        print(f"Face crop      : {face.shape[1]} x {face.shape[0]}")
        print(f"Prediction     : {predicted_emotion}")

        print("\nEmotion probabilities:")

        for emotion, probability in sorted(
            zip(classes, probabilities),
            key=lambda x: -x[1]
        ):
            print(f"{emotion:>8}: {probability:.4f}")

        print("\nFace crop saved to:")
        print(face_image_path)

        print("=======================================\n")

    finally:
        detector.close()


if __name__ == "__main__":
    main()