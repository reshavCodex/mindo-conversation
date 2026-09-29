import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config_loader import load_config
from src.data.transforms import get_eval_transforms
from src.inference.predict import load_trained_model, predict_image


MODEL_PATH = Path(__file__).parent / "saved_models" / "emotion_model_best.pth"

IMAGE_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\FER2013_CLEAN\angry\row_00000.png"
)


def main():
    cfg = load_config()
    device = torch.device("cpu")

    model, classes = load_trained_model(
        cfg,
        str(MODEL_PATH),
        device
    )

    transform = get_eval_transforms(cfg)

    label, probabilities = predict_image(
        model,
        classes,
        str(IMAGE_PATH),
        transform,
        device
    )

    print("\n========== FER TEST ==========")
    print(f"Image: {IMAGE_PATH.name}")
    print("Expected emotion: angry")
    print(f"Predicted emotion: {label}")

    print("\nEmotion probabilities:")

    for emotion, probability in sorted(
        probabilities.items(),
        key=lambda x: -x[1]
    ):
        print(f"{emotion:>8}: {probability:.4f}")

    print("==============================\n")


if __name__ == "__main__":
    main()