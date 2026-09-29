import sys
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config_loader import load_config
from src.data.transforms import get_eval_transforms
from src.inference.predict import load_trained_model


MODEL_PATH = Path(__file__).parent / "saved_models" / "emotion_model_best.pth"
ONNX_PATH = Path(__file__).parent / "saved_models" / "emotion_model_best.onnx"

IMAGE_PATH = Path(
    r"C:\Users\user\Desktop\LUCY\AI Unleashed 2026\datasets\FER2013_CLEAN\angry\row_00000.png"
)


def softmax_numpy(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def main():
    cfg = load_config()
    device = torch.device("cpu")

    transform = get_eval_transforms(cfg)

    image = Image.open(IMAGE_PATH).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    # --------------------------------------------------
    # PyTorch
    # --------------------------------------------------

    model, classes = load_trained_model(
        cfg,
        str(MODEL_PATH),
        device
    )

    with torch.no_grad():
        pytorch_logits = model(tensor)
        pytorch_probs = torch.softmax(
            pytorch_logits,
            dim=1
        ).cpu().numpy()[0]

    pytorch_index = np.argmax(pytorch_probs)
    pytorch_prediction = classes[pytorch_index]

    # --------------------------------------------------
    # ONNX
    # --------------------------------------------------

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    onnx_logits = session.run(
        None,
        {
            input_name: tensor.numpy()
        }
    )[0][0]

    onnx_probs = softmax_numpy(onnx_logits)

    onnx_index = np.argmax(onnx_probs)
    onnx_prediction = classes[onnx_index]

    # --------------------------------------------------
    # Comparison
    # --------------------------------------------------

    differences = np.abs(
        pytorch_probs - onnx_probs
    )

    print("\n========== PYTORCH vs ONNX ==========")

    print(f"Image: {IMAGE_PATH.name}")

    print("\nPredictions:")
    print(f"PyTorch : {pytorch_prediction}")
    print(f"ONNX    : {onnx_prediction}")

    print("\nProbability comparison:")

    print(
        f"{'Emotion':>10} "
        f"{'PyTorch':>12} "
        f"{'ONNX':>12} "
        f"{'Difference':>12}"
    )

    print("-" * 50)

    for emotion, pt, ox, diff in zip(
        classes,
        pytorch_probs,
        onnx_probs,
        differences
    ):
        print(
            f"{emotion:>10} "
            f"{pt:>12.6f} "
            f"{ox:>12.6f} "
            f"{diff:>12.6f}"
        )

    max_difference = np.max(differences)

    print("\nMaximum probability difference:")
    print(f"{max_difference:.8f}")

    print("\nPrediction match:")
    print(pytorch_prediction == onnx_prediction)

    print("=====================================\n")


if __name__ == "__main__":
    main()