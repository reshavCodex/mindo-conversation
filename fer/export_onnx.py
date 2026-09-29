import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config_loader import load_config
from src.inference.predict import load_trained_model


MODEL_PATH = Path(__file__).parent / "saved_models" / "emotion_model_best.pth"
OUTPUT_PATH = Path(__file__).parent / "saved_models" / "emotion_model_best.onnx"


def main():
    cfg = load_config()
    device = torch.device("cpu")

    print("Loading trained FER model...")

    model, classes = load_trained_model(
        cfg,
        str(MODEL_PATH),
        device
    )

    model.eval()

    print("Model loaded successfully.")
    print("Classes:", classes)

    dummy_input = torch.randn(
        1,
        3,
        cfg.model.input_size,
        cfg.model.input_size
    )

    print(
        f"Exporting ONNX model with input shape: "
        f"{tuple(dummy_input.shape)}"
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {
                0: "batch_size"
            },
            "logits": {
                0: "batch_size"
            }
        },
        opset_version=17,
        dynamo=False
    )

    print("\n========== ONNX EXPORT ==========")
    print("Export successful.")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Classes: {classes}")
    print("=================================\n")


if __name__ == "__main__":
    main()