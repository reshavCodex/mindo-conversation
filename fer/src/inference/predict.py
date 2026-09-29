"""Single-image inference before webcam integration (Phase 8)."""
import argparse
import torch
from PIL import Image

from src.config_loader import load_config
from src.data.transforms import get_eval_transforms
from src.model.efficientnet_b0 import build_model


def load_trained_model(cfg, checkpoint_path, device):
    model = build_model(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    classes = checkpoint.get("classes", cfg.classes)
    return model, classes


@torch.no_grad()
def predict_image(model, classes, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = probs.argmax()
    return classes[pred_idx], {c: float(p) for c, p in zip(classes, probs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--checkpoint", default="saved_models/emotion_model_best.pth")
    args = parser.parse_args()

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes = load_trained_model(cfg, args.checkpoint, device)
    transform = get_eval_transforms(cfg)

    label, probs = predict_image(model, classes, args.image, transform, device)
    print(f"Predicted emotion: {label}")
    for c, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {c:10s} {p:.4f}")


if __name__ == "__main__":
    main()
