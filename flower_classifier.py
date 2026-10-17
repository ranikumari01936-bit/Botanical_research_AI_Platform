from pathlib import Path

from PIL import Image
import streamlit as st


# ---------- Project paths ----------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "plant_ml" / "models"

MODEL_PATH = MODEL_DIR / "flower_classifier_v2.pth"
CLASSES_PATH = MODEL_DIR / "flower_classes.json"


# ---------- Load trained model ----------

@st.cache_resource(show_spinner=False, max_entries=1)
def load_flower_model():

    import json
    import torch
    import torch.nn as nn
    from torchvision import models

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint)
    model.to("cpu")
    model.eval()

    return model, class_names


@st.cache_resource(show_spinner=False, max_entries=1)
def load_image_transform():
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------- Predict flower ----------

def predict_flower(image: Image.Image):
    import torch

    model, class_names = load_flower_model()

    image = image.convert("RGB")
    image_tensor = load_image_transform()(image).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1,
        )

    flower_name = class_names[predicted_index.item()]
    confidence_score = confidence.item() * 100

    return {
        "flower": flower_name,
        "confidence": confidence_score,
    }