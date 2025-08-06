import argparse
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import yaml

with open("./configs/predict_ef_resnet.yaml", "r") as f:
    config = yaml.safe_load(f)

# --------------------- Model ---------------------
class VolumeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.keypoint_fc = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.area_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(512 + 128 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, area, keypoints):
        x_img = self.resnet_backbone(image).squeeze()
        x_kp = self.keypoint_fc(keypoints)
        x_area = self.area_fc(area)
        x = torch.cat([x_img, x_kp, x_area], dim=1)
        return self.combined_fc(x).squeeze(1)



# --------------------------
# Load model from checkpoint
# --------------------------
def load_model(checkpoint_path, device):
    model = VolumeRegressor().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# ---------------------
# Load and preprocess image
# ---------------------
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# -------------------------
# Predict EF = (EDV - ESV)/EDV
# -------------------------
def predict_ef(esv_image,edv_image, area_esv, keypoints_esv, area_edv, keypoints_edv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model_esv = load_model(config["model_esv_path"], device)
    model_edv = load_model(config["model_edv_path"], device)

    # Load image
    esv_image_tensor = load_image(esv_image.unsqueeze(0).to(device)
    edv_image_tensor = load_image(edv_image).unsqueeze(0).to(device)

    # Prepare ESV input
    area_esv_tensor = torch.tensor([[area_esv]], dtype=torch.float32).to(device)
    keypoints_esv_tensor = torch.tensor([keypoints_esv.flatten()], dtype=torch.float32).to(device)

    # Prepare EDV input
    area_edv_tensor = torch.tensor([[area_edv]], dtype=torch.float32).to(device)
    keypoints_edv_tensor = torch.tensor([keypoints_edv.flatten()], dtype=torch.float32).to(device)

    # Predict volumes
    with torch.no_grad():
        pred_esv = model_esv(esv_image_tensor, area_esv_tensor, keypoints_esv_tensor).item()
        pred_edv = model_edv(edv_image_tensor, area_edv_tensor, keypoints_edv_tensor).item()

    # Compute EF
    ef = (pred_edv - pred_esv) / pred_edv * 100 if pred_edv != 0 else 0
    return {
        "ESV": pred_esv,
        "EDV": pred_edv,
        "EF": ef
    }