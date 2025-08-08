import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import json
import numpy as np
import yaml


# ─── CONFIG ────────────────────────────────────────────────────────────────────
# Load YOLO config
with open("./configs/predict_ef_resnet.yaml", "r") as f:
    config = yaml.safe_load(f)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── MODEL ─────────────────────────────────────────────────────────────────────
class DualVolumeRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared image backbone
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # modify stem to accept grayscale if needed:
        # backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove FC

        # Shared keypoint MLP
        self.kp_fc = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3)
        )
        # Shared area MLP
        self.area_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Dropout(0.3)
        )

        # Combined regressor head
        total_feat = 2 * (512 + 128 + 16)
        self.head = nn.Sequential(
            nn.Linear(total_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),   # predict [esv, edv]
            nn.ReLU()
        )

    def forward(self, img_esv, area_esv, kp_esv, img_edv, area_edv, kp_edv):
        f_esv = self.img_encoder(img_esv).flatten(1)
        k_esv = self.kp_fc(kp_esv)
        a_esv = self.area_fc(area_esv)

        f_edv = self.img_encoder(img_edv).flatten(1)
        k_edv = self.kp_fc(kp_edv)
        a_edv = self.area_fc(area_edv)

        x = torch.cat([f_esv, k_esv, a_esv, f_edv, k_edv, a_edv], dim=1)
        return self.head(x)

# ─── DATASET ───────────────────────────────────────────────────────────────────
class DualEFVolumeDataset(Dataset):
    def __init__(self, metadata_path, image_dir, stats_path="input_stats.yaml"):
        with open(metadata_path, 'r') as f:
            self.entries = json.load(f)
        self.img_dir = Path(image_dir)

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        stats = config['stats_path']
        self.kp_mean   = torch.tensor(stats['kp_mean'], dtype=torch.float32)
        self.kp_std    = torch.tensor(stats['kp_std'],  dtype=torch.float32)
        self.area_mean = float(stats['area_mean'])
        self.area_std  = float(stats['area_std'])

    def __len__(self):
        return len(self.entries) // 2

    def __getitem__(self, idx):
        esv_entry = self.entries[2 * idx]
        edv_entry = self.entries[2 * idx + 1]

        def load(e):
            name = f"{e['video_id']}_{e['label']}_f{e['frame_num']}.jpg"
            img = Image.open(self.img_dir / name).convert("L")
            img = self.transform(img)
            kp = (torch.tensor(e["keypoints"], dtype=torch.float32) - self.kp_mean) / self.kp_std
            area = (torch.tensor([e["area"]], dtype=torch.float32) - self.area_mean) / self.area_std
            vol = torch.tensor([e["volume"]], dtype=torch.float32)
            return img, area, kp, vol

        img_esv, area_esv, kp_esv, vol_esv = load(esv_entry)
        img_edv, area_edv, kp_edv, vol_edv = load(edv_entry)

        return (img_esv, area_esv, kp_esv, img_edv, area_edv, kp_edv), \
               torch.tensor([vol_esv, vol_edv], dtype=torch.float32)
    
# ─── PREDICTION FUNCTION ────────────────────────────────────────────────────────
def predict_ef(esv_image, edv_image, area_esv, keypoints_esv, area_edv, keypoints_edv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load normalization stats
    with open(config['stats_path'], "r") as f:
        stats = yaml.safe_load(f)
    kp_mean = torch.tensor(stats['kp_mean'], dtype=torch.float32).to(device)
    kp_std = torch.tensor(stats['kp_std'], dtype=torch.float32).to(device)
    area_mean = float(stats['area_mean'])
    area_std = float(stats['area_std'])

    # Load model
    model = DualVolumeRegressor().to(device)
    model.load_state_dict(torch.load(config["best_model_path"], map_location=device))
    model.eval()

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def prep_image(img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)
        return img

    esv_img_tensor = prep_image(esv_image)
    edv_img_tensor = prep_image(edv_image)

    # Normalize keypoints and area
    kp_esv_tensor = (torch.tensor(keypoints_esv, dtype=torch.float32).flatten().to(device) - kp_mean) / kp_std
    kp_edv_tensor = (torch.tensor(keypoints_edv, dtype=torch.float32).flatten().to(device) - kp_mean) / kp_std

    area_esv_tensor = torch.tensor([(area_esv - area_mean) / area_std], dtype=torch.float32).to(device)
    area_edv_tensor = torch.tensor([(area_edv - area_mean) / area_std], dtype=torch.float32).to(device)

    # Ensure batch dimension
    if kp_esv_tensor.dim() == 1:
        kp_esv_tensor = kp_esv_tensor.unsqueeze(0)
    if kp_edv_tensor.dim() == 1:
        kp_edv_tensor = kp_edv_tensor.unsqueeze(0)
    if area_esv_tensor.dim() == 1:
        area_esv_tensor = area_esv_tensor.unsqueeze(0)
    if area_edv_tensor.dim() == 1:
        area_edv_tensor = area_edv_tensor.unsqueeze(0)

    with torch.no_grad():
        preds = model(esv_img_tensor, area_esv_tensor, kp_esv_tensor,
                      edv_img_tensor, area_edv_tensor, kp_edv_tensor)
        pred_esv = preds[0, 0].item()
        pred_edv = preds[0, 1].item()

    ef = (pred_edv - pred_esv) / pred_edv * 100 if pred_edv != 0 else 0

    return {
        "ESV": pred_esv,
        "EDV": pred_edv,
        "EF": ef
    }