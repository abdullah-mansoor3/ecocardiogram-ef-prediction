import yaml
from scripts.yolo_keypoint import get_min_max_area_frames
from scripts.predict_ef_resnet import predict_ef

# Load YOLO config
with open("./configs/yolo_keypoint.yaml", "r") as f:
    yolo_cfg = yaml.safe_load(f)

# Get min/max frames + areas
min_frame, max_frame = get_min_max_area_frames(
    video_path=yolo_cfg["video_path"],
    model_path=yolo_cfg["model_path"],
    image_size=yolo_cfg["image_size"],
    interpolation_points=yolo_cfg["interpolation_points"]
)

min_area = min_frame[1]
max_area = max_frame[1]
min_keypoints = min_frame[-2]
max_keypoints = max_frame[-2]
min_image = min_frame[-1]
max_image = max_frame[-1]

print(f"🟢 Min area frame: #{min_frame[0]} with area {min_area:.2f}")
print(f"🔴 Max area frame: #{max_frame[0]} with area {max_area:.2f}")

# Load EF prediction config
with open("./configs/predict_ef.yaml", "r") as f:
    ef_cfg = yaml.safe_load(f)

# Predict EDV, ESV, EF
# edv, esv, ef = predict_ef(
#     min_area=min_area,
#     max_area=max_area,
#     edv_model_path=ef_cfg["edv_model_path"],
#     esv_model_path=ef_cfg["esv_model_path"]
# )
res = predict_ef(
    esv_image=min_image,
    edv_image=max_image,
    area_esv=min_area,
    area_edv=max_area,
    keypoints_esv=min_keypoints,
    keypoints_edv=max_keypoints
)


print(f"\n📈 EDV: {res['EDV']:.2f} mL")
print(f"📉 ESV: {res['ESV']:.2f} mL")
print(f"🫀 EF:  {res['EF']:.2f}%")
