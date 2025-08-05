import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def extract_keypoints(img, model, imgsz):
    results = model.predict(source=img, save=False, imgsz=imgsz, verbose=False)
    keypoints = results[0].keypoints.data.cpu().numpy()
    if keypoints.shape[0] == 0:
        return None
    return keypoints[0][:, :2]  # shape (N, 2)

def sort_polygon_points(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    return points[sorted_idx]

def interpolate_polygon(points, target_len):
    if len(points) < target_len:
        idxs = np.linspace(0, len(points) - 1, num=target_len)
        interp_x = interp1d(np.arange(len(points)), points[:, 0], kind='linear')(idxs)
        interp_y = interp1d(np.arange(len(points)), points[:, 1], kind='linear')(idxs)
        return np.stack([interp_x, interp_y], axis=-1)
    else:
        return points[:target_len]

def get_min_max_area_frames(video_path, model_path, image_size=128, interpolation_points=42):
    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_areas = []

    for frame_num in tqdm(range(total_frames), desc=f"Processing {video_path}"):
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(frame, model, image_size)
        if keypoints is None:
            continue

        sorted_points = sort_polygon_points(keypoints)
        polygon = interpolate_polygon(sorted_points, interpolation_points)
        area = cv2.contourArea(polygon.astype(np.int32))

        frame_areas.append((frame_num, area))

    cap.release()

    if len(frame_areas) == 0:
        return None, None  # No valid frames found

    min_frame = min(frame_areas, key=lambda x: x[1])
    max_frame = max(frame_areas, key=lambda x: x[1])

    return min_frame, max_frame  # (frame_num, area)
