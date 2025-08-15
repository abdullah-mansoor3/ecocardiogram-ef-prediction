import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from ultralytics import YOLO
from pathlib import Path
import os

def load_model(model_path):
    return YOLO(model_path)

def draw_keypoints(image, keypoints, labels=True):
    """
    Draw only the polygon (no individual keypoint markers). Returns annotated image.
    """
    img_copy = image.copy()
    if keypoints is None:
        return img_copy

    # polygon only, convert to int
    pts = keypoints.astype(np.int32)
    if pts.shape[0] >= 2:
        # draw a semi-transparent filled polygon + thin outline
        overlay = img_copy.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
        alpha = 0.18
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
        cv2.polylines(img_copy, [pts], True, (0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
    return img_copy

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

def get_min_max_area_frames(video_path, model_path, image_size=128, interpolation_points=42, 
                            output_dir="output", labels=True, upscale_factor=1.0):
    import shutil
    import subprocess

    # Prepare output dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frame_images"
    frames_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # Restore AVI (MJPG) writer as primary output
    codec, ext = 'MJPG', '.avi'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_out_path = output_dir / f"keypoints_visualization{ext}"
    out = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height), True)
    if out is None or not out.isOpened():
        out = None
        video_out_path = None
        print("Could not open MJPG writer. PNG frames will still be saved to output/frame_images/")

    frame_areas = []
    min_area = float('inf')
    max_area = float('-inf')
    min_frame_data = None
    max_frame_data = None

    try:
        for frame_num in tqdm(range(total_frames), desc=f"Processing {video_path}"):
            ret, frame = cap.read()
            if not ret:
                break

            # sanitize frame
            if frame is None:
                continue
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

            # get keypoints on original-size frame
            keypoints = extract_keypoints(frame, model, image_size)

            # compute polygon and area in original scale
            area = None
            polygon_i = None
            if keypoints is not None:
                sorted_points = sort_polygon_points(keypoints)
                polygon = interpolate_polygon(sorted_points, interpolation_points)
                if polygon is not None and polygon.size != 0 and np.isfinite(polygon).all():
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                    polygon_i = polygon.astype(np.int32)
                    if polygon_i.shape[0] >= 3:
                        area = float(cv2.contourArea(polygon_i))
                    else:
                        area = None
                else:
                    polygon_i = None

            # annotated frame = original resolution (no upscaling)
            annotated_frame = frame.copy()

            if polygon_i is not None and polygon_i.size and np.isfinite(polygon_i).all():
                # ensure contiguous int32 array shaped (N,2)
                try:
                    scaled_i = np.ascontiguousarray(polygon_i.astype(np.int32))
                    if scaled_i.ndim == 2 and scaled_i.shape[1] == 2 and scaled_i.shape[0] >= 3:
                        overlay = annotated_frame.copy()
                        cv2.fillPoly(overlay, [scaled_i], color=(0, 255, 0))
                        cv2.addWeighted(overlay, 0.18, annotated_frame, 0.82, 0, annotated_frame)
                        cv2.polylines(annotated_frame, [scaled_i], True, (0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
                    elif scaled_i.shape[0] == 2:
                        cv2.polylines(annotated_frame, [scaled_i], False, (0, 200, 0), thickness=1, lineType=cv2.LINE_AA)
                    else:
                        x, y = int(scaled_i[0, 0]), int(scaled_i[0, 1])
                        cv2.rectangle(annotated_frame, (x-1, y-1), (x+1, y+1), (0, 255, 0), -1)
                except Exception:
                    pass
            else:
                cv2.putText(annotated_frame, "No LV polygon", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # area text: small and bottom-right
            if area is not None:
                area_text = f"Area: {area:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (tw, th), _ = cv2.getTextSize(area_text, font, font_scale, thickness)
                margin = 8
                x = width - tw - margin
                y = height - margin
                cv2.rectangle(annotated_frame, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
                cv2.putText(annotated_frame, area_text, (x, y), font, font_scale, (200, 255, 200), thickness, lineType=cv2.LINE_AA)

            # Write to AVI if writer available
            if out is not None:
                try:
                    out.write(annotated_frame)
                except Exception:
                    out.release()
                    out = None
                    video_out_path = None
                    print("AVI VideoWriter write failed; continuing and keeping PNG frames.")

            # Always save PNG frame for robust fallback
            frame_file = frames_dir / f"frame_{frame_num:06d}.png"
            cv2.imwrite(str(frame_file), annotated_frame)

            # Update min/max using valid areas only
            if area is not None:
                if area < min_area:
                    min_area = area
                    min_frame_data = (frame_num, area, keypoints, frame.copy())
                if area > max_area:
                    max_area = area
                    max_frame_data = (frame_num, area, keypoints, frame.copy())
                frame_areas.append((frame_num, area, keypoints, frame.copy()))

    except Exception as e:
        print(f"Error during frame processing: {e}")

    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

    if len(frame_areas) == 0:
        print("No frames with keypoints were processed.")
        return None, None

    # Save ESV (min) and EDV (max) frames (original size, polygon-only)
    def save_selected_frame(frame_data, name):
        fn, a, kps, img = frame_data
        if kps is None:
            return
        sorted_points = sort_polygon_points(kps)
        poly = interpolate_polygon(sorted_points, interpolation_points)
        if poly is None or poly.size == 0 or not np.isfinite(poly).all():
            return
        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
        poly_i = poly.astype(np.int32)
        out_img = img.copy()
        try:
            if poly_i.shape[0] >= 3:
                overlay = out_img.copy()
                cv2.fillPoly(overlay, [poly_i], color=(0, 255, 0))
                cv2.addWeighted(overlay, 0.18, out_img, 0.82, 0, out_img)
                cv2.polylines(out_img, [poly_i], True, (0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
            elif poly_i.shape[0] == 2:
                cv2.polylines(out_img, [poly_i], False, (0, 200, 0), thickness=1, lineType=cv2.LINE_AA)
        except Exception:
            pass
        (tw, th), _ = cv2.getTextSize(f"Area: {a:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        margin = 8
        x = width - tw - margin
        y = height - margin
        cv2.rectangle(out_img, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(out_img, f"Area: {a:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(output_dir / name), out_img)

    if min_frame_data is not None:
        save_selected_frame(min_frame_data, "esv_frame.png")
    else:
        print("No ESV frame found to save.")

    if max_frame_data is not None:
        save_selected_frame(max_frame_data, "edv_frame.png")
    else:
        print("No EDV frame found to save.")

    print(f"Saved visualization to: {video_out_path if video_out_path is not None else 'PNG frames in output/frame_images/ (no AVI)'}")
    return min_frame_data, max_frame_data
