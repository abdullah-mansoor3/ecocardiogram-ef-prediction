import yaml
from scripts.yolo_keypoint import get_min_max_area_frames
from scripts.predict_ef_resnet import predict_ef

def analyse_echo_video(video_path):
    """
    Analyze an echocardiogram video to predict ESV, EDV, EF and provide clinical analysis.
    
    Args:
        video_path (str): Path to the echo video file.
    
    Returns:
        dict: {
            'ESV': float,
            'EDV': float,
            'EF': float,
            'analysis': str
        }
    """
    # Load YOLO config to get model_path and other params
    with open("./configs/yolo_keypoint.yaml", "r") as f:
        yolo_cfg = yaml.safe_load(f)
    yolo_cfg["video_path"] = video_path  # override video path dynamically

    # Get min and max LV area frames
    min_frame, max_frame = get_min_max_area_frames(
        video_path=yolo_cfg["video_path"],
        model_path=yolo_cfg["model_path"],
        image_size=yolo_cfg["image_size"],
        interpolation_points=yolo_cfg["interpolation_points"],
        output_dir=yolo_cfg.get("output_dir", "output")  # Default to "output" if not specified
    )

    if min_frame is None or max_frame is None:
        print("Error: failed to extract ESV/EDV frames — no valid polygons found. Check model or input video.")
        return {
            'ESV': None,
            'EDV': None,
            'EF': None,
            'analysis': "Failed: no valid LV polygons detected."
        }

    min_area = min_frame[1]
    max_area = max_frame[1]
    min_keypoints = min_frame[-2]
    max_keypoints = max_frame[-2]
    min_image = min_frame[-1]
    max_image = max_frame[-1]

    # Predict ESV, EDV, EF using your regression model
    res = predict_ef(
        esv_image=min_image,
        edv_image=max_image,
        area_esv=min_area,
        area_edv=max_area,
        keypoints_esv=min_keypoints,
        keypoints_edv=max_keypoints
    )

    esv = res['ESV']
    edv = res['EDV']
    ef = res['EF']

    # Define normal clinical ranges (literature backed):
    # EF normal: 55-70% (American Heart Association)
    # ESV normal: ~20-50 mL (depending on body size; approximate for adults)
    # EDV normal: ~70-120 mL (approximate adult range)
    # Source references:
    # https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.108.190473
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4337419/

    ef_norm_low, ef_norm_high = 55, 70
    esv_norm_low, esv_norm_high = 20, 50
    edv_norm_low, edv_norm_high = 70, 120

    analysis = f"--- Echocardiogram Analysis ---\n"
    analysis += f"Predicted EDV: {edv:.2f} mL (Normal range: {edv_norm_low}-{edv_norm_high} mL)\n"
    analysis += f"Predicted ESV: {esv:.2f} mL (Normal range: {esv_norm_low}-{esv_norm_high} mL)\n"
    analysis += f"Predicted EF:  {ef:.2f}% (Normal range: {ef_norm_low}-{ef_norm_high}%)\n\n"

    # Analyze EF
    if ef < ef_norm_low:
        analysis += ("EF is below normal, indicating reduced left ventricular systolic function, "
                     "which may suggest heart failure, cardiomyopathy, or ischemic heart disease.\n")
    elif ef > ef_norm_high:
        analysis += ("EF is above normal range, which can sometimes be seen in hyperdynamic states "
                     "such as sepsis or anemia.\n")
    else:
        analysis += "EF is within the normal range, suggesting normal left ventricular function.\n"

    # Analyze ESV
    if esv < esv_norm_low:
        analysis += ("ESV is below the typical range, which may occur in small ventricles or "
                     "due to measurement variability.\n")
    elif esv > esv_norm_high:
        analysis += ("ESV is elevated, indicating poor systolic emptying, which is commonly seen in "
                     "heart failure or dilated cardiomyopathy.\n")
    else:
        analysis += "ESV is within the normal range.\n"

    # Analyze EDV
    if edv < edv_norm_low:
        analysis += ("EDV is below normal, possibly indicating reduced preload or hypovolemia.\n")
    elif edv > edv_norm_high:
        analysis += ("EDV is elevated, which can be due to volume overload, dilated ventricles, or "
                     "heart failure.\n")
    else:
        analysis += "EDV is within the normal range.\n"

    return {
        'ESV': esv,
        'EDV': edv,
        'EF': ef,
        'analysis': analysis
    }

if __name__ == "__main__":
    with open("./configs/yolo_keypoint.yaml", "r") as f:
        yolo_cfg = yaml.safe_load(f)

    result = analyse_echo_video(yolo_cfg["video_path"])
    print(result['analysis'])

