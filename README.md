# ecocardiogram-ef-prediction

This project predicts the ejection fraction (EF) from echocardiogram videos using a deep learning pipeline. The EF is a key clinical metric for assessing cardiac function.

## Methodology

The pipeline consists of several stages:

### 1. YOLO Keypoint Estimation

- A YOLO model is trained to detect keypoints outlining the left ventricle in each frame of the echocardiogram video.
- The model is loaded and used to process each frame, extracting keypoints and calculating the area enclosed by the ventricle contour.

### 2. Frame Selection

- For each video, the pipeline identifies the frames with minimum and maximum left ventricular area, corresponding to end-systolic (ESV) and end-diastolic (EDV) phases.
- This is done using [`get_min_max_area_frames`](full-pipeline/scripts/yolo_keypoint.py) from [full-pipeline/scripts/yolo_keypoint.py](full-pipeline/scripts/yolo_keypoint.py).

### 3. Regression Model Prediction

- The selected ESV and EDV frames, their areas, and keypoints are passed to a regression model (ResNet-based) to predict the actual ESV, EDV volumes, and the ejection fraction (EF).
- This is implemented in [`predict_ef`](full-pipeline/scripts/predict_ef_resnet.py) from [full-pipeline/scripts/predict_ef_resnet.py](full-pipeline/scripts/predict_ef_resnet.py).

### 4. Clinical Analysis

- The predicted values are compared to normal clinical ranges.
- The pipeline generates a textual analysis explaining whether the values are normal or indicative of possible cardiac conditions.
