import joblib

def load_model(model_path):
    return joblib.load(model_path)

def predict_volume(area, model):
    return model.predict([[area]])[0]

def calculate_ef(edv, esv):
    return 100 * (edv - esv) / edv if edv != 0 else 0

def predict_ef(min_area, max_area, edv_model_path, esv_model_path):
    # Load models
    edv_model = load_model(edv_model_path)
    esv_model = load_model(esv_model_path)

    # Predict volumes
    edv = predict_volume(max_area, edv_model)  # max area → EDV
    esv = predict_volume(min_area, esv_model)  # min area → ESV

    # Calculate EF
    ef = calculate_ef(edv, esv)

    return edv, esv, ef
