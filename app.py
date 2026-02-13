import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import json
from flask import Flask, request, jsonify, render_template

#reqirements.txt
# numpy
# pandas
# joblib
# Flask
# scikit-learn
# xgboost

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Create flask app
flask_app = Flask(__name__)

class_names = ['Absence', 'Presence']
n_active = 6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Function to Load and Preprocess an dataset ---
def prepare_dataset(age, sex, bp, max_hr, cp, fbs, chol, ekg, ex_ang, st_dep, slope_st, thal, no_fluro):
    # Creating data as a dictionary
    data = {
        "Age": age,
        "Sex": sex,
        "Chest pain type": cp,
        "BP": bp,
        "Cholesterol": chol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": ex_ang,
        "ST depression": st_dep,
        "Slope of ST": slope_st,
        "Number of vessels fluro": no_fluro,
        "Thallium": thal
    }
    newDf = pd.DataFrame(data, index=[0])

    # Select Features From MI
    with open(os.path.join(BASE_DIR, 'Models', 'selected_features.json'), 'r') as f:
        required_features = json.load(f)
    newDF = newDf[required_features]

    # Load the Clustering model
    bgm = joblib.load(os.path.join(BASE_DIR, 'Models', 'bgm_clustering_model.joblib'))

    # NOW ADD COLUMNS TO TEST SET
    bgm_probs_newDF = bgm.predict_proba(newDF)
    bgm_labels_newDF = bgm.predict(newDF)

    for i in range(n_active):
        newDF[f'bgm_prob_{i}'] = bgm_probs_newDF[:, i]
    newDF['bgm_cluster'] = bgm_labels_newDF

    # 1. Load everything
    cat_cols = joblib.load(os.path.join(BASE_DIR, 'Models', 'categorical_cols.pkl'))
    final_cols = joblib.load(os.path.join(BASE_DIR, 'Models', 'model_columns.pkl'))

    # 2. Encode strings to dummies
    new_data_encoded = pd.get_dummies(newDF, columns=cat_cols)

    # 3. Force match the training schema (Adds missing columns as 0, drops extras)
    processed_data = new_data_encoded.reindex(columns=final_cols, fill_value=0)
    processed_data = processed_data.apply(pd.to_numeric, errors='coerce')
    return processed_data


@flask_app.route("/")
def Home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    # Get the data uploaded
    age = float(request.form.get('age', 0))
    sex = float(request.form.get('sex', 0))
    bp = float(request.form.get('bp', 0))
    max_hr = float(request.form.get('max_hr', 0))
    cp = float(request.form.get('cp', 0))
    fbs = float(request.form.get('fbs', 0))
    chol = float(request.form.get('chol', 0))
    ekg = float(request.form.get('ekg', 0))
    ex_ang = float(request.form.get('ex_ang', 0))
    st_dep = float(request.form.get('st_dep', 0))
    slope_st = float(request.form.get('slope_st', 0))
    thal = float(request.form.get('thal', 0))
    no_fluro = float(request.form.get('no_fluro', 0))

    processed_data = prepare_dataset(age, sex, bp, max_hr, cp, fbs, chol, ekg, ex_ang, st_dep, slope_st, thal,
                                     no_fluro)
    print(processed_data)
    # Load XGBClassifier Model
    xgb_model = joblib.load(os.path.join(BASE_DIR, 'Models', 'best_xgboost_model.pkl'))

    # Predict
    preds_xgb = xgb_model.predict(processed_data)

    proba_xgb = xgb_model.predict_proba(processed_data)[:, 1]
    # Convert to percentage
    proba_xgb = (proba_xgb * 100).round(3)
    print(preds_xgb)
    print(proba_xgb)

    # Map indices to "Yes" or "No"
    # Typically: 0 = 'Absence',  1 = 'Presence'
    final_newDF_prediction = [class_names[idx] for idx in preds_xgb]
    print(final_newDF_prediction[0])


    return render_template('index.html',
                           status=final_newDF_prediction,
                           probability=proba_xgb[0])


if __name__ == "__main__":
    flask_app.run(debug=True)

#For Hugging face
# if __name__ == "__main__":
#     # Hugging Face Spaces uses port 7860 by default
#     flask_app.run(host="0.0.0.0", port=7860)

# https://huggingface.co/spaces/Dijeh08/heart_disease_prediction