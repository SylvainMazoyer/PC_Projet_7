# import urllib
# from urllib import response

import pickle
import json

import flask
from flask import Flask, jsonify, request

import pandas as pd
import numpy as np

import sklearn
from lightgbm import LGBMClassifier
import shap

# from pathlib import Path
# from sklearn.preprocessing import RobustScaler
# from imblearn.pipeline import Pipeline


# PATH
DATA_PATH = "DATA/"
MODEL_PATH = "MODEL/"

# FILES
MODEL_FILE = "lgbmc_final_V2.pkl"
DATA_FILE = "df_10pc.csv"
EXPLAINER_FILE = "explainer_lgbmc_final_V2.pkl"


# READ OBJECTS

df = pd.read_csv(DATA_PATH + DATA_FILE)  # df
with open(MODEL_PATH + MODEL_FILE, "rb") as f:  # estimator
    estimator = pickle.load(f)
with open(MODEL_PATH + EXPLAINER_FILE, "rb") as f:  # explainer
    explainer = pickle.load(f)


# vars
list_columns = df.columns.tolist()
list_columns.remove("SK_ID_CURR")
list_columns.remove("TARGET")
select_features = list_columns
# sku = 0


# init flask
app = Flask(__name__)


# route test
@app.route("/hello", methods=["GET"])
def hello():

    return ("<h1> Welcome </h1>", 200)


# send client id
@app.route("/send_sku", methods=["GET"])
def send_sku():

    serie_sku = df.iloc[0:50, 0]
    dict_sku = serie_sku.to_dict()
    json_dict_sku = jsonify(dict_sku)

    return json_dict_sku


@app.route("/predict", methods=["GET"])
def predict():

    # recupere id du client
    json_sku = json.loads(request.data)  # remplacé par request.json()
    sku = int(json_sku["sku"])
    print("sku :" + str(sku))

    # extract features for the sku
    X_sku = df.loc[(df["SK_ID_CURR"] == sku), select_features]

    # make preds
    result_pred = estimator.predict(X_sku)[0]
    result_pred_proba = estimator.predict_proba(X_sku)[0]

    print("result pred :" + str(result_pred))
    if result_pred == 0:
        result_reimbursement = "Ok"
    else:
        result_reimbursement = "Not ok"

    # Utiliser module Loging, log écrit dans un fichier log
    print("Reimbursement : " + result_reimbursement)

    # build json
    pred_dict = {"pred": int(result_pred), "proba_0": result_pred_proba[0]}
    json_pred = json.dumps(pred_dict)
    print(pred_dict)

    return json_pred


# send feature importance
@app.route("/return_shap_data", methods=["GET"])
def return_shap_data():

    # recupere id du client
    json_sku = json.loads(request.data)  # remplacé par request.json()

    sku = int(json_sku["sku"])
    print("sku :" + str(sku))

    # extract features for the sku
    X_sku = df.loc[(df["SK_ID_CURR"] == sku), select_features]

    # shap
    shap_value = explainer(X_sku)[0]

    shap_data = pd.DataFrame(
        np.array([abs(shap_value.values), shap_value.values, shap_value.data.round(3)]).T,
        index=shap_value.feature_names,
        columns=["SHAP_Strength", "SHAP", "Data"],
    )
    shap_data = shap_data.sort_values(by="SHAP_Strength", ascending=False)
    shap_data = shap_data["SHAP"]
    json_shap_data = json.dumps({"SHAP_data": shap_data.to_json()})
    return json_shap_data


if __name__ == "__main__":
    app.run(debug=True)
