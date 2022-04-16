from urllib import response
import sklearn
import flask
import shap
import pickle
import json
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from flask import Flask, jsonify,request
import pandas as pd
import numpy as np
from pathlib import Path

# Chargement du Df, du modèle et de l'explainer
URL="https://raw.githubusercontent.com/SylvainMazoyer/PC_Projet_7/main/"
path_transformed=URL+'DATA/'
df_csv=path_transformed+'df_10pc.csv'
df = pd.read_csv(df_csv)

path_results='MODEL/'
model_pkl=URL+path_results+"lgbmc_full.pkl"
estimator=pickle.load(open(model_pkl, 'rb'))

explainer_pkl=URL+path_results+"explainer_lgbmc_full.pkl"
explainer=pickle.load(open(explainer_pkl, 'rb'))


URL = "http://127.0.0.1:5000/"
app=Flask(__name__)

list_columns=df.columns.tolist()
list_columns.remove('SK_ID_CURR')
list_columns.remove('TARGET')
select_features=list_columns
sku=0

@app.route("/predict", methods=['GET','POST'])
def predict():
    
    json_sku = json.loads(request.data)
    sku=int(json_sku['sku'])
    print("sku :" + str(sku))
    X_sku=df.loc[(df['SK_ID_CURR']==sku), select_features]
    result_pred=estimator.predict(X_sku)[0]
    result_pred_proba=estimator.predict_proba(X_sku)[0]
    print("result pred :" +str(result_pred))
    if result_pred==0:
        result_reimbursement="Ok"
    else:
        result_reimbursement="Not ok"
    # Utiliser module Loging, log écrit dans un fichier log

    print("Reimbursement : " + result_reimbursement)

    pred_dict={'pred' : int(result_pred), "proba_0" : result_pred_proba[0]}
    json_pred=json.dumps(pred_dict)
    print(pred_dict)
    return json_pred

#send feature importance
@app.route("/return_shap_data", methods=['GET','POST'])
def return_shap_data() :
    json_sku = json.loads(request.data)
    sku=int(json_sku['sku'])
    print("sku :" + str(sku))
    X_sku=df.loc[(df['SK_ID_CURR']==sku), select_features]
    shap_value = explainer(X_sku)[0] 
    shap_data = pd.DataFrame(np.array([abs(shap_value.values), shap_value.values, shap_value.data.round(3)]).T, 
                                  index=shap_value.feature_names, 
                                  columns=["SHAP_Strength","SHAP", "Data"])
    shap_data = shap_data.sort_values(by="SHAP_Strength", ascending=False)
    shap_data = shap_data["SHAP"]
    json_shap_data=json.dumps({'SHAP_data' : shap_data.to_json()})
    return json_shap_data


# route test
@app.route("/hello", methods=['GET'])
def hello():
    return ("<h1> Welcome </h1>",200)


# send client id
@app.route("/send_sku", methods=['GET'])
def send_sku():
    serie_sku=df.iloc[0:50,0]
    dict_sku=serie_sku.to_dict()
    json_dict_sku=jsonify(dict_sku)
    return json_dict_sku



if __name__ == '__main__':
    app.run()
