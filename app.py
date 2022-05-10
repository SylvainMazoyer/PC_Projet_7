import pickle
import json

import pandas as pd
import numpy as np

import os

import flask
from flask import Flask, jsonify,request

import sklearn
from sklearn.ensemble import RandomForestClassifier
import shap




# PATH
script_dir = os.path.dirname(__file__)
DATA_PATH='DATA/'
MODEL_PATH="MODEL/"

# FILES
DATA_FILE='df_train_full_imp_1000.csv'
MODEL_FILE='rfc_2.pkl'
EXPLAINER_FILE="rfc_explainer_2.pkl.pkl"
SHAP_FILE='shap_df_1000.csv'

# ABS PATHS
DATA_ABS_PATH=os.path.join(script_dir, DATA_PATH+DATA_FILE)
SHAP_ABS_PATH=os.path.join(script_dir, DATA_PATH+SHAP_FILE)
ESTIMATOR_ABS_PATH=os.path.join(script_dir, MODEL_PATH+MODEL_FILE)

# READ FILES
df = pd.read_csv(DATA_ABS_PATH)
shap_df=pd.read_csv(SHAP_ABS_PATH)

with open(ESTIMATOR_ABS_PATH, 'rb') as f:
    estimator=pickle.load(f)

#with open(MODEL_PATH+EXPLAINER_FILE, 'rb') as f:
#    explainer=pickle.load(f)


# VARS
list_columns=df.columns.tolist()
list_columns.remove('SK_ID_CURR')
list_columns.remove('TARGET')
select_features=list_columns

# INIT FLASK
app=Flask(__name__)

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


@app.route("/predict", methods=['GET'])
def predict():
    
    # récupère id du client
    json_sku = json.loads(request.data)
    sku=int(json_sku['sku'])
    print("sku :" + str(sku))

    # extract features from the sku
    X_sku=df.loc[(df['SK_ID_CURR']==sku), select_features]
    #X_sku=np.array(X_sku)

    # make prediction
    result_pred=estimator.predict(X_sku)[0]
    result_pred_proba=estimator.predict_proba(X_sku)[0]
    print("result pred :" +str(result_pred))

    if result_pred==0:
        result_reimbursement="Ok"
    else:
        result_reimbursement="Not ok"
    
    # Utiliser module Loging, log écrit dans un fichier log
    print("Reimbursement : " + result_reimbursement)

    #génerate json 
    pred_dict={'pred' : int(result_pred), "proba_0" : result_pred_proba[0]}
    json_pred=json.dumps(pred_dict)
    print(pred_dict)

    return json_pred


# send feature importance
@app.route("/return_shap_data", methods=['GET'])
def return_shap_data() :

    # recupère l'id du client
    json_sku = json.loads(request.data)
    sku=int(json_sku['sku'])
    print("sku :" + str(sku))


    # extract features from the sku
    shap_sku=shap_df.loc[(df['SK_ID_CURR']==sku), select_features]

    # shap

    #shap_value = explainer(X_sku)[0] 

    #shap_data = pd.DataFrame(np.array([abs(shap_value.values), shap_value.values, shap_value.data.round(3)]).T, 
    #                              index=shap_value.feature_names, 
    #                              columns=["SHAP_Strength","SHAP", "Data"])
    #shap_data = shap_data.sort_values(by="SHAP_Strength", ascending=False)

    shap_data=shap_sku.transpose()
    shap_data=pd.Series(-shap_data.iloc[:,0])
    shap_data=shap_data.sort_values(ascending=False, key=lambda x: abs(x))
    

    # build json
    json_shap_data=json.dumps({'SHAP_data' : shap_data.to_json()})

    return json_shap_data


if __name__ == '__main__':
    app.run()