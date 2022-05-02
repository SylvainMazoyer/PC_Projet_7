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


from app import *


def test_df():
    df.head()


def test_send_sku():

    serie_sku = df.iloc[0:50, 0]
    dict_sku = serie_sku.to_dict()
    print(dict_sku)


def test_predict(sku=100002):

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


def test_return_shap_data(sku=100002):

    # extract features for the sku
    X_sku = df.loc[(df["SK_ID_CURR"] == sku), select_features]

    # shap
    shap_value = explainer(X_sku)[0]


if __name__ == "__main__":
    test_df()
    test_send_sku()
    test_predict()