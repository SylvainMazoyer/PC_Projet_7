# HTTP communication
from flask import Flask
import requests

# OS path
import os

# Data reading
import numpy as np
import json

# Displaying libraries
import streamlit as st
from PIL import Image
import plotly.graph_objs as go
import plotly.express as px
import ast


# Loading intro image
PATH_IMG='IMG/'
PATH_DOLLARS  = PATH_IMG+'dollars.jpg'

with  Image.open(PATH_DOLLARS) as dollars_img:
    dollars_img=np.array(dollars_img)

st.image(dollars_img, output_format="JPEG")
st.title('Credit reimbursement prediction')

URL = "http://127.0.0.1:5000/"
app=Flask(__name__)

list_sku=[]
prediction=0

def ask_for_list_sku():

    json_list_sku = requests.get(URL+"/send_sku")
    json_text=json_list_sku.text
    dict_sku=eval(json_text)
    for key, value in dict_sku.items():   
        list_sku.append(value)
    return (list_sku)

def ask_user_sku():

    selected_sku=0
    selected_sku=int(st.number_input("enter cient  sk Id : ",value=100002))
    if selected_sku not in list_sku:
        st.header("This sku doesn't correspond to any client.")
        return -1
    else:
        button=st.button("Ok")
    if button==True:
        predict_selected_sku(sku=selected_sku)
        

def predict_selected_sku(sku):
    
    sku=int(sku)
    sku_dict={'sku' : sku}
    json_data=json.dumps(sku_dict)
    json_response = requests.get(URL+"/predict", data=json_data)
    
    # Print reimbursement advice
    json_pred = json_response.json()
    pred=json_pred['pred']
    if pred==0:
        st.header("Predicted reimbursement : "+"Ok")
    else:
        st.header("Predicted reimbursement : "+"Warning")
    
    proba_0=json_pred['proba_0']
    proba_0=int(proba_0*1000.)/1000.
    
    #plot Probability gauge
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = proba_0,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Reimbursement probability"},
    gauge={ 'axis': {'range' : [0,1]}}))

    threshold=0.5

    if (proba_0 > threshold):
        fig.update_traces(gauge_bar_color='green')
    else:
        fig.update_traces(gauge_bar_color='red')
    st.write(fig)

    # Plot shap values
    json_response = requests.get(URL+"/return_shap_data", data=json_data)
    json_shap = json_response.json()
    shap_data_all=json_shap['SHAP_data']
    shap_data_dict=ast.literal_eval(shap_data_all)
    list_features_shap=[]
    for key, value in shap_data_dict.items():   
        list_features_shap.append([key, value])

    list_features_shap.sort(key=lambda tup: abs(tup[1]), reverse=True)
    list_features_shap=list_features_shap[0:10]
    list_features=[]
    list_shap_values=[]
    for shap in list_features_shap:
        list_features.append(shap[0])
        list_shap_values.append(abs(shap[1]))

    # displaying shap values
    st.header("Importance of features for help to decision")
    fig = px.bar(x=list_features, y=list_shap_values)#, title="Importance of features")
    st.plotly_chart(fig, use_container_width=True)

    # displaying shap importance for 1000 clients
    PATH_SHAP_1000  = PATH_IMG+'hist_abs_shap_1000.png'
    with  Image.open( PATH_SHAP_1000) as shap_100_img:
        shap_1000_img=np.array(shap_100_img)


    st.header("Importance of features for  1000 clients for comparison")
    st.image(shap_1000_img, output_format="JPEG")

    return json_data


result=ask_for_list_sku()
predict_response=ask_user_sku()


