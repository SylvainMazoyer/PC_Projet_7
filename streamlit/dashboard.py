# HTTP communication
from flask import Flask
import requests

# Data reading
import numpy as np
import json

# os
import os
from pathlib import Path

# Displaying libraries
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import ast
import shap
import seaborn as sns

path_img='IMG/'
script_dir = os.path.dirname(__file__)
rel_path = path_img+'dollars.jpg'
dollars_jpg = os.path.join(script_dir, rel_path)

with  Image.open(dollars_jpg) as dollars_img:
    dollars_img=np.array(dollars_img)

st.image(dollars_img, output_format="JPEG")
st.title('Credit reimbursement prediction')

#URL = "http://127.0.0.1:5000/"
URL = "http://194.233.168.72/"
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
        list_features, list_shap_values =predict_selected_sku(sku=selected_sku)
        st.header("4. Select a feature to know its importance")
        # Scrolling list
        feature = st.selectbox("Feature list :", list_features, key="feature list") 

        # Plotting shap values for one feature
        index=list_features.index(feature)
        shap_feature=list_shap_values[index]
        fig, ax = plt.subplots()
        histplot = sns.distplot(a=list_shap_values)
        histplot.axvline(shap_feature, color='red')
        st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
        

        

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
    list_features=[]
    list_shap_values=[]
    list_abs_shap_values=[]
    for shap_features in list_features_shap:
        list_features.append(shap_features[0])
        list_shap_values.append(shap_features[1])
        list_abs_shap_values.append(abs(shap_features[1]))

    shap_values=np.array(list_shap_values)
    # displaying shap values as a waterfall plot
    st.header("1. Importance of features as a waterfall plot for help to decision")
    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(-0.05, shap_values, feature_names=list_features)
    st.pyplot(fig, bbox_inches='tight',dpi=300,pad_inches=0)
    #plt.clf()


    # displaying shap values as a bar plot
    list_features_10=list_features[0:10]
    list_abs_shap_values_10=list_abs_shap_values[0:10]
    st.header("2. Absolute importance of features for help to decision")
    fig = px.bar(x=list_features_10, y=list_abs_shap_values_10)#, title="Importance of features")
    st.plotly_chart(fig, use_container_width=True)


    # displaying shap importance for 1000 clients
    rel_path = path_img+'hist_abs_shap_1000.png'
    shap_1000_png = os.path.join(script_dir, rel_path)
    with  Image.open(shap_1000_png) as shap_1000_img:
        shap_1000_img=np.array(shap_1000_img)

    st.header("3. Absolute importance of features for  1000 clients for comparison")
    st.image(shap_1000_img, output_format="JPEG")

    return (list_features,list_shap_values)


result=ask_for_list_sku()
list_features=ask_user_sku()



