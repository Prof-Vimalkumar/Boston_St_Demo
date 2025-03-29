# create simple python programe to do a linear regression on boston housing dataset
# using sklearn
# and plot the results using matplotlib
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Boston Dataset
fname = "data/boston.csv"
df = pd.read_csv(fname) 

#Glimpse of data
df.head()

#print(df.columns)
#print(df.info())

#Create Independent and Dependent Variables
X = df[['CRIM','CHAS','NOX', 'RM','AGE', 'DIS']]
y = df['MEDV']

#Create Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_regression = LinearRegression()

lrMod = lin_regression.fit(X, y)

#Predict the price for test data
lrPred = lrMod.predict(X[1:5])
#print(lrPred)

#Let's convert this into a streamlit app
import streamlit as st
import plotly.express as px
import altair as alt #pip install altair
from plotly import graph_objs as go

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

#create a streamlit header as "Boston Housing"
st.title("Boston Housing")
st.subheader("Predicting the price of a house in Boston")

#display image 'BostanHousing//data//boston_house.png'
st.image("data/boston_house.png")

#display the data in a table format
#st.dataframe(df)    

#create streamlit sidebar with radio button for navigation
nav = st.sidebar.radio("Navigation",["Home", "Prediction"])

#create if statement for navigation 
if nav == 'Home':
    #display the data in a table format
    if st.checkbox("Show data"):
        #Show data
        st.dataframe(df)
        # st.table(df)
        # st.write(df)

    if st.checkbox("Show map"):
        val = st.slider("Filter data based on Median Value",0,40)
        fdata = df.loc[df["MEDV"]>= val]
        city_data = fdata[["LON","LAT","MEDV"]]
        city_data.columns = ['longitude','latitude', 'Medv']
        st.map(city_data)

if nav == 'Prediction':
    st.header("Prediction")

    #create a streamlit input form for user to input values for CRIM, CHAS, NOX, RM, AGE, DIS
    v_room = st.number_input("Number of Rooms", value=3, min_value=0, max_value=10, step=1)
    v_age = st.slider("Age of House", 0,100, value=10, step=1)
    v_dist = st.slider("Distance from the office",0.0,15.0,step=0.5)
    on = st.toggle("Next to charls river?")
    if on:
        v_chas = 1
    else:
        v_chas = 0

    v_crim = st.number_input("Enter the preferred crime rate:",0.00, 10.00, step = 0.100, value= 3.00)
    v_NOX = st.number_input("Enter the NOX value in the neighborhood:",0.00, 1.00, step = 0.010, value= 0.10)

    #create a button to predict the price of the house
    if st.button("Predict Price"):
        #create a dataframe with the input values
        input_data = pd.DataFrame([[v_crim, v_chas, v_NOX, v_room, v_age, v_dist]], columns=['CRIM','CHAS','NOX', 'RM','AGE', 'DIS'])
        #predict the price of the house
        price = lrMod.predict(input_data)[0] * 1000
        #display the predicted price in dollars
        st.success(f"The predicted price of the house is ${price:,.2f}")

# For requirements.txt install pip pipreqs