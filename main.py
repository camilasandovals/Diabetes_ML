#Program that detects if a person has Diabetes (1) or not (1) using ML

#Import libraries

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#create a title and a subtitle
st.write("""
# Diabetes Detection WebSite
 Helps to detect if a person has diabetes or not using machine learning.
""")

#import main image
#image = Image.open("C:/Users/Camila/Documents/Programacion/ML/PyCharm/Diabetes Web Site/diabetes.jpg")
image = Image.open("diabetes.jpg")
st.image(image, caption='Diabetes Mellitus (DM) is a condition induced by unregulated diabetes that may lead to multi-organ failure in patients. Advances in machine learning and AI, enables the early detection and diagnosis of DM through an automated process which is more advantageous than a manual diagnosis.',use_column_width=True)

#Get the data
#df = pd.read_csv("C:/Users/Camila/Documents/Programacion/ML/PyCharm/Diabetes Web Site/diabetes.csv")
df = pd.read_csv("diabetes.csv")

#Set a subheader
st.subheader('Data Information:')

#Show the data as a table (you can also use st.write(df))
st.dataframe(df)

#Get statistics on the data
st.write(df.describe())

# Show the data as a chart.
chart = st.line_chart(df)

#Split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y= df.iloc[:,-1].values

# Split the dataset into 80% Training set and 20% Testing set
#random states is used to keep the states of the data, here we are not using it
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


# Get the feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 10, 2)
    glucose = st.sidebar.slider('glucose', 0, 225, 115)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 130, 75)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 110, 25)
    insulin = st.sidebar.slider('insulin', 0.0, 850.0, 32.0)
    BMI = st.sidebar.slider('BMI', 0.0, 65.1, 30.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 18, 90, 35)

#Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin': insulin,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }

    #transform the data into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store the user input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('User Input :')
st.write(user_input)

#Create and train the model (RFC)
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )

#Store the models prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader and display the results
st.subheader('Classification: ')
st.write(prediction)