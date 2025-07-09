import streamlit as st
import pickle 
import numpy as np

model=pickle.load(open("linear_regression_model.pkl",'rb'))
st.title("Salary Prediction App")
st.write("This app predicts your salary based on years of experiance using a simple linear regression model")

years_experiance=st.number_input("Enter Years of Experiance:",min_value=0.0,max_value=50.0,value=1.0,step=0.5)

if st.button("Predict Salary"):
    experience_input=np.array([[years_experiance]])
    prediction=model.predict(experience_input)
    st.success(f"The predicted salary for {years_experiance} years of experiance is: ${prediction[0]:,.2f}")
st.write("The model was trained using a dataset of salary and year of expeiance")
