from os import write
import streamlit as st
import pickle
import numpy as np


def load_module():

    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_module()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("SOFTWARE DEVELOPER SALARY PREDICTION")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden"
    )

    education = (
        'Bachelor’s degree',
        'Master’s degree',
        'Less than a Bachelors',
        'Post grad'
    )

    country = st.selectbox("Country", countries)
    edu = st.selectbox("Education Level", education)

    experience = st.slider("Experience", 0, 50, 3)

    clk = st.button("Calculate Salary")

    if clk:
        X = np.array([[country, edu, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor_loaded.predict(X)

        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
