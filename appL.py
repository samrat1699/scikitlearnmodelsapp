import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def generate_data():
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    return X, y

def plot_regression_line(X, y, model):
    plt.scatter(X, y, color='b', label='Data')
    plt.plot(X, model.predict(X), color='r', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    return plt

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Linear Regression")

fit_intercept = st.sidebar.checkbox('Fit Intercept', value=True)

copy_X = st.sidebar.checkbox('Copy X', value=True)

positive = st.sidebar.checkbox('Positive', value=False)

X, y = generate_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, positive=positive)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

st.subheader("Model Parameters")
st.text(f"Fit Intercept: {fit_intercept}")
st.text(f"Copy X: {copy_X}")
st.text(f"Positive: {positive}")

st.subheader("Model Performance")
st.text(f"R-squared Score: {r2}")

st.subheader("Regression Line")
fig = plot_regression_line(X, y, model)
st.pyplot(fig)
