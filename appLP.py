import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def generate_data():
    np.random.seed(0)
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = 0.5 * X**3 - 2 * X**2 + X + 10 + np.random.randn(100, 1) * 5
    return X, y

def plot_polynomial_regression(X, y, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    X_poly = polynomial_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    X_pred = np.linspace(-5, 5, 100).reshape(-1, 1)
    X_pred_poly = polynomial_features.transform(X_pred)
    y_pred = model.predict(X_pred_poly)

    plt.scatter(X, y, color='b', label='Data')
    plt.plot(X_pred, y_pred, color='r', label='Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.legend()
    return plt

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Polynomial Regression")

degree = st.sidebar.slider('Degree', min_value=1, max_value=10, value=2, step=1)

X, y = generate_data()

st.subheader("Original Data")
fig = plt.figure()
plt.scatter(X, y, color='b')
plt.xlabel('X')
plt.ylabel('y')
st.pyplot(fig)

if st.sidebar.button('Run Regression'):
    st.subheader(f"Polynomial Regression (Degree {degree})")
    fig = plot_polynomial_regression(X, y, degree)
    st.pyplot(fig)
