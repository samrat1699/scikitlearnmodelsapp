import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def generate_data():
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                               n_clusters_per_class=1, random_state=42)
    return X, y

def draw_meshgrid(X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    input_array = np.c_[xx.ravel(), yy.ravel()]

    return xx, yy, input_array

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Support Vector Classifier")

C = st.sidebar.slider('C', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
kernel = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
degree = st.sidebar.slider('Degree', min_value=1, max_value=10, value=3, step=1)
gamma = st.sidebar.selectbox('Gamma', ('scale', 'auto'))
coef0 = st.sidebar.slider('Coef0', min_value=0.0, max_value=10.0, value=0.0, step=0.1)
shrinking = st.sidebar.checkbox('Shrinking', value=True)
probability = st.sidebar.checkbox('Probability', value=False)
tol = st.sidebar.slider('Tolerance', min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
cache_size = st.sidebar.slider('Cache Size', min_value=100, max_value=1000, value=200, step=100)
class_weight = st.sidebar.selectbox('Class Weight', (None, 'balanced'))
verbose = st.sidebar.checkbox('Verbose', value=False)
max_iter = st.sidebar.slider('Max Iterations', min_value=-1, max_value=10000, value=-1, step=100)
decision_function_shape = st.sidebar.selectbox('Decision Function Shape', ('ovr', 'ovo'))
break_ties = st.sidebar.checkbox('Break Ties', value=False)

X, y = generate_data()

st.subheader("Original Data")
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
st.pyplot(fig)

predict_button = st.button("Predict")

if predict_button:
    if decision_function_shape == 'ovo':
        break_ties = False

    model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking,
                probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight,
                verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape,
                break_ties=break_ties)

    model.fit(X, y)

    xx, yy, input_array = draw_meshgrid(X)
    Z = model.predict(input_array)
    Z = Z.reshape(xx.shape)

    st.subheader("Decision Boundary")
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.5, cmap='rainbow')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(fig)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    #st.subheader("Accuracy")
    st.subheader("Accuracy for Decision Tree: " + str(round(accuracy_score(y, y_pred), 2)))
    #st.write(accuracy)
