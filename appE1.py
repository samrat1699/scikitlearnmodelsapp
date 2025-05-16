import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier")

dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)

max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=10, value=5, step=1)

min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=10, value=2, step=1)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', min_value=1, max_value=5, value=1, step=1)

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
X, y = load_initial_graph(dataset, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):
    orig.empty()

    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree: " + str(round(accuracy_score(y_test, y_pred), 2)))
