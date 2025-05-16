import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1, random_state=42)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
        return X, y

def draw_meshgrid():
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    input_array = np.c_[xx.ravel(), yy.ravel()]

    return xx, yy, input_array


plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Decision Tree Classifier")

dataset = st.sidebar.selectbox(
    'Select Dataset',
    ('Binary', 'Multiclass')
)

max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=10, value=5, step=1)

min_samples_split = st.sidebar.slider('Min Samples Split', min_value=2, max_value=10, value=2, step=1)

min_samples_leaf = st.sidebar.slider('Min Samples Leaf', min_value=1, max_value=5, value=1, step=1)
st.title("Decision Tree Classifier")
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

    xx, yy, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(xx, yy, labels.reshape(xx.shape), alpha=0.5, cmap='rainbow')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow', edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    orig = st.pyplot(fig)
    st.subheader("Accuracy for Decision Tree: " + str(round(accuracy_score(y_test, y_pred), 2)))

    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(clf, filled=True, feature_names=["Feature 1", "Feature 2"], class_names=["0", "1", "2"])
    st.pyplot(fig)
