import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define datasets
datasets = {
    'U-Shaped': make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                                     n_clusters_per_class=1, flip_y=0.1, class_sep=2, random_state=42),
    'Linearly Separable': make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                                               n_clusters_per_class=1, flip_y=0.1, class_sep=0.5, random_state=42),
    'Outliers': make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                                     n_clusters_per_class=1, flip_y=0.1, class_sep=2, random_state=42),
    'Two Spirals': make_moons(n_samples=500, noise=0.5, random_state=42),
    'Concentric Circles': make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
}

# Define models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Streamlit app
def main():
    st.title('Ensemble Classifier App')
    st.write('This app demonstrates an Ensemble Classifier using scikit-learn and Streamlit.')

    # Sidebar
    st.sidebar.title('Options')
    dataset_name = st.sidebar.selectbox('Select Dataset:', list(datasets.keys()))
    selected_models = st.sidebar.multiselect('Select Models:', list(models.keys()))
    voting_type = st.sidebar.radio('Voting Type:', ['hard', 'soft'])

    # Choose dataset
    dataset = datasets[dataset_name]
    X, y = dataset

    # Train models
    trained_models = {}
    for model_name in selected_models:
        model = models[model_name]
        model.fit(X, y)
        trained_models[model_name] = model

    # Visualization
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    ax.set_title(f'Dataset: {dataset_name}')
    st.pyplot(fig)

    # Prediction input and decision boundary visualization
    st.subheader('Predict')
    features = st.text_input('Enter feature values (comma-separated)', '5.0,3.0')

    if st.button('Predict'):
        feature_values = [float(val) for val in features.split(',')]
        new_data = [feature_values]
        ensemble_estimators = [(model_name, model) for model_name, model in trained_models.items()]
        ensemble = VotingClassifier(ensemble_estimators, voting=voting_type)
        ensemble.fit(X, y)
        prediction = ensemble.predict(new_data)
        st.write(f'Predicted Class: {prediction[0]}')

        # Decision boundary visualization
        plot_decision_boundary(X, y, ensemble)

def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')

if __name__ == '__main__':
    main()
