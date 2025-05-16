import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
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
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Streamlit app
def main():
    st.title('Ensemble Classifier App')
    st.write('This app demonstrates an Ensemble Classifier using scikit-learn and Streamlit.')

    # Choose dataset
    dataset_name = st.selectbox('Select Dataset:', list(datasets.keys()))
    dataset = datasets[dataset_name]
    X, y = dataset

    # Choose models
    selected_models = st.multiselect('Select Models:', list(models.keys()))

    # Voting type
    voting_type = st.radio('Voting Type:', ['hard', 'soft'])

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

    # Prediction and results
    st.subheader('Prediction Results:')
    for model_name, model in trained_models.items():
        prediction = model.predict(X)
        accuracy = accuracy_score(y, prediction)
        st.write(f'{model_name} Accuracy: {accuracy:.2f}')

    # Ensemble Voting
    if selected_models:
        st.subheader('Ensemble Voting:')
        ensemble_estimators = [(model_name, model) for model_name, model in trained_models.items()]
        ensemble = VotingClassifier(ensemble_estimators, voting=voting_type)
        ensemble.fit(X, y)
        ensemble_prediction = ensemble.predict(X)
        ensemble_accuracy = accuracy_score(y, ensemble_prediction)
        st.write(f'Ensemble Voting ({voting_type}) Accuracy: {ensemble_accuracy:.2f}')

if __name__ == '__main__':
    main()
