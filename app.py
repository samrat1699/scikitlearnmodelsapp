import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#Student marks performance
data = pd.DataFrame({
    "Hours Studied": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "Marks Obtained": [25, 45, 56, 64, 73, 82, 88, 92, 95, 98]
})

# Set page layout width
st.markdown('<style>body{max-width: 800px; margin: auto;}</style>', unsafe_allow_html=True)

# Header
st.markdown('<h1 style="text-align: center;">My Streamlit App</h1>', unsafe_allow_html=True)

# Subheader
st.markdown('<h2>Regression Model</h2>', unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Set Parameters")
test_size = st.sidebar.slider('Test size', 0.1, 0.5, 0.2, step=0.1)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X = data[['Hours Studied']]
y = data['Marks Obtained']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.header('Student Performance Regression')
st.write('Mean Squared Error:', mse)
st.write('R^2 Score:', r2)

# Create a form for user input
st.header('Make a Prediction')
hours = st.number_input('Hours Studied:', value=5)

# Make a prediction based on user input
input_data = [[hours]]
prediction = model.predict(input_data)

st.write('Predicted Marks Obtained:', prediction[0])

# Plot the regression line and data points
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Student Performance Regression')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Box plot
st.subheader('Box Plot')
plt.figure(figsize=(8, 6))
sns.boxplot(y=data['Marks Obtained'])
plt.ylabel('Marks Obtained')
plt.title('Student Performance')
st.pyplot(plt)

# Distribution plot
st.subheader('Distribution Plot')
plt.figure(figsize=(8, 6))
sns.histplot(data['Marks Obtained'], kde=True)
plt.xlabel('Marks Obtained')
plt.title('Student Performance')
st.pyplot(plt)