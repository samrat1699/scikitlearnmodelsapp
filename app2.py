import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dummy data for student performance marks
data = pd.DataFrame({
    'Hours Studied': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'Marks Obtained': [25, 45, 56, 64, 73, 82, 88, 92, 95, 98]
})

# Sidebar inputs
st.sidebar.header('Set Parameters')
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

# Live line chart
st.subheader('Live Performance Visualization')

# Create empty lists to store data
hours_list = []
marks_list = []

# Update data based on user input
for i in range(len(data)):
    hours = data['Hours Studied'][i]
    marks = data['Marks Obtained'][i]
    hours_list.append(hours)
    marks_list.append(marks)
    plt.plot(hours_list, marks_list, '-o')
    plt.xlabel('Hours Studied')
    plt.ylabel('Marks Obtained')
    plt.title('Student Performance')
    plt.grid(True)
    st.pyplot(plt)
    plt.pause(0.5)  # Pause for 0.5 seconds before updating the plot

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
