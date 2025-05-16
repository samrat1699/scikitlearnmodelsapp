import pandas as pd
import numpy as np

# Generate random dummy data
num_rows = 100  # Number of rows in the dataset

# Create dummy data
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], num_rows),
    'Feature1': np.random.randint(1, 100, num_rows),
    'Feature2': np.random.randint(1, 100, num_rows),
})

# Save the dummy dataset to a CSV file
data.to_csv('dummy_dataset.csv', index=False)



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('dummy_dataset.csv')
    return data

data = load_data()

# App title
st.title('Real-Life Data Analysis App')

# Sidebar for data exploration options
st.sidebar.header('Data Exploration')
show_data = st.sidebar.checkbox('Show Raw Data')

# Display raw data if selected
if show_data:
    st.subheader('Raw Data')
    st.write(data)

# Data analysis and visualization
st.header('Data Analysis and Visualization')

# Summary statistics
st.subheader('Summary Statistics')
st.write(data.describe())

# Histogram
st.subheader('Histogram')
column_to_plot = st.selectbox('Select a column', data.columns)
plt.figure(figsize=(10, 6))
sns.histplot(data[column_to_plot])
plt.xlabel(column_to_plot)
plt.ylabel('Count')
plt.title(f'Distribution of {column_to_plot}')
st.pyplot(plt)

# Bar chart
st.subheader('Bar Chart')
bar_chart_data = data['Category'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=bar_chart_data.index, y=bar_chart_data.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Data Distribution by Category')
st.pyplot(plt)

# Interactive component
st.header('Interactive Component')

# User input for filtering data
category_filter = st.selectbox('Filter by Category', ['All'] + data['Category'].unique().tolist())
filtered_data = data if category_filter == 'All' else data[data['Category'] == category_filter]

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_data)
