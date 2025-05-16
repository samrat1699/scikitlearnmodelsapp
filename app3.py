import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dummy data for performance time series
data = pd.DataFrame({
    'Time': pd.date_range('2023-06-16', periods=100, freq='H'),
    'Performance': [25, 45, 56, 64, 73, 82, 88, 92, 95, 98] * 10
})

# App title and description
st.title('Real-time Performance Time Series')
st.write('This app visualizes the real-time performance time series.')

# Create an empty figure
fig = go.Figure()

# Initialize the figure with the initial data
fig.add_trace(go.Scatter(x=data['Time'], y=data['Performance'], mode='lines', name='Performance'))

# Configure the layout of the figure
fig.update_layout(
    xaxis_title='Time',
    yaxis_title='Performance',
    template='plotly_white'
)

# Add a real-time update to the figure
performance_chart = st.plotly_chart(fig)

# Real-time update function
def update_chart():
    while True:
        # Get the latest performance data
        latest_time = pd.date_range('2023-06-16', periods=2, freq='H')[-1:]
        latest_performance = [st.number_input('Enter Performance', value=0, key=f'performance{i}') for i in range(2)]
        
        # Create the latest data DataFrame
        latest_data = pd.DataFrame({
            'Time': latest_time,
            'Performance': latest_performance
        })

        # Append the latest data to the existing data
        updated_data = pd.concat([data, latest_data])

        # Update the figure with the updated data
        fig.update_traces(x=updated_data['Time'], y=updated_data['Performance'])

        # Update the chart with the new figure
        performance_chart.plotly_chart(fig)

# Call the real-time update function
update_chart()
