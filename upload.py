import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web Title
st.markdown(''' 
# ** Exploratory Data Analysis Web Application**
''')

# How to upload file from pc
with st.sidebar.header(" Upload your dataset (.csv)"):
    upload_file = st.sidebar.file_uploader("Upload your file", type=['csv'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[Example CSV file](df)")

# Profiling report for pandas
if upload_file is not None:
    @st.cache_data 
    def load_csv():
        csv = pd.read_csv(upload_file)
        return csv
    
    df = load_csv()  # Call the function to load CSV data
    pr = ProfileReport(df, explorative=True)
    
    st.header('**Input DF**')
    st.write(df)
    st.write('---')
    
    st.header('**Profiling report with pandas**')
    st_profile_report(pr)

else:
    st.info('Awaiting for CSV file, upload file') 
    
    if st.button('Press to use example data'):
        # Example data set
        @st.cache_data 
        def load_data():
            a = pd.DataFrame(np.random.rand(100, 5),
                             columns=['age', 'banana', 'candies', 'dunants', 'rae'])
            return a
        
        df = load_data()  # Call the function to load example data
        pr = ProfileReport(df, explorative=True)
        
        st.header('**Input DF**')
        st.write(df)
        st.write('---')
        
        st.header('**Profiling report with pandas**')
        st_profile_report(pr)
