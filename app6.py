import streamlit as st
import nbconvert
import tempfile
import webbrowser

# Convert Jupyter Notebook to HTML
def convert_notebook_to_html(notebook_path):
    html_exporter = nbconvert.HTMLExporter()
    html_output, _ = html_exporter.from_filename(notebook_path)
    return html_output

# App title
st.title('Jupyter Notebook Viewer')

# Sidebar for selecting the notebook file
st.sidebar.header('Select Notebook File')
notebook_file = st.sidebar.file_uploader('Upload a Jupyter Notebook', type=['ipynb'])

if notebook_file is not None:
    # Save the uploaded notebook file
    with open('temp_notebook.ipynb', 'wb') as f:
        f.write(notebook_file.getvalue())

    # Convert the notebook to HTML
    html_output = convert_notebook_to_html('temp_notebook.ipynb')

    # Save the HTML to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        temp_file.write(html_output.encode('utf-8'))
        temp_file_path = temp_file.name

    # Display the notebook preview using Streamlit's HTML component
    st.sidebar.markdown('**Notebook Preview**')
    st.sidebar.write('Use the slider to adjust the height of the notebook preview')
    height = st.sidebar.slider('Preview Height', min_value=200, max_value=1000, value=600, step=100)
    st.sidebar.components.v1.html(open(temp_file_path).read(), height=height)
    # Open the temporary HTML file in a new browser tab
    webbrowser.open_new_tab(temp_file_path)
