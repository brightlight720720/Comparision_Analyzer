import streamlit as st
import importlib

st.set_page_config(page_title="Multi-format Analysis App", layout="wide")

# Dynamic page loading
page = st.selectbox("Select a page", ["Home", "C-Index", "IDI/NRI"])

if page == "Home":
    home = importlib.import_module("pages.1_Home")
    home.app()
elif page == "C-Index":
    c_index = importlib.import_module("pages.2_C_Index")
    c_index.app()
elif page == "IDI/NRI":
    idi_nri = importlib.import_module("pages.3_IDI_NRI")
    idi_nri.app()

# Main content
st.title("Multi-format Analysis App")
st.write("Welcome to the Multi-format Analysis App. This application allows you to upload CSV, Excel, or JSON files and perform various statistical analyses.")

st.markdown('''
### Instructions:
1. Use the page selector at the top to navigate between different pages.
2. Start by uploading your CSV, Excel, or JSON file on the Home page.
3. Select the required columns for analysis on the Home page.
4. Perform C-index computations on the C-Index page.
5. Calculate IDI and NRI on the IDI/NRI page.

### Supported File Formats:
- CSV (.csv)
- Excel (.xlsx)
- JSON (.json)

### Overview of Metrics:

#### C-Index (Concordance Index):
The C-Index measures the predictive accuracy of a survival model. It ranges from 0.5 to 1.0, where 0.5 indicates random predictions and 1.0 indicates perfect predictions. A higher C-Index suggests better model performance.

#### IDI (Integrated Discrimination Improvement):
IDI quantifies the improvement in prediction performance between two models. A positive IDI indicates that the new model improves risk prediction compared to the old model.

#### NRI (Net Reclassification Improvement):
NRI assesses the improvement in risk classification offered by a new model compared to an old model. A positive NRI suggests that the new model improves risk classification.

### Navigation:
- **Home**: Upload file (CSV, Excel, or JSON), preview data, split dataset, and select columns
- **C-Index**: Compute and compare C-index for old and new models
- **IDI/NRI**: Calculate Integrated Discrimination Improvement and Net Reclassification Improvement

Please select a page from the dropdown menu to begin your analysis.
''')

# Add author information to the bottom of the sidebar
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")
st.sidebar.markdown(" ")
st.sidebar.markdown("### Author Information")
st.sidebar.write("Author: YuHsuan")
st.sidebar.write("Email:\n brightlight720720@gmail.com")
