import streamlit as st

st.set_page_config(page_title="CSV Analysis App", layout="wide")

st.title("CSV Analysis App")
st.write("Welcome to the CSV Analysis App. This application allows you to upload a CSV file and perform various statistical analyses.")

st.markdown("""
### Instructions:
1. Use the sidebar to navigate between different pages.
2. Start by uploading your CSV file on the Home page.
3. Select the required columns for analysis on the Home page.
4. Perform C-index computations on the C-Index page.
5. Calculate IDI and NRI on the IDI/NRI page.

### Navigation:
- **Home**: Upload CSV, preview data, split dataset, and select columns
- **C-Index**: Compute and compare C-index for old and new models
- **IDI/NRI**: Calculate Integrated Discrimination Improvement and Net Reclassification Improvement

Please select a page from the sidebar to begin your analysis.
""")

st.sidebar.title("Navigation")
st.sidebar.info("""
1. Start with the Home page
2. Then go to C-Index page
3. Finally, visit the IDI/NRI page
""")
