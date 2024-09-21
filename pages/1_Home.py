import streamlit as st
import pandas as pd
import io
import json

st.set_page_config(page_title="Home - Multi-format Analysis App", layout="wide")

st.title("Home - Multi-format Analysis App")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

def read_file(file):
    file_extension = file.name.split(".")[-1].lower()
    if file_extension == "csv":
        return pd.read_csv(file)
    elif file_extension == "xlsx":
        return pd.read_excel(file, engine="openpyxl")
    elif file_extension == "json":
        return pd.read_json(file)
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
        return None

if uploaded_file is not None:
    try:
        df = read_file(uploaded_file)
        if df is not None:
            # File summary
            st.subheader("File Summary")
            st.write(df.describe())
            
            # Data preview
            st.subheader("Data Preview")
            st.write(df.head())
            
            # Download button for the original dataset
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == "csv":
                data = df.to_csv(index=False)
                mime = "text/csv"
            elif file_extension == "xlsx":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False)
                data = buffer.getvalue()
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif file_extension == "json":
                data = df.to_json(orient="records")
                mime = "application/json"
            
            st.download_button(
                label=f"Download Original Dataset",
                data=data,
                file_name=f"original_dataset.{file_extension}",
                mime=mime,
            )
            
            # File splitting based on a column condition
            st.subheader("Split File")
            split_column = st.selectbox("Select a column to split the data", df.columns)
            split_value = st.text_input("Enter the value to split on (e.g., median, mean, or a specific value)")
            
            if st.button("Split File"):
                try:
                    if split_value.lower() == "median":
                        split_point = df[split_column].median()
                    elif split_value.lower() == "mean":
                        split_point = df[split_column].mean()
                    else:
                        split_point = float(split_value)
                    
                    df1 = df[df[split_column] <= split_point]
                    df2 = df[df[split_column] > split_point]
                    
                    st.write(f"Split point: {split_point}")
                    st.write(f"Dataset 1 shape: {df1.shape}")
                    st.write(f"Dataset 2 shape: {df2.shape}")
                    
                    # Download buttons for both split datasets
                    col1, col2 = st.columns(2)
                    with col1:
                        if file_extension == "csv":
                            data1 = df1.to_csv(index=False)
                        elif file_extension == "xlsx":
                            buffer1 = io.BytesIO()
                            with pd.ExcelWriter(buffer1, engine="openpyxl") as writer:
                                df1.to_excel(writer, index=False)
                            data1 = buffer1.getvalue()
                        elif file_extension == "json":
                            data1 = df1.to_json(orient="records")
                        
                        st.download_button(
                            label=f"Download Dataset 1",
                            data=data1,
                            file_name=f"dataset1.{file_extension}",
                            mime=mime,
                        )
                    with col2:
                        if file_extension == "csv":
                            data2 = df2.to_csv(index=False)
                        elif file_extension == "xlsx":
                            buffer2 = io.BytesIO()
                            with pd.ExcelWriter(buffer2, engine="openpyxl") as writer:
                                df2.to_excel(writer, index=False)
                            data2 = buffer2.getvalue()
                        elif file_extension == "json":
                            data2 = df2.to_json(orient="records")
                        
                        st.download_button(
                            label=f"Download Dataset 2",
                            data=data2,
                            file_name=f"dataset2.{file_extension}",
                            mime=mime,
                        )
                except ValueError:
                    st.error("Invalid split value. Please enter a numeric value or 'median'/'mean'.")
            
            # Clear instructions for column selection
            st.subheader("Column Selection")
            st.write("Please select the columns for your analysis:")
            
            duration_column = st.selectbox("Select the Duration column", df.columns)
            dead_column = st.selectbox("Select the Dead column", df.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                old_model_columns = st.multiselect("Select the old model columns", df.columns)
            with col2:
                new_model_columns = st.multiselect("Select the new model columns", df.columns)
            
            # Visual indicators for the number of columns selected for each model
            st.write(f"Old model columns selected: {len(old_model_columns)}")
            st.write(f"New model columns selected: {len(new_model_columns)}")
            
            # Save column selections to session state
            if st.button("Save Column Selections"):
                st.session_state.duration_column = duration_column
                st.session_state.dead_column = dead_column
                st.session_state.old_model_columns = old_model_columns
                st.session_state.new_model_columns = new_model_columns
                st.success("Column selections saved. You can now proceed to the C-Index and IDI/NRI pages for analysis.")
        
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload a CSV, Excel, or JSON file to begin the analysis.")
