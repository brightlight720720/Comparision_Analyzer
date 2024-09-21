import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Home - CSV Analysis App", layout="wide")

st.title("Home - CSV Analysis App")

# CSV file upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # CSV summary
        st.subheader("CSV Summary")
        st.write(df.describe())
        
        # Data preview
        st.subheader("Data Preview")
        st.write(df.head())
        
        # Download button for the original dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Original Dataset",
            data=csv,
            file_name="original_dataset.csv",
            mime="text/csv",
        )
        
        # CSV splitting based on a column condition
        st.subheader("Split CSV")
        split_column = st.selectbox("Select a column to split the data", df.columns)
        split_value = st.text_input("Enter the value to split on (e.g., median, mean, or a specific value)")
        
        if st.button("Split CSV"):
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
                csv1 = df1.to_csv(index=False)
                csv2 = df2.to_csv(index=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Dataset 1",
                        data=csv1,
                        file_name="dataset1.csv",
                        mime="text/csv",
                    )
                with col2:
                    st.download_button(
                        label="Download Dataset 2",
                        data=csv2,
                        file_name="dataset2.csv",
                        mime="text/csv",
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
        st.error(f"An error occurred while processing the CSV file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")
