import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io
import importlib
from lifelines import CoxPHFitter
from sklearn.impute import SimpleImputer

def app():
    st.title("IDI and NRI Computation")

    # Function to read different file formats
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

    # File upload section
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"], key="idi_nri_uploader")

    st.markdown("""
    ### What are IDI and NRI?

    #### Integrated Discrimination Improvement (IDI)
    IDI measures the improvement in prediction performance between two models. It quantifies the difference in discrimination slopes between the new and old models.

    - Positive IDI: The new model improves risk prediction
    - Negative IDI: The new model worsens risk prediction
    - IDI = 0: No improvement in risk prediction

    #### Net Reclassification Improvement (NRI)
    NRI assesses the improvement in risk classification offered by a new model compared to an old model. It quantifies the net proportion of individuals with and without the event of interest who are correctly reclassified by the new model.

    - Positive NRI: The new model improves risk classification
    - Negative NRI: The new model worsens risk classification
    - NRI = 0: No improvement in risk classification

    ### Interpretation:
    - For both IDI and NRI, larger positive values indicate greater improvement in the new model's predictive performance.
    - The confidence interval (CI) and p-value help determine if the improvement is statistically significant.
    - NRI can be further broken down into NRI for events and non-events, providing insight into how the new model performs for different outcomes.
    """)

    def compute_idi_nri(df, duration_col, event_col, old_model_columns, new_model_columns):
        # Preprocess data
        df = df.replace([np.inf, -np.inf], np.nan)
       
        # Check for NaN values and inform the user
        nan_columns = df[old_model_columns + new_model_columns + [duration_col, event_col]].columns[df[old_model_columns + new_model_columns + [duration_col, event_col]].isna().any()].tolist()
        if nan_columns:
            raise ValueError(f"NaN values detected in columns: {', '.join(nan_columns)}. Please handle missing values before proceeding.")
       
        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        cols_to_impute = old_model_columns + new_model_columns + [duration_col]
        df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
       
        # Remove rows with NaN values in the event column
        df = df.dropna(subset=[event_col])
       
        # Fit old model
        cox_model_1 = CoxPHFitter()
        cox_model_1.fit(df[old_model_columns + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)
       
        # Fit new model
        cox_model_2 = CoxPHFitter()
        cox_model_2.fit(df[new_model_columns + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)
       
        # Compute predicted probabilities (risk scores) for both models
        df['pred_model_1'] = cox_model_1.predict_partial_hazard(df[old_model_columns + [duration_col, event_col]])
        df['pred_model_2'] = cox_model_2.predict_partial_hazard(df[new_model_columns + [duration_col, event_col]])
       
        # Calculate IDI
        diff_event = df.loc[df[event_col] == 1, 'pred_model_2'].mean() - df.loc[df[event_col] == 1, 'pred_model_1'].mean()
        diff_nonevent = df.loc[df[event_col] == 0, 'pred_model_2'].mean() - df.loc[df[event_col] == 0, 'pred_model_1'].mean()
        idi = diff_event - diff_nonevent
       
        # Calculate NRI
        improved = ((df['pred_model_2'] > df['pred_model_1']) & (df[event_col] == 1)).sum()
        worsened = ((df['pred_model_2'] < df['pred_model_1']) & (df[event_col] == 1)).sum()
        total_events = df[event_col].sum()
       
        improved_censored = ((df['pred_model_2'] < df['pred_model_1']) & (df[event_col] == 0)).sum()
        worsened_censored = ((df['pred_model_2'] > df['pred_model_1']) & (df[event_col] == 0)).sum()
        total_censored = len(df[event_col]) - total_events
       
        nri_events = (improved - worsened) / total_events
        nri_censored = (improved_censored - worsened_censored) / total_censored
        nri = nri_events + nri_censored
       
        return idi, nri, nri_events, nri_censored

    def bootstrap_nri_idi(df, duration_col, event_col, old_model_columns, new_model_columns, n_iterations=1000):
        nri_values = []
        idi_values = []
        
        for _ in range(n_iterations):
            # Bootstrap resample the data
            boot_df = df.sample(n=len(df), replace=True)
            
            # Calculate IDI and NRI on bootstrap sample
            idi_bootstrap, nri_bootstrap, _, _ = compute_idi_nri(boot_df, duration_col, event_col, old_model_columns, new_model_columns)
            
            nri_values.append(nri_bootstrap)
            idi_values.append(idi_bootstrap)
        
        # Convert lists to numpy arrays for easier calculations
        nri_values = np.array(nri_values)
        idi_values = np.array(idi_values)
        
        # Compute original IDI and NRI
        idi_original, nri_original, nri_events, nri_nonevents = compute_idi_nri(df, duration_col, event_col, old_model_columns, new_model_columns)
        
        # Calculate 95% confidence intervals
        nri_ci = np.percentile(nri_values, [2.5, 97.5])
        idi_ci = np.percentile(idi_values, [2.5, 97.5])
        
        # Calculate p-value (two-tailed test)
        nri_p_value = min(np.mean(nri_values >= nri_original), np.mean(nri_values <= nri_original)) * 2
        idi_p_value = min(np.mean(idi_values >= idi_original), np.mean(idi_values <= idi_original)) * 2
        
        return {
            'nri_original': nri_original, 'nri_ci': nri_ci, 'nri_p_value': nri_p_value,
            'idi_original': idi_original, 'idi_ci': idi_ci, 'idi_p_value': idi_p_value,
            'nri_events': nri_events, 'nri_nonevents': nri_nonevents
        }

    if uploaded_file is not None:
        try:
            df = read_file(uploaded_file)
            if df is not None:
                # Column selection section
                st.subheader("Column Selection")
                duration_column = st.selectbox("Select the Duration column", df.columns)
                dead_column = st.selectbox("Select the Dead column", df.columns)
                
                st.info("You can select different numbers of columns for the old and new models.")
                
                col1, col2 = st.columns(2)
                with col1:
                    old_model_columns = st.multiselect("Select the old model columns", df.columns, key="old_model_columns_idi_nri")
                with col2:
                    new_model_columns = st.multiselect("Select the new model columns", df.columns, key="new_model_columns_idi_nri")
                
                # Visual indicators for the number of columns selected for each model
                st.write(f"Old model columns selected: {len(old_model_columns)}")
                st.write(f"New model columns selected: {len(new_model_columns)}")

                if st.button("Compute IDI and NRI"):
                    if len(old_model_columns) == 0 or len(new_model_columns) == 0:
                        st.error("Please select at least one column for both old and new models.")
                    else:
                        try:
                            with st.spinner("Computing IDI and NRI..."):
                                results = bootstrap_nri_idi(df, duration_column, dead_column, old_model_columns, new_model_columns)
                            
                            st.subheader("IDI and NRI Results")
                            
                            # Display results in a table
                            results_table = pd.DataFrame({
                                'Metric': ['IDI', 'NRI', 'NRI (Events)', 'NRI (Non-events)'],
                                'Value': [f"{results['idi_original']:.4f}", f"{results['nri_original']:.4f}", f"{results['nri_events']:.4f}", f"{results['nri_nonevents']:.4f}"],
                                'P-value': [f"{results['idi_p_value']:.4f}", f"{results['nri_p_value']:.4f}", "-", "-"],
                                'Confidence Interval': [f"({results['idi_ci'][0]:.4f}, {results['idi_ci'][1]:.4f})", f"({results['nri_ci'][0]:.4f}, {results['nri_ci'][1]:.4f})", "-", "-"]
                            })

                            st.table(results_table)
                            
                            st.markdown("""
                            ### Interpretation of IDI Results:
                            - Positive IDI indicates that the new model improves risk prediction compared to the old model.
                            - The magnitude of IDI represents the degree of improvement.
                            - The 95% Confidence Interval (CI) indicates the range where the true IDI likely lies.
                            - A p-value < 0.05 suggests that the improvement is statistically significant.
                            
                            ### Interpretation of NRI Results:
                            - Positive NRI indicates that the new model improves risk classification compared to the old model.
                            - NRI for events shows the net improvement in classifying individuals who experience the event.
                            - NRI for non-events shows the net improvement in classifying individuals who do not experience the event.
                            - The total NRI is the sum of NRI for events and non-events.
                            - The 95% Confidence Interval (CI) indicates the range where the true NRI likely lies.
                            - A p-value < 0.05 suggests that the improvement in classification is statistically significant.
                            """)
                            
                            # Export results
                            st.download_button(
                                label="Download IDI/NRI Results",
                                data=results_table.to_csv(index=False),
                                file_name="idi_nri_results.csv",
                                mime="text/csv",
                            )
                        except ValueError as ve:
                            st.error(f"Error: {str(ve)}")
                            st.info("Please check your data for missing values and handle them appropriately.")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during computation: {str(e)}")
                            st.info("Please check your data for any inconsistencies or contact support if the issue persists.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.info("Please upload a CSV, Excel, or JSON file to begin the analysis.")

if __name__ == "__main__":
    app()
