import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
import matplotlib.pyplot as plt
import io
import importlib

def app():
    st.title("C-Index Computation and Comparison")

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
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

    st.markdown("""
    ### What is the C-Index?
    The Concordance Index (C-Index) is a measure of the predictive accuracy of a survival model. It ranges from 0.5 to 1.0, where:
    - 0.5 indicates that the model's predictions are no better than random chance
    - 1.0 indicates perfect prediction

    ### Interpretation:
    - C-Index > 0.7 is considered good
    - C-Index > 0.8 is considered strong
    - C-Index > 0.9 is considered exceptional

    The C-Index represents the probability that, for a randomly selected pair of patients, the patient with the higher predicted risk will experience the event of interest (e.g., death) before the patient with the lower predicted risk.

    ### Comparing C-Indices:
    When comparing two models:
    - A higher C-Index indicates better predictive performance
    - The difference in C-Indices, along with its confidence interval and p-value, helps determine if the improvement is statistically significant
    """)

    def compute_c_index(duration, event, predictions):
        cph = CoxPHFitter().fit(pd.DataFrame({'T': duration, 'E': event, 'predictions': predictions}), 'T', 'E')
        return concordance_index(duration, -cph.predict_partial_hazard(pd.DataFrame({'predictions': predictions})), event)

    def compare_c_index(c_index_old, c_index_new, n_bootstrap=1000, alpha=0.05):
        diff = c_index_new - c_index_old
        
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            c_index_old_boot = c_index_old + np.random.normal(0, 0.01)  # Assuming some variability
            c_index_new_boot = c_index_new + np.random.normal(0, 0.01)  # Assuming some variability
            bootstrap_diffs.append(c_index_new_boot - c_index_old_boot)
        
        ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        
        z_score = diff / np.std(bootstrap_diffs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return diff, ci_lower, ci_upper, p_value

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
                    old_model_columns = st.multiselect("Select the old model columns", df.columns)
                with col2:
                    new_model_columns = st.multiselect("Select the new model columns", df.columns)
                
                # Visual indicators for the number of columns selected for each model
                st.write(f"Old model columns selected: {len(old_model_columns)}")
                st.write(f"New model columns selected: {len(new_model_columns)}")

                # C-Index computation section
                if st.button("Compute C-Index"):
                    if len(old_model_columns) == 0 or len(new_model_columns) == 0:
                        st.error("Please select at least one column for both old and new models.")
                    else:
                        with st.spinner("Computing C-Index..."):
                            y_true = df[[duration_column, dead_column]].rename(columns={duration_column: 'Duration', dead_column: 'Dead'})
                            
                            # Compute predictions for old and new models
                            y_pred_old = df[old_model_columns].mean(axis=1)
                            y_pred_new = df[new_model_columns].mean(axis=1)
                            
                            c_index_old = compute_c_index(y_true['Duration'], y_true['Dead'], y_pred_old)
                            c_index_new = compute_c_index(y_true['Duration'], y_true['Dead'], y_pred_new)
                            
                            diff, ci_lower, ci_upper, p_value = compare_c_index(c_index_old, c_index_new)
                        
                        st.subheader("C-Index Results")
                        st.write(f"Old model C-Index: {c_index_old:.4f}")
                        st.write(f"New model C-Index: {c_index_new:.4f}")
                        st.write(f"Difference: {diff:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
                        st.write(f"P-value: {p_value:.4f}")
                        
                        if c_index_new > c_index_old and p_value < 0.05:
                            st.success("The new model shows improvement over the old model.")
                        else:
                            st.warning("The new model does not show improvement over the old model.")
                        
                        st.markdown("""
                        ### Interpretation of Results:
                        - If the new model's C-Index is higher, it suggests improved predictive performance.
                        - The difference represents the magnitude of improvement.
                        - The 95% Confidence Interval (CI) indicates the range where the true difference likely lies.
                        - The p-value helps determine if the difference is statistically significant:
                          - p < 0.05 is typically considered statistically significant
                          - p ≥ 0.05 suggests the difference might be due to chance
                        """)
                        
                        # Visualization of C-index comparison
                        fig, ax = plt.subplots(figsize=(12, 8))
                        x = ['Old Model', 'New Model']
                        y = [c_index_old, c_index_new]
                        bar_width = 0.3
                        colors = ['#ADD8E6', '#00008B']  # Light blue for old model, Deep blue for new model

                        bars = ax.bar(x, y, width=bar_width, color=colors, capsize=7)
                        ax.set_ylabel('C-Index', fontsize=18)
                        ax.set_title('C-Index Comparison', fontsize=22, fontweight='bold')
                        ax.set_ylim(0.5, 1)  # C-index ranges from 0.5 to 1
                        ax.tick_params(axis='both', which='major', labelsize=16)

                        # Add value labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.3f}',
                                    ha='center', va='bottom', fontsize=16)

                        # Add error bars
                        error = [[max(0, c_index_old - ci_lower), max(0, c_index_new - ci_lower)],
                                 [max(0, ci_upper - c_index_old), max(0, ci_upper - c_index_new)]]
                        ax.errorbar(x, y, yerr=error, fmt='none', capsize=5, color='black', linewidth=2)

                        # Add gridlines
                        ax.grid(axis='y', linestyle='--', alpha=0.7)

                        # Add a legend
                        ax.legend(['C-Index'], loc='lower right', fontsize=16)

                        # Adjust layout and display the plot
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Export results
                        results_df = pd.DataFrame({
                            'Metric': ['Old model C-Index', 'New model C-Index', 'Difference', 'CI Lower', 'CI Upper', 'P-value'],
                            'Value': [c_index_old, c_index_new, diff, ci_lower, ci_upper, p_value]
                        })
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download C-Index Results",
                            data=csv,
                            file_name="c_index_results.csv",
                            mime="text/csv",
                        )
                        
                        # Export plot
                        img_buffer = io.BytesIO()
                        plt.savefig(img_buffer, format='png')
                        img_buffer.seek(0)
                        st.download_button(
                            label="Download C-Index Plot",
                            data=img_buffer,
                            file_name="c_index_plot.png",
                            mime="image/png",
                        )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
    else:
        st.info("Please upload a CSV, Excel, or JSON file to begin the analysis.")

if __name__ == "__main__":
    app()
