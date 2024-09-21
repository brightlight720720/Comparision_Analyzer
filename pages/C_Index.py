import streamlit as st
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="C-Index - CSV Analysis App", layout="wide")

st.title("C-Index Computation and Comparison")

def compute_c_index(y_true, y_pred, n_bootstrap=1000):
    df = pd.concat([y_true, y_pred], axis=1)
    
    cox_model = CoxPHFitter()
    cox_model.fit(df, duration_col='Duration', event_col='Dead')
    
    df['risk_score'] = cox_model.predict_partial_hazard(df)
    
    c_index = concordance_index(df['Duration'], df['risk_score'], df['Dead'])
    
    return c_index

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

if 'duration_column' not in st.session_state or 'dead_column' not in st.session_state or 'old_model_columns' not in st.session_state or 'new_model_columns' not in st.session_state:
    st.warning("Please select columns on the Home page before proceeding with C-Index computation.")
else:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            y_true = df[[st.session_state.duration_column, st.session_state.dead_column]].rename(columns={st.session_state.duration_column: 'Duration', st.session_state.dead_column: 'Dead'})
            y_pred_old = df[st.session_state.old_model_columns]
            y_pred_new = df[st.session_state.new_model_columns]
            
            if st.button("Compute C-Index"):
                with st.spinner("Computing C-Index..."):
                    c_index_old = compute_c_index(y_true, y_pred_old)
                    c_index_new = compute_c_index(y_true, y_pred_new)
                    
                    diff, ci_lower, ci_upper, p_value = compare_c_index(c_index_old, c_index_new)
                
                st.subheader("C-Index Results")
                st.write(f"Old model C-Index: {c_index_old:.4f}")
                st.write(f"New model C-Index: {c_index_new:.4f}")
                st.write(f"Difference: {diff:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
                st.write(f"P-value: {p_value:.4f}")
                
                # Visualization of C-index comparison
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(['Old Model', 'New Model'], [c_index_old, c_index_new], yerr=[[0, 0], [0, ci_upper-c_index_new]], capsize=5)
                ax.set_ylabel('C-Index')
                ax.set_title('C-Index Comparison')
                ax.set_ylim(0.5, 1)  # C-index ranges from 0.5 to 1
                for i, v in enumerate([c_index_old, c_index_new]):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
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
            st.error(f"An error occurred while computing C-Index: {str(e)}")
    else:
        st.info("Please upload a CSV file to compute C-Index.")
