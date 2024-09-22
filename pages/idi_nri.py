import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import io

st.set_page_config(page_title="IDI/NRI - Multi-format Analysis App", layout="wide")

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
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

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

def compute_idi(y_true, y_pred_old, y_pred_new, n_bootstrap=1000, alpha=0.05):
    idi = np.mean((y_pred_new - y_pred_old) * (y_true - y_pred_old)) - np.mean((y_pred_new - y_pred_old) * (y_true - y_pred_new))
    
    bootstrap_idis = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_resampled = y_true.iloc[indices]
        y_pred_old_resampled = y_pred_old.iloc[indices]
        y_pred_new_resampled = y_pred_new.iloc[indices]
        bootstrap_idi = np.mean((y_pred_new_resampled - y_pred_old_resampled) * (y_true_resampled - y_pred_old_resampled)) - np.mean((y_pred_new_resampled - y_pred_old_resampled) * (y_true_resampled - y_pred_new_resampled))
        bootstrap_idis.append(bootstrap_idi)
    
    ci_lower = np.percentile(bootstrap_idis, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_idis, (1 - alpha/2) * 100)
    
    se = np.std(bootstrap_idis)
    z_score = idi / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return idi, ci_lower, ci_upper, p_value

def compute_nri(y_true, y_pred_old, y_pred_new, threshold=0.5, n_bootstrap=1000, alpha=0.05):
    def categorize(y_pred):
        return (y_pred > threshold).astype(int)
    
    old_categories = categorize(y_pred_old)
    new_categories = categorize(y_pred_new)
    
    events_improved = np.sum((new_categories > old_categories) & (y_true == 1))
    events_worsened = np.sum((new_categories < old_categories) & (y_true == 1))
    nonevents_improved = np.sum((new_categories < old_categories) & (y_true == 0))
    nonevents_worsened = np.sum((new_categories > old_categories) & (y_true == 0))
    
    n_events = np.sum(y_true == 1)
    n_nonevents = np.sum(y_true == 0)
    
    nri_events = (events_improved - events_worsened) / n_events
    nri_nonevents = (nonevents_improved - nonevents_worsened) / n_nonevents
    nri = nri_events + nri_nonevents
    
    bootstrap_nris = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_resampled = y_true.iloc[indices]
        y_pred_old_resampled = y_pred_old.iloc[indices]
        y_pred_new_resampled = y_pred_new.iloc[indices]
        
        old_categories_resampled = categorize(y_pred_old_resampled)
        new_categories_resampled = categorize(y_pred_new_resampled)
        
        events_improved = np.sum((new_categories_resampled > old_categories_resampled) & (y_true_resampled == 1))
        events_worsened = np.sum((new_categories_resampled < old_categories_resampled) & (y_true_resampled == 1))
        nonevents_improved = np.sum((new_categories_resampled < old_categories_resampled) & (y_true_resampled == 0))
        nonevents_worsened = np.sum((new_categories_resampled > old_categories_resampled) & (y_true_resampled == 0))
        
        n_events_resampled = np.sum(y_true_resampled == 1)
        n_nonevents_resampled = np.sum(y_true_resampled == 0)
        
        nri_events_resampled = (events_improved - events_worsened) / n_events_resampled
        nri_nonevents_resampled = (nonevents_improved - nonevents_worsened) / n_nonevents_resampled
        bootstrap_nris.append(nri_events_resampled + nri_nonevents_resampled)
    
    ci_lower = np.percentile(bootstrap_nris, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_nris, (1 - alpha/2) * 100)
    
    se = np.std(bootstrap_nris)
    z_score = nri / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return nri, nri_events, nri_nonevents, ci_lower, ci_upper, p_value

if uploaded_file is not None:
    try:
        df = read_file(uploaded_file)
        if df is not None:
            # Column selection section
            st.subheader("Column Selection")
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

            if st.button("Compute IDI and NRI"):
                if len(old_model_columns) == 0 or len(new_model_columns) == 0:
                    st.error("Please select at least one column for both old and new models.")
                else:
                    with st.spinner("Computing IDI and NRI..."):
                        y_true = df[dead_column]
                        y_pred_old = df[old_model_columns].mean(axis=1)
                        y_pred_new = df[new_model_columns].mean(axis=1)
                        
                        idi, idi_ci_lower, idi_ci_upper, idi_p_value = compute_idi(y_true, y_pred_old, y_pred_new)
                        nri, nri_events, nri_nonevents, nri_ci_lower, nri_ci_upper, nri_p_value = compute_nri(y_true, y_pred_old, y_pred_new)
                    
                    st.subheader("IDI and NRI Results")
                    
                    # Display results in a table
                    results_table = pd.DataFrame({
                        'Metric': ['IDI', 'NRI', 'NRI (Events)', 'NRI (Non-events)'],
                        'Value': [f"{idi:.4f}", f"{nri:.4f}", f"{nri_events:.4f}", f"{nri_nonevents:.4f}"],
                        'P-value': [f"{idi_p_value:.4f}", f"{nri_p_value:.4f}", "-", "-"],
                        'Confidence Interval': [f"({idi_ci_lower:.4f}, {idi_ci_upper:.4f})", f"({nri_ci_lower:.4f}, {nri_ci_upper:.4f})", "-", "-"]
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
    except Exception as e:
        st.error(f"An error occurred while processing the file: {str(e)}")
else:
    st.info("Please upload a CSV, Excel, or JSON file to begin the analysis.")
