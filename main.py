import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import io
from scipy import stats
from sklearn.utils import resample
from lifelines.utils import concordance_index

# Set page title and layout
st.set_page_config(page_title="CSV Analysis App", layout="wide")

# Function to compute C-index with confidence interval and p-value
def compute_c_index(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    results = []
    for col in y_pred.columns:
        c_index = concordance_index(y_true['Duration'], y_pred[col], y_true['Dead'])
        
        # Bootstrap for confidence interval
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            indices = resample(range(len(y_true)))
            y_true_resampled = y_true.iloc[indices]
            y_pred_resampled = y_pred[col].iloc[indices]
            bootstrap_scores.append(concordance_index(y_true_resampled['Duration'], y_pred_resampled, y_true_resampled['Dead']))
        
        ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        
        # Compute p-value (two-tailed test against null hypothesis of C-index = 0.5)
        z_score = (c_index - 0.5) / np.std(bootstrap_scores)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        results.append({
            'column': col,
            'c_index': c_index,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value
        })
    
    return results

# Function to compute IDI with confidence interval and p-value
def compute_idi(y_true, y_pred_old, y_pred_new, n_bootstrap=1000, alpha=0.05):
    results = []
    for old_col, new_col in zip(y_pred_old.columns, y_pred_new.columns):
        idi = np.mean(y_pred_new[new_col] - y_pred_old[old_col])
        
        # Bootstrap for confidence interval and p-value
        bootstrap_idis = []
        for _ in range(n_bootstrap):
            indices = resample(range(len(y_true)))
            y_true_resampled = y_true.iloc[indices]
            y_pred_old_resampled = y_pred_old[old_col].iloc[indices]
            y_pred_new_resampled = y_pred_new[new_col].iloc[indices]
            bootstrap_idis.append(np.mean(y_pred_new_resampled - y_pred_old_resampled))
        
        ci_lower = np.percentile(bootstrap_idis, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_idis, (1 - alpha/2) * 100)
        
        # Compute p-value (two-tailed test against null hypothesis of IDI = 0)
        t_statistic = idi / (np.std(bootstrap_idis) / np.sqrt(n_bootstrap))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n_bootstrap-1))
        
        results.append({
            'idi': idi,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'mean_old': np.mean(y_pred_old[old_col]),
            'mean_new': np.mean(y_pred_new[new_col])
        })
    
    return results

# Function to compute NRI with confidence interval and p-value
def compute_nri(y_true, y_pred_old, y_pred_new, threshold=0.5, n_bootstrap=1000, alpha=0.05):
    results = []
    for old_col, new_col in zip(y_pred_old.columns, y_pred_new.columns):
        events_improved = np.sum((y_pred_new[new_col] > threshold) & (y_pred_old[old_col] <= threshold) & (y_true == 1))
        events_worsened = np.sum((y_pred_new[new_col] <= threshold) & (y_pred_old[old_col] > threshold) & (y_true == 1))
        nonevents_improved = np.sum((y_pred_new[new_col] <= threshold) & (y_pred_old[old_col] > threshold) & (y_true == 0))
        nonevents_worsened = np.sum((y_pred_new[new_col] > threshold) & (y_pred_old[old_col] <= threshold) & (y_true == 0))
        
        nri_events = (events_improved - events_worsened) / np.sum(y_true == 1)
        nri_nonevents = (nonevents_improved - nonevents_worsened) / np.sum(y_true == 0)
        nri = nri_events + nri_nonevents
        
        # Bootstrap for confidence interval and p-value
        bootstrap_nris = []
        for _ in range(n_bootstrap):
            indices = resample(range(len(y_true)))
            y_true_resampled = y_true.iloc[indices]
            y_pred_old_resampled = y_pred_old[old_col].iloc[indices]
            y_pred_new_resampled = y_pred_new[new_col].iloc[indices]
            
            events_improved = np.sum((y_pred_new_resampled > threshold) & (y_pred_old_resampled <= threshold) & (y_true_resampled == 1))
            events_worsened = np.sum((y_pred_new_resampled <= threshold) & (y_pred_old_resampled > threshold) & (y_true_resampled == 1))
            nonevents_improved = np.sum((y_pred_new_resampled <= threshold) & (y_pred_old_resampled > threshold) & (y_true_resampled == 0))
            nonevents_worsened = np.sum((y_pred_new_resampled > threshold) & (y_pred_old_resampled <= threshold) & (y_true_resampled == 0))
            
            nri_events = (events_improved - events_worsened) / np.sum(y_true_resampled == 1)
            nri_nonevents = (nonevents_improved - nonevents_worsened) / np.sum(y_true_resampled == 0)
            bootstrap_nris.append(nri_events + nri_nonevents)
        
        ci_lower = np.percentile(bootstrap_nris, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_nris, (1 - alpha/2) * 100)
        
        # Compute p-value (two-tailed test against null hypothesis of NRI = 0)
        t_statistic = nri / (np.std(bootstrap_nris) / np.sqrt(n_bootstrap))
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n_bootstrap-1))
        
        results.append({
            'nri': nri,
            'nri_events': nri_events,
            'nri_nonevents': nri_nonevents,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value
        })
    
    return results

# Function to split CSV based on column condition
def split_csv(dataframe, column_name, condition_value, condition_operator):
    if condition_operator == "equal to":
        condition = dataframe[column_name] == condition_value
    elif condition_operator == "greater than":
        condition = dataframe[column_name] > condition_value
    elif condition_operator == "less than":
        condition = dataframe[column_name] < condition_value
    else:
        raise ValueError("Invalid condition operator")
    
    df_true = dataframe[condition]
    df_false = dataframe[~condition]
    
    return df_true, df_false

# Modified function to generate CSV summary as a pandas DataFrame
def generate_csv_summary(dataframe):
    summary = {
        'Column': [],
        'Data Type': [],
        'Missing Values': [],
        'Unique Values': [],
        'Mean': [],
        'Median': [],
        'Min': [],
        'Max': [],
        'Std': [],
        'Top 5 Values': []
    }
    
    for column in dataframe.columns:
        summary['Column'].append(column)
        summary['Data Type'].append(str(dataframe[column].dtype))
        summary['Missing Values'].append(dataframe[column].isnull().sum())
        summary['Unique Values'].append(dataframe[column].nunique())
        
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            summary['Mean'].append(dataframe[column].mean())
            summary['Median'].append(dataframe[column].median())
            summary['Min'].append(dataframe[column].min())
            summary['Max'].append(dataframe[column].max())
            summary['Std'].append(dataframe[column].std())
            summary['Top 5 Values'].append('')
        else:
            summary['Mean'].append('')
            summary['Median'].append('')
            summary['Min'].append('')
            summary['Max'].append('')
            summary['Std'].append('')
            top_5 = dataframe[column].value_counts().nlargest(5).to_dict()
            summary['Top 5 Values'].append(', '.join([f"{k}: {v}" for k, v in top_5.items()]))
    
    return pd.DataFrame(summary)

# Function to convert dataframe to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Main app
def main():
    st.title("CSV Analysis App")
    st.write("Upload a CSV file, select columns, and compute C index, IDI, and NRI")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Generate and display CSV summary
            st.subheader("CSV Summary")
            summary_df = generate_csv_summary(df)
            st.table(summary_df)
            
            # Add download button for original dataset
            st.download_button(
                label="Download original dataset",
                data=convert_df_to_csv(df),
                file_name="original_dataset.csv",
                mime="text/csv",
            )
            
            st.subheader("Data Preview")
            st.write(df.head())

            # New section for CSV splitting
            st.subheader("Split CSV")
            split_column = st.selectbox("Select column to split on", df.columns)
            condition_operator = st.selectbox("Select condition operator", ["equal to", "greater than", "less than"])
            condition_value = st.text_input("Enter condition value")

            if st.button("Split CSV"):
                try:
                    condition_value = float(condition_value) if condition_value.replace('.', '').isdigit() else condition_value
                    df_true, df_false = split_csv(df, split_column, condition_value, condition_operator)
                    
                    st.write("Rows that meet the condition:")
                    st.table(generate_csv_summary(df_true))
                    st.write(df_true.head())
                    
                    # Add download button for rows that meet the condition
                    st.download_button(
                        label="Download rows that meet the condition",
                        data=convert_df_to_csv(df_true),
                        file_name="rows_meet_condition.csv",
                        mime="text/csv",
                    )
                    
                    st.write("Rows that don't meet the condition:")
                    st.table(generate_csv_summary(df_false))
                    st.write(df_false.head())
                    
                    # Add download button for rows that don't meet the condition
                    st.download_button(
                        label="Download rows that don't meet the condition",
                        data=convert_df_to_csv(df_false),
                        file_name="rows_dont_meet_condition.csv",
                        mime="text/csv",
                    )
                except ValueError as e:
                    st.error(f"Error: {str(e)}")

            # Column selection
            st.subheader("Column Selection")
            st.write("Please select the required columns for analysis.")
            
            duration_column = st.selectbox("Select the Duration column", df.columns)
            dead_column = st.selectbox("Select the Dead column", df.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                old_model_columns = st.multiselect("Select the old model predictions columns", df.columns)
                st.write(f"Old model columns selected: {len(old_model_columns)}")
            
            with col2:
                new_model_columns = st.multiselect("Select the new model predictions columns", df.columns)
                st.write(f"New model columns selected: {len(new_model_columns)}")

            if st.button("Compute Metrics"):
                try:
                    # Prepare data
                    y_true = df[[duration_column, dead_column]].rename(columns={duration_column: 'Duration', dead_column: 'Dead'})
                    y_pred_old = df[old_model_columns]
                    y_pred_new = df[new_model_columns]

                    # Compute metrics
                    with st.spinner("Computing metrics..."):
                        c_index_results_old = compute_c_index(y_true, y_pred_old)
                        c_index_results_new = compute_c_index(y_true, y_pred_new)
                        idi_results = compute_idi(y_true['Dead'], y_pred_old, y_pred_new)
                        nri_results = compute_nri(y_true['Dead'], y_pred_old, y_pred_new)

                    # Display results
                    st.subheader("Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("C-index:")
                        for old_result, new_result in zip(c_index_results_old, c_index_results_new):
                            old_col = old_result['column']
                            new_col = new_result['column']
                            st.write(f"{old_col} vs {new_col}:")
                            st.write(f"Old model: {old_result['c_index']:.4f} (95% CI: {old_result['ci_lower']:.4f} - {old_result['ci_upper']:.4f}, p-value: {old_result['p_value']:.4f})")
                            st.write(f"New model: {new_result['c_index']:.4f} (95% CI: {new_result['ci_lower']:.4f} - {new_result['ci_upper']:.4f}, p-value: {new_result['p_value']:.4f})")
                            st.write(f"Improvement: {new_result['c_index'] - old_result['c_index']:.4f}")
                            st.write("")

                        st.write("Integrated Discrimination Improvement (IDI):")
                        for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                            st.write(f"{old_col} vs {new_col}:")
                            st.write(f"IDI: {idi_results[i]['idi']:.4f} (95% CI: {idi_results[i]['ci_lower']:.4f} - {idi_results[i]['ci_upper']:.4f}, p-value: {idi_results[i]['p_value']:.4f})")
                            st.write(f"Mean predicted probability (old model): {idi_results[i]['mean_old']:.4f}")
                            st.write(f"Mean predicted probability (new model): {idi_results[i]['mean_new']:.4f}")
                            st.write("")

                    with col2:
                        st.write("Net Reclassification Improvement (NRI):")
                        for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                            st.write(f"{old_col} vs {new_col}:")
                            st.write(f"NRI: {nri_results[i]['nri']:.4f} (95% CI: {nri_results[i]['ci_lower']:.4f} - {nri_results[i]['ci_upper']:.4f}, p-value: {nri_results[i]['p_value']:.4f})")
                            st.write(f"NRI for events: {nri_results[i]['nri_events']:.4f}")
                            st.write(f"NRI for non-events: {nri_results[i]['nri_nonevents']:.4f}")
                            st.write("")

                    # Visualizations
                    st.subheader("Visualizations")
                    
                    # ROC curve
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
                    
                    for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                        fpr_old, tpr_old, _ = roc_curve(y_true['Dead'], y_pred_old[old_col])
                        fpr_new, tpr_new, _ = roc_curve(y_true['Dead'], y_pred_new[new_col])
                        
                        ax.plot(fpr_old, tpr_old, label=f'Old Model ({old_col})')
                        ax.plot(fpr_new, tpr_new, label=f'New Model ({new_col})')
                        
                        # Add error bars for C-index
                        ax.errorbar(0.5, c_index_results_old[i]['c_index'], 
                                    yerr=[[c_index_results_old[i]['c_index'] - c_index_results_old[i]['ci_lower']], 
                                          [c_index_results_old[i]['ci_upper'] - c_index_results_old[i]['c_index']]], 
                                    fmt='o', capsize=5, label=f'Old Model CI ({old_col})')
                        ax.errorbar(0.55, c_index_results_new[i]['c_index'], 
                                    yerr=[[c_index_results_new[i]['c_index'] - c_index_results_new[i]['ci_lower']], 
                                          [c_index_results_new[i]['ci_upper'] - c_index_results_new[i]['c_index']]], 
                                    fmt='o', capsize=5, label=f'New Model CI ({new_col})')
                    
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend()
                    st.pyplot(fig)

                    # Calibration plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
                    
                    for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                        prob_true_old, prob_pred_old = calibration_curve(y_true['Dead'], y_pred_old[old_col], n_bins=10)
                        prob_true_new, prob_pred_new = calibration_curve(y_true['Dead'], y_pred_new[new_col], n_bins=10)
                        
                        ax.plot(prob_pred_old, prob_true_old, marker='o', label=f'Old Model ({old_col})')
                        ax.plot(prob_pred_new, prob_true_new, marker='o', label=f'New Model ({new_col})')
                    
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.set_title('Calibration Plot')
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred while computing metrics: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {str(e)}")

if __name__ == "__main__":
    main()