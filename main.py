import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="CSV Analysis App", layout="wide")

# Function to compute C-index (ROC AUC) for multiple columns
def compute_c_index(y_true, y_pred):
    return [roc_auc_score(y_true, y_pred[col]) for col in y_pred.columns]

# Function to compute IDI for multiple columns
def compute_idi(y_true, y_pred_old, y_pred_new):
    idi_values = []
    mean_old_values = []
    mean_new_values = []
    
    for old_col, new_col in zip(y_pred_old.columns, y_pred_new.columns):
        mean_old = np.mean(y_pred_old[old_col])
        mean_new = np.mean(y_pred_new[new_col])
        
        idi_events = np.mean(y_pred_new[new_col][y_true == 1]) - np.mean(y_pred_old[old_col][y_true == 1])
        idi_nonevents = np.mean(y_pred_old[old_col][y_true == 0]) - np.mean(y_pred_new[new_col][y_true == 0])
        idi = idi_events + idi_nonevents
        
        idi_values.append(idi)
        mean_old_values.append(mean_old)
        mean_new_values.append(mean_new)
    
    return idi_values, mean_old_values, mean_new_values

# Function to compute NRI for multiple columns
def compute_nri(y_true, y_pred_old, y_pred_new, threshold=0.5):
    nri_values = []
    nri_events_values = []
    nri_nonevents_values = []
    
    for old_col, new_col in zip(y_pred_old.columns, y_pred_new.columns):
        events_improved = np.sum((y_pred_new[new_col] > threshold) & (y_pred_old[old_col] <= threshold) & (y_true == 1))
        events_worsened = np.sum((y_pred_new[new_col] <= threshold) & (y_pred_old[old_col] > threshold) & (y_true == 1))
        nonevents_improved = np.sum((y_pred_new[new_col] <= threshold) & (y_pred_old[old_col] > threshold) & (y_true == 0))
        nonevents_worsened = np.sum((y_pred_new[new_col] > threshold) & (y_pred_old[old_col] <= threshold) & (y_true == 0))
        
        nri_events = (events_improved - events_worsened) / np.sum(y_true == 1)
        nri_nonevents = (nonevents_improved - nonevents_worsened) / np.sum(y_true == 0)
        nri = nri_events + nri_nonevents
        
        nri_values.append(nri)
        nri_events_values.append(nri_events)
        nri_nonevents_values.append(nri_nonevents)
    
    return nri_values, nri_events_values, nri_nonevents_values

# New function to split CSV based on column condition
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

# New function to generate CSV summary
def generate_csv_summary(dataframe):
    summary = {}
    summary['num_rows'] = dataframe.shape[0]
    summary['num_columns'] = dataframe.shape[1]
    summary['column_info'] = []
    
    for column in dataframe.columns:
        col_info = {
            'name': column,
            'dtype': str(dataframe[column].dtype),
            'missing_values': dataframe[column].isnull().sum()
        }
        
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            col_info.update({
                'mean': dataframe[column].mean(),
                'median': dataframe[column].median(),
                'min': dataframe[column].min(),
                'max': dataframe[column].max(),
                'std': dataframe[column].std()
            })
        else:
            col_info.update({
                'unique_values': dataframe[column].nunique(),
                'top_5_values': dataframe[column].value_counts().nlargest(5).to_dict()
            })
        
        summary['column_info'].append(col_info)
    
    return summary

# Main app
def main():
    st.title("CSV Analysis App")
    st.write("Upload a CSV file, select columns, and compute C index, IDI, and NRI")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Generate and display CSV summary
        st.subheader("CSV Summary")
        summary = generate_csv_summary(df)
        st.write(f"Number of rows: {summary['num_rows']}")
        st.write(f"Number of columns: {summary['num_columns']}")
        
        for col_info in summary['column_info']:
            st.write(f"\nColumn: {col_info['name']}")
            st.write(f"Data type: {col_info['dtype']}")
            st.write(f"Missing values: {col_info['missing_values']}")
            
            if 'mean' in col_info:
                st.write(f"Mean: {col_info['mean']:.2f}")
                st.write(f"Median: {col_info['median']:.2f}")
                st.write(f"Min: {col_info['min']:.2f}")
                st.write(f"Max: {col_info['max']:.2f}")
                st.write(f"Standard deviation: {col_info['std']:.2f}")
            else:
                st.write(f"Unique values: {col_info['unique_values']}")
                st.write("Top 5 most frequent values:")
                for value, count in col_info['top_5_values'].items():
                    st.write(f"  {value}: {count}")
        
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
                st.write(df_true)
                st.write("Summary for rows that meet the condition:")
                st.write(generate_csv_summary(df_true))
                
                st.write("Rows that don't meet the condition:")
                st.write(df_false)
                st.write("Summary for rows that don't meet the condition:")
                st.write(generate_csv_summary(df_false))
            except ValueError as e:
                st.error(f"Error: {str(e)}")

        # Column selection
        st.subheader("Column Selection")
        target_column = st.selectbox("Select the target column", df.columns)
        old_model_columns = st.multiselect("Select the old model predictions columns", df.columns)
        new_model_columns = st.multiselect("Select the new model predictions columns", df.columns)

        if st.button("Compute Metrics") and len(old_model_columns) == len(new_model_columns):
            # Prepare data
            y_true = df[target_column]
            y_pred_old = df[old_model_columns]
            y_pred_new = df[new_model_columns]

            # Compute metrics
            c_index_old = compute_c_index(y_true, y_pred_old)
            c_index_new = compute_c_index(y_true, y_pred_new)
            idi_values, mean_old_values, mean_new_values = compute_idi(y_true, y_pred_old, y_pred_new)
            nri_values, nri_events_values, nri_nonevents_values = compute_nri(y_true, y_pred_old, y_pred_new)

            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.write("C-index (ROC AUC):")
                for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                    st.write(f"{old_col} vs {new_col}:")
                    st.write(f"Old model: {c_index_old[i]:.4f}")
                    st.write(f"New model: {c_index_new[i]:.4f}")
                    st.write(f"Improvement: {c_index_new[i] - c_index_old[i]:.4f}")
                    st.write("")

                st.write("Integrated Discrimination Improvement (IDI):")
                for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                    st.write(f"{old_col} vs {new_col}:")
                    st.write(f"IDI: {idi_values[i]:.4f}")
                    st.write(f"Mean predicted probability (old model): {mean_old_values[i]:.4f}")
                    st.write(f"Mean predicted probability (new model): {mean_new_values[i]:.4f}")
                    st.write("")

            with col2:
                st.write("Net Reclassification Improvement (NRI):")
                for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                    st.write(f"{old_col} vs {new_col}:")
                    st.write(f"NRI: {nri_values[i]:.4f}")
                    st.write(f"NRI for events: {nri_events_values[i]:.4f}")
                    st.write(f"NRI for non-events: {nri_nonevents_values[i]:.4f}")
                    st.write("")

            # Visualizations
            st.subheader("Visualizations")
            
            # ROC curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
            
            for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                fpr_old, tpr_old, _ = roc_curve(y_true, y_pred_old[old_col])
                fpr_new, tpr_new, _ = roc_curve(y_true, y_pred_new[new_col])
                
                ax.plot(fpr_old, tpr_old, label=f'Old Model ({old_col})')
                ax.plot(fpr_new, tpr_new, label=f'New Model ({new_col})')
            
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend()
            st.pyplot(fig)

            # Calibration plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
            
            for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                prob_true_old, prob_pred_old = calibration_curve(y_true, y_pred_old[old_col], n_bins=10)
                prob_true_new, prob_pred_new = calibration_curve(y_true, y_pred_new[new_col], n_bins=10)
                
                ax.plot(prob_pred_old, prob_true_old, marker='o', label=f'Old Model ({old_col})')
                ax.plot(prob_pred_new, prob_true_new, marker='o', label=f'New Model ({new_col})')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
