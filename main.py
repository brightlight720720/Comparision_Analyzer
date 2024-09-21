import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="CSV Analysis App", layout="wide")

# Function to compute C-index (ROC AUC)
def compute_c_index(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# Function to compute IDI
def compute_idi(y_true, y_pred_old, y_pred_new):
    # Compute mean predicted probabilities for each model
    mean_old = np.mean(y_pred_old)
    mean_new = np.mean(y_pred_new)
    
    # Compute IDI
    idi_events = np.mean(y_pred_new[y_true == 1]) - np.mean(y_pred_old[y_true == 1])
    idi_nonevents = np.mean(y_pred_old[y_true == 0]) - np.mean(y_pred_new[y_true == 0])
    idi = idi_events + idi_nonevents
    
    return idi, mean_old, mean_new

# Function to compute NRI
def compute_nri(y_true, y_pred_old, y_pred_new, threshold=0.5):
    # Compute reclassification for events and non-events
    events_improved = np.sum((y_pred_new > threshold) & (y_pred_old <= threshold) & (y_true == 1))
    events_worsened = np.sum((y_pred_new <= threshold) & (y_pred_old > threshold) & (y_true == 1))
    nonevents_improved = np.sum((y_pred_new <= threshold) & (y_pred_old > threshold) & (y_true == 0))
    nonevents_worsened = np.sum((y_pred_new > threshold) & (y_pred_old <= threshold) & (y_true == 0))
    
    # Compute NRI
    nri_events = (events_improved - events_worsened) / np.sum(y_true == 1)
    nri_nonevents = (nonevents_improved - nonevents_worsened) / np.sum(y_true == 0)
    nri = nri_events + nri_nonevents
    
    return nri, nri_events, nri_nonevents

# Main app
def main():
    st.title("CSV Analysis App")
    st.write("Upload a CSV file, select columns, and compute C index, IDI, and NRI")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())

        # Column selection
        st.subheader("Column Selection")
        target_column = st.selectbox("Select the target column", df.columns)
        old_model_column = st.selectbox("Select the old model predictions column", df.columns)
        new_model_column = st.selectbox("Select the new model predictions column", df.columns)

        if st.button("Compute Metrics"):
            # Prepare data
            y_true = df[target_column]
            y_pred_old = df[old_model_column]
            y_pred_new = df[new_model_column]

            # Compute metrics
            c_index_old = compute_c_index(y_true, y_pred_old)
            c_index_new = compute_c_index(y_true, y_pred_new)
            idi, mean_old, mean_new = compute_idi(y_true, y_pred_old, y_pred_new)
            nri, nri_events, nri_nonevents = compute_nri(y_true, y_pred_old, y_pred_new)

            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.write("C-index (ROC AUC):")
                st.write(f"Old model: {c_index_old:.4f}")
                st.write(f"New model: {c_index_new:.4f}")
                st.write(f"Improvement: {c_index_new - c_index_old:.4f}")

                st.write("\nIntegrated Discrimination Improvement (IDI):")
                st.write(f"IDI: {idi:.4f}")
                st.write(f"Mean predicted probability (old model): {mean_old:.4f}")
                st.write(f"Mean predicted probability (new model): {mean_new:.4f}")

            with col2:
                st.write("Net Reclassification Improvement (NRI):")
                st.write(f"NRI: {nri:.4f}")
                st.write(f"NRI for events: {nri_events:.4f}")
                st.write(f"NRI for non-events: {nri_nonevents:.4f}")

            # Visualizations
            st.subheader("Visualizations")
            
            # ROC curve
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
            ax.plot(roc_curve(y_true, y_pred_old)[0], roc_curve(y_true, y_pred_old)[1], label='Old Model')
            ax.plot(roc_curve(y_true, y_pred_new)[0], roc_curve(y_true, y_pred_new)[1], label='New Model')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax.legend()
            st.pyplot(fig)

            # Calibration plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
            ax.plot(*calibration_curve(y_true, y_pred_old, n_bins=10), marker='o', label='Old Model')
            ax.plot(*calibration_curve(y_true, y_pred_new, n_bins=10), marker='o', label='New Model')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
