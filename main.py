import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import io
from scipy import stats
from sklearn.utils import resample
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import seaborn as sns

st.set_page_config(page_title="CSV Analysis App", layout="wide")

def compute_c_index(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    results = []
    
    df = pd.concat([y_true, y_pred], axis=1)
    
    cox_model = CoxPHFitter()
    cox_model.fit(df, duration_col='Duration', event_col='Dead')
    
    df['risk_score'] = cox_model.predict_partial_hazard(df)
    
    c_index = concordance_index(df['Duration'], df['risk_score'], df['Dead'])
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        indices = resample(range(len(df)))
        df_resampled = df.iloc[indices]
        cox_model_boot = CoxPHFitter()
        cox_model_boot.fit(df_resampled, duration_col='Duration', event_col='Dead')
        df_resampled['risk_score'] = cox_model_boot.predict_partial_hazard(df_resampled)
        bootstrap_scores.append(concordance_index(df_resampled['Duration'], df_resampled['risk_score'], df_resampled['Dead']))
    
    ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    
    z_score = (c_index - 0.5) / np.std(bootstrap_scores)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    results.append({
        'c_index': c_index,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value
    })
    
    return results

def compute_idi(y_true, y_pred_old, y_pred_new, n_bootstrap=1000, alpha=0.05):
    results = []
    for old_col, new_col in zip(y_pred_old.columns, y_pred_new.columns):
        idi = np.mean(y_pred_new[new_col] - y_pred_old[old_col])
        
        bootstrap_idis = []
        for _ in range(n_bootstrap):
            indices = resample(range(len(y_true)))
            y_true_resampled = y_true.iloc[indices]
            y_pred_old_resampled = y_pred_old[old_col].iloc[indices]
            y_pred_new_resampled = y_pred_new[new_col].iloc[indices]
            bootstrap_idis.append(np.mean(y_pred_new_resampled - y_pred_old_resampled))
        
        ci_lower = np.percentile(bootstrap_idis, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_idis, (1 - alpha/2) * 100)
        
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

def plot_kaplan_meier(df, duration_col, event_col):
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col], label="Kaplan-Meier Estimate")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf.plot(ax=ax)
    ax.set_title('Kaplan-Meier Survival Curve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    return fig

def plot_cox_summary(df, duration_col, event_col, covariates):
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    
    summary = cph.summary
    fig, ax = plt.subplots(figsize=(12, len(covariates) * 0.5))
    sns.heatmap(summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].sort_values(by='exp(coef)'), 
                annot=True, fmt='.3f', cmap='coolwarm', center=1, ax=ax)
    ax.set_title('Cox Proportional Hazards Model Summary')
    return fig

def plot_hazard_ratio_forest(df, duration_col, event_col, covariates):
    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    
    summary = cph.summary.sort_values(by='exp(coef)')
    fig, ax = plt.subplots(figsize=(10, len(covariates) * 0.5))
    
    y_pos = range(len(summary))
    ax.errorbar(summary['exp(coef)'], y_pos, xerr=[summary['exp(coef)'] - summary['exp(coef) lower 95%'], 
                                                   summary['exp(coef) upper 95%'] - summary['exp(coef)']],
                fmt='o', capsize=5, capthick=2, ecolor='gray')
    
    ax.axvline(x=1, color='r', linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel('Hazard Ratio (95% CI)')
    ax.set_title('Hazard Ratio Forest Plot')
    
    return fig

def generate_report(c_index_results_old, c_index_results_new, idi_results, nri_results, old_model_columns, new_model_columns):
    report = []
    
    report.append(["Metric", "Old Model", "New Model", "Improvement"])
    report.append(["C-index", 
                   f"{c_index_results_old[0]['c_index']:.4f} (95% CI: {c_index_results_old[0]['ci_lower']:.4f} - {c_index_results_old[0]['ci_upper']:.4f}, p-value: {c_index_results_old[0]['p_value']:.4f})",
                   f"{c_index_results_new[0]['c_index']:.4f} (95% CI: {c_index_results_new[0]['ci_lower']:.4f} - {c_index_results_new[0]['ci_upper']:.4f}, p-value: {c_index_results_new[0]['p_value']:.4f})",
                   f"{c_index_results_new[0]['c_index'] - c_index_results_old[0]['c_index']:.4f}"])
    
    for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
        report.append([f"IDI ({old_col} vs {new_col})",
                       f"Mean prob: {idi_results[i]['mean_old']:.4f}",
                       f"Mean prob: {idi_results[i]['mean_new']:.4f}",
                       f"{idi_results[i]['idi']:.4f} (95% CI: {idi_results[i]['ci_lower']:.4f} - {idi_results[i]['ci_upper']:.4f}, p-value: {idi_results[i]['p_value']:.4f})"])
    
    for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
        report.append([f"NRI ({old_col} vs {new_col})",
                       f"Events: {nri_results[i]['nri_events']:.4f}",
                       f"Non-events: {nri_results[i]['nri_nonevents']:.4f}",
                       f"{nri_results[i]['nri']:.4f} (95% CI: {nri_results[i]['ci_lower']:.4f} - {nri_results[i]['ci_upper']:.4f}, p-value: {nri_results[i]['p_value']:.4f})"])
    
    return pd.DataFrame(report[1:], columns=report[0])

def main():
    st.title("CSV Analysis App")
    st.write("Upload a CSV file, select columns, and compute advanced statistical analyses")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Preview")
            st.write(df.head())

            st.subheader("Column Selection")
            st.write("Please select the required columns for analysis.")
            
            duration_column = st.selectbox("Select the Duration column", df.columns)
            dead_column = st.selectbox("Select the Dead column", df.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                old_model_columns = st.multiselect("Select the old model columns", df.columns)
                st.write(f"Old model columns selected: {len(old_model_columns)}")
            
            with col2:
                new_model_columns = st.multiselect("Select the new model columns", df.columns)
                st.write(f"New model columns selected: {len(new_model_columns)}")

            if st.button("Compute Metrics and Visualizations"):
                try:
                    y_true = df[[duration_column, dead_column]].rename(columns={duration_column: 'Duration', dead_column: 'Dead'})
                    y_pred_old = df[old_model_columns]
                    y_pred_new = df[new_model_columns]

                    with st.spinner("Computing metrics and generating visualizations..."):
                        c_index_results_old = compute_c_index(y_true, y_pred_old)
                        c_index_results_new = compute_c_index(y_true, y_pred_new)
                        idi_results = compute_idi(y_true['Dead'], y_pred_old, y_pred_new)
                        nri_results = compute_nri(y_true['Dead'], y_pred_old, y_pred_new)

                    st.subheader("Results")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("C-index:")
                        st.write("Old model:")
                        st.write(f"C-index: {c_index_results_old[0]['c_index']:.4f} (95% CI: {c_index_results_old[0]['ci_lower']:.4f} - {c_index_results_old[0]['ci_upper']:.4f}, p-value: {c_index_results_old[0]['p_value']:.4f})")
                        st.write("New model:")
                        st.write(f"C-index: {c_index_results_new[0]['c_index']:.4f} (95% CI: {c_index_results_new[0]['ci_lower']:.4f} - {c_index_results_new[0]['ci_upper']:.4f}, p-value: {c_index_results_new[0]['p_value']:.4f})")
                        st.write(f"Improvement: {c_index_results_new[0]['c_index'] - c_index_results_old[0]['c_index']:.4f}")

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

                    st.subheader("Visualizations")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
                    
                    for i, (old_col, new_col) in enumerate(zip(old_model_columns, new_model_columns)):
                        fpr_old, tpr_old, _ = roc_curve(y_true['Dead'], y_pred_old[old_col])
                        fpr_new, tpr_new, _ = roc_curve(y_true['Dead'], y_pred_new[new_col])
                        
                        ax.plot(fpr_old, tpr_old, label=f'Old Model ({old_col})')
                        ax.plot(fpr_new, tpr_new, label=f'New Model ({new_col})')
                    
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend()
                    st.pyplot(fig)

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

                    st.subheader("Advanced Statistical Analyses")

                    km_fig = plot_kaplan_meier(df, duration_column, dead_column)
                    st.pyplot(km_fig)

                    cox_summary_fig = plot_cox_summary(df, duration_column, dead_column, new_model_columns)
                    st.pyplot(cox_summary_fig)

                    hr_forest_fig = plot_hazard_ratio_forest(df, duration_column, dead_column, new_model_columns)
                    st.pyplot(hr_forest_fig)

                    report_df = generate_report(c_index_results_old, c_index_results_new, idi_results, nri_results, old_model_columns, new_model_columns)
                    
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="Download Analysis Report",
                        data=csv,
                        file_name="analysis_report.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"An error occurred while computing metrics: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred while processing the CSV file: {str(e)}")

if __name__ == "__main__":
    main()