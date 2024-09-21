import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="IDI/NRI - CSV Analysis App", layout="wide")

st.title("IDI and NRI Computation")

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
    idi = np.mean(y_pred_new - y_pred_old)
    
    bootstrap_idis = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_resampled = y_true.iloc[indices]
        y_pred_old_resampled = y_pred_old.iloc[indices]
        y_pred_new_resampled = y_pred_new.iloc[indices]
        bootstrap_idis.append(np.mean(y_pred_new_resampled - y_pred_old_resampled))
    
    ci_lower = np.percentile(bootstrap_idis, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_idis, (1 - alpha/2) * 100)
    
    t_statistic = idi / (np.std(bootstrap_idis) / np.sqrt(n_bootstrap))
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n_bootstrap-1))
    
    return idi, ci_lower, ci_upper, p_value

def compute_nri(y_true, y_pred_old, y_pred_new, threshold=0.5, n_bootstrap=1000, alpha=0.05):
    events_improved = np.sum((y_pred_new > threshold) & (y_pred_old <= threshold) & (y_true == 1))
    events_worsened = np.sum((y_pred_new <= threshold) & (y_pred_old > threshold) & (y_true == 1))
    nonevents_improved = np.sum((y_pred_new <= threshold) & (y_pred_old > threshold) & (y_true == 0))
    nonevents_worsened = np.sum((y_pred_new > threshold) & (y_pred_old <= threshold) & (y_true == 0))
    
    nri_events = (events_improved - events_worsened) / np.sum(y_true == 1)
    nri_nonevents = (nonevents_improved - nonevents_worsened) / np.sum(y_true == 0)
    nri = nri_events + nri_nonevents
    
    bootstrap_nris = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_resampled = y_true.iloc[indices]
        y_pred_old_resampled = y_pred_old.iloc[indices]
        y_pred_new_resampled = y_pred_new.iloc[indices]
        
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
    
    return nri, nri_events, nri_nonevents, ci_lower, ci_upper, p_value

if 'duration_column' not in st.session_state or 'dead_column' not in st.session_state or 'old_model_columns' not in st.session_state or 'new_model_columns' not in st.session_state:
    st.warning("Please select columns on the Home page before proceeding with IDI/NRI computation.")
else:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            y_true = df[st.session_state.dead_column]
            y_pred_old = df[st.session_state.old_model_columns]
            y_pred_new = df[st.session_state.new_model_columns]
            
            if st.button("Compute IDI and NRI"):
                with st.spinner("Computing IDI and NRI..."):
                    idi_results = []
                    nri_results = []
                    
                    for old_col, new_col in zip(st.session_state.old_model_columns, st.session_state.new_model_columns):
                        idi, idi_ci_lower, idi_ci_upper, idi_p_value = compute_idi(y_true, y_pred_old[old_col], y_pred_new[new_col])
                        nri, nri_events, nri_nonevents, nri_ci_lower, nri_ci_upper, nri_p_value = compute_nri(y_true, y_pred_old[old_col], y_pred_new[new_col])
                        
                        idi_results.append({
                            'old_model': old_col,
                            'new_model': new_col,
                            'idi': idi,
                            'ci_lower': idi_ci_lower,
                            'ci_upper': idi_ci_upper,
                            'p_value': idi_p_value
                        })
                        
                        nri_results.append({
                            'old_model': old_col,
                            'new_model': new_col,
                            'nri': nri,
                            'nri_events': nri_events,
                            'nri_nonevents': nri_nonevents,
                            'ci_lower': nri_ci_lower,
                            'ci_upper': nri_ci_upper,
                            'p_value': nri_p_value
                        })
                
                st.subheader("IDI Results")
                for result in idi_results:
                    st.write(f"{result['old_model']} vs {result['new_model']}:")
                    st.write(f"IDI: {result['idi']:.4f} (95% CI: {result['ci_lower']:.4f} - {result['ci_upper']:.4f})")
                    st.write(f"P-value: {result['p_value']:.4f}")
                    st.write("")
                
                st.markdown("""
                ### Interpretation of IDI Results:
                - Positive IDI indicates that the new model improves risk prediction compared to the old model.
                - The magnitude of IDI represents the degree of improvement.
                - The 95% Confidence Interval (CI) indicates the range where the true IDI likely lies.
                - A p-value < 0.05 suggests that the improvement is statistically significant.
                """)
                
                st.subheader("NRI Results")
                for result in nri_results:
                    st.write(f"{result['old_model']} vs {result['new_model']}:")
                    st.write(f"NRI: {result['nri']:.4f} (95% CI: {result['ci_lower']:.4f} - {result['ci_upper']:.4f})")
                    st.write(f"NRI for events: {result['nri_events']:.4f}")
                    st.write(f"NRI for non-events: {result['nri_nonevents']:.4f}")
                    st.write(f"P-value: {result['p_value']:.4f}")
                    st.write("")
                
                st.markdown("""
                ### Interpretation of NRI Results:
                - Positive NRI indicates that the new model improves risk classification compared to the old model.
                - NRI for events shows the net improvement in classifying individuals who experience the event.
                - NRI for non-events shows the net improvement in classifying individuals who do not experience the event.
                - The total NRI is the sum of NRI for events and non-events.
                - The 95% Confidence Interval (CI) indicates the range where the true NRI likely lies.
                - A p-value < 0.05 suggests that the improvement in classification is statistically significant.
                """)
                
                # Export results
                idi_df = pd.DataFrame(idi_results)
                nri_df = pd.DataFrame(nri_results)
                
                results_df = pd.concat([idi_df, nri_df], axis=1, keys=['IDI', 'NRI'])
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download IDI/NRI Results",
                    data=csv,
                    file_name="idi_nri_results.csv",
                    mime="text/csv",
                )
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # IDI plot
                ax1.bar(range(len(idi_results)), [r['idi'] for r in idi_results], yerr=[[r['idi']-r['ci_lower'] for r in idi_results], [r['ci_upper']-r['idi'] for r in idi_results]], capsize=5)
                ax1.set_xlabel('Model Comparison')
                ax1.set_ylabel('IDI')
                ax1.set_title('Integrated Discrimination Improvement (IDI)')
                ax1.set_xticks(range(len(idi_results)))
                ax1.set_xticklabels([f"{r['old_model']} vs {r['new_model']}" for r in idi_results], rotation=45, ha='right')
                
                # NRI plot
                ax2.bar(range(len(nri_results)), [r['nri'] for r in nri_results], yerr=[[r['nri']-r['ci_lower'] for r in nri_results], [r['ci_upper']-r['nri'] for r in nri_results]], capsize=5)
                ax2.set_xlabel('Model Comparison')
                ax2.set_ylabel('NRI')
                ax2.set_title('Net Reclassification Improvement (NRI)')
                ax2.set_xticks(range(len(nri_results)))
                ax2.set_xticklabels([f"{r['old_model']} vs {r['new_model']}" for r in nri_results], rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Export plot
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png')
                img_buffer.seek(0)
                st.download_button(
                    label="Download IDI/NRI Plot",
                    data=img_buffer,
                    file_name="idi_nri_plot.png",
                    mime="image/png",
                )
                
        except Exception as e:
            st.error(f"An error occurred while computing IDI and NRI: {str(e)}")
    else:
        st.info("Please upload a CSV file to compute IDI and NRI.")
