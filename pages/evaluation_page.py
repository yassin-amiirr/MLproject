import streamlit as st
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix,
    r2_score, mean_squared_error,
    silhouette_score, davies_bouldin_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def show_evaluation_page():
    st.title("📊 Model Evaluation Dashboard")

    if 'executed_models' not in st.session_state or not st.session_state['executed_models']:
        st.info("No models have been executed yet. Please go to the 'Model Selection' page and run a model first.")
        return

    executed_models = st.session_state['executed_models']
    
    # 1. Model Selection
    model_names = list(executed_models.keys())
    selected_model_name = st.selectbox("Select a Model to Evaluate", model_names)
    
    model_data = executed_models[selected_model_name]
    model_type = model_data['type']
    
    st.subheader(f"Evaluating: {selected_model_name} ({model_type})")
    
    # 2. Dynamic Metric Selection based on Model Type
    if model_type == 'Classification':
        metric_options = ['Accuracy', 'Precision', 'Recall', 'Confusion Matrix']
        selected_metric = st.selectbox("Select Metric to Display", metric_options)
        
        y_true = model_data.get('y_true')
        y_pred = model_data.get('y_pred')
        
        if st.button("Calculate Metric"):
            if y_true is None or y_pred is None:
                st.error("Missing true labels or predicted labels for this model.")
            else:
                if selected_metric == 'Accuracy':
                    st.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
                elif selected_metric == 'Precision':
                    st.metric("Precision (Macro)", round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4))
                elif selected_metric == 'Recall':
                    st.metric("Recall (Macro)", round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4))
                elif selected_metric == 'Confusion Matrix':
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)
                
    elif model_type == 'Regression':
        metric_options = ['R-squared', 'Mean Squared Error (MSE)']
        selected_metric = st.selectbox("Select Metric to Display", metric_options)
        
        y_true = model_data.get('y_true')
        y_pred = model_data.get('y_pred')
        
        if st.button("Calculate Metric"):
            if y_true is None or y_pred is None:
                st.error("Missing true values or predicted values for this model.")
            else:
                if selected_metric == 'R-squared':
                    st.metric("R-squared", round(r2_score(y_true, y_pred), 4))
                elif selected_metric == 'Mean Squared Error (MSE)':
                    st.metric("MSE", round(mean_squared_error(y_true, y_pred), 4))
                
    elif model_type == 'K-Means':
        metric_options = ['Silhouette Score', 'Davies-Bouldin Index']
        selected_metric = st.selectbox("Select Metric to Display", metric_options)
        
        X = model_data.get('X')
        labels = model_data.get('labels')
        
        if st.button("Calculate Metric"):
            if X is None or labels is None:
                st.error("Missing features (X) or labels for this model.")
            elif len(set(labels)) < 2:
                st.warning("Cannot calculate metrics for less than 2 clusters.")
            else:
                if selected_metric == 'Silhouette Score':
                    score = silhouette_score(X, labels)
                    st.metric("Silhouette Score", round(score, 4))
                    st.caption("Values range from -1 to 1. Higher values indicate better defined clusters.")
                elif selected_metric == 'Davies-Bouldin Index':
                    score = davies_bouldin_score(X, labels)
                    st.metric("Davies-Bouldin Index", round(score, 4))
                    st.caption("Lower values indicate better clustering (less dispersion, more separation).")

    st.markdown("---")
    
    # 3. Comparison Table
    st.subheader("Model Performance Comparison")
    if st.checkbox("Show Comparison Table"):
        # We group by model type to create meaningful comparisons
        types_present = list(set([data['type'] for data in executed_models.values()]))
        
        for t in types_present:
            st.write(f"**{t} Models**")
            comparison_data = []
            
            for name, data in executed_models.items():
                if data['type'] == t:
                    row = {'Model': name}
                    if t == 'Classification':
                        y_t, y_p = data.get('y_true'), data.get('y_pred')
                        if y_t is not None and y_p is not None:
                            row['Accuracy'] = round(accuracy_score(y_t, y_p), 4)
                            row['Precision'] = round(precision_score(y_t, y_p, average='macro', zero_division=0), 4)
                            row['Recall'] = round(recall_score(y_t, y_p, average='macro', zero_division=0), 4)
                    elif t == 'Regression':
                        y_t, y_p = data.get('y_true'), data.get('y_pred')
                        if y_t is not None and y_p is not None:
                            row['R-squared'] = round(r2_score(y_t, y_p), 4)
                            row['MSE'] = round(mean_squared_error(y_t, y_p), 4)
                    elif t == 'K-Means':
                        X_data, lbls = data.get('X'), data.get('labels')
                        if X_data is not None and lbls is not None and len(set(lbls)) > 1:
                            row['Silhouette Score'] = round(silhouette_score(X_data, lbls), 4)
                            row['Davies-Bouldin Index'] = round(davies_bouldin_score(X_data, lbls), 4)
                    comparison_data.append(row)
            
            if comparison_data:
                df_comp = pd.DataFrame(comparison_data)
                st.dataframe(df_comp, use_container_width=True)
            else:
                st.info(f"No valid models to compare for {t}.")
