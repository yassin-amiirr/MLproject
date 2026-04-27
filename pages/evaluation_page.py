import streamlit as st

def show_evaluation_page():
    st.title("📊 Model Evaluation")
    
    if "evaluation_results" not in st.session_state or st.session_state["evaluation_results"] is None:
        st.info("No model has been evaluated yet. Please go to the 'Model Selection' page to train a model first.")
        return

    results = st.session_state["evaluation_results"]
    
    task = results.get("task", "Unknown Task")
    model_name = results.get("model_name", "Unknown Model")
    metrics = results.get("metrics", {})
    
    st.markdown(f"### 🎯 Results for **{model_name}** ({task})")
    st.markdown("---")
    
    # Display metrics dynamically in columns
    if metrics:
        cols = st.columns(len(metrics))
        for i, (metric_name, value) in enumerate(metrics.items()):
            cols[i].metric(metric_name, f"{value:.4f}")
            
    # Display clustered data preview if task is Clustering
    if task == "Clustering" and "clustered_data" in results:
        st.markdown("---")
        st.subheader("🧩 Clustered Data Preview")
        st.dataframe(results["clustered_data"], use_container_width=True)