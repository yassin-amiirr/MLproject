# ********************* K-Means Clustering *********************
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_kmeans_model(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    centers = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=250, marker='X', label='Centroids')
    
    ax.set_title(f"K-Means Clustering Result (K={n_clusters})")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    return labels, kmeans, fig

import streamlit as st

st.title("Machine Learning Models")

if 'executed_models' not in st.session_state:
    st.session_state['executed_models'] = {}

model_choice = st.selectbox("Select a Model", ["K-Means Clustering", "Other..."])

if model_choice == "K-Means Clustering":
    st.subheader("K-Means Configuration")
    
    # Slider for user to choose K
    k = st.slider("Select Number of Clusters (K)", min_value=2, max_value=20, value=3)
    
    if st.button("Run K-Means Model"):
        # Check if data exists in session state (uploaded data)
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = st.session_state['df']
            
            # Select only numeric columns and drop NaNs for KMeans
            numeric_df = df.select_dtypes(include=['number']).dropna()
            
            if numeric_df.empty or numeric_df.shape[1] < 2:
                st.error("The dataset must contain at least two numeric columns without missing values to run K-Means and PCA.")
            else:
                X = numeric_df.values
                
                with st.spinner("Running K-Means..."):
                    labels, kmeans_model, fig = run_kmeans_model(X, n_clusters=k)
                    
                    # Store model details for evaluation page
                    model_id = f"K-Means (K={k})"
                    st.session_state['executed_models'][model_id] = {
                        'type': 'K-Means',
                        'model': kmeans_model,
                        'X': X,
                        'labels': labels,
                        'k': k
                    }
                    
                st.success("Model executed successfully!")
                st.pyplot(fig)
        else:
            st.error("No dataset found! Please go to the 'File Upload' page and upload your data first.")
