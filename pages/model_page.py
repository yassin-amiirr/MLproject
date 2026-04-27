import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score
)

def create_sample_data():
    data = {
        "year": [2018, 2020, 2017, 2019, 2021, 2016, 2015, 2022, 2018, 2020],
        "km_driven": [50000, 30000, 70000, 40000, 20000, 90000, 100000, 15000, 60000, 35000],
        "engine": [1200, 1500, 1300, 1400, 1600, 1100, 1000, 1800, 1250, 1550],
        "fuel": ["Petrol", "Diesel", "Petrol", "Diesel", "Petrol", "CNG", "Diesel", "Petrol", "CNG", "Diesel"],
        "transmission": ["Manual", "Automatic", "Manual", "Automatic", "Manual", "Manual", "Manual", "Automatic", "Manual", "Automatic"],
        "price": [350000, 500000, 300000, 450000, 600000, 250000, 200000, 700000, 320000, 520000]
    }
    return pd.DataFrame(data)

def encode_categorical_columns(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def add_price_category(df, target_col="price"):
    df_new = df.copy()
    if target_col in df_new.columns:
        if len(df_new[target_col].unique()) > 3:
            df_new[f"{target_col}_category"] = pd.qcut(df_new[target_col], q=3, labels=["Cheap", "Medium", "Expensive"], duplicates='drop')
        else:
            df_new[f"{target_col}_category"] = df_new[target_col]
    return df_new

def get_regression_models():
    return {
        "Linear Regression": LinearRegression()
    }

def get_classification_models():
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Bayesian Classifier": GaussianNB()
    }

def show_model_page():
    st.title("🤖 Model Selection & Training")
    st.markdown("Select a machine learning task, configure features, and train your model seamlessly.")

    # Load Data
    with st.expander("📁 Data Source", expanded=True):
        use_sample = st.checkbox("Use sample car dataset", value=False)
        
        if use_sample:
            df = create_sample_data()
            st.info("Sample data loaded successfully.")
        else:
            if "df" in st.session_state and st.session_state["df"] is not None:
                df = st.session_state["df"]
                st.success(f"Using uploaded dataset: {st.session_state.get('file_name', 'dataset')}")
            else:
                st.warning("No dataset uploaded. Falling back to sample data.")
                df = create_sample_data()
                
        st.dataframe(df.head(), use_container_width=True)

    # Preprocessing for Models
    df_encoded = encode_categorical_columns(df)

    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        model_type = st.radio(
            "Select Task Type",
            ["Regression", "Classification", "Clustering (K-Means)"],
            help="Choose the type of machine learning problem you want to solve."
        )

    with col2:
        if model_type == "Regression":
            run_regression(df, df_encoded)
        elif model_type == "Classification":
            run_classification(df, df_encoded)
        elif model_type == "Clustering (K-Means)":
            run_clustering(df_encoded)

def run_regression(df, df_encoded):
    st.subheader("📈 Regression Task")
    
    models = get_regression_models()
    selected_model_name = st.selectbox("Choose Model", list(models.keys()))
    
    col_feat, col_targ = st.columns(2)
    with col_targ:
        target_col = st.selectbox("Select Target Variable", df_encoded.columns, index=len(df_encoded.columns) - 1)
    with col_feat:
        feature_cols = st.multiselect(
            "Select Features",
            [col for col in df_encoded.columns if col != target_col],
            default=[col for col in df_encoded.columns if col != target_col]
        )

    test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100.0

    if st.button("Train Regression Model", type="primary"):
        if not feature_cols:
            st.error("Please select at least one feature.")
            return

        X = df_encoded[feature_cols]
        y = df_encoded[target_col]

        if X.isnull().values.any() or y.isnull().values.any():
            st.error("Data contains missing values. Please preprocess data first.")
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = models[selected_model_name]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success(f"{selected_model_name} trained successfully! Please go to the Evaluation page to see the results.")
            
            st.session_state["evaluation_results"] = {
                "task": "Regression",
                "model_name": selected_model_name,
                "metrics": {
                    "R² Score": r2,
                    "RMSE": rmse,
                    "MSE": mse,
                    "MAE": mae
                }
            }
        except Exception as e:
            st.error(f"Model training failed: {e}")

def run_classification(df, df_encoded):
    st.subheader("🎯 Classification Task")

    models = get_classification_models()
    selected_model_name = st.selectbox("Choose Model", list(models.keys()))

    target_options = list(df.columns)
    
    col_feat, col_targ = st.columns(2)
    with col_targ:
        target_col_raw = st.selectbox("Select Target Variable", target_options, index=len(target_options) - 1)
        
    is_continuous = pd.api.types.is_numeric_dtype(df[target_col_raw]) and len(df[target_col_raw].unique()) > 10
    
    df_clf = df_encoded.copy()
    target_col = target_col_raw
    
    if is_continuous:
        st.warning(f"'{target_col_raw}' seems continuous. Converting to categories (Cheap/Medium/Expensive) for classification.")
        df_temp = add_price_category(df, target_col_raw)
        cat_col_name = f"{target_col_raw}_category"
        le = LabelEncoder()
        df_clf[cat_col_name] = le.fit_transform(df_temp[cat_col_name].astype(str))
        target_col = cat_col_name

    with col_feat:
        feature_options = [col for col in df_clf.columns if col != target_col and col != target_col_raw]
        feature_cols = st.multiselect("Select Features", feature_options, default=feature_options[:3] if len(feature_options) >= 3 else feature_options)

    test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100.0

    if st.button("Train Classification Model", type="primary"):
        if not feature_cols:
            st.error("Please select at least one feature.")
            return

        X = df_clf[feature_cols]
        y = df_clf[target_col]

        if X.isnull().values.any() or y.isnull().values.any():
            st.error("Data contains missing values. Please preprocess data first.")
            return

        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if selected_model_name in ["Logistic Regression", "SVM", "K-Nearest Neighbors"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = models[selected_model_name]
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            st.success(f"{selected_model_name} trained successfully! Please go to the Evaluation page to see the results.")
            
            st.session_state["evaluation_results"] = {
                "task": "Classification",
                "model_name": selected_model_name,
                "metrics": {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1 Score": f1
                }
            }
        except Exception as e:
            st.error(f"Model training failed: {e}")

def run_clustering(df_encoded):
    st.subheader("🧩 K-Means Clustering")

    numeric_cols = list(df_encoded.select_dtypes(include=[np.number]).columns)
    
    col_feat, col_params = st.columns(2)
    with col_feat:
        feature_cols = st.multiselect("Select Features for Clustering", numeric_cols, default=numeric_cols[:3] if len(numeric_cols)>=3 else numeric_cols)
    with col_params:
        n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)

    if st.button("Run K-Means", type="primary"):
        if len(feature_cols) < 2:
            st.error("Please select at least 2 features for meaningful clustering.")
            return

        X = df_encoded[feature_cols]

        if X.isnull().values.any():
            st.error("Data contains missing values. Please preprocess data first.")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = model.fit_predict(X_scaled)

            sil_score = silhouette_score(X_scaled, clusters)

            st.success("K-Means clustering completed! Please go to the Evaluation page to see the results.")
            
            df_preview = df_encoded.copy()
            df_preview["Cluster"] = clusters
            st.success("K-Means ran successfully.")
            st.write(f"**Silhouette Score:** {sil_score:.4f}")
            st.subheader("Clustered Data Preview")
            st.dataframe(df_preview.head())
            st.session_state["evaluation_results"] = {
            "task": "Clustering",
            "model_name": f"K-Means (K={n_clusters})",
            "metrics": {
                "Silhouette Score": sil_score
            },
            "clustered_data": df_preview
            }
        except Exception as e:
            st.error(f"K-Means clustering failed: {e}")
        