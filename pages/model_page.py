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
    accuracy_score, precision_score, recall_score, f1_score
)


def encode_categorical_columns(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


def add_price_category(df, target_col="price"):
    df_new = df.copy()
    if target_col in df_new.columns:
        df_new[f"{target_col}_category"] = pd.qcut(
            df_new[target_col],
            q=3,
            labels=["Cheap", "Medium", "Expensive"],
            duplicates="drop"
        )
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
    st.title("🤖 Model Selection")

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("Please upload a dataset first from the File Upload page.")
        return

    df = st.session_state["df"]

    st.success(f"Using uploaded dataset: {st.session_state.get('file_name', 'dataset')}")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    df_encoded = encode_categorical_columns(df)

    model_type = st.radio(
        "Select Task Type",
        ["Regression", "Classification", "Clustering (K-Means)"]
    )

    if model_type == "Regression":
        run_regression(df_encoded)

    elif model_type == "Classification":
        run_classification(df, df_encoded)

    elif model_type == "Clustering (K-Means)":
        run_clustering(df_encoded)


def run_regression(df_encoded):
    st.subheader("📈 Regression")

    models = get_regression_models()
    selected_model_name = st.selectbox("Choose Model", list(models.keys()))

    col1, col2 = st.columns(2)

    with col2:
        target_col = st.selectbox("Select Target Variable", df_encoded.columns)

    with col1:
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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = models[selected_model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if "executed_models" not in st.session_state:
            st.session_state["executed_models"] = {}

        st.session_state["executed_models"][selected_model_name] = {
            "type": "Regression",
            "y_true": y_test,
            "y_pred": y_pred
        }

        st.success(f"{selected_model_name} trained successfully. Go to Evaluation page.")


def run_classification(df, df_encoded):
    st.subheader("🎯 Classification")

    models = get_classification_models()
    selected_model_name = st.selectbox("Choose Model", list(models.keys()))

    target_options = list(df.columns)

    col1, col2 = st.columns(2)

    with col2:
        target_col_raw = st.selectbox("Select Target Variable", target_options)

    is_continuous = (
        pd.api.types.is_numeric_dtype(df[target_col_raw])
        and len(df[target_col_raw].unique()) > 10
    )

    df_clf = df_encoded.copy()
    target_col = target_col_raw

    if is_continuous:
        st.warning(f"'{target_col_raw}' is continuous, so it will be converted to categories.")
        df_temp = add_price_category(df, target_col_raw)
        cat_col_name = f"{target_col_raw}_category"
        le = LabelEncoder()
        df_clf[cat_col_name] = le.fit_transform(df_temp[cat_col_name].astype(str))
        target_col = cat_col_name

    with col1:
        feature_options = [col for col in df_clf.columns if col != target_col and col != target_col_raw]
        feature_cols = st.multiselect(
            "Select Features",
            feature_options,
            default=feature_options[:3] if len(feature_options) >= 3 else feature_options
        )

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if selected_model_name in ["Logistic Regression", "SVM", "K-Nearest Neighbors"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = models[selected_model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if "executed_models" not in st.session_state:
            st.session_state["executed_models"] = {}

        st.session_state["executed_models"][selected_model_name] = {
            "type": "Classification",
            "y_true": y_test,
            "y_pred": y_pred
        }

        st.success(f"{selected_model_name} trained successfully. Go to Evaluation page.")


def run_clustering(df_encoded):
    st.subheader("🧩 K-Means Clustering")

    numeric_cols = list(df_encoded.select_dtypes(include=[np.number]).columns)

    col1, col2 = st.columns(2)

    with col1:
        feature_cols = st.multiselect(
            "Select Features for Clustering",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )

    with col2:
        n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)

    if st.button("Run K-Means", type="primary"):
        if len(feature_cols) < 2:
            st.error("Please select at least 2 features.")
            return

        X = df_encoded[feature_cols]

        if X.isnull().values.any():
            st.error("Data contains missing values. Please preprocess data first.")
            return

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X_scaled)

        if "executed_models" not in st.session_state:
            st.session_state["executed_models"] = {}

        st.session_state["executed_models"]["K-Means"] = {
            "type": "K-Means",
            "X": X_scaled,
            "labels": clusters
        }

        st.success("K-Means completed successfully. Go to Evaluation page.")
