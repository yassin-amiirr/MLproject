import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error, r2_score,
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
    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def add_price_category(df, price_col="price"):
    df = df.copy()
    df["price_category"] = pd.qcut(df[price_col], q=3, labels=["Cheap", "Medium", "Expensive"])
    return df


def show_model_page():
    st.title("Model Selection")
    st.markdown("### Choose a suitable ML algorithm manually")

    use_sample = st.checkbox("Use sample data for testing", value=True)

    if use_sample:
        df = create_sample_data()
        st.success("Sample data loaded successfully.")
    else:
        if "df" in st.session_state:
            df = st.session_state["df"]
            st.success("Using uploaded dataset.")
        else:
            st.warning("No uploaded dataset found, so sample data is being used.")
            df = create_sample_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    model_type = st.selectbox(
        "Select Model Type",
        ["Regression", "Classification", "Clustering (K-Means)"]
    )

    if model_type == "Regression":
        st.subheader("Regression")
        selected_model = st.selectbox("Choose Model", ["Linear Regression"])

        df_encoded = encode_categorical_columns(df)

        target_col = st.selectbox("Select Target Column", df_encoded.columns, index=len(df_encoded.columns) - 1)
        feature_cols = st.multiselect(
            "Select Feature Columns",
            [col for col in df_encoded.columns if col != target_col],
            default=[col for col in df_encoded.columns if col != target_col]
        )

        if st.button("Run Regression Model"):
            if not feature_cols:
                st.error("Please select at least one feature column.")
            else:
                X = df_encoded[feature_cols]
                y = df_encoded[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                st.success("Regression model ran successfully.")
                st.write(f"**MSE:** {mse:.4f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**R² Score:** {r2:.4f}")

    elif model_type == "Classification":
        st.subheader("Classification")

        selected_model = st.selectbox(
            "Choose Model",
            [
                "Decision Tree",
                "Logistic Regression",
                "SVM",
                "Random Forest",
                "K-Nearest Neighbors",
                "Bayesian Classifier"
            ]
        )

        if "price" in df.columns:
            df = add_price_category(df, "price")
            st.info("price_category was created automatically from price.")

        df_encoded = encode_categorical_columns(df)

        target_options = list(df_encoded.columns)
        default_index = target_options.index("price_category") if "price_category" in target_options else len(target_options) - 1

        target_col = st.selectbox("Select Target Column", target_options, index=default_index)
        feature_cols = st.multiselect(
            "Select Feature Columns",
            [col for col in df_encoded.columns if col != target_col],
            default=[col for col in df_encoded.columns if col != target_col][:3]
        )

        if st.button("Run Classification Model"):
            if not feature_cols:
                st.error("Please select at least one feature column.")
            else:
                X = df_encoded[feature_cols]
                y = df_encoded[target_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                if selected_model == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif selected_model == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif selected_model == "SVM":
                    model = SVC()
                elif selected_model == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif selected_model == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()
                else:
                    model = GaussianNB()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                st.success("Classification model ran successfully.")
                st.write(f"**Accuracy:** {accuracy:.4f}")
                st.write(f"**Precision:** {precision:.4f}")
                st.write(f"**Recall:** {recall:.4f}")
                st.write(f"**F1 Score:** {f1:.4f}")

    elif model_type == "Clustering (K-Means)":
        st.subheader("K-Means Clustering")

        df_encoded = encode_categorical_columns(df)
        numeric_cols = list(df_encoded.select_dtypes(include=[np.number]).columns)

        feature_cols = st.multiselect(
            "Select Columns for Clustering",
            numeric_cols,
            default=numeric_cols[:3]
        )

        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        if st.button("Run K-Means"):
            if len(feature_cols) < 2:
                st.error("Please select at least 2 columns.")
            else:
                X = df_encoded[feature_cols]

                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = model.fit_predict(X)

                sil_score = silhouette_score(X, clusters)

                df_result = df.copy()
                df_result["Cluster"] = clusters

                st.success("K-Means ran successfully.")
                st.write(f"**Silhouette Score:** {sil_score:.4f}")
                st.subheader("Clustered Data Preview")
                st.dataframe(df_result.head())