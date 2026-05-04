import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import numpy as np
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import zscore


# ── Imputation helpers ────────────────────────────────────────────────────────

def simple_imputation(df, numeric_cols, categorical_cols):
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])


def KNN_imputation(df, numeric_cols, categorical_cols):
    from sklearn.impute import KNNImputer

    encoders = {}
    for col in categorical_cols:
        codes, unique_vals = pd.factorize(df[col], sort=True)
        encoders[col] = unique_vals
        df[col] = pd.Series(codes, index=df.index)

    imputer = KNNImputer(n_neighbors=5)
    cols_to_impute = numeric_cols + categorical_cols
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    for col in categorical_cols:
        df[col] = np.clip(df[col], 0, len(encoders[col]) - 1)
        df[col] = df[col].round().astype(int)
        df[col] = encoders[col][df[col]]


def iterative_imputation(df, numeric_cols, categorical_cols):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    encoders = {}
    for col in categorical_cols:
        codes, unique_vals = pd.factorize(df[col], sort=True)
        encoders[col] = unique_vals
        df[col] = pd.Series(codes, index=df.index)

    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        verbose=0
    )

    cols_to_impute = numeric_cols + categorical_cols
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    for col in categorical_cols:
        df[col] = np.clip(df[col], 0, len(encoders[col]) - 1)
        df[col] = df[col].round().astype(int)
        df[col] = encoders[col][df[col]]


# ── Outlier helpers ───────────────────────────────────────────────────────────

def IQR_method(df, col):
    """Return the indices of outliers in a numeric column using the IQR method."""
    data = df[col].dropna()
    if data.empty:
        return data.index
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)].index


def z_score_method(df, col):
    """Return the indices of outliers in a numeric column using z-score."""
    data = df[col].dropna()
    if data.empty:
        return data.index
    z_scores = zscore(data)
    outlier_indices = data[(z_scores < -3) | (z_scores > 3)].index
    return outlier_indices


def winsorization(df, col):
    """Replace extreme values with IQR-based caps."""
    data = df[col].dropna()
    if data.empty:
        return
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


def clipping(df, col):
    """Clip values to the 1st and 99th percentile range."""
    data = df[col].dropna()
    if data.empty:
        return
    lower_bound = data.quantile(0.01)
    upper_bound = data.quantile(0.99)
    df[col] = np.clip(df[col], lower_bound, upper_bound)


# ── Main page ─────────────────────────────────────────────────────────────────

def show_preprocessing_page():
    st.title("⚙️ 3- Data Preprocessing")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    df = st.session_state.df.copy()

    st.subheader("🔍 Current Data")
    st.dataframe(df.head(), use_container_width=True)

    # ================= MISSING VALUES & IMPUTATION =================
    st.markdown("---")
    st.subheader("🩹 Missing Values & Imputation")

    total_missing = st.session_state.df.isnull().sum().sum()
    if total_missing > 0:
        st.write(f"**Total missing values detected:** {total_missing}")
        missing_summary = st.session_state.df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].rename("Missing Count")
        st.dataframe(missing_summary.to_frame(), use_container_width=True)
    else:
        st.info("✅ No missing values found in the dataset.")

    imputation_method = st.selectbox(
        "Choose Imputation Method:",
        ["None", "Simple Imputation (Median / Mode)", "KNN Imputation", "Iterative Imputation (MICE)"],
        key="imputation_method"
    )

    if st.button("Apply Imputation"):
        if imputation_method == "None":
            st.warning("No imputation method selected.")
        else:
            df_imp = st.session_state.df.copy()
            imp_numeric_cols = df_imp.select_dtypes(include=["int64", "float64"]).columns.tolist()
            imp_categorical_cols = df_imp.select_dtypes(include=["object"]).columns.tolist()

            if imputation_method == "Simple Imputation (Median / Mode)":
                simple_imputation(df_imp, imp_numeric_cols, imp_categorical_cols)
                st.success("✅ Simple Imputation Applied (Median for numeric, Mode for categorical)")

            elif imputation_method == "KNN Imputation":
                KNN_imputation(df_imp, imp_numeric_cols, imp_categorical_cols)
                st.success("✅ KNN Imputation Applied")

            elif imputation_method == "Iterative Imputation (MICE)":
                iterative_imputation(df_imp, imp_numeric_cols, imp_categorical_cols)
                st.success("✅ Iterative (MICE) Imputation Applied")

            remaining = df_imp.isnull().sum().sum()
            st.write(f"**Missing values after imputation:** {remaining}")
            st.session_state.df = df_imp

    # ================= OUTLIER DETECTION & HANDLING =================
    st.markdown("---")
    st.subheader("🚨 Outlier Detection & Handling")

    outlier_numeric_cols = st.session_state.df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(outlier_numeric_cols) > 0:
        selected_outlier_cols = st.multiselect(
            "Select numeric columns for outlier analysis:",
            outlier_numeric_cols,
            key="outlier_cols"
        )

        detection_method = st.selectbox(
            "Choose Detection Method:",
            ["None", "IQR Method", "Z-Score Method"],
            key="outlier_detection"
        )

        if st.button("Detect Outliers") and selected_outlier_cols:
            if detection_method == "None":
                st.warning("No detection method selected.")
            else:
                df_detect = st.session_state.df.copy()
                outlier_summary = {}
                for col in selected_outlier_cols:
                    if detection_method == "IQR Method":
                        idx = IQR_method(df_detect, col)
                    else:
                        idx = z_score_method(df_detect, col)
                    outlier_summary[col] = len(idx)
                summary_df = pd.DataFrame.from_dict(
                    outlier_summary, orient="index", columns=["Outlier Count"]
                )
                st.write("**Outliers detected per column:**")
                st.dataframe(summary_df, use_container_width=True)

        st.markdown("**Outlier Treatment:**")
        treatment_method = st.selectbox(
            "Choose Treatment Method:",
            ["None", "Remove Outliers (IQR)", "Remove Outliers (Z-Score)", "Winsorization", "Clipping"],
            key="outlier_treatment"
        )

        if st.button("Apply Outlier Treatment") and selected_outlier_cols:
            if treatment_method == "None":
                st.warning("No treatment method selected.")
            else:
                df_out = st.session_state.df.copy()

                if treatment_method == "Remove Outliers (IQR)":
                    all_outlier_idx = set()
                    for col in selected_outlier_cols:
                        all_outlier_idx.update(IQR_method(df_out, col))
                    df_out = df_out.drop(index=list(all_outlier_idx)).reset_index(drop=True)
                    st.success(f"✅ Removed {len(all_outlier_idx)} outlier row(s) using IQR Method")

                elif treatment_method == "Remove Outliers (Z-Score)":
                    all_outlier_idx = set()
                    for col in selected_outlier_cols:
                        all_outlier_idx.update(z_score_method(df_out, col))
                    df_out = df_out.drop(index=list(all_outlier_idx)).reset_index(drop=True)
                    st.success(f"✅ Removed {len(all_outlier_idx)} outlier row(s) using Z-Score Method")

                elif treatment_method == "Winsorization":
                    for col in selected_outlier_cols:
                        winsorization(df_out, col)
                    st.success("✅ Winsorization Applied (IQR-based caps)")

                elif treatment_method == "Clipping":
                    for col in selected_outlier_cols:
                        clipping(df_out, col)
                    st.success("✅ Clipping Applied (1st–99th percentile)")

                st.session_state.df = df_out

    else:
        st.info("No numeric columns found for outlier analysis.")

    # ================= ENCODING =================
    st.markdown("---")
    st.subheader("🔤 Data Encoding")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(categorical_cols) > 0:
        selected_cols = st.multiselect(
            "Select categorical columns:",
            categorical_cols
        )

        encoding_method = st.selectbox(
            "Choose Encoding Method:",
            ["None", "Label Encoding", "One-Hot Encoding"]
        )

        if st.button("Apply Encoding"):
            if encoding_method == "Label Encoding":
                le = LabelEncoder()
                for col in selected_cols:
                    df[col] = le.fit_transform(df[col])
                st.success("✅ Label Encoding Applied")

            elif encoding_method == "One-Hot Encoding":
                df = pd.get_dummies(df, columns=selected_cols)
                st.success("✅ One-Hot Encoding Applied")

            else:
                st.warning("No encoding selected")

            st.session_state.df = df

    else:
        st.info("No categorical columns found.")

    # ================= SCALING =================
    st.markdown("---")
    st.subheader("📏 Data Scaling")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) > 0:
        selected_num_cols = st.multiselect(
            "Select numeric columns:",
            numeric_cols
        )

        scaling_method = st.selectbox(
            "Choose Scaling Method:",
            ["None", "Standard Scaler", "MinMax Scaler"]
        )

        if st.button("Apply Scaling"):
            if scaling_method == "Standard Scaler":
                scaler = StandardScaler()
                df[selected_num_cols] = scaler.fit_transform(df[selected_num_cols])
                st.success("✅ Standard Scaling Applied")

            elif scaling_method == "MinMax Scaler":
                scaler = MinMaxScaler()
                df[selected_num_cols] = scaler.fit_transform(df[selected_num_cols])
                st.success("✅ MinMax Scaling Applied")

            else:
                st.warning("No scaling selected")

            st.session_state.df = df

    else:
        st.info("No numeric columns found.")

    # ================= FEATURE TRANSFORMATION =================
    st.markdown("---")
    st.subheader("🔀 Feature Transformation")


    numeric_cols_ft = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols_ft) > 0:
        selected_ft_cols = st.multiselect(
            "Select numeric columns for transformation:",
            numeric_cols_ft,
            key="ft_cols"
        )

        transformation_method = st.selectbox(
            "Choose Transformation Method:",
            ["None", "Log Transformation", "Box-Cox Transformation", "Power Transformation", "Polynomial Features"],
            key="ft_method"
        )

        if st.button("Apply Feature Transformation"):
            df_ft = st.session_state.df.copy()

            if transformation_method == "Log Transformation":
                for col in selected_ft_cols:
                    if (df_ft[col] > 0).all():
                        df_ft[col] = np.log(df_ft[col])
                    else:
                        st.warning(f"⚠️ Column '{col}' has non-positive values. Skipping Log Transformation for it.")
                st.success("✅ Log Transformation Applied")

            elif transformation_method == "Box-Cox Transformation":
                pt = PowerTransformer(method='box-cox')
                valid_cols = [col for col in selected_ft_cols if (df_ft[col] > 0).all()]
                if valid_cols:
                    df_ft[valid_cols] = pt.fit_transform(df_ft[valid_cols])
                    st.success("✅ Box-Cox Transformation Applied")
                else:
                    st.warning("⚠️ Box-Cox requires all positive values. No valid columns found.")

            elif transformation_method == "Power Transformation":
                pt = PowerTransformer(method='yeo-johnson')
                df_ft[selected_ft_cols] = pt.fit_transform(df_ft[selected_ft_cols])
                st.success("✅ Power (Yeo-Johnson) Transformation Applied")

            elif transformation_method == "Polynomial Features":
                degree = st.session_state.get("poly_degree", 2)
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_array = poly.fit_transform(df_ft[selected_ft_cols])
                poly_feature_names = poly.get_feature_names_out(selected_ft_cols)
                poly_df = pd.DataFrame(poly_array, columns=poly_feature_names, index=df_ft.index)
                df_ft = df_ft.drop(columns=selected_ft_cols)
                df_ft = pd.concat([df_ft, poly_df], axis=1)
                st.success("✅ Polynomial Features Applied")

            else:
                st.warning("No transformation selected")

            st.session_state.df = df_ft

        if transformation_method == "Polynomial Features":
            st.session_state["poly_degree"] = st.slider("Polynomial Degree:", min_value=2, max_value=5, value=2, key="poly_slider")

    else:
        st.info("No numeric columns found for transformation.")

    # ================= FEATURE SELECTION & DIMENSIONALITY REDUCTION =================
    st.markdown("---")
    st.subheader("🎯 Feature Selection & Dimensionality Reduction")

    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA

    numeric_cols_fs = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols_fs) > 1:
        fs_method = st.selectbox(
            "Choose Feature Selection / Reduction Method:",
            ["None", "Recursive Feature Elimination (RFE)", "PCA"],
            key="fs_method"
        )

        if fs_method == "Recursive Feature Elimination (RFE)":
            target_col_rfe = st.selectbox("Select Target Column:", st.session_state.df.columns.tolist(), key="rfe_target")
            n_features = st.slider("Number of features to select:", min_value=1, max_value=len(numeric_cols_fs) - 1, value=5, key="rfe_n")

            if st.button("Apply RFE"):
                df_rfe = st.session_state.df.copy()
                feature_cols = [c for c in numeric_cols_fs if c != target_col_rfe]
                X = df_rfe[feature_cols]
                y = df_rfe[target_col_rfe]

                model = RandomForestClassifier()
                rfe = RFE(model, n_features_to_select=n_features)
                X_selected = rfe.fit_transform(X, y)

                selected_features = rfe.support_
                ranking = rfe.ranking_

                st.write("**Selected Features:**", selected_features)
                st.write("**Feature Ranking:**", ranking)

                selected_col_names = [feature_cols[i] for i in range(len(feature_cols)) if selected_features[i]]
                st.success(f"✅ RFE Applied — Selected columns: {selected_col_names}")

                df_result = pd.DataFrame(X_selected, columns=selected_col_names)
                if target_col_rfe in df_rfe.columns:
                    df_result[target_col_rfe] = y.values
                st.session_state.df = df_result

        elif fs_method == "PCA":
            n_components = st.slider("Number of PCA components:", min_value=1, max_value=len(numeric_cols_fs), value=2, key="pca_n")

            if st.button("Apply PCA"):
                df_pca = st.session_state.df.copy()
                X = df_pca[numeric_cols_fs]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)

                st.write("**Explained Variance Ratio for each principal component:**")
                st.write(pca.explained_variance_ratio_)

                pca_columns = [f'PC{i+1}' for i in range(n_components)]
                pca_df = pd.DataFrame(data=X_pca, columns=pca_columns)

                non_numeric_cols = st.session_state.df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
                for col in non_numeric_cols:
                    pca_df[col] = st.session_state.df[col].values

                st.success(f"✅ PCA Applied — Reduced to {n_components} components")
                st.session_state.df = pca_df

        elif fs_method == "None":
            st.info("No feature selection method selected.")

    else:
        st.info("Not enough numeric columns for feature selection.")

    # ================= HANDLING IMBALANCED DATA =================
    st.markdown("---")
    st.subheader("⚖️ Handling Imbalanced Data")

    if st.button("Check Class Distribution"):
        target_candidates = st.session_state.df.columns.tolist()
        st.session_state["imb_target_candidates"] = target_candidates

    imb_target = st.selectbox(
        "Select Target Column for Imbalance Handling:",
        st.session_state.df.columns.tolist(),
        key="imb_target"
    )

    if imb_target:
        st.write("**Current Class Distribution:**")
        st.bar_chart(st.session_state.df[imb_target].value_counts())

    imbalance_method = st.selectbox(
        "Choose Imbalance Handling Method:",
        ["None", "Oversampling (SMOTE)", "Undersampling"],
        key="imb_method"
    )

    if st.button("Apply Imbalance Handling"):
        df_imb = st.session_state.df.copy()
        feature_cols_imb = [c for c in df_imb.columns if c != imb_target]
        numeric_feature_cols = df_imb[feature_cols_imb].select_dtypes(include=['int64', 'float64']).columns.tolist()

        X = df_imb[numeric_feature_cols]
        y = df_imb[imb_target]

        if imbalance_method == "Oversampling (SMOTE)":
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X_res, y_res = smote.fit_resample(X, y)
                df_result = pd.DataFrame(X_res, columns=numeric_feature_cols)
                df_result[imb_target] = y_res
                st.session_state.df = df_result
                st.success("✅ SMOTE Oversampling Applied")
                st.write("**New Class Distribution:**")
                st.bar_chart(df_result[imb_target].value_counts())
            except ImportError:
                st.error("❌ Please install imbalanced-learn: `pip install imbalanced-learn`")

        elif imbalance_method == "Undersampling":
            from collections import Counter
            class_counts = Counter(y)
            min_count = min(class_counts.values())
            df_undersampled = pd.concat([
                df_imb[df_imb[imb_target] == cls].sample(min_count, random_state=42)
                for cls in class_counts.keys()
            ]).reset_index(drop=True)
            st.session_state.df = df_undersampled
            st.success("✅ Undersampling Applied")
            st.write("**New Class Distribution:**")
            st.bar_chart(df_undersampled[imb_target].value_counts())

        else:
            st.warning("No imbalance handling method selected")

    # ================= RESULT =================
    st.markdown("---")
    st.subheader("📊 Updated Data")

    st.dataframe(st.session_state.df.head(), use_container_width=True)

    # Optional download
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Processed Data",
        csv,
        "processed_data.csv",
        "text/csv"
    )
