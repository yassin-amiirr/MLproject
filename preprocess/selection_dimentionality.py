import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def dimensionality_reduction_ui(data):
    st.header(" تقليل الأبعاد (PCA)")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("لا توجد أعمدة رقمية لتطبيق PCA.")
        return data
    target = None
    if st.session_state.get('target_column'):
        target = st.session_state['target_column']
        st.info(f"عمود الهدف الحالي: {target}")
    else:
        target = st.selectbox("حدد عمود الهدف (Target):", data.columns, key="pca_target")
    cols_wout_target = [c for c in numeric_cols if c != target]

    selected_cols = st.multiselect("اختر الأعمدة للضغط (PCA):", cols_wout_target, default=cols_wout_target)
    
    if len(selected_cols) < 2:
        st.info("اختار عمودين على الأقل لتشغيل الـ PCA.")
        return data

    n_components = st.slider("عدد المكونات الأساسية (Components):", 1, len(selected_cols), min(len(selected_cols), 2))
    
    if st.button("تشغيل PCA"):
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data[selected_cols])
        
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(pca_data, columns=pca_cols, index=data.index)
        
        remaining_df = data.drop(columns=selected_cols)
        transformed_df = pd.concat([remaining_df, pca_df], axis=1)
        
        st.success(f" تم الضغط بنجاح! نسبة التباين: {np.sum(pca.explained_variance_ratio_):.2f} with ratio: {pca.explained_variance_ratio_}")
        st.dataframe(transformed_df.head())
        return transformed_df
    return data

def feature_selection_ui(data):
    st.header(" اختيار الخصائص (RFE)")

    target_col = st.selectbox("حدد عمود الهدف (Target):", data.columns, key="rfe_target")
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)

    selected_cols = st.multiselect("اختر الميزات الرقمية للتقييم:", numeric_features, default=numeric_features)
    
    if not selected_cols:
        st.warning("ال RFE يتطلب أعمدة رقمية. يرجى اختيار ميزات رقمية.")
        return data

    # تحديد نوع الموديل تلقائياً
    is_classification = data[target_col].nunique() < 20 
    if len(selected_cols) > 1:
        n_features = st.slider(
            "عدد الميزات المطلوب الاحتفاظ بها:", 
            min_value=1, 
            max_value=len(selected_cols), 
            value=min(len(selected_cols), 5)
        )
    else:
        n_features = 1
        st.info("ملاحظة: لديك ميزة واحدة فقط مختارة، سيتم الاحتفاظ بها.") 
    if st.button("بدء اختيار الخصائص"):
        model =None
        if is_classification:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)

        rfe = RFE(estimator=model, n_features_to_select=n_features)
        
        X = data[selected_cols].fillna(0).astype(float)
        y = data[target_col]
        selected_features = rfe.fit_transform(X, y)
        selected_features = [col for col, support in zip(selected_cols, rfe.support_) if support]

        st.success(f"✅ الخصائص المختارة: {', '.join(selected_features)}")

        chosen_features = [col for col, supp in zip(selected_cols, rfe.support_) if supp]

        final_df = pd.concat([data.drop(columns=selected_cols), data[chosen_features]], axis=1)
        
        st.success(f"✅ تم حذف الأعمدة الضعيفة والاحتفاظ بـ: {', '.join(chosen_features)}")
        st.dataframe(final_df.head())
        return final_df
    return data