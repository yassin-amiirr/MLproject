import streamlit as st
import sys
import os

# حل مشكلة المسارات
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocess.transformation import transform_data
from preprocess.selection_dimentionality import dimensionality_reduction_ui, feature_selection_ui

st.set_page_config(page_title="Preprocessing", layout="wide")


if "df" not in st.session_state or st.session_state.df is None:
    st.warning("يرجى رفع ملف البيانات أولاً.")
else:
    df = st.session_state.get("df", None)
    tab1, tab2, tab3 = st.tabs(["التحويل (Transformation)", "اختيار الخصائص (RFE)", "تقليل الأبعاد (PCA)"])

    with tab1:
        st.session_state.df = transform_data(st.session_state.df)

    with tab2:
        st.session_state.df = feature_selection_ui(st.session_state.df)

    with tab3:
        st.session_state.df = dimensionality_reduction_ui(st.session_state.df)


    st.divider()
    if st.button("حفظ البيانات النهائية للتدريب 💾"):
        st.success("تم حفظ البيانات! جاهزة الآن لمرحلة الـ Model Selection.")