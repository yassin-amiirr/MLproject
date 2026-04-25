import streamlit as st
import pandas as pd

st.title("1- File Upload")
st.write("Upload dataset files with multiple supported extensions: CSV, Excel")

if "df" not in st.session_state:
    st.session_state.df = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.session_state.file_name = uploaded_file.name
    except Exception as e:
        st.error(f"Error reading file: {e}")

if st.session_state.df is not None:
    st.success(f"✅ File loaded successfully: {st.session_state.file_name}")
    st.subheader("Dataset Preview")
    st.dataframe(st.session_state.df.head())
    st.write("Rows, Columns:", st.session_state.df.shape)
    
    if st.button("Clear Dataset"):
        st.session_state.df = None
        st.session_state.file_name = None
        st.rerun()