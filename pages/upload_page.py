import streamlit as st
import pandas as pd


def show_upload_page():
    st.title("📁 File Upload")
    st.markdown("Upload your dataset here. Supported formats: CSV, XLSX")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state["df"] = df
            st.session_state["file_name"] = uploaded_file.name

            st.success(f"File uploaded successfully: {uploaded_file.name}")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)

            st.subheader("Dataset Info")
            st.write("Shape:", df.shape)
            st.write("Columns:", list(df.columns))

        except Exception as e:
            st.error(f"Error while reading file: {e}")

    else:
        if "df" in st.session_state and st.session_state["df"] is not None:
            st.info(f"Current uploaded file: {st.session_state.get('file_name', 'dataset')}")
            st.dataframe(st.session_state["df"].head(), use_container_width=True)
        else:
            st.warning("No dataset uploaded yet.")
