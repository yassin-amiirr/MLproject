import streamlit as st
import pandas as pd

# ------------------ Page Config ------------------
st.set_page_config(page_title="ML Project", layout="wide")

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #1f2937;
        margin-bottom: 20px;
    }

    .section-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .sub-title {
        font-size: 26px;
        font-weight: bold;
        color: #2563eb;
        margin-bottom: 15px;
    }

    .info-text {
        font-size: 17px;
        color: #374151;
        margin-bottom: 10px;
    }

    .success-box {
        background-color: #dcfce7;
        color: #166534;
        padding: 12px;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
st.markdown('<div class="main-title">ML Project</div>', unsafe_allow_html=True)

# ------------------ Sidebar Navigation ------------------
page = st.sidebar.radio(
    "Go to Page:",
    ["File Upload", "Visualization", "Preprocessing", "Model Selection", "Evaluation"]
)

# ------------------ Page 1: File Upload ------------------
if page == "File Upload":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">1- File Upload (Page 1)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-text">Upload dataset files with multiple supported extensions: CSV, Excel</div>',
        unsafe_allow_html=True
    )

    # Initialize session state values if they don't exist
    if "df" not in st.session_state:
        st.session_state.df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            
            st.session_state.file_name = uploaded_file.name

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Display the data if it exists in session_state (whether just uploaded or previously uploaded)
    if st.session_state.df is not None:
        # Confirmation Message
        st.markdown(
            f'<div class="success-box">✅ File loaded successfully: {st.session_state.file_name}</div>',
            unsafe_allow_html=True
        )

        # Show dataset preview
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head())

        # Optional basic info
        st.subheader("Dataset Info")
        st.write("Rows, Columns:", st.session_state.df.shape)
        st.write("Columns:", list(st.session_state.df.columns))

        # Button to clear data
        if st.button("Clear Dataset"):
            st.session_state.df = None
            st.session_state.file_name = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ باقي الصفحات Placeholder ------------------
elif page == "Visualization":
    st.title("2- Data Visualization (Page 2)")
    st.info("This page will be implemented next.")

elif page == "Preprocessing":
    st.title("3- Preprocessing (Page 3)")
    st.info("This page will be implemented next.")

elif page == "Model Selection":
    st.title("4- Model Selection (Page 4)")
    st.info("This page will be implemented next.")

elif page == "Evaluation":
    st.title("5- Model Evaluation (Page 5)")
    st.info("This page will be implemented next.")