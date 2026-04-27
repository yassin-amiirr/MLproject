import streamlit as st
import pandas as pd
from pages.model_page import show_model_page
from pages.preprocessing_page import show_preprocessing_page

st.set_page_config(page_title="ML Project", layout="wide")

if "df" in st.session_state:
    st.dataframe(st.session_state.df)
else:
    st.warning("Please upload a data file first.")

st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 36px; font-weight: bold; color: #1f2937; margin-bottom: 20px; }
    .info-text { font-size: 18px; color: #374151; margin-bottom: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Welcome to the ML Project</div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">Please select a page from the sidebar to begin.</div>', unsafe_allow_html=True)

# ------------------ Navigation State ------------------
PAGES = ["File Upload", "Visualization", "Preprocessing", "Model Selection", "Evaluation"]

if "sidebar_page" not in st.session_state:
    st.session_state.sidebar_page = PAGES[0]

# ------------------ Sidebar Navigation ------------------
page = st.sidebar.radio(
    "Go to Page:",
    PAGES,
    key="sidebar_page"
)

# ------------------ Page 1: File Upload ------------------
if page == "File Upload":
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">📁 1- File Upload & Data Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-text">Upload your dataset to get started. Supported formats: CSV, Excel (XLSX, XLS).</div>',
        unsafe_allow_html=True
    )

    # Initialize session state values if they don't exist
    if "df" not in st.session_state:
        st.session_state.df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None

    uploaded_file = st.file_uploader(
        "Choose a dataset file",
        type=["csv", "xlsx", "xls"],
        help="Make sure your file has a header row."
    )

    if uploaded_file is not None:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith(".csv"):
                df_temp = pd.read_csv(uploaded_file)
            else:
                df_temp = pd.read_excel(uploaded_file)
            
            st.session_state.df = df_temp
            st.session_state.file_name = uploaded_file.name

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Display the data if it exists in session_state (whether just uploaded or previously uploaded)
    if st.session_state.df is not None:
        st.success(f"✅ Active Dataset: **{st.session_state.file_name}**")

        col1, col2 = st.columns([3, 1])
        with col1:
            # Show dataset preview
            with st.expander("🔍 Dataset Preview", expanded=True):
                st.dataframe(st.session_state.df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 rows of {len(st.session_state.df)} total rows.")
        
        with col2:
            # Optional basic info
            with st.expander("📊 Dataset Info", expanded=True):
                st.write("**Rows:**", st.session_state.df.shape[0])
                st.write("**Columns:**", st.session_state.df.shape[1])
                st.write("**Missing Values:**", st.session_state.df.isnull().sum().sum())
                
                # Button to clear data
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Dataset", use_container_width=True, type="secondary"):
                    st.session_state.df = None
                    st.session_state.file_name = None
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ باقي الصفحات Placeholder ------------------
elif page == "Visualization":
    st.title("2- Data Visualization (Page 2)")
    st.info("This page will be implemented next.")

elif page == "Preprocessing":
    show_preprocessing_page()
elif page == "Model Selection":
    show_model_page()

elif page == "Evaluation":
    from pages.evaluation_page import show_evaluation_page
    show_evaluation_page()

# ------------------ Bottom Navigation Buttons ------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

current_index = PAGES.index(st.session_state.sidebar_page)

def go_previous():
    st.session_state.sidebar_page = PAGES[current_index - 1]

def go_next():
    st.session_state.sidebar_page = PAGES[current_index + 1]

with col1:
    if current_index > 0:
        st.button("⬅️ Previous", use_container_width=True, on_click=go_previous)

with col3:
    if current_index < len(PAGES) - 1:
        st.button("Next ➡️", use_container_width=True, on_click=go_next)
