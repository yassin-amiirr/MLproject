import streamlit as st

st.set_page_config(page_title="ML Project", layout="wide")

if "df" in st.session_state:
    st.dataframe(st.session_state.df)
else:
    st.warning("يرجى رفع ملف البيانات أولاً.")

# Custom CSS
st.markdown("""
    <style>
    .main-title { text-align: center; font-size: 36px; font-weight: bold; color: #1f2937; margin-bottom: 20px; }
    .info-text { font-size: 18px; color: #374151; margin-bottom: 10px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Welcome to the ML Project</div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">Please select a page from the sidebar to begin.</div>', unsafe_allow_html=True)

# يمكنك إضافة أسماء الفريق هنا كما هو مطلوب في المشروع
st.info("Team Members: Omar, Yousef, Mohamed, Mostafa, Yassin, Abdelrahman, Yassir")