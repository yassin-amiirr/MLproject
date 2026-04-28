import streamlit as st

from pages.upload_page import show_upload_page
from pages.visualization_page import show_visualization_page
from pages.preprocessing_page import show_preprocessing_page
from pages.model_page import show_model_page
from pages.evaluation_page import show_evaluation_page


st.set_page_config(
    page_title="ML Project",
    page_icon="🤖",
    layout="wide"
)


def main():
    st.title("ML Project")

    page = st.sidebar.radio(
        "Go to Page:",
        [
            "File Upload",
            "Visualization",
            "Preprocessing",
            "Model Selection",
            "Evaluation"
        ]
    )

    if page == "File Upload":
        show_upload_page()

    elif page == "Visualization":
        show_visualization_page()

    elif page == "Preprocessing":
        show_preprocessing_page()

    elif page == "Model Selection":
        show_model_page()

    elif page == "Evaluation":
        show_evaluation_page()


if __name__ == "__main__":
    main()