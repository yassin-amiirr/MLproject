import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_visualization_page():
    st.title("📊 Data Visualization")

    # Checking if data exists in session_state from the upload page
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset first in the 'File Upload' page.")
        return

    df = st.session_state['df']

    st.subheader("Explore your data through visual charts")

    # Layout: Top Row (Line Plot)
    st.markdown("### 1. Price Trends")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.lineplot(x="Year", y="Price", data=df, ax=ax1, marker='o', color='royalblue')
    ax1.set_title("Price over Years")
    st.pyplot(fig1)

    st.divider()

    # Layout: Bottom Row (Two columns for Scatter and Box plots)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 2. Mileage vs Price")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x="Mileage", y="Price", data=df, hue="Condition", ax=ax2)
        ax2.set_title("Correlation between Mileage and Price")
        st.pyplot(fig2)

    with col2:
        st.markdown("### 3. Price Distribution")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="Condition", y="Price", data=df, palette="Set2", ax=ax3)
        ax3.set_title("Price by Vehicle Condition")
        st.pyplot(fig3)
