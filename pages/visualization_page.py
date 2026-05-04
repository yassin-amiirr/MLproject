import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def show_visualization_page():
    st.title(":bar_chart: Dynamic Data Visualization")
    if 'df' not in st.session_state:
        st.warning("Please upload a dataset first.")
        return
    df = st.session_state['df']
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    # إذا لم يجد أعمدة تصنيفية، استخدم كل الأعمدة
    if not categorical_columns:
        categorical_columns = all_columns
    if not numeric_columns:
        st.error("The uploaded dataset doesn't contain any numerical columns for plotting.")
        return
    st.info("Customize your plots by selecting columns below.")
    # --- Section 1: Line Plot ---
    st.markdown("### 1. Trend Analysis (Line Plot)")
    col_x_line = st.selectbox("Select X-axis:", all_columns, key='x_line')
    col_y_line = st.selectbox("Select Y-axis:", numeric_columns, key='y_line')
    if col_x_line and col_y_line:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=col_x_line, y=col_y_line, data=df, ax=ax1, marker='o')
        st.pyplot(fig1)
    st.divider()
    # --- Section 2: Box Plot ---
    st.markdown("### 2. Distribution Analysis (Box Plot)")
    col_box_x = st.selectbox("Select Category (X-axis):", categorical_columns, key='box_x')
    col_box_y = st.selectbox("Select Value (Y-axis):", numeric_columns, key='box_y')

    if col_box_x and col_box_y:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.boxplot(x=col_box_x, y=col_box_y, data=df, palette="Set3", ax=ax2)
        plt.xticks(rotation=45) # لضمان ظهور الأسماء الطويلة بشكل واضح
        st.pyplot(fig2)

    st.divider()
    # --- Section 3: Scatter Plot ---
    st.markdown("### 3. Relationship Analysis (Scatter Plot)")
    c1, c2 = st.columns(2)
    with c1:
        x_scatter = st.selectbox("X-axis:", numeric_columns, key='x_scat')
    with c2:
        y_scatter = st.selectbox("Y-axis:", numeric_columns, key='y_scat')

    if x_scatter and y_scatter:
        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=x_scatter, y=y_scatter, data=df, ax=ax3)
        st.pyplot(fig3)
