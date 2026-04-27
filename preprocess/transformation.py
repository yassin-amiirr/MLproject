import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures

def transform_data(data):
    st.header("Feature Transformation")
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns found in the dataset.")
        return data

    selected_cols = st.multiselect("Select numerical columns to transform:", numeric_cols, default=numeric_cols)
    
    if not selected_cols:
        return data

    transformation_method = st.selectbox(
        "Select a transformation method:",
        ["Log Transformation", "Box-Cox Transformation", "Yeo-Johnson (Power) Transformation", "Polynomial Features"]
    )

    transformed_df = data.copy()
    if st.button("Apply Transformation"):
        try:
            if transformation_method == "Log Transformation":
                for col in selected_cols:
                    if (transformed_df[col] <= 0).any():
                        transformed_df[col] = transformed_df[col] - transformed_df[col].min() + 1
                    transformed_df[col] = np.log1p(transformed_df[col])

            elif transformation_method == "Box-Cox Transformation":
                for col in selected_cols:
                    if (transformed_df[col] <= 0).any():
                        transformed_df[col] = transformed_df[col] - transformed_df[col].min() + 1
                transformer = PowerTransformer(method="box-cox")
                transformed_df[selected_cols] = transformer.fit_transform(transformed_df[selected_cols])

            elif transformation_method == "Yeo-Johnson (Power) Transformation":
                transformer = PowerTransformer(method="yeo-johnson")
                transformed_df[selected_cols] = transformer.fit_transform(transformed_df[selected_cols])

            elif transformation_method == "Polynomial Features":
                degree = st.slider("Select the degree of polynomial features:", min_value=2, max_value=5, value=2)
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                
                poly_data = poly.fit_transform(transformed_df[selected_cols])
                poly_cols = poly.get_feature_names_out(selected_cols)
                
                poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=transformed_df.index)
                
                transformed_df = transformed_df.drop(columns=selected_cols)
                transformed_df = pd.concat([transformed_df, poly_df], axis=1)

            st.success(f" Data transformed successfully using {transformation_method}!")
            st.dataframe(transformed_df.head())
            
            return transformed_df

        except Exception as e:
            st.error(f"An error occurred during transformation: {e}")
            return data
    return data