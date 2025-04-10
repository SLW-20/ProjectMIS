import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# عرض شعار جامعة الملك خالد
st.image("images/kku.logo.png", width=150)

# Page config
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# App title
st.title("Real Estate Price Prediction App")
st.info("This app predicts real estate prices based on property features!")

# Load data from Github
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/SLW-20/ProjectMIS/master/ahbaha%20real%20estate.csv"
        df = pd.read_csv(url)

        # Data validation
        required_columns = ['neighborhood_name', 'classification_name',
                            'property_type_name', 'area', 'price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    # Sidebar filters
    st.sidebar.header("Filter Options")
    neighborhood = st.sidebar.selectbox("Select Neighborhood", df['neighborhood_name'].unique())
    classification = st.sidebar.selectbox("Select Classification", df['classification_name'].unique())
    property_type = st.sidebar.selectbox("Select Property Type", df['property_type_name'].unique())
    area = st.sidebar.slider("Select Area (sq m)", int(df['area'].min()), int(df['area'].max()), int(df['area'].mean()))

    # Filter data
    input_df = pd.DataFrame([{
        'neighborhood_name': neighborhood,
        'classification_name': classification,
        'property_type_name': property_type,
        'area': area
    }])

    df_filtered = df.dropna()
    df_filtered = pd.get_dummies(df_filtered, drop_first=True)

    # Prepare training data
    feature_columns = [col for col in df_filtered.columns if col != 'price']
    X = df_filtered[feature_columns]
    y = df_filtered['price']

    model = RandomForestRegressor()
    model.fit(X, y)

    # Generate dummy features
    input_processed = pd.get_dummies(input_df, drop_first=True)

    # Align with training features
    for col in feature_columns:
        if col not in input_processed.columns:
            input_processed[col] = 0
    input_processed = input_processed[feature_columns]

    # Make prediction
    try:
        prediction = model.predict(input_processed)[0]
        st.markdown(f"## Predicted Price: ${prediction:,.2f}")
        
        # Feature importance
        with st.expander("Feature Importance"):
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

    # Similar properties section
    with st.expander("Similar Properties"):
        similar = df[df['neighborhood_name'] == neighborhood]
        if not similar.empty:
            st.dataframe(similar.head())
            fig = px.scatter(similar, x='area', y='price', 
                            hover_data=['classification_name', 'property_type_name'])
            st.plotly_chart(fig)
        else:
            st.warning("No similar properties found in this neighborhood")

else:
    st.error("Failed to load data. Please check the data source and try again.")
