import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px

# Page config
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# App title
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices based on property features!')

# Load data from GitHub
@st.cache_data
def load_data():
    # Replace this URL with your GitHub raw data URL
    url = "https://raw.githubusercontent.com/1Hani-77/TEST/refs/heads/main/abha%20real%20estate.csv"
    df = pd.read_csv(url)
    
    # Data validation
    required_columns = ['neighborhood_name', 'classification_name', 'property_type_name', 'area', 'price']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert price and area to numeric, removing any currency symbols or commas
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    df['area'] = pd.to_numeric(df['area'].astype(str).str.replace(r'[^\d.]', ''), errors='coerce')
    
    # Remove any rows with null values
    df = df.dropna(subset=['price', 'area'])
    
    return df

try:
    df = load_data()
    st.success("Data loaded successfully!")
    
    # Debug information
    st.write("### Data Shape:", df.shape)
    st.write("### Columns:", df.columns.tolist())
    
    # Data Overview
    with st.expander("Data Overview"):
        st.write("### Raw Data Sample")
        st.dataframe(df.head())
        
        st.write("### Data Statistics")
        st.dataframe(df.describe())

        # Safe plotting with error handling
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Price Distribution")
                fig = px.histogram(df, x='price', title='Price Distribution',
                                 labels={'price': 'Price', 'count': 'Count'})
                st.plotly_chart(fig)
            
            with col2:
                st.write("### Area vs Price")
                fig = px.scatter(df, x='area', y='price', color='neighborhood_name',
                               title='Area vs Price by Neighborhood',
                               labels={'area': 'Area (m²)', 'price': 'Price', 
                                     'neighborhood_name': 'Neighborhood'})
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating plots: {str(e)}")

    # Sidebar inputs
    with st.sidebar:
        st.header("Enter Property Details")
        
        neighborhood = st.selectbox(
            "Select Neighborhood",
            options=sorted(df['neighborhood_name'].unique())
        )
        
        classification = st.selectbox(
            "Select Classification",
            options=sorted(df['classification_name'].unique())
        )
        
        property_type = st.selectbox(
            "Select Property Type",
            options=sorted(df['property_type_name'].unique())
        )
        
        area = st.slider(
            "Area (m²)",
            min_value=float(df['area'].min()),
            max_value=float(df['area'].max()),
            value=float(df['area'].mean())
        )

    # Prepare features
    input_data = pd.DataFrame({
        'neighborhood_name': [neighborhood],
        'classification_name': [classification],
        'property_type_name': [property_type],
        'area': [area]
    })

    # Model training
    @st.cache_resource
    def train_model(df):
        # Prepare features
        X = pd.get_dummies(df[['neighborhood_name', 'classification_name', 
                              'property_type_name', 'area']], drop_first=True)
        y = df['price']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, X.columns

    # Train model
    model, feature_names = train_model(df)

    # Make prediction
    X_input = pd.get_dummies(input_data, drop_first=True)
    # Ensure input has same columns as training data
    for col in feature_names:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[feature_names]

    prediction = model.predict(X_input)[0]

    # Display prediction
    st.write("## Predicted Price")
    st.write(f"### ${prediction:,.2f}")

    # Feature importance
    with st.expander("Model Insights"):
        st.write("### Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance, x='importance', y='feature', 
                     orientation='h', title='Feature Importance')
        st.plotly_chart(fig)

    # Similar properties
    with st.expander("Similar Properties"):
        st.write("### Properties in the same neighborhood")
        similar_properties = df[df['neighborhood_name'] == neighborhood].head()
        st.dataframe(similar_properties)
        
        fig = px.scatter(similar_properties, x='area', y='price',
                        title='Similar Properties: Area vs Price',
                        hover_data=['classification_name', 'property_type_name'])
        st.plotly_chart(fig)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Please check your data format and ensure all required columns are present.")
