import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from supabase import create_client
import os

# Page config
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# App title
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices based on property features!')

# Supabase connection
@st.cache_resource
def init_connection():
    # Using the provided Supabase credentials
    supabase_url = "https://imdnhiwyfgjdgextvrkj.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0.9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk"
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

# Load data from Supabase
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data():
    try:
        # Initialize Supabase client
        supabase = init_connection()
        if not supabase:
            return pd.DataFrame()
        
        # Fetch data from the 'real_estate' table (adjust table name as needed)
        response = supabase.table('real_estate').select('*').execute()
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        
        # Data validation
        required_columns = ['neighborhood_name', 'classification_name', 
                          'property_type_name', 'area', 'price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert price and area to numeric if needed
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        
        # Clean data
        df = df.dropna(subset=['price', 'area'])
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Initialize a placeholder for database connection status
if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

# Load data
df = load_data()

if not df.empty:
    st.session_state['db_connected'] = True
    st.success("Data loaded successfully from Supabase!")
    
    # Data Overview
    with st.expander("Data Overview"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Raw Data Sample")
            st.dataframe(df.head())
        with col2:
            st.write("### Data Statistics")
            st.dataframe(df.describe())

        # Visualizations
        try:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='price', title='Price Distribution')
                st.plotly_chart(fig)
            with col2:
                fig = px.scatter(df, x='area', y='price', color='neighborhood_name',
                               title='Area vs Price by Neighborhood')
                st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    # Sidebar inputs
    with st.sidebar:
        st.header("Enter Property Details")
        neighborhood = st.selectbox("Neighborhood", sorted(df['neighborhood_name'].unique()))
        classification = st.selectbox("Classification", sorted(df['classification_name'].unique()))
        property_type = st.selectbox("Property Type", sorted(df['property_type_name'].unique()))
        
        # Modified area slider with max 1500
        area_min = float(df['area'].min())
        area_max = 1500.0  # Hard-coded maximum
        default_area = min(float(df['area'].median()), area_max)  # Ensure default doesn't exceed max
        area = st.slider("Area (m²)", 
                        min_value=area_min, 
                        max_value=area_max,
                        value=default_area)

    # Model training
    @st.cache_resource
    def train_model(data):
        try:
            X = pd.get_dummies(data[['neighborhood_name', 'classification_name',
                                   'property_type_name', 'area']], drop_first=True)
            y = data['price']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            return None, None

    model, feature_columns = train_model(df)

    if model and feature_columns:
        # Prepare input features
        input_df = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': classification,
            'property_type_name': property_type,
            'area': area
        }])

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
            st.markdown(f"## Predicted Price: **${prediction:,.2f}**")
            
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
    st.error("Failed to load data from Supabase. Please check your database connection and table structure.")
    
    # Display configuration help if not connected
    if not st.session_state['db_connected']:
        st.warning("""
        ### Supabase Table Setup Guide
        
        It appears your connection to Supabase is not returning any data. Make sure you've:
        
        1. Created a 'real_estate' table in your Supabase project with these columns:
           - neighborhood_name (text)
           - classification_name (text)
           - property_type_name (text)
           - area (numeric)
           - price (numeric)
        
        2. Imported your real estate data into the table
        
        You can create this table with the following SQL:
        
        ```sql
        CREATE TABLE real_estate (
          id SERIAL PRIMARY KEY,
          neighborhood_name TEXT NOT NULL,
          classification_name TEXT NOT NULL,
          property_type_name TEXT NOT NULL,
          area NUMERIC NOT NULL,
          price NUMERIC NOT NULL
        );
        ```
        
        3. Verify your table name matches 'real_estate' in the code (or update the code to match your table name)
        """)
