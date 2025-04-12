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

# Load reference tables to get names from IDs
@st.cache_data(ttl=600)
def load_reference_data():
    try:
        supabase = init_connection()
        if not supabase:
            return {}, {}, {}
            
        # Try to load neighborhood reference table
        try:
            neighborhood_response = supabase.table('neighborhoods').select('*').execute()
            neighborhood_dict = {item['id']: item['name'] for item in neighborhood_response.data} if neighborhood_response.data else {}
        except Exception:
            st.warning("Could not load neighborhoods table. Will use IDs instead of names.")
            neighborhood_dict = {}
            
        # Try to load property type reference table
        try:
            property_type_response = supabase.table('property_types').select('*').execute()
            property_type_dict = {item['id']: item['name'] for item in property_type_response.data} if property_type_response.data else {}
        except Exception:
            st.warning("Could not load property_types table. Will use IDs instead of names.")
            property_type_dict = {}
            
        # Try to load classification reference table
        try:
            classification_response = supabase.table('classifications').select('*').execute()
            classification_dict = {item['id']: item['name'] for item in classification_response.data} if classification_response.data else {}
        except Exception:
            st.warning("Could not load classifications table. Will use IDs instead of names.")
            classification_dict = {}
            
        return neighborhood_dict, property_type_dict, classification_dict
    except Exception as e:
        st.error(f"Failed to load reference data: {str(e)}")
        return {}, {}, {}

# Load data from Supabase with ID columns
@st.cache_data(ttl=600)
def load_data():
    try:
        # Initialize Supabase client
        supabase = init_connection()
        if not supabase:
            return pd.DataFrame()
        
        # Fetch data from the 'properties' table
        response = supabase.table('properties').select('*').execute()
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        
        if df.empty:
            raise ValueError("No data returned from database")
            
        # Verify required columns
        required_columns = ['neighborhood_id', 'classification_id', 'property_type_id', 'area', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert price and area to numeric if needed
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        
        # Clean data
        df = df.dropna(subset=['price', 'area'])
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        # Load lookup tables to convert IDs to names
        neighborhood_dict, property_type_dict, classification_dict = load_reference_data()
        
        # Create name columns from IDs
        if neighborhood_dict:
            df['neighborhood_name'] = df['neighborhood_id'].map(neighborhood_dict).fillna('Unknown')
        else:
            df['neighborhood_name'] = df['neighborhood_id'].astype(str)
            
        if property_type_dict:
            df['property_type_name'] = df['property_type_id'].map(property_type_dict).fillna('Unknown')
        else:
            df['property_type_name'] = df['property_type_id'].astype(str)
            
        if classification_dict:
            df['classification_name'] = df['classification_id'].map(classification_dict).fillna('Unknown')
        else:
            df['classification_name'] = df['classification_id'].astype(str)
            
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
    
    # Display database inspection tool
    st.warning("""
    ### Database Structure
    
    Your properties table has these columns:
    - property_id
    - area
    - price
    - property_type_id
    - classification_id
    - neighborhood_id
    
    But the app needs name values, not just IDs. Please make sure you have:
    
    1. A 'neighborhoods' table with 'id' and 'name' columns
    2. A 'property_types' table with 'id' and 'name' columns
    3. A 'classifications' table with 'id' and 'name' columns
    
    These tables should connect to your properties table via their ID fields.
    """)
    
    # Create table structure tool
    with st.expander("Create Reference Tables"):
        st.write("""
        If you don't already have reference tables, you can create them with this SQL:
        
        ```sql
        -- Create neighborhoods table
        CREATE TABLE neighborhoods (
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL
        );
        
        -- Create property_types table
        CREATE TABLE property_types (
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL
        );
        
        -- Create classifications table
        CREATE TABLE classifications (
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL
        );
        ```
        
        Then populate them with your data.
        """)
        
        # Simple tool to create test data
        st.write("### Quick Reference Data Creator")
        st.write("Use this to create simple reference data for testing:")
        
        table_option = st.selectbox(
            "Select table to create:",
            ["neighborhoods", "property_types", "classifications"]
        )
        
        num_items = st.number_input("Number of items to create:", min_value=1, max_value=20, value=5)
        
        if st.button(f"Create {table_option} data"):
            try:
                supabase = init_connection()
                
                # Create example data
                data = []
                for i in range(1, num_items + 1):
                    data.append({
                        "id": i,
                        "name": f"{table_option.title()[:-1]} {i}"
                    })
                
                # Insert data
                response = supabase.table(table_option).insert(data).execute()
                
                if response.data:
                    st.success(f"Created {len(response.data)} items in {table_option} table")
                    st.write("Data:", response.data)
                else:
                    st.error("Failed to create data")
            except Exception as e:
                st.error(f"Error creating reference data: {str(e)}")




