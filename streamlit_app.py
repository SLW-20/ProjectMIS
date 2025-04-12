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

# First, fetch table schema to understand what columns are available
@st.cache_data(ttl=600)
def get_table_columns():
    try:
        supabase = init_connection()
        if not supabase:
            return None
            
        # First attempt to get column info by querying a single row
        response = supabase.table('properties').select('*').limit(1).execute()
        if response.data and len(response.data) > 0:
            return list(response.data[0].keys())
        else:
            return None
    except Exception as e:
        st.error(f"Failed to fetch table schema: {str(e)}")
        return None

# Load data from Supabase with column mapping
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
            
        # Get actual columns from the table
        columns = list(df.columns)
        st.write("Available columns in properties table:", columns)
        
        # Try to map columns to expected names
        column_mapping = {}
        
        # Here we'll try different variations of column names
        neighborhood_options = ['neighborhood_name', 'neighborhood', 'area_name', 'location', 'district']
        classification_options = ['classification_name', 'classification', 'property_classification', 'class']
        property_type_options = ['property_type_name', 'property_type', 'type', 'building_type']
        area_options = ['area', 'size', 'square_meters', 'sqm', 'area_sqm']
        price_options = ['price', 'value', 'cost', 'sale_price']
        
        # Find matching columns
        for col in columns:
            col_lower = col.lower()
            if any(opt.lower() == col_lower for opt in neighborhood_options):
                column_mapping['neighborhood_name'] = col
            elif any(opt.lower() == col_lower for opt in classification_options):
                column_mapping['classification_name'] = col
            elif any(opt.lower() == col_lower for opt in property_type_options):
                column_mapping['property_type_name'] = col
            elif any(opt.lower() == col_lower for opt in area_options):
                column_mapping['area'] = col
            elif any(opt.lower() == col_lower for opt in price_options):
                column_mapping['price'] = col
        
        # Check if we found all necessary columns
        required_columns = ['neighborhood_name', 'classification_name', 'property_type_name', 'area', 'price']
        missing_columns = [col for col in required_columns if col not in column_mapping]
        
        if missing_columns:
            # If columns are missing, ask user to manually map them
            st.warning(f"Missing required columns: {', '.join(missing_columns)}")
            st.info("Please select which columns from your database correspond to the required fields:")
            
            # Use dropdown to let user map columns
            for missing in missing_columns:
                column_mapping[missing] = st.selectbox(
                    f"Map '{missing}' to:", 
                    options=columns,
                    key=f"map_{missing}"
                )
        
        # Apply column mapping to the dataframe
        mapped_df = df.copy()
        for target_col, source_col in column_mapping.items():
            if source_col in df.columns:
                mapped_df[target_col] = df[source_col]
            else:
                raise ValueError(f"Column mapping failed: {source_col} not found in data")
        
        # Ensure we now have all required columns
        for col in required_columns:
            if col not in mapped_df.columns:
                raise ValueError(f"Missing required column after mapping: {col}")
        
        # Convert price and area to numeric if needed
        mapped_df['price'] = pd.to_numeric(mapped_df['price'], errors='coerce')
        mapped_df['area'] = pd.to_numeric(mapped_df['area'], errors='coerce')
        
        # Clean data
        mapped_df = mapped_df.dropna(subset=['price', 'area'])
        if mapped_df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        return mapped_df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Initialize a placeholder for database connection status
if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

# Show table structure information
columns = get_table_columns()
if columns:
    st.success("Successfully connected to Supabase!")
    st.write("Columns in your 'properties' table:", columns)
else:
    st.error("Could not retrieve columns from your properties table.")

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
    with st.expander("Database Inspection Tool"):
        st.write("Let's check what's in your Supabase database:")
        
        if st.button("Show Available Tables"):
            try:
                supabase = init_connection()
                if supabase:
                    # This would require admin access, so it might not work with anon key
                    st.warning("Cannot list tables with anon key. Please check your Supabase dashboard directly.")
                    st.write("Try entering your table name manually:")
                    table_name = st.text_input("Table name:", value="properties")
                    
                    if st.button(f"Inspect '{table_name}' Table"):
                        response = supabase.table(table_name).select('*').limit(5).execute()
                        if response.data:
                            st.write(f"Found data in '{table_name}' table:")
                            st.write("Columns:", list(response.data[0].keys()))
                            st.dataframe(pd.DataFrame(response.data))
                        else:
                            st.error(f"No data found in '{table_name}' table or table doesn't exist.")
            except Exception as e:
                st.error(f"Database inspection failed: {str(e)}")
                
        st.write("""
        ### Manual Setup
        
        If you're seeing column errors, make sure your Supabase table has these columns or use the mapping tool above:
        
        ```sql
        -- Example SQL to create a properties table with the expected columns
        CREATE TABLE properties (
          id SERIAL PRIMARY KEY,
          neighborhood_name TEXT NOT NULL,
          classification_name TEXT NOT NULL,
          property_type_name TEXT NOT NULL,
          area NUMERIC NOT NULL,
          price NUMERIC NOT NULL
        );
        ```
        
        Or update your existing table to include these columns.
        """)
