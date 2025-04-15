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

# Debug function to show table structure
def debug_table_structure(table_name):
    try:
        supabase = init_connection()
        if not supabase:
            return
        
        # Get the first row to see columns
        response = supabase.table(table_name).select('*').limit(1).execute()
        
        if response.data:
            sample_row = response.data[0]
            st.write(f"### {table_name} columns:")
            st.json(sample_row)
        else:
            st.warning(f"No data in {table_name} table")
    except Exception as e:
        st.error(f"Error inspecting {table_name}: {str(e)}")

# Load reference tables to get names from IDs
@st.cache_data(ttl=600)
def load_reference_data():
    try:
        supabase = init_connection()
        if not supabase:
            return {}, {}, {}
        
        # Debug flag to show detailed error information
        debug_mode = True
            
        # Try to load neighborhood reference table
        neighborhood_dict = {}
        try:
            neighborhood_response = supabase.table('neighborhoods').select('*').execute()
            if neighborhood_response.data:
                # First, attempt with expected column names
                if 'neighborhood_id' in neighborhood_response.data[0] and 'neighborhood_name' in neighborhood_response.data[0]:
                    neighborhood_dict = {item['neighborhood_id']: item['neighborhood_name'] for item in neighborhood_response.data}
                # Fallback to generic 'id' and 'name'
                elif 'id' in neighborhood_response.data[0] and 'name' in neighborhood_response.data[0]:
                    neighborhood_dict = {item['id']: item['name'] for item in neighborhood_response.data}
                # Last resort: print actual keys to help debug
                else:
                    if debug_mode:
                        st.warning(f"Unexpected neighborhood columns: {list(neighborhood_response.data[0].keys())}")
        except Exception as e:
            if debug_mode:
                st.warning(f"Could not load neighborhoods table: {type(e).__name__}: {str(e)}")
            
        # Try to load property type reference table
        property_type_dict = {}
        try:
            # Try both possible table names
            try:
                property_type_response = supabase.table('property_type').select('*').execute()
            except:
                property_type_response = supabase.table('property_types').select('*').execute()
                
            if property_type_response.data:
                # First, attempt with expected column names
                if 'property_type_id' in property_type_response.data[0] and 'property_type_name' in property_type_response.data[0]:
                    property_type_dict = {item['property_type_id']: item['property_type_name'] for item in property_type_response.data}
                # Fallback to generic 'id' and 'name'
                elif 'id' in property_type_response.data[0] and 'name' in property_type_response.data[0]:
                    property_type_dict = {item['id']: item['name'] for item in property_type_response.data}
                # Last resort: print actual keys to help debug
                else:
                    if debug_mode:
                        st.warning(f"Unexpected property type columns: {list(property_type_response.data[0].keys())}")
        except Exception as e:
            if debug_mode:
                st.warning(f"Could not load property type table: {type(e).__name__}: {str(e)}")
            
        # Try to load classification reference table
        classification_dict = {}
        try:
            # Try both possible table names
            try:
                classification_response = supabase.table('property_classifications').select('*').execute()
            except:
                try:
                    classification_response = supabase.table('classifications').select('*').execute()
                except:
                    classification_response = supabase.table('property_classification').select('*').execute()
                
            if classification_response.data:
                # First, attempt with expected column names
                if 'classification_id' in classification_response.data[0] and 'classification_name' in classification_response.data[0]:
                    classification_dict = {item['classification_id']: item['classification_name'] for item in classification_response.data}
                # Fallback to generic 'id' and 'name'
                elif 'id' in classification_response.data[0] and 'name' in classification_response.data[0]:
                    classification_dict = {item['id']: item['name'] for item in classification_response.data}
                # Last resort: print actual keys to help debug
                else:
                    if debug_mode:
                        st.warning(f"Unexpected classification columns: {list(classification_response.data[0].keys())}")
        except Exception as e:
            if debug_mode:
                st.warning(f"Could not load classification table: {type(e).__name__}: {str(e)}")
            
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
        
        # Debug: Show actual columns in the properties table
        st.write("Properties table columns:", list(df.columns))
            
        # Identify ID columns - handle different possible naming conventions
        neighborhood_id_col = next((col for col in df.columns if col in ['neighborhood_id', 'neighborhood']), None)
        property_type_id_col = next((col for col in df.columns if col in ['property_type_id', 'property_type']), None)
        classification_id_col = next((col for col in df.columns if col in ['classification_id', 'classification']), None)
        
        # Verify required columns exist in some form
        missing_columns = []
        if not neighborhood_id_col:
            missing_columns.append("neighborhood_id")
        if not property_type_id_col:
            missing_columns.append("property_type_id")
        if not classification_id_col:
            missing_columns.append("classification_id")
        if 'area' not in df.columns:
            missing_columns.append("area")
        if 'price' not in df.columns:
            missing_columns.append("price")
        
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
        if neighborhood_dict and neighborhood_id_col:
            df['neighborhood_name'] = df[neighborhood_id_col].map(neighborhood_dict).fillna('Unknown')
        else:
            df['neighborhood_name'] = df[neighborhood_id_col].astype(str) if neighborhood_id_col else 'Unknown'
            
        if property_type_dict and property_type_id_col:
            df['property_type_name'] = df[property_type_id_col].map(property_type_dict).fillna('Unknown')
        else:
            df['property_type_name'] = df[property_type_id_col].astype(str) if property_type_id_col else 'Unknown'
            
        if classification_dict and classification_id_col:
            df['classification_name'] = df[classification_id_col].map(classification_dict).fillna('Unknown')
        else:
            df['classification_name'] = df[classification_id_col].astype(str) if classification_id_col else 'Unknown'
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# Initialize a placeholder for database connection status
if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

# Show debug tools
with st.expander("Database Debug Tools"):
    st.write("Use these tools to inspect your database tables")
    debug_table = st.selectbox("Select table to inspect", 
                          ["neighborhoods", "property_type", "property_classifications", "properties"])
    if st.button("Inspect Table Structure"):
        debug_table_structure(debug_table)

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
    
    Your properties table should have these columns:
    - property_id
    - area (numeric)
    - price (numeric)
    - property_type_id
    - classification_id
    - neighborhood_id
    
    And you need these reference tables:
    
    1. A 'neighborhoods' table with 'neighborhood_id' and 'neighborhood_name' columns
    2. A 'property_type' (or 'property_types') table with 'property_type_id' and 'property_type_name' columns
    3. A 'property_classifications' (or 'classifications') table with 'classification_id' and 'classification_name' columns
    
    These tables should connect to your properties table via their ID fields.
    """)
    
    # Create table structure tool
    with st.expander("Create Reference Tables"):
        st.write("""
        If you don't already have reference tables, you can create them with this SQL:
        
        ```sql
        -- Create neighborhoods table (if it doesn't exist)
        CREATE TABLE IF NOT EXISTS neighborhoods (
          neighborhood_id TEXT PRIMARY KEY,
          neighborhood_name VARCHAR NOT NULL
        );
        
        -- Create property_type table (if it doesn't exist)
        CREATE TABLE IF NOT EXISTS property_type (
          property_type_id TEXT PRIMARY KEY,
          property_type_name VARCHAR NOT NULL
        );
        
        -- Create property_classifications table (if it doesn't exist)
        CREATE TABLE IF NOT EXISTS property_classifications (
          classification_id TEXT PRIMARY KEY,
          classification_name VARCHAR NOT NULL
        );
        ```
        
        Then populate them with your data.
        """)
        
        # Simple tool to create test data
        st.write("### Quick Reference Data Creator")
        st.write("Use this to create simple reference data for testing:")
        
        table_options = {
            "neighborhoods": ("neighborhood_id", "neighborhood_name"),
            "property_type": ("property_type_id", "property_type_name"),
            "property_classifications": ("classification_id", "classification_name")
        }
        
        table_option = st.selectbox(
            "Select table to create:",
            list(table_options.keys())
        )
        
        id_col, name_col = table_options[table_option]
        
        num_items = st.number_input("Number of items to create:", min_value=1, max_value=20, value=5)
        
        if st.button(f"Create {table_option} data"):
            try:
                supabase = init_connection()
                
                # Create example data
                data = []
                prefix = table_option[0:3].upper()
                for i in range(1, num_items + 1):
                    data.append({
                        id_col: f"{prefix}-{i:03d}",
                        name_col: f"{table_option.title().replace('_', ' ')} {i}"
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
