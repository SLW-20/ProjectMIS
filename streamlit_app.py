import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from supabase import create_client
import os
from PIL import Image

try:
    import statsmodels
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False

st.set_page_config(
    page_title="Real Estate Price Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    if os.path.exists('kku.logo.jpg'):
        logo = Image.open('kku.logo.jpg')
    else:
        possible_paths = [
            'kku_logo.jpg',
            'kku_logo.png',
            'kku.logo.png',
            'logo.jpg',
            'logo.png',
            './kku.logo.jpg',
            './images/kku.logo.jpg'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logo = Image.open(path)
                st.success(f"Found KKU logo at: {path}")
                break
        else:
            raise FileNotFoundError("KKU logo image file not found. Please ensure 'kku.logo.jpg' is in the same directory as the app.")
    
    st.markdown(
        """
        <style>
        .logo-container {
            position: fixed;
            top: 18px;
            right: 30px;
            z-index: 1000;
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        .logo-text {
            margin-top: 5px;
            font-size: 20px;
            font-weight: bold;
            color: black;
        }
        .main-header {
            margin-top: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(logo, width=200)
    st.markdown('<div class="logo-text">MIS Graduation Project</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state['logo_displayed'] = True
    
except Exception as e:
    st.error(f"Error loading KKU logo: {str(e)}")
    st.info("Please ensure the KKU logo file (kku.logo.jpg) is in the same directory as this app.")
    st.session_state['logo_displayed'] = False

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 1000;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #374151;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #10B981;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #F8FAFC;
        padding: 2rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin: 1.5rem 0;
    }
    .sidebar .block-container {
        padding-top: 2rem;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1E3A8A;
    }
    .sidebar .block-container {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 Real Estate Price Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">This app predicts real estate prices based on property features!</div>', unsafe_allow_html=True)

@st.cache_resource
def init_connection():
    supabase_url = "https://imdnhiwyfgjdgextvrkj.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

@st.cache_data(ttl=600)
def load_reference_data():
    try:
        supabase = init_connection()
        if not supabase:
            return {}, {}, {}
            
        neighborhood_dict, property_type_dict, classification_dict = {}, {}, {}

        try:
            neighborhood_response = supabase.table('neighborhoods').select('*').execute()
            if neighborhood_response.data:
                neighborhood_dict = {item.get('neighborhood_id', item.get('id')): item.get('neighborhood_name', item.get('name')) for item in neighborhood_response.data}
        except: pass
            
        try:
            try:
                property_type_response = supabase.table('property_type').select('*').execute()
            except:
                property_type_response = supabase.table('property_types').select('*').execute()
            if property_type_response.data:
                property_type_dict = {item.get('property_type_id', item.get('id')): item.get('property_type_name', item.get('name')) for item in property_type_response.data}
        except: pass
            
        try:
            try:
                classification_response = supabase.table('property_classifications').select('*').execute()
            except:
                classification_response = supabase.table('classifications').select('*').execute()
            if classification_response.data:
                classification_dict = {item.get('classification_id', item.get('id')): item.get('classification_name', item.get('name')) for item in classification_response.data}
        except: pass
            
        return neighborhood_dict, property_type_dict, classification_dict
    except Exception as e:
        st.error(f"Failed to load reference data: {str(e)}")
        return {}, {}, {}

@st.cache_data(ttl=600)
def load_data():
    try:
        supabase = init_connection()
        if not supabase:
            return pd.DataFrame()
        
        response = supabase.table('properties').select('*').execute()
        df = pd.DataFrame(response.data)
        
        if df.empty:
            raise ValueError("No data returned from database")

        neighborhood_id_col = next((col for col in df.columns if col in ['neighborhood_id', 'neighborhood']), None)
        property_type_id_col = next((col for col in df.columns if col in ['property_type_id', 'property_type']), None)
        classification_id_col = next((col for col in df.columns if col in ['classification_id', 'classification']), None)
        
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

        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df = df.dropna(subset=['price', 'area'])
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        neighborhood_dict, property_type_dict, classification_dict = load_reference_data()
        
        df['neighborhood_name'] = df[neighborhood_id_col].map(neighborhood_dict).fillna('Unknown')
        df['property_type_name'] = df[property_type_id_col].map(property_type_dict).fillna('Unknown')
        df['classification_name'] = df[classification_id_col].map(classification_dict).fillna('Unknown')
        
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

df = load_data()

if not df.empty:
    st.session_state['db_connected'] = True
    col1, col2 = st.columns([1, 2])
    
    with st.sidebar:
        st.markdown('<div class="sub-header">Enter Property Details</div>', unsafe_allow_html=True)
        neighborhood = st.selectbox("Neighborhood", sorted(df['neighborhood_name'].unique()))
        classification = st.selectbox("Classification", sorted(df['classification_name'].unique()))
        property_type = st.selectbox("Property Type", sorted(df['property_type_name'].unique()))
        
        area_min = float(df['area'].min())
        area_max = 1500.0
        default_area = float(np.clip(df['area'].mean(), area_min, area_max))
        area = st.slider("Area (m²)", min_value=area_min, max_value=area_max, value=default_area)
