import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from supabase import create_client
import os
from PIL import Image

# Check if statsmodels is available for OLS trendline in plots
try:
    import statsmodels
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False

# --------------------------#
# --- Page Configuration --#
# --------------------------#
st.set_page_config(
    page_title="Real Estate Price Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------#
# --- Custom CSS Styles  ---#
# --------------------------#
custom_css = """
<style>
    /* Logo container style */
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
    /* Main header and subheader styles */
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
    /* Styled info and success boxes */
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
    /* Prediction box styling */
    .prediction-box {
        background-color: #F8FAFC;
        padding: 2rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 
                    0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        margin: 1.5rem 0;
    }
    /* Sidebar style adjustments */
    .sidebar .block-container {
        padding-top: 2rem;
        background-color: #F8FAFC;
    }
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1E3A8A;
    }
    /* Button styling */
    div.stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# --------------------------#
# --- Display KKU Logo   ---#
# --------------------------#
def load_logo():
    """
    Attempts to load the KKU logo image from one of several possible paths.
    Returns an Image object if found, otherwise raises FileNotFoundError.
    """
    logo_filename = 'kku.logo.jpg'
    possible_paths = [
        'kku_logo.jpg',
        'kku_logo.png',
        'kku.logo.png',
        'logo.jpg',
        'logo.png',
        './kku.logo.jpg',
        './images/kku.logo.jpg'
    ]
    
    # Check current directory first with primary filename
    if os.path.exists(logo_filename):
        return Image.open(logo_filename), logo_filename

    # Check alternative paths if primary not found
    for path in possible_paths:
        if os.path.exists(path):
            return Image.open(path), path

    raise FileNotFoundError("KKU logo image file not found. Ensure 'kku.logo.jpg' or one of the alternate filenames is in the same directory as the app.")

try:
    logo, logo_path = load_logo()
    st.success(f"Found KKU logo at: {logo_path}")
    # Position the logo in the top-right corner with text
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(logo, width=200)
    st.markdown('<div class="logo-text">MIS Graduation Project</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state['logo_displayed'] = True
except Exception as e:
    st.error(f"Error loading KKU logo: {str(e)}")
    st.info("Please ensure the KKU logo file is available in your app directory.")
    st.session_state['logo_displayed'] = False

# --------------------------#
# --- App Header         ---#
# --------------------------#
st.markdown('<div class="main-header">🏠 Real Estate Price Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">This app predicts real estate prices based on property features!</div>', unsafe_allow_html=True)

# --------------------------#
# --- Supabase Connection ---#
# --------------------------#
@st.cache_resource
def init_connection():
    """
    Establishes a connection to the Supabase database.
    """
    supabase_url = "https://imdnhiwyfgjdgextvrkj.supabase.co"
    supabase_key = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
                    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0."
                    "9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk")
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Failed to connect to Supabase: {str(e)}")
        return None

# --------------------------#
# --- Load Reference Data ---#
# --------------------------#
@st.cache_data(ttl=600)
def load_reference_data():
    """
    Loads reference tables for neighborhoods, property types, and classifications
    from Supabase.
    """
    supabase = init_connection()
    if not supabase:
        return {}, {}, {}
    
    def load_table(table_names, id_fields, name_fields):
        """
        Attempt to load table using possible field names.
        Returns a dictionary mapping ID to name.
        """
        for table_name in table_names:
            try:
                response = supabase.table(table_name).select('*').execute()
                if response.data:
                    # Check available field names in the first item
                    id_field = next((f for f in id_fields if f in response.data[0]), None)
                    name_field = next((f for f in name_fields if f in response.data[0]), None)
                    if id_field and name_field:
                        return {item[id_field]: item[name_field] for item in response.data}
            except Exception:
                continue
        return {}
    
    neighborhood_dict = load_table(['neighborhoods'], ['neighborhood_id', 'id'], ['neighborhood_name', 'name'])
    property_type_dict = load_table(['property_type', 'property_types'], ['property_type_id', 'id'], ['property_type_name', 'name'])
    classification_dict = load_table(['property_classifications', 'classifications', 'property_classification'], ['classification_id', 'id'], ['classification_name', 'name'])
    
    return neighborhood_dict, property_type_dict, classification_dict

# --------------------------#
# --- Load Property Data  ---#
# --------------------------#
@st.cache_data(ttl=600)
def load_data():
    """
    Loads property data from the Supabase 'properties' table, converts numeric
    fields, and maps reference IDs to names.
    """
    supabase = init_connection()
    if not supabase:
        return pd.DataFrame()
    
    try:
        response = supabase.table('properties').select('*').execute()
        df = pd.DataFrame(response.data)
        if df.empty:
            raise ValueError("No data returned from database")
        
        # Determine columns for IDs
        neighborhood_id_col = next((col for col in df.columns if col in ['neighborhood_id', 'neighborhood']), None)
        property_type_id_col = next((col for col in df.columns if col in ['property_type_id', 'property_type']), None)
        classification_id_col = next((col for col in df.columns if col in ['classification_id', 'classification']), None)
        
        missing_columns = []
        for col, name in zip(
            [neighborhood_id_col, property_type_id_col, classification_id_col, 'area', 'price'], 
            ['neighborhood_id', 'property_type_id', 'classification_id', 'area', 'price']
        ):
            if col is None or col not in df.columns:
                missing_columns.append(name)
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Ensure numeric columns are cast properly
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df = df.dropna(subset=['price', 'area'])
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
        
        # Map reference data
        neighborhood_dict, property_type_dict, classification_dict = load_reference_data()
        df['neighborhood_name'] = (df[neighborhood_id_col]
                                   .map(neighborhood_dict)
                                   .fillna(df[neighborhood_id_col].astype(str) if neighborhood_id_col else 'Unknown'))
        df['property_type_name'] = (df[property_type_id_col]
                                    .map(property_type_dict)
                                    .fillna(df[property_type_id_col].astype(str) if property_type_id_col else 'Unknown'))
        df['classification_name'] = (df[classification_id_col]
                                     .map(classification_dict)
                                     .fillna(df[classification_id_col].astype(str) if classification_id_col else 'Unknown'))
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

# --------------------------#
# --- Data Loading        ---#
# --------------------------#
df = load_data()
if not df.empty:
    st.session_state['db_connected'] = True

    # ------------------------------#
    # --- Layout: Two Main Columns ---#
    # ------------------------------#
    col1, col2 = st.columns([1, 2])
    
    # --------------------------#
    # --- Sidebar: Input Form ---#
    # --------------------------#
    with st.sidebar:
        st.markdown('<div class="sub-header">Enter Property Details</div>', unsafe_allow_html=True)
        neighborhood = st.selectbox("Neighborhood", sorted(df['neighborhood_name'].unique()))
        classification = st.selectbox("Classification", sorted(df['classification_name'].unique()))
        property_type = st.selectbox("Property Type", sorted(df['property_type_name'].unique()))
        
        # Configure area slider using dataset limits
        area_min = float(df['area'].min())
        area_max = 1500.0
        default_area = min(float(df['area'].median()), area_max)
        st.markdown("### Area (m²)")
        area = st.slider("", min_value=area_min, max_value=area_max,
                         value=default_area, format="%.2f m²")
        st.markdown("<br>", unsafe_allow_html=True)
        calculate_button = st.button("Calculate Price Prediction", use_container_width=True)

    # --------------------------#
    # --- Model Training      ---#
    # --------------------------#
    @st.cache_resource
    def train_model(data):
        """
        Trains a RandomForestRegressor on the one-hot encoded property data.
        Returns the trained model and the feature column names.
        """
        try:
            X = pd.get_dummies(data[['neighborhood_name', 'classification_name',
                                     'property_type_name', 'area']])
            y = data['price']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            return None, None

    model, feature_columns = train_model(df)
    
    if model and feature_columns:
        # Create input dataframe based on sidebar selection
        input_df = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': classification,
            'property_type_name': property_type,
            'area': area
        }])
        # One-hot encode input data
        input_processed = pd.get_dummies(input_df)
        # Ensure all expected features are present
        for col in feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[feature_columns]
        
        try:
            prediction = model.predict(input_processed)[0]
            # --------------------------#
            # --- Prediction Display ---#
            # --------------------------#
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.5rem; color: #6B7280;">Estimated Property Price</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 3rem; font-weight: bold; color: #1E3A8A; margin: 1rem 0;">${prediction:,.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.875rem; color: #6B7280;">Based on property attributes and market data</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed Property Summary
            details_html = f"""
            <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);">
                <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; color: #1E3A8A;">Property Details</div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">Neighborhood</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{neighborhood}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">Classification</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{classification}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">Property Type</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{property_type}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; color: #6B7280;">Area</td>
                        <td style="padding: 0.5rem; font-weight: 500;">{area:.2f} m²</td>
                    </tr>
                </table>
            </div>
            """
            st.markdown(details_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    # --------------------------#
    # --- Market Analysis Tabs ---#
    # --------------------------#
    st.markdown('<div class="sub-header">Market Analysis</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Area vs Price", "Model Performance"])
    
    # Tab 1: Price Distribution Histogram
    with tab1:
        try:
            fig = px.histogram(
                df, x='price', 
                title='Price Distribution in the Market',
                labels={'price': 'Price ($)', 'count': 'Number of Properties'},
                color_discrete_sequence=['#3B82F6']
            )
            fig.update_layout(
                title_font_size=20,
                plot_bgcolor='white',
                paper_bgcolor='white',
                bargap=0.1,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error in Price Distribution: {str(e)}")
    
    # Tab 2: Area vs Price Scatter Plot with Trendline
    with tab2:
        try:
            trendline_arg = "ols" if STATS_MODELS_AVAILABLE else None
            trendline_note = "Trendline: OLS (statsmodels installed)" if STATS_MODELS_AVAILABLE else "Trendline: Not available (statsmodels not installed)"
            fig = px.scatter(
                df,
                x='area',
                y='price',
                color='neighborhood_name',
                title='Area vs Price by Neighborhood',
                labels={'area': 'Area (m²)', 'price': 'Price ($)', 'neighborhood_name': 'Neighborhood'},
                hover_data=['classification_name', 'property_type_name'],
                trendline=trendline_arg,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(
                title_font_size=20,
                legend_title_font_size=14,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                annotations=[{
                    'text': trendline_note,
                    'x': 0.5, 'y': -0.15,
                    'xref': 'paper', 'yref': 'paper',
                    'showarrow': False, 'font': {'color': "gray", 'size': 12}
                }]
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error in Area vs Price: {str(e)}")
    
    # Tab 3: Model Performance Evaluation
    with tab3:
        try:
            # Prepare training features for performance evaluation
            X_train = pd.get_dummies(df[['neighborhood_name', 'classification_name', 'property_type_name', 'area']])
            for col in feature_columns:
                if col not in X_train.columns:
                    X_train[col] = 0
            X_train = X_train[feature_columns]
            y_actual = df['price']
            y_pred = model.predict(X_train)
            
            performance_fig = px.scatter(
                x=y_actual,
                y=y_pred,
                labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                title='Model Performance: Actual vs Predicted Prices',
                color_discrete_sequence=['#3B82F6']
            )
            performance_fig.add_shape(
                type='line',
                x0=y_actual.min(), y0=y_actual.min(),
                x1=y_actual.max(), y1=y_actual.max(),
                line=dict(color='red', dash='dash')
            )
            performance_fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                title_font_size=20
            )
            st.plotly_chart(performance_fig, use_container_width=True)
            
            # Calculate and display RMSE
            rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
            st.markdown(f"<div style='font-size:1.1rem; color: #374151;'>Model RMSE: <strong>${rmse:,.2f}</strong></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Model performance visualization error: {str(e)}")
    
    # --------------------------#
    # --- Footer              ---#
    # --------------------------#
    st.markdown("""
    <div style="margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>Real Estate Price Prediction App | Powered by Machine Learning</p>
        <p>Data is updated daily from our real estate database</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to load data from Supabase. Please check your database connection and table structure.")
