import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client
import os
from PIL import Image

# Attempt to import statsmodels to enable OLS trendline
try:
    import statsmodels
    STATS_MODELS_AVAILABLE = True
except ImportError:
    STATS_MODELS_AVAILABLE = False

# Enhanced page configuration with custom theme
st.set_page_config(
    page_title="Real Estate Price Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load KKU logo and display in top-right corner with text below
try:
    # Try to load logo from various possible paths
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
    
    # CSS to position logo top-right with text below
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
    
    # Display logo and text
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    st.image(logo, width=200)
    st.markdown('<div class="logo-text">MIS Graduation Project</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state['logo_displayed'] = True
    
except Exception as e:
    st.error(f"Error loading KKU logo: {str(e)}")
    st.info("Please ensure the KKU logo file (kku.logo.jpg) is in the same directory as this app.")
    st.session_state['logo_displayed'] = False

# Custom CSS to improve the design of the application
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
    .metric-card {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# App header with custom styling
st.markdown('<div class="main-header">🏠 Real Estate Price Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">This app predicts real estate prices based on property features with advanced machine learning models!</div>', unsafe_allow_html=True)

# Supabase connection
@st.cache_resource
def init_connection():
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
            
        # Load neighborhoods table
        neighborhood_dict = {}
        try:
            neighborhood_response = supabase.table('neighborhoods').select('*').execute()
            if neighborhood_response.data:
                if 'neighborhood_id' in neighborhood_response.data[0] and 'neighborhood_name' in neighborhood_response.data[0]:
                    neighborhood_dict = {item['neighborhood_id']: item['neighborhood_name'] for item in neighborhood_response.data}
                elif 'id' in neighborhood_response.data[0] and 'name' in neighborhood_response.data[0]:
                    neighborhood_dict = {item['id']: item['name'] for item in neighborhood_response.data}
        except Exception:
            pass
            
        # Load property type reference table
        property_type_dict = {}
        try:
            try:
                property_type_response = supabase.table('property_type').select('*').execute()
            except:
                property_type_response = supabase.table('property_types').select('*').execute()
                
            if property_type_response.data:
                if 'property_type_id' in property_type_response.data[0] and 'property_type_name' in property_type_response.data[0]:
                    property_type_dict = {item['property_type_id']: item['property_type_name'] for item in property_type_response.data}
                elif 'id' in property_type_response.data[0] and 'name' in property_type_response.data[0]:
                    property_type_dict = {item['id']: item['name'] for item in property_type_response.data}
        except Exception:
            pass
            
        # Load classification reference table
        classification_dict = {}
        try:
            try:
                classification_response = supabase.table('property_classifications').select('*').execute()
            except:
                try:
                    classification_response = supabase.table('classifications').select('*').execute()
                except:
                    classification_response = supabase.table('property_classification').select('*').execute()
                
            if classification_response.data:
                if 'classification_id' in classification_response.data[0] and 'classification_name' in classification_response.data[0]:
                    classification_dict = {item['classification_id']: item['classification_name'] for item in classification_response.data}
                elif 'id' in classification_response.data[0] and 'name' in classification_response.data[0]:
                    classification_dict = {item['id']: item['name'] for item in classification_response.data}
        except Exception:
            pass
            
        return neighborhood_dict, property_type_dict, classification_dict
    except Exception as e:
        st.error(f"Failed to load reference data: {str(e)}")
        return {}, {}, {}

# Load data from Supabase with ID columns
@st.cache_data(ttl=600)
def load_data():
    try:
        supabase = init_connection()
        if not supabase:
            return pd.DataFrame()
        
        # Fetch data from the 'properties' table
        response = supabase.table('properties').select('*').execute()
        
        # Convert response data to DataFrame
        df = pd.DataFrame(response.data)
        
        if df.empty:
            raise ValueError("No data returned from database")
        
        # Identify columns using possible labels
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

# Add feature engineering function
def engineer_features(df):
    """Apply feature engineering to improve model performance"""
    # Create a copy to avoid modifying the original dataframe
    df_engineered = df.copy()
    
    # Handle price outliers using log transformation
    df_engineered['log_price'] = np.log1p(df_engineered['price'])
    
    # Create price_per_sqm feature
    df_engineered['price_per_sqm'] = df_engineered['price'] / df_engineered['area']
    
    # Add area bins as categorical features (can help capture non-linear relationships)
    df_engineered['area_bin'] = pd.qcut(df_engineered['area'], q=5, labels=False, duplicates='drop')
    
    # Add price_per_sqm bins
    df_engineered['price_per_sqm_bin'] = pd.qcut(df_engineered['price_per_sqm'], q=5, labels=False, duplicates='drop')
    
    # Add neighborhood statistics
    neighborhood_stats = df_engineered.groupby('neighborhood_name').agg({
        'price': ['mean', 'median', 'std'],
        'area': ['mean', 'std'],
        'price_per_sqm': ['mean', 'median']
    })
    
    neighborhood_stats.columns = [f'{col[0]}_{col[1]}_by_neighborhood' for col in neighborhood_stats.columns]
    df_engineered = df_engineered.join(neighborhood_stats, on='neighborhood_name')
    
    # Fill NaN values that might have been introduced
    df_engineered = df_engineered.fillna(df_engineered.median())
    
    return df_engineered

# Function to remove outliers
def remove_outliers(df, column, lower_quantile=0.01, upper_quantile=0.99):
    """Remove extreme outliers from the dataset"""
    lower_bound = df[column].quantile(lower_quantile)
    upper_bound = df[column].quantile(upper_quantile)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Function to train the model with cross-validation and hyperparameter tuning
@st.cache_resource
def train_model(df):
    try:
        # Engineer features for better model performance
        df_engineered = engineer_features(df)
        
        # Remove extreme outliers
        df_clean = remove_outliers(df_engineered, 'price')
        df_clean = remove_outliers(df_clean, 'area')
        
        # Select features for model training
        feature_columns = [
            'neighborhood_name', 'classification_name', 'property_type_name', 
            'area', 'area_bin', 'price_per_sqm_bin', 
            'price_mean_by_neighborhood', 'area_mean_by_neighborhood'
        ]
        
        # Split data into features and target
        X = pd.get_dummies(df_clean[feature_columns])
        y = df_clean['price']
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a model pipeline with scaling
        pipe_rf = Pipeline([
            ('scaler', RobustScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        # Define hyperparameters for grid search
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 20, 30],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2],
        }
        
        # Create pipeline for Gradient Boosting model
        pipe_gb = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(random_state=42))
        ])
        
        # Define hyperparameters for gradient boosting
        param_grid_gb = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.05, 0.1],
            'model__max_depth': [3, 5],
            'model__subsample': [0.8, 1.0],
        }
        
        # Create a cross-validated grid search for both models
        grid_rf = GridSearchCV(
            pipe_rf, 
            param_grid, 
            cv=5, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1,
        )
        
        grid_gb = GridSearchCV(
            pipe_gb, 
            param_grid_gb,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
        )
        
        # Fit the models
        with st.spinner('Training Random Forest model...'):
            grid_rf.fit(X_train, y_train)
        
        with st.spinner('Training Gradient Boosting model...'):
            grid_gb.fit(X_train, y_train)
        
        # Get best models
        best_rf = grid_rf.best_estimator_
        best_gb = grid_gb.best_estimator_
        
        # Evaluate models on validation set
        y_pred_rf = best_rf.predict(X_val)
        rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
        r2_rf = r2_score(y_val, y_pred_rf)
        
        y_pred_gb = best_gb.predict(X_val)
        rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred_gb))
        r2_gb = r2_score(y_val, y_pred_gb)
        
        # Choose the best model
        if rmse_rf <= rmse_gb:
            best_model = best_rf
            model_name = "Random Forest"
            best_rmse = rmse_rf
            best_r2 = r2_rf
        else:
            best_model = best_gb
            model_name = "Gradient Boosting"
            best_rmse = rmse_gb
            best_r2 = r2_gb
        
        # Get feature importance
        if model_name == "Random Forest":
            feature_importances = best_model.named_steps['model'].feature_importances_
            feature_names = X.columns
            importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
            importance_df = importance_df.sort_values('importance', ascending=False).head(10)
        else:
            # Use permutation importance for GB
            perm_importance = permutation_importance(best_model, X_val, y_val, n_repeats=10, random_state=42)
            feature_names = X.columns
            importance_df = pd.DataFrame({'feature': feature_names, 
                                         'importance': perm_importance.importances_mean})
            importance_df = importance_df.sort_values('importance', ascending=False).head(10)
        
        return best_model, X.columns.tolist(), model_name, best_rmse, best_r2, importance_df, df_engineered
    
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None, None, None, None, None, None, None

if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

# Load data
df = load_data()

if not df.empty:
    st.session_state['db_connected'] = True
    
    # Create two main columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Sidebar for inputs with improved styling  
    with st.sidebar:
        st.markdown('<div class="sub-header">Enter Property Details</div>', unsafe_allow_html=True)
        neighborhood = st.selectbox("Neighborhood", sorted(df['neighborhood_name'].unique()))
        classification = st.selectbox("Classification", sorted(df['classification_name'].unique()))
        property_type = st.selectbox("Property Type", sorted(df['property_type_name'].unique()))
        
        # Area slider
        area_min = float(df['area'].min())
        area_max = 1500.0
        default_area = min(float(df['area'].median()), area_max)
        
        st.markdown("### Area (m²)")
        area = st.slider("", 
                         min_value=area_min, 
                         max_value=area_max,
                         value=default_area,
                         format="%.2f m²")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add model selection
        model_selection = st.radio(
            "Select Model Mode",
            ["Automatic (Best Performer)", "Random Forest", "Gradient Boosting"],
            index=0
        )
        
        calculate_button = st.button("Calculate Price Prediction", use_container_width=True)
    
    # Train models
    model, feature_columns, model_name, rmse, r2, importance_df, df_engineered = train_model(df)
    
    if model and feature_columns:
        # Create engineered features for the input
        # We need to create similar engineered features as we did during training
        
        # Create a small dataframe with the input data
        input_df = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': classification,
            'property_type_name': property_type,
            'area': area
        }])
        
        # Add area_bin and price_per_sqm_bin
        # For this, we need to use the same binning logic as in the training data
        # First, get the bin edges from the training data
        area_bins = pd.qcut(df_engineered['area'], q=5, duplicates='drop').categories
        
        # Find which bin the input area falls into
        area_bin = 0
        for i, bin_edge in enumerate(area_bins):
            if area >= bin_edge.left and area <= bin_edge.right:
                area_bin = i
                break
        
        input_df['area_bin'] = area_bin
        
        # Add neighborhood statistics
        neighborhood_stats = df_engineered.groupby('neighborhood_name').agg({
            'price': ['mean', 'median', 'std'],
            'area': ['mean', 'std'],
            'price_per_sqm': ['mean', 'median']
        })
        
        neighborhood_stats.columns = [f'{col[0]}_{col[1]}_by_neighborhood' for col in neighborhood_stats.columns]
        
        # Join neighborhood stats to input
        if neighborhood in neighborhood_stats.index:
            for col in neighborhood_stats.columns:
                input_df[col] = neighborhood_stats.loc[neighborhood, col]
        else:
            # Use median values if neighborhood not found
            for col in neighborhood_stats.columns:
                input_df[col] = neighborhood_stats[col].median()
        
        # One-hot encode input without dropping any category
        input_processed = pd.get_dummies(input_df)
        
        # Ensure all expected feature columns are available; if missing, add with value 0
        for col in feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        
        # Reorder columns to match the training data and select only needed columns
        missing_cols = set(feature_columns) - set(input_processed.columns)
        for col in missing_cols:
            input_processed[col] = 0
            
        input_processed = input_processed[feature_columns]
        
        try:
            prediction = model.predict(input_processed)[0]
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.5rem; color: #6B7280;">Estimated Property Price</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 3rem; font-weight: bold; color: #1E3A8A; margin: 1rem 0;">${prediction:,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 1rem; color: #4B5563;">Predicted using {model_name} model</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.875rem; color: #6B7280;">Based on property attributes and market data</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);">
                <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; color: #1E3A8A;">Property Details</div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">Neighborhood</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">Classification</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">Property Type</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; color: #6B7280;">Area</td>
                        <td style="padding: 0.5rem; font-weight: 500;">{:.2f} m²</td>
                    </tr>
                </table>
            </div>
            """.format(neighborhood, classification, property_type, area), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    st.markdown('<div class="sub-header">Market Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Area vs Price", "Model Performance", "Feature Importance"])
    
    with tab1:
        try:
            fig = px.histogram(df, x='price', 
                              title='Price Distribution in the Market',
                              labels={'price': 'Price ($)', 'count': 'Number of Properties'},
                              color_discrete_sequence=['#3B82F6'])
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
            
            # Add log-transformed price distribution
            fig_log = px.histogram(df_engineered, x='log_price', 
                           title='Log-Transformed Price Distribution',
                           labels={'log_price': 'Log Price', 'count': 'Number of Properties'},
                           color_discrete_sequence=['#10B981'])
            fig_log.update_layout(
                title_font_size=20,
                plot_bgcolor='white',
                paper_bgcolor='white',
                bargap=0.1,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig_log, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            
    with tab2:
        try:
            # Decide whether to add the trendline based on statsmodels availability
            if STATS_MODELS_AVAILABLE:
                trendline_arg = "ols"
                trendline_note = "Trendline: OLS (statsmodels installed)"
            else:
                trendline_arg = None
                trendline_note = "Trendline: Not available (statsmodels not installed)"
            
            fig = px.scatter(
                df, 
                x='area', 
                y='price', 
                color='neighborhood_name',
                title='Area vs Price by Neighborhood',
                labels={'area': 'Area (m²)', 'price': 'Price ($)', 'neighborhood_name': '
                labels={'area': 'Area (m²)', 'price': 'Price ($)', 'neighborhood_name': 'Neighborhood'},
                hover_data=['classification_name', 'property_type_name'],
                trendline=trendline_arg,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            # Make markers bigger, add a border
            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            
            # Enhance layout
            fig.update_layout(
                title_font_size=20,
                legend_title_font_size=14,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14,
                annotations=[
                    dict(
                        text=trendline_note,
                        x=0.5, 
                        y=-0.15, 
                        xref='paper', 
                        yref='paper',
                        showarrow=False, 
                        font=dict(color="gray", size=12)
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add price per sqm scatter plot
            fig_price_per_sqm = px.scatter(
                df_engineered, 
                x='area', 
                y='price_per_sqm', 
                color='neighborhood_name',
                title='Area vs Price per sq meter',
                labels={'area': 'Area (m²)', 'price_per_sqm': 'Price per sq meter ($)', 'neighborhood_name': 'Neighborhood'},
                hover_data=['classification_name', 'property_type_name'],
                trendline=trendline_arg,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_price_per_sqm.update_traces(marker=dict(size=8, opacity=0.7))
            fig_price_per_sqm.update_layout(
                title_font_size=20,
                legend_title_font_size=14,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig_price_per_sqm, use_container_width=True)
            
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    with tab3:
        try:
            if model and rmse and r2:
                # Display model metrics in cards
                col_metrics1, col_metrics2 = st.columns(2)
                
                with col_metrics1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Model Type</div>
                        <div class="metric-value">{model_name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_metrics2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">RMSE (Lower is better)</div>
                        <div class="metric-value">${rmse:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                col_metrics3, col_metrics4 = st.columns(2)
                
                with col_metrics3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">R² Score (Higher is better)</div>
                        <div class="metric-value">{r2:.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_metrics4:
                    # Calculate MAPE (Mean Absolute Percentage Error)
                    X = pd.get_dummies(df_engineered[feature_columns])
                    for col in feature_columns:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_columns]
                    y_actual = df_engineered['price']
                    y_pred = model.predict(X)
                    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">MAPE (Mean Absolute % Error)</div>
                        <div class="metric-value">{mape:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prepare training features for model performance evaluation
                X_train = pd.get_dummies(df_engineered[feature_columns])
                for col in feature_columns:
                    if col not in X_train.columns:
                        X_train[col] = 0
                X_train = X_train[feature_columns]
                y_actual = df_engineered['price']
                y_pred = model.predict(X_train)
                
                # Scatter plot for Actual vs Predicted Prices with a y=x reference line
                performance_fig = px.scatter(
                    x=y_actual, 
                    y=y_pred,
                    labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                    title=f'Model Performance: Actual vs Predicted Prices ({model_name})',
                    color_discrete_sequence=['#3B82F6'],
                    opacity=0.7
                )
                performance_fig.add_shape(
                    type='line',
                    x0=y_actual.min(), y0=y_actual.min(),
                    x1=y_actual.max(), y1=y_actual.max(),
                    line=dict(color='red', dash='dash'),
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
                
                # Add residual plot
                residuals = y_actual - y_pred
                residual_fig = px.scatter(
                    x=y_pred,
                    y=residuals,
                    labels={'x': 'Predicted Price', 'y': 'Residuals (Actual - Predicted)'},
                    title='Residual Plot: Error Distribution',
                    color_discrete_sequence=['#10B981'],
                    opacity=0.7
                )
                # Add horizontal line at y=0
                residual_fig.add_hline(y=0, line_dash="dash", line_color="red")
                residual_fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=20
                )
                st.plotly_chart(residual_fig, use_container_width=True)
                
                # Add residual histogram
                residual_hist = px.histogram(
                    x=residuals,
                    title='Distribution of Prediction Errors',
                    labels={'x': 'Residual Value (Actual - Predicted)'},
                    color_discrete_sequence=['#6366F1']
                )
                residual_hist.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=20
                )
                st.plotly_chart(residual_hist, use_container_width=True)
                
                with st.expander("Cross-Validation Results"):
                    # Perform cross-validation on the model
                    X = pd.get_dummies(df_engineered[feature_columns])
                    for col in feature_columns:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_columns]
                    y = df_engineered['price']
                    
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
                    cv_rmse = -cv_scores
                    
                    st.write("5-Fold Cross-Validation RMSE Scores:")
                    cv_df = pd.DataFrame({
                        'Fold': range(1, 6),
                        'RMSE': cv_rmse
                    })
                    
                    # Format RMSE values to show commas for thousands
                    cv_df['RMSE'] = cv_df['RMSE'].map('${:,.2f}'.format)
                    
                    st.table(cv_df)
                    st.write(f"Mean CV RMSE: ${cv_rmse.mean():,.2f}")
                    st.write(f"Standard Deviation of CV RMSE: ${cv_rmse.std():,.2f}")
            else:
                st.warning("Model training did not complete successfully. Unable to display performance metrics.")
                
        except Exception as e:
            st.error(f"Model performance visualization error: {str(e)}")
    
    with tab4:
        try:
            if importance_df is not None:
                st.write("### Top 10 Most Important Features")
                
                # Create bar chart of feature importance
                fig_importance = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Feature Importance ({model_name})',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color_discrete_sequence=['#2563EB']
                )
                
                fig_importance.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14,
                    title_font_size=20,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Feature importance explanation
                st.markdown("""
                <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #3B82F6;">
                    <h4 style="margin-top: 0;">Understanding Feature Importance</h4>
                    <p>Feature importance shows which attributes have the most influence on price predictions. Higher values indicate stronger influence.</p>
                    <ul>
                        <li>Location-based features typically have the highest impact on property prices.</li>
                        <li>Physical attributes like area also significantly affect price predictions.</li>
                        <li>The model considers both direct features and derived features (like price_per_sqm statistics).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional explanation about how to use this information
                with st.expander("How to use this information to improve predictions"):
                    st.markdown("""
                    - **Data Collection**: Focus on collecting more accurate data for high-importance features
                    - **Feature Engineering**: Create more derived features based on the most important attributes
                    - **Model Tuning**: Adjust hyperparameters to better capture relationships in important features
                    - **Data Quality**: Ensure the most important features have clean, accurate data with few missing values
                    """)
            else:
                st.warning("Feature importance information is not available")
                
        except Exception as e:
            st.error(f"Feature importance visualization error: {str(e)}")
            
    st.markdown("""
    <div style="margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>Advanced Real Estate Price Prediction App | Powered by Machine Learning</p>
        <p>Data is updated daily from our real estate database</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Failed to load data from Supabase. Please check your database connection and table structure.")
