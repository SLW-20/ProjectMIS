import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Supabase Connection Setup
SUPABASE_URL = "https://imdnhiwyfgjdgextvrkj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0.9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load data from Supabase
@st.cache_data(ttl=3600)
def load_data():
    try:
        # Load properties table (main data)
        response = supabase.table('properties').select('*').execute()
        df = pd.DataFrame(response.data)
        
        # Load reference tables
        neighborhoods = pd.DataFrame(supabase.table('neighborhoods').select('*').execute().data)
        classifications = pd.DataFrame(supabase.table('property_classifications').select('*').execute().data)
        types = pd.DataFrame(supabase.table('property_type').select('*').execute().data)
        
        # Merge data
        df = df.merge(neighborhoods, left_on='neighborhood_id', right_on='id', how='left')
        df = df.merge(classifications, left_on='classification_id', right_on='id', how='left')
        df = df.merge(types, left_on='property_type_id', right_on='id', how='left')
        
        # Clean data
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df = df.dropna(subset=['price', 'area'])
        
        # Rename columns for display
        df = df.rename(columns={
            'name_x': 'neighborhood',
            'name_y': 'classification',
            'name': 'property_type'
        })
        
        return df[['neighborhood', 'classification', 'property_type', 'area', 'price']]
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# App Interface
st.set_page_config(page_title="Real Estate Predictor", layout="wide")
st.title("🏠 Real Estate Price Prediction")
st.info("Predict property prices based on features")

# Load data
df = load_data()

if not df.empty:
    # Data Preview
    with st.expander("Data Preview"):
        st.dataframe(df.head())
    
    # Input Section (Now clearly visible in main panel)
    st.sidebar.header("Input Features")
    
    neighborhood = st.sidebar.selectbox("Neighborhood", df['neighborhood'].unique())
    classification = st.sidebar.selectbox("Classification", df['classification'].unique())
    property_type = st.sidebar.selectbox("Property Type", df['property_type'].unique())
    
    # Slider for area input (now more prominent)
    area = st.sidebar.slider(
        "Area (sqm)",
        min_value=float(df['area'].min()),
        max_value=float(df['area'].max()),
        value=float(df['area'].median()),
        step=10.0
    )
    
    # Train Model
    @st.cache_resource
    def train_model(data):
        try:
            X = pd.get_dummies(data[['neighborhood', 'classification', 'property_type', 'area']])
            y = data['price']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        except Exception as e:
            st.error(f"Model training failed: {str(e)}")
            return None, None
    
    model, features = train_model(df)
    
    if model:
        # Prepare input
        input_data = pd.DataFrame([{
            'neighborhood': neighborhood,
            'classification': classification,
            'property_type': property_type,
            'area': area
        }])
        
        input_processed = pd.get_dummies(input_data)
        for col in features:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[features]
        
        # Prediction
        prediction = model.predict(input_processed)[0]
        st.success(f"## Predicted Price: ${prediction:,.2f}")
        
        # Feature Importance
        with st.expander("Feature Importance"):
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance.head(10), x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig)
    
    # Similar Properties
    with st.expander("Similar Properties"):
        similar = df[
            (df['neighborhood'] == neighborhood) & 
            (df['classification'] == classification) & 
            (df['property_type'] == property_type)
        ]
        
        if not similar.empty:
            st.dataframe(similar)
            fig = px.scatter(similar, x='area', y='price', trendline="ols")
            st.plotly_chart(fig)
        else:
            st.warning("No similar properties found")
else:
    st.error("Failed to load data. Please check your database connection.")
