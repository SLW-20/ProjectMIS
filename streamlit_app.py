import streamlit as st
import pandas as pd
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# بيانات الاتصال بـ Supabase
SUPABASE_URL = "https://imdnhiwyfgjdgextvrkj.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0.9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk"

# إنشاء العميل Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# أسماء الجداول – ضيفها هنا يدويًا
table_names = ["students", "grades", "real_estate_data"]  # استبدلها باسم الجدول الذي يحتوي على البيانات العقارية

# Page config
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# App title
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices based on property features!')

# تحميل البيانات من Supabase
@st.cache_data
def load_data_from_supabase(table_name):
    try:
        # استعلام لجلب البيانات من الجدول المحدد
        response = supabase.table(table_name).select("*").execute()
        df = pd.DataFrame(response.data)
        
        # التحقق من الأعمدة المطلوبة
        required_columns = ['neighborhood_name', 'classification_name', 
                            'property_type_name', 'area', 'price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # تحويل السعر والمساحة إلى قيم عددية
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df['area'] = pd.to_numeric(df['area'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')

        # تنظيف البيانات
        df = df.dropna(subset=['price', 'area'])
        if df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        return df
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return pd.DataFrame()

# اختيار الجدول من Supabase
selected_table = st.selectbox("اختر جدول لعرض البيانات:", table_names)

# تحميل البيانات من الجدول المحدد
df = load_data_from_supabase(selected_table)

if not df.empty:
    st.success("Data loaded successfully!")
    
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
