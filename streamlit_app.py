import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from supabase import create_client
import os
from PIL import Image

# Enhanced page configuration with custom theme
st.set_page_config(
    page_title="Real Estate Price Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحميل شعار جامعة الملك خالد وعرضه في الزاوية العلوية اليمنى مع النص تحته
try:
    # محاولة تحميل الشعار من الملف الأساسي
    if os.path.exists('kku.logo.jpg'):
        logo = Image.open('kku.logo.jpg')
    else:
        # قائمة بالمسارات المحتملة لاسم الصورة
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
    
    # تطبيق CSS لتثبيت الشعار في الزاوية العلوية اليمنى مع النص تحته
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
            font-size: 3rem;
            font-weight: 1000;
            color: #1E3A8A;
            margin-bottom: 1rem;
            text-align: right;
            direction: rtl;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # عرض الشعار والنص أسفله داخل نفس الحاوية
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
</style>
""", unsafe_allow_html=True)

# App header with custom styling
st.markdown('<div class="main-header">🏠 تطبيق توقع أسعار العقارات</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">هذا التطبيق يتوقع أسعار العقارات بناءً على خصائص العقار!</div>', unsafe_allow_html=True)

# باقي الكود الأصلي يبقى كما هو بدون تغيير
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
        
        # تحديد الأعمدة باستخدام تسميات محتملة
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

if 'db_connected' not in st.session_state:
    st.session_state['db_connected'] = False

# Load data
df = load_data()

if not df.empty:
    st.session_state['db_connected'] = True
    st.markdown('<div class="success-box">✅ تم تحميل البيانات بنجاح من Supabase!</div>', unsafe_allow_html=True)
    
    # Create two main columns for the layout
    col1, col2 = st.columns([1, 2])
    
    # Sidebar for inputs with improved styling
    with st.sidebar:
        st.markdown('<div class="sub-header">أدخل تفاصيل العقار</div>', unsafe_allow_html=True)
        neighborhood = st.selectbox("الحي", sorted(df['neighborhood_name'].unique()))
        classification = st.selectbox("التصنيف", sorted(df['classification_name'].unique()))
        property_type = st.selectbox("نوع العقار", sorted(df['property_type_name'].unique()))
        
        # Area slider
        area_min = float(df['area'].min())
        area_max = 1500.0
        default_area = min(float(df['area'].median()), area_max)
        
        st.markdown("### المساحة (م²)")
        area = st.slider("", 
                         min_value=area_min, 
                         max_value=area_max,
                         value=default_area,
                         format="%.2f م²")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        calculate_button = st.button("حساب السعر المتوقع", use_container_width=True)
    
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
        input_df = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': classification,
            'property_type_name': property_type,
            'area': area
        }])
        
        input_processed = pd.get_dummies(input_df, drop_first=True)
        
        for col in feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[feature_columns]
        
        try:
            prediction = model.predict(input_processed)[0]
            
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 1.5rem; color: #6B7280;">السعر المتوقع للعقار</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 3rem; font-weight: bold; color: #1E3A8A; margin: 1rem 0;">${prediction:,.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.875rem; color: #6B7280;">بناءً على خصائص العقار وبيانات السوق</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: #F8FAFC; padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 2rem; box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);">
                <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; color: #1E3A8A;">تفاصيل العقار</div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">الحي</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">التصنيف</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; color: #6B7280;">نوع العقار</td>
                        <td style="padding: 0.5rem; border-bottom: 1px solid #E5E7EB; font-weight: 500;">{}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.5rem; color: #6B7280;">المساحة</td>
                        <td style="padding: 0.5rem; font-weight: 500;">{:.2f} م²</td>
                    </tr>
                </table>
            </div>
            """.format(neighborhood, classification, property_type, area), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
    
    st.markdown('<div class="sub-header">تحليل السوق</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["توزيع الأسعار", "المساحة مقابل السعر", "أهمية الخصائص"])
    
    with tab1:
        try:
            fig = px.histogram(df, x='price', 
                              title='توزيع الأسعار في السوق',
                              labels={'price': 'السعر ($)', 'count': 'عدد العقارات'},
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
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
            
    with tab2:
        try:
            fig = px.scatter(df, x='area', y='price', 
                             color='neighborhood_name',
                             title='المساحة مقابل السعر حسب الحي',
                             labels={'area': 'المساحة (م²)', 'price': 'السعر ($)', 'neighborhood_name': 'الحي'},
                             hover_data=['classification_name', 'property_type_name'])
            fig.update_layout(
                title_font_size=20,
                legend_title_font_size=14,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title_font_size=14,
                yaxis_title_font_size=14
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    with tab3:
        try:
            if model and feature_columns:
                importance_df = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                             orientation='h',
                             title='أهمية الخصائص في توقع السعر',
                             labels={'Importance': 'درجة الأهمية', 'Feature': 'خاصية العقار'},
                             color_discrete_sequence=['#3B82F6'])
                fig.update_layout(
                    title_font_size=20,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title_font_size=14,
                    yaxis_title_font_size=14
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Feature importance visualization error: {str(e)}")
            
    st.markdown('<div class="sub-header">عقارات مشابهة</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.875rem; color: #6B7280; margin-bottom: 1rem;">عقارات في نفس الحي للمقارنة</div>', unsafe_allow_html=True)
    
    similar = df[df['neighborhood_name'] == neighborhood]
    if not similar.empty:
        st.dataframe(
            similar[['property_type_name', 'classification_name', 'area', 'price']].head(5),
            column_config={
                'property_type_name': 'نوع العقار',
                'classification_name': 'التصنيف',
                'area': st.column_config.NumberColumn('المساحة (م²)', format="%.2f"),
                'price': st.column_config.NumberColumn('السعر ($)', format="$%d")
            },
            use_container_width=True,
            hide_index=True
        )
        
        fig = px.scatter(similar, x='area', y='price', 
                         title=f'السعر مقابل المساحة في حي {neighborhood}',
                         labels={'area': 'المساحة (م²)', 'price': 'السعر ($)'},
                         hover_data=['classification_name', 'property_type_name'],
                         color_discrete_sequence=['#3B82F6'])
        fig.update_layout(
            title_font_size=18,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        fig.add_scatter(
            x=[area], 
            y=[prediction] if 'prediction' in locals() else [0],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='اختيارك',
            hoverinfo='name'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("لا توجد عقارات مشابهة في هذا الحي")

    st.markdown("""
    <div style="margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #E5E7EB; text-align: center; color: #6B7280; font-size: 0.875rem;">
        <p>تطبيق توقع أسعار العقارات | مدعوم بتعلم الآلة</p>
        <p>يتم تحديث البيانات يوميًا من قاعدة بيانات العقارات</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("فشل تحميل البيانات من Supabase. يرجى التحقق من اتصال قاعدة البيانات وهيكل الجدول")
