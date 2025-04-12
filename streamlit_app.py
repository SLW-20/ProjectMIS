import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# إعداد اتصال Supabase
SUPABASE_URL = "https://imdnhiwyfgjdgextvrkj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0.9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# تحميل البيانات من Supabase
@st.cache_data(ttl=3600)
def load_data():
    try:
        # جلب البيانات من الجداول الأربعة
        neighborhoods = supabase.table('neighborhoods').select('*').execute()
        properties = supabase.table('properties').select('*').execute()
        classifications = supabase.table('property_classifications').select('*').execute()
        property_types = supabase.table('property_type').select('*').execute()
        
        # تحويل إلى DataFrames
        df_neighborhoods = pd.DataFrame(neighborhoods.data)
        df_properties = pd.DataFrame(properties.data)
        df_classifications = pd.DataFrame(classifications.data)
        df_types = pd.DataFrame(property_types.data)
        
        # دمج البيانات (يمكن تعديل هذا الجزء حسب علاقات الجداول لديك)
        df = df_properties.merge(
            df_neighborhoods, 
            left_on='neighborhood_id', 
            right_on='id', 
            how='left'
        ).merge(
            df_classifications,
            left_on='classification_id',
            right_on='id',
            how='left'
        ).merge(
            df_types,
            left_on='property_type_id',
            right_on='id',
            how='left'
        )
        
        # تنظيف البيانات
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df = df.dropna(subset=['price', 'area'])
        
        # إعادة تسمية الأعمدة للواجهة
        df = df.rename(columns={
            'name_x': 'neighborhood_name',
            'name_y': 'classification_name',
            'name': 'property_type_name'
        })
        
        return df
    
    except Exception as e:
        st.error(f"خطأ في تحميل البيانات: {str(e)}")
        return pd.DataFrame()

# واجهة التطبيق
st.set_page_config(page_title="توقع أسعار العقارات", layout="wide")
st.title("🏠 تطبيق توقع أسعار العقارات")
st.info("هذا التطبيق يتنبأ بأسعار العقارات بناءً على خصائصها")

# تحميل البيانات
df = load_data()

if not df.empty:
    # قسم عرض البيانات
    with st.expander("عرض البيانات الخام"):
        st.dataframe(df.head())
    
    # قسم التنبؤ
    st.header("تنبؤ السعر")
    
    # عناصر التحكم في الشريط الجانبي
    with st.sidebar:
        st.header("إدخال خصائص العقار")
        neighborhood = st.selectbox("الحي", df['neighborhood_name'].unique())
        classification = st.selectbox("التصنيف", df['classification_name'].unique())
        property_type = st.selectbox("نوع العقار", df['property_type_name'].unique())
        area = st.slider("المساحة (م²)", float(df['area'].min()), float(df['area'].max()), float(df['area'].median()))
    
    # تدريب النموذج
    @st.cache_resource
    def train_model(data):
        try:
            X = pd.get_dummies(data[['neighborhood_name', 'classification_name', 'property_type_name', 'area']])
            y = data['price']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        except Exception as e:
            st.error(f"خطأ في تدريب النموذج: {str(e)}")
            return None, None
    
    model, features = train_model(df)
    
    if model:
        # إعداد بيانات الإدخال للتنبؤ
        input_data = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': classification,
            'property_type_name': property_type,
            'area': area
        }])
        
        # معالجة البيانات بنفس طريقة التدريب
        input_processed = pd.get_dummies(input_data)
        for col in features:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[features]
        
        # التنبؤ
        prediction = model.predict(input_processed)[0]
        st.success(f"## السعر المتوقع: {prediction:,.2f} ريال")
        
        # عرض أهمية الميزات
        with st.expander("أهمية الخصائص في التنبؤ"):
            importance = pd.DataFrame({
                'الخاصية': features,
                'الأهمية': model.feature_importances_
            }).sort_values('الأهمية', ascending=False)
            
            fig = px.bar(importance.head(10), x='الأهمية', y='الخاصية', orientation='h')
            st.plotly_chart(fig)
    
    # قسم العقارات المشابهة
    with st.expander("عقارات مشابهة"):
        similar = df[
            (df['neighborhood_name'] == neighborhood) & 
            (df['classification_name'] == classification) & 
            (df['property_type_name'] == property_type)
        ]
        
        if not similar.empty:
            st.dataframe(similar[['neighborhood_name', 'classification_name', 'property_type_name', 'area', 'price']])
            fig = px.scatter(similar, x='area', y='price', trendline="ols")
            st.plotly_chart(fig)
        else:
            st.warning("لا توجد عقارات مشابهة في قاعدة البيانات")
else:
    st.error("فشل تحميل البيانات. يرجى التحقق من اتصال قاعدة البيانات.")
