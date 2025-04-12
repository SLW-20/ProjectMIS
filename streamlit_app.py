import streamlit as st
import pandas as pd
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.express as px
import numpy as np

# تكوين الصفحة
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# عنوان التطبيق
st.title('🏠 تطبيق توقع أسعار العقارات')
st.info('هذا التطبيق يتنبأ بأسعار العقارات بناءً على خصائص العقار!')

# بيانات الاتصال بـ Supabase - يجب تخزينها بشكل آمن
@st.cache_resource
def init_supabase():
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://imdnhiwyfgjdgextvrkj.supabase.co")
    SUPABASE_API_KEY = st.secrets.get("SUPABASE_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0.9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk")
    return create_client(SUPABASE_URL, SUPABASE_API_KEY)

supabase = init_supabase()

# أسماء الجداول
TABLE_NAME = "real_estate_data"  # جدول البيانات العقارية

# تحميل البيانات من Supabase
@st.cache_data(ttl=3600)  # تحديث البيانات كل ساعة
def load_data():
    try:
        response = supabase.table(TABLE_NAME).select("*").execute()
        df = pd.DataFrame(response.data)
        
        # التحقق من الأعمدة المطلوبة
        required_columns = ['neighborhood_name', 'classification_name', 
                           'property_type_name', 'area', 'price']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"الأعمدة المطلوبة مفقودة: {', '.join(missing_cols)}")
            return pd.DataFrame()

        # تنظيف البيانات
        df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        df['area'] = pd.to_numeric(df['area'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
        # إزالة القيم غير الصالحة
        df = df.dropna(subset=['price', 'area'])
        df = df[(df['price'] > 0) & (df['area'] > 0)]
        
        if df.empty:
            st.error("لا توجد بيانات صالحة بعد التنظيف")
            return pd.DataFrame()
            
        return df
    except Exception as e:
        st.error(f"فشل تحميل البيانات: {str(e)}")
        return pd.DataFrame()

# تحميل البيانات
df = load_data()

if not df.empty:
    st.success("تم تحميل البيانات بنجاح!")
    
    # نظرة عامة على البيانات
    with st.expander("نظرة عامة على البيانات"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("### عينة من البيانات الخام")
            st.dataframe(df.head(), height=300)
        with col2:
            st.write("### إحصائيات البيانات")
            st.dataframe(df.describe(), height=300)

        # تصورات البيانات
        st.write("### تصورات البيانات")
        try:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='price', title='توزيع الأسعار')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(df, x='area', y='price', color='neighborhood_name',
                               title='المساحة مقابل السعر حسب الحي')
                st.plotly_chart(fig, use_container_width=True)
                
            # مخطط إضافي
            fig = px.box(df, x='property_type_name', y='price', 
                        title='توزيع الأسعار حسب نوع العقار')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"خطأ في التصور: {str(e)}")

    # تدريب النموذج
    @st.cache_resource
    def train_model(data):
        try:
            # تحضير البيانات
            X = pd.get_dummies(data[['neighborhood_name', 'classification_name',
                                   'property_type_name', 'area']], drop_first=True)
            y = data['price']
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # تدريب النموذج
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X_train, y_train)
            
            # تقييم النموذج
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            mae = mean_absolute_error(y_test, model.predict(X_test))
            
            return model, X.columns.tolist(), {'train_score': train_score, 
                                             'test_score': test_score, 
                                             'mae': mae}
        except Exception as e:
            st.error(f"فشل تدريب النموذج: {str(e)}")
            return None, None, None

    model, feature_columns, model_metrics = train_model(df)

    # واجهة المستخدم للتنبؤ
    st.sidebar.header("أدخل تفاصيل العقار")
    
    # تحديد القيم المتاحة بناءً على البيانات
    neighborhoods = sorted(df['neighborhood_name'].unique())
    classifications = sorted(df['classification_name'].unique())
    property_types = sorted(df['property_type_name'].unique())
    
    neighborhood = st.sidebar.selectbox("الحي", neighborhoods)
    classification = st.sidebar.selectbox("التصنيف", classifications)
    property_type = st.sidebar.selectbox("نوع العقار", property_types)
    
    # تحديد المساحة
    area_min = float(df['area'].min())
    area_max = float(min(df['area'].max(), 2000))  # حد أقصى 2000 متر مربع
    default_area = float(df['area'].median())
    area = st.sidebar.slider("المساحة (م²)", 
                            min_value=area_min, 
                            max_value=area_max,
                            value=default_area)

    if model and feature_columns:
        # إعداد بيانات الإدخال
        input_df = pd.DataFrame([{
            'neighborhood_name': neighborhood,
            'classification_name': classification,
            'property_type_name': property_type,
            'area': area
        }])

        # معالجة البيانات
        input_processed = pd.get_dummies(input_df, drop_first=True)
        
        # محاذاة مع ميزات التدريب
        for col in feature_columns:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[feature_columns]

        # التنبؤ بالسعر
        try:
            prediction = model.predict(input_processed)[0]
            
            # عرض النتائج
            st.markdown(f"## السعر المتوقع: ${prediction:,.2f}")
            
            # معلومات عن أداء النموذج
            with st.expander("أداء النموذج"):
                st.write(f"**دقة التدريب:** {model_metrics['train_score']:.2%}")
                st.write(f"**دقة الاختبار:** {model_metrics['test_score']:.2%}")
                st.write(f"**متوسط الخطأ المطلق:** ${model_metrics['mae']:,.2f}")
                
                # أهمية الميزات
                st.write("### أهمية الخصائص في التنبؤ")
                importance_df = pd.DataFrame({
                    'الخاصية': feature_columns,
                    'الأهمية': model.feature_importances_
                }).sort_values('الأهمية', ascending=False)
                
                fig = px.bar(importance_df.head(10), x='الأهمية', y='الخاصية', 
                             orientation='h', title='أهم 10 خصائص مؤثرة')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"فشل في التنبؤ: {str(e)}")

    # عرض عقارات مشابهة
    with st.expander("عقارات مشابهة"):
        similar = df[
            (df['neighborhood_name'] == neighborhood) &
            (df['classification_name'] == classification) &
            (df['property_type_name'] == property_type)
        ]
        
        if not similar.empty:
            st.write(f"تم العثور على {len(similar)} عقار مشابه:")
            
            # حساب متوسط السعر للمساحة المحددة
            similar['price_per_sqm'] = similar['price'] / similar['area']
            avg_price_per_sqm = similar['price_per_sqm'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("متوسط السعر للمتر المربع", f"${avg_price_per_sqm:,.2f}")
            with col2:
                st.metric("السعر المتوقع للمساحة المحددة", f"${avg_price_per_sqm * area:,.2f}")
            
            st.dataframe(similar.sort_values('price', ascending=False))
            
            # تصور العقارات المشابهة
            fig = px.scatter(similar, x='area', y='price', 
                            hover_data=['classification_name', 'property_type_name'],
                            title='العقارات المشابهة: المساحة مقابل السعر')
            fig.add_vline(x=area, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("لا توجد عقارات مشابهة في قاعدة البيانات")

else:
    st.error("فشل تحميل البيانات. يرجى التحقق من مصدر البيانات والمحاولة مرة أخرى.")
