import streamlit as st
import pandas as pd
from supabase import create_client
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# 1. إعداد اتصال Supabase بشكل آمن
@st.cache_resource
def init_supabase():
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"فشل الاتصال بقاعدة البيانات: {str(e)}")
        return None

supabase = init_supabase()

# 2. دالة محسنة لجلب البيانات مع التحقق من كل خطوة
@st.cache_data(ttl=3600)
def load_merged_data():
    if supabase is None:
        return pd.DataFrame()

    try:
        # جلب الجدول الرئيسي
        properties = supabase.table("properties").select("*").execute()
        df = pd.DataFrame(properties.data)
        
        if df.empty:
            st.warning("لا توجد بيانات في جدول العقارات")
            return pd.DataFrame()

        # الجداول المرجعية
        tables = {
            "neighborhoods": ("neighborhood_id", "name"),
            "property_classifications": ("classification_id", "name"),
            "property_type": ("property_type_id", "name")
        }

        for table, (id_col, name_col) in tables.items():
            try:
                ref_data = supabase.table(table).select("*").execute()
                ref_df = pd.DataFrame(ref_data.data)
                if not ref_df.empty:
                    df = df.merge(
                        ref_df[["id", name_col]], 
                        left_on=id_col,
                        right_on="id",
                        how="left"
                    ).rename(columns={name_col: table}).drop(columns=["id"])
            except Exception as e:
                st.error(f"خطأ في جلب جدول {table}: {str(e)}")

        # تنظيف البيانات النهائية
        df = df[["neighborhoods", "property_classifications", "property_type", "area", "price"]]
        df.columns = ["neighborhood", "classification", "property_type", "area", "price"]
        
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["area"] = pd.to_numeric(df["area"], errors="coerce")
        return df.dropna()

    except Exception as e:
        st.error(f"خطأ في معالجة البيانات: {str(e)}")
        return pd.DataFrame()

# 3. واجهة التطبيق
st.set_page_config(page_title="توقع أسعار العقارات", layout="wide")
st.title("🏠 نظام توقع أسعار العقارات")

# تحميل البيانات
df = load_merged_data()

if not df.empty:
    # قسم الإدخال
    with st.sidebar:
        st.header("خصائص العقار")
        neighborhood = st.selectbox("الحي", df["neighborhood"].unique())
        classification = st.selectbox("التصنيف", df["classification"].unique())
        property_type = st.selectbox("نوع العقار", df["property_type"].unique())
        area = st.slider("المساحة (م²)", 
                        min_value=int(df["area"].min()),
                        max_value=int(df["area"].max()),
                        value=int(df["area"].median()))

    # تدريب النموذج
    @st.cache_resource
    def train_model(_df):
        try:
            X = pd.get_dummies(_df[["neighborhood", "classification", "property_type", "area"]])
            y = _df["price"]
            model = RandomForestRegressor(n_estimators=150, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        except Exception as e:
            st.error(f"خطأ في التدريب: {str(e)}")
            return None, None

    model, features = train_model(df)

    if model:
        # التنبؤ
        input_data = pd.DataFrame([{
            "neighborhood": neighborhood,
            "classification": classification,
            "property_type": property_type,
            "area": area
        }])

        input_processed = pd.get_dummies(input_data)
        for col in features:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[features]

        prediction = model.predict(input_processed)[0]
        st.success(f"## السعر المتوقع: {prediction:,.2f} ريال")

        # عرض البيانات
        with st.expander("عرض البيانات"):
            st.dataframe(df)

        # العقارات المشابهة
        similar = df[
            (df["neighborhood"] == neighborhood) &
            (df["classification"] == classification) &
            (df["property_type"] == property_type)
        ]
        if not similar.empty:
            st.plotly_chart(px.scatter(
                similar, x="area", y="price",
                title="العقارات المشابهة"
            ))
else:
    st.error("فشل تحميل البيانات. يرجى التحقق من اتصال قاعدة البيانات.")
