# تعديل دالة load_reference_data لتتعامل بشكل أفضل مع حالة عدم وجود الجداول
@st.cache_data(ttl=600)
def load_reference_data():
    try:
        supabase = init_connection()
        if not supabase:
            return {}, {}, {}
            
        # محاولة تحميل بيانات الأحياء مع معالجة أفضل للأخطاء
        neighborhood_dict = {}
        try:
            neighborhood_response = supabase.table('neighborhoods').select('*').execute()
            if neighborhood_response.data:
                neighborhood_dict = {item['id']: item['name'] for item in neighborhood_response.data}
                st.success("تم تحميل بيانات الأحياء بنجاح")
            else:
                st.warning("جدول الأحياء فارغ - سيتم عرض المعرفات بدلاً من الأسماء")
        except Exception as e:
            st.warning(f"لم يتم العثور على جدول الأحياء: {str(e)}")
            
        # محاولة تحميل بيانات أنواع العقارات
        property_type_dict = {}
        try:
            property_type_response = supabase.table('property_types').select('*').execute()
            if property_type_response.data:
                property_type_dict = {item['id']: item['name'] for item in property_type_response.data}
                st.success("تم تحميل بيانات أنواع العقارات بنجاح")
            else:
                st.warning("جدول أنواع العقارات فارغ - سيتم عرض المعرفات بدلاً من الأسماء")
        except Exception as e:
            st.warning(f"لم يتم العثور على جدول أنواع العقارات: {str(e)}")
            
        # محاولة تحميل بيانات التصنيفات
        classification_dict = {}
        try:
            classification_response = supabase.table('classifications').select('*').execute()
            if classification_response.data:
                classification_dict = {item['id']: item['name'] for item in classification_response.data}
                st.success("تم تحميل بيانات التصنيفات بنجاح")
            else:
                st.warning("جدول التصنيفات فارغ - سيتم عرض المعرفات بدلاً من الأسماء")
        except Exception as e:
            st.warning(f"لم يتم العثور على جدول التصنيفات: {str(e)}")
            
        return neighborhood_dict, property_type_dict, classification_dict
    except Exception as e:
        st.error(f"فشل تحميل البيانات المرجعية: {str(e)}")
        return {}, {}, {}

# إضافة دالة جديدة لإنشاء الجداول المرجعية تلقائياً إذا كانت غير موجودة
def create_reference_tables_if_needed():
    try:
        supabase = init_connection()
        if not supabase:
            return False
        
        # التحقق من وجود الجداول وإنشائها إذا لم تكن متوفرة
        tables_to_check = ['neighborhoods', 'property_types', 'classifications']
        created_tables = []
        
        for table in tables_to_check:
            try:
                # محاولة الاستعلام عن الجدول للتحقق من وجوده
                test_query = supabase.table(table).select('id').limit(1).execute()
                st.info(f"جدول {table} موجود بالفعل")
            except Exception:
                st.warning(f"جدول {table} غير موجود. سيتم محاولة إنشائه...")
                created_tables.append(table)
                
        # إذا كانت هناك جداول تحتاج للإنشاء، عرض رسالة للمستخدم
        if created_tables:
            st.warning(f"""
            يرجى إنشاء الجداول التالية في Supabase من خلال واجهة SQL:
            
            ```sql
            -- إنشاء جدول الأحياء
            CREATE TABLE IF NOT EXISTS neighborhoods (
              id SERIAL PRIMARY KEY,
              name TEXT NOT NULL
            );
            
            -- إنشاء جدول أنواع العقارات
            CREATE TABLE IF NOT EXISTS property_types (
              id SERIAL PRIMARY KEY,
              name TEXT NOT NULL
            );
            
            -- إنشاء جدول التصنيفات
            CREATE TABLE IF NOT EXISTS classifications (
              id SERIAL PRIMARY KEY,
              name TEXT NOT NULL
            );
            ```
            """)
            return False
        
        return True
    except Exception as e:
        st.error(f"حدث خطأ أثناء التحقق من الجداول: {str(e)}")
        return False

# تعديل دالة load_data للتعامل مع حالة عدم وجود البيانات المرجعية
@st.cache_data(ttl=600)
def load_data():
    try:
        # التحقق من وجود الجداول المرجعية
        tables_exist = create_reference_tables_if_needed()
        
        # تهيئة Supabase
        supabase = init_connection()
        if not supabase:
            return pd.DataFrame()
        
        # جلب البيانات من جدول العقارات
        response = supabase.table('properties').select('*').execute()
        
        # تحويل إلى DataFrame
        df = pd.DataFrame(response.data)
        
        if df.empty:
            raise ValueError("لا توجد بيانات في جدول العقارات")
            
        # التحقق من العمدة المطلوبة
        required_columns = ['neighborhood_id', 'classification_id', 'property_type_id', 'area', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"عمدة مفقودة: {', '.join(missing_columns)}")
        
        # تحويل السعر والمساحة إلى أرقام
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        
        # تنظيف البيانات
        df = df.dropna(subset=['price', 'area'])
        if df.empty:
            raise ValueError("لا توجد بيانات صالحة بعد التنظيف")
            
        # تحميل جداول البحث لتحويل المعرفات إلى أسماء
        neighborhood_dict, property_type_dict, classification_dict = load_reference_data()
        
        # إنشاء أعمدة الأسماء من المعرفات مع إضافة زر لتحديث البيانات المرجعية
        if st.button("تحديث البيانات المرجعية"):
            st.cache_data.clear()
            neighborhood_dict, property_type_dict, classification_dict = load_reference_data()
            st.success("تم تحديث البيانات المرجعية")
            
        # تحويل معرفات الأحياء
        if neighborhood_dict:
            df['neighborhood_name'] = df['neighborhood_id'].map(neighborhood_dict).fillna('غير معروف')
        else:
            df['neighborhood_name'] = "حي " + df['neighborhood_id'].astype(str)
            
        # تحويل معرفات أنواع العقارات
        if property_type_dict:
            df['property_type_name'] = df['property_type_id'].map(property_type_dict).fillna('غير معروف')
        else:
            df['property_type_name'] = "نوع " + df['property_type_id'].astype(str)
            
        # تحويل معرفات التصنيفات
        if classification_dict:
            df['classification_name'] = df['classification_id'].map(classification_dict).fillna('غير معروف')
        else:
            df['classification_name'] = "تصنيف " + df['classification_id'].astype(str)
            
        return df
    except Exception as e:
        st.error(f"فشل تحميل البيانات: {str(e)}")
        return pd.DataFrame()

# إضافة واجهة لإدارة الجداول المرجعية
def manage_reference_tables():
    st.header("إدارة الجداول المرجعية")
    
    table_option = st.selectbox(
        "اختر الجدول:",
        ["neighborhoods", "property_types", "classifications"],
        format_func=lambda x: {
            "neighborhoods": "الأحياء",
            "property_types": "أنواع العقارات", 
            "classifications": "التصنيفات"
        }[x]
    )
    
    # واجهة الإضافة
    with st.expander("إضافة عناصر جديدة"):
        col1, col2 = st.columns(2)
        with col1:
            new_id = st.number_input("المعرف:", min_value=1, step=1)
        with col2:
            new_name = st.text_input("الاسم:")
            
        if st.button("إضافة"):
            if new_name:
                try:
                    supabase = init_connection()
                    response = supabase.table(table_option).insert({"id": new_id, "name": new_name}).execute()
                    if response.data:
                        st.success(f"تمت إضافة: {new_name}")
                        # مسح الذاكرة المؤقتة لتحديث البيانات
                        st.cache_data.clear()
                    else:
                        st.error("فشلت عملية الإضافة")
                except Exception as e:
                    st.error(f"خطأ: {str(e)}")
            else:
                st.warning("يرجى إدخال اسم")
    
    # عرض البيانات الحالية
    try:
        supabase = init_connection()
        response = supabase.table(table_option).select('*').order('id').execute()
        if response.data:
            st.write("### البيانات الحالية")
            df = pd.DataFrame(response.data)
            st.dataframe(df)
        else:
            st.info("لا توجد بيانات في هذا الجدول")
    except Exception as e:
        st.error(f"خطأ في عرض البيانات: {str(e)}")

# إضافة هذه الدالة في أي مكان مناسب في التطبيق (مثلاً في الشريط الجانبي)
with st.sidebar:
    st.header("إدارة قاعدة البيانات")
    if st.checkbox("إظهار أدوات إدارة البيانات المرجعية"):
        manage_reference_tables()
