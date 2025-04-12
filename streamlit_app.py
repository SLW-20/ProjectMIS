import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from supabase import create_client

# Page configuration
st.set_page_config(page_title="Real Estate Price Prediction", layout="wide")

# App title and description
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices based on property features using Supabase data!')

# Supabase connection
@st.cache_resource
def init_connection():
    try:
        return create_client(
            "https://imdnhiwyfgjdgextvrkj.supabase.co",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltZG5oaXd5ZmdqZGdleHR2cmtqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Mzk3MTM5NzksImV4cCI6MjA1NTI4OTk3OX0.9hIzkJYKrOTsKTKwjAyHRWBG2Rqe2Sgwq7WgddqLTDk"
        )
    except Exception as e:
        st.error(f"🔌 Connection failed: {str(e)}")
        return None

# Enhanced data loader with relational handling
@st.cache_data(ttl=600)
def load_data():
    try:
        supabase = init_connection()
        if not supabase:
            return pd.DataFrame()

        # Attempt to load data with joined tables
        try:
            response = supabase.table('properties').select('*, neighborhoods(name), property_types(name), classifications(name)').execute()
            df = pd.DataFrame(response.data)
            
            # Extract nested names from joined tables
            df['neighborhood_name'] = df['neighborhoods'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
            df['property_type_name'] = df['property_types'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
            df['classification_name'] = df['classifications'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
            
        except Exception as join_error:
            st.warning("Using manual ID mapping for names")
            response = supabase.table('properties').select('*').execute()
            df = pd.DataFrame(response.data)
            
            # Load reference tables
            ref_tables = {
                'neighborhoods': ('id', 'name'),
                'property_types': ('id', 'name'),
                'classifications': ('id', 'name')
            }
            
            for table, (id_col, name_col) in ref_tables.items():
                try:
                    ref_response = supabase.table(table).select(f'{id_col}, {name_col}').execute()
                    ref_df = pd.DataFrame(ref_response.data)
                    if not ref_df.empty:
                        mapper = dict(zip(ref_df[id_col], ref_df[name_col]))
                        df[f'{table[:-1]}_name'] = df[f'{table[:-1]}_id'].map(mapper)
                    else:
                        st.warning(f"Empty reference table: {table}")
                        df[f'{table[:-1]}_name'] = df[f'{table[:-1]}_id'].astype(str)
                except Exception as e:
                    st.error(f"Missing {table} table: Using IDs")
                    df[f'{table[:-1]}_name'] = df[f'{table[:-1]}_id'].astype(str)

        # Validate required columns
        required = ['neighborhood_name', 'property_type_name', 
                   'classification_name', 'area', 'price']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Data cleaning
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df = df.dropna(subset=['price', 'area'])
        
        return df.drop(columns=['neighborhoods', 'property_types', 'classifications'], errors='ignore')

    except Exception as e:
        st.error(f"🚨 Data loading error: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

if not df.empty:
    st.success("✅ Data loaded successfully from Supabase!")
    
    # Data overview section
    with st.expander("📊 Data Overview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Raw Data Preview")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.write("### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

        # Visualizations
        try:
            st.write("### Data Distributions")
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='price', title='Price Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(df, x='area', y='price', color='neighborhood_name',
                               title='Price vs Area by Neighborhood')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"📈 Visualization error: {str(e)}")

    # Prediction interface
    with st.sidebar:
        st.header("🔧 Prediction Parameters")
        neighborhood = st.selectbox("Neighborhood", sorted(df['neighborhood_name'].unique()))
        property_type = st.selectbox("Property Type", sorted(df['property_type_name'].unique()))
        classification = st.selectbox("Classification", sorted(df['classification_name'].unique()))
        
        area = st.slider("Area (m²)", 
                        min_value=float(df['area'].min()),
                        max_value=1500.0,
                        value=float(df['area'].median()))

    # Model training
    @st.cache_resource
    def train_model(_df):
        try:
            X = pd.get_dummies(_df[['neighborhood_name', 'property_type_name', 
                                  'classification_name', 'area']], drop_first=True)
            y = _df['price']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        except Exception as e:
            st.error(f"🤖 Model training failed: {str(e)}")
            return None, None

    model, features = train_model(df)

    if model and features:
        # Prediction handling
        try:
            input_data = pd.DataFrame([{
                'neighborhood_name': neighborhood,
                'property_type_name': property_type,
                'classification_name': classification,
                'area': area
            }])
            
            # Process input features
            processed = pd.get_dummies(input_data, drop_first=True)
            for col in features:
                if col not in processed.columns:
                    processed[col] = 0
            processed = processed[features]
            
            # Make prediction
            prediction = model.predict(processed)[0]
            st.markdown(f"## 🏷 Predicted Price: **${prediction:,.2f}**")
            
            # Feature importance visualization
            with st.expander("📈 Feature Importance Analysis"):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title='Feature Importance Ranking')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"🔮 Prediction failed: {str(e)}")

    # Similar properties section
    with st.expander("🏘 Similar Properties in Neighborhood"):
        similar = df[df['neighborhood_name'] == neighborhood]
        if not similar.empty:
            st.dataframe(similar.head(10), use_container_width=True)
            fig = px.scatter(similar, x='area', y='price',
                            hover_data=['classification_name', 'property_type_name'],
                            title=f'Properties in {neighborhood}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No similar properties found in this neighborhood")

else:
    st.error("❌ Failed to load data. Please check your database setup.")
    
    # Database setup guide
    with st.expander("🔧 Database Configuration Guide", expanded=True):
        st.markdown("""
        ### Required Database Structure
        
        1. **Properties Table** (`properties`)
           - Columns:
             - `neighborhood_id` (INT)
             - `property_type_id` (INT)
             - `classification_id` (INT)
             - `area` (NUMERIC)
             - `price` (NUMERIC)
        
        2. **Reference Tables** (with proper relationships):
           ```sql
           CREATE TABLE neighborhoods (
               id SERIAL PRIMARY KEY,
               name TEXT NOT NULL
           );
           
           CREATE TABLE property_types (
               id SERIAL PRIMARY KEY,
               name TEXT NOT NULL
           );
           
           CREATE TABLE classifications (
               id SERIAL PRIMARY KEY,
               name TEXT NOT NULL
           );
           ```
        
        3. **Foreign Key Relationships**:
           - `properties.neighborhood_id` → `neighborhoods.id`
           - `properties.property_type_id` → `property_types.id`
           - `properties.classification_id` → `classifications.id`
        """)
        
        # Reference table creator
        st.markdown("### 🛠 Quick Setup Tool")
        table_choice = st.selectbox("Select table to create:", 
                                  ["neighborhoods", "property_types", "classifications"])
        
        if st.button("🔄 Initialize Table"):
            try:
                supabase = init_connection()
                response = supabase.table(table_choice).insert([{"name": "Sample Entry"}]).execute()
                if response.data:
                    st.success(f"Successfully created {table_choice} table!")
                else:
                    st.error("Table creation failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("🔍 Check the sidebar to modify prediction parameters")
