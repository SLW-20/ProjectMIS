import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title('🏠 AI-Powered Property Valuation')
st.markdown("""
**Predict property prices** based on location, features, and market trends.
Explore market dynamics through interactive visualizations.
""")

# Data loading and preprocessing with enhanced validation
@st.cache_data
def load_and_clean_data():
    """Load and preprocess real estate data with robust type checking"""
    url = "https://raw.githubusercontent.com/1Hani-77/TEST/main/abha%20real%20estate.csv"
    
    try:
        df = pd.read_csv(url)
        
        # Validate required columns
        required_columns = ['neighborhood_name', 'classification_name', 
                           'property_type_name', 'area', 'price']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")

        # Enhanced numeric conversion with error tracking
        numeric_cols = ['price', 'area']
        
        for col in numeric_cols:
            # Remove non-numeric characters and convert to float
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[^\d.]', '', regex=True)
                .replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
                .apply(pd.to_numeric, errors='coerce')
            )  # Added closing parenthesis here
            
        # Remove rows with invalid numeric values
        df_clean = df.dropna(subset=numeric_cols).copy()
        
        # Validate numeric columns
        for col in numeric_cols:
            if not np.issubdtype(df_clean[col].dtype, np.number):
                raise TypeError(f"Column {col} contains non-numeric values after cleaning")

        # Calculate IQR ranges using clean numeric data
        Q1 = df_clean[numeric_cols].quantile(0.05)
        Q3 = df_clean[numeric_cols].quantile(0.95)
        IQR = Q3 - Q1

        # Identify outliers with proper parentheses
        outlier_mask = (
            (df_clean[numeric_cols] < (Q1 - 1.5 * IQR)) | 
            (df_clean[numeric_cols] > (Q3 + 1.5 * IQR))
        ).any(axis=1)

        final_df = df_clean[~outlier_mask]
        
        # Post-cleaning validation
        if final_df.empty:
            raise ValueError("No valid data remaining after cleaning")
            
        return final_df

    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        st.stop()

# Model training pipeline
@st.cache_resource
def train_price_model(_df):
    """Train and evaluate pricing model with data validation"""
    try:
        X = pd.get_dummies(
            _df[['neighborhood_name', 'classification_name', 
                'property_type_name', 'area']],
            drop_first=True
        )
        y = _df['price'].astype(float)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        return model, X.columns, metrics

    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        st.stop()

# Main application
try:
    df = load_and_clean_data()
    
    # Sidebar inputs
    with st.sidebar:
        st.header("🔍 Property Details")
        neighborhood = st.selectbox(
            "Neighborhood",
            options=sorted(df['neighborhood_name'].unique())
        )
        classification = st.selectbox(
            "Classification",
            options=sorted(df['classification_name'].unique())
        )
        property_type = st.selectbox(
            "Property Type",
            options=sorted(df['property_type_name'].unique())
        )
        area = st.slider(
            "Living Area (m²)",
            min_value=float(df['area'].quantile(0.05)),
            max_value=float(df['area'].quantile(0.95)),
            value=float(df['area'].median()),
            step=1.0
        )

    # Model training
    model, feature_names, metrics = train_price_model(df)

    # Prediction
    input_data = pd.DataFrame([{
        'neighborhood_name': neighborhood,
        'classification_name': classification,
        'property_type_name': property_type,
        'area': area
    }])
    
    X_input = pd.get_dummies(input_data).reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(X_input)[0]
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Value", f"${prediction:,.0f}")
    with col2:
        avg_price = df[df['neighborhood_name'] == neighborhood]['price'].mean()
        diff = prediction - avg_price
        st.metric("Neighborhood Average", f"${avg_price:,.0f}", 
                 delta=f"{diff:+,.0f} vs Average")

    # Market insights
    with st.expander("📊 Market Analysis"):
        tab1, tab2, tab3 = st.tabs(["Distributions", "Model Metrics", "Comparables"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x='price', title="Price Distribution")
                st.plotly_chart(fig)
            with col2:
                fig = px.scatter(df, x='area', y='price', color='neighborhood_name',
                               title="Price vs Area")
                st.plotly_chart(fig)
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{metrics['r2']:.1%}")
            col2.metric("MAE", f"${metrics['mae']:,.0f}")
            col3.metric("RMSE", f"${metrics['rmse']:,.0f}")
            
            fig = px.scatter(x=y_test, y=y_pred, 
                            labels={'x': 'Actual', 'y': 'Predicted'},
                            title="Actual vs Predicted Prices")
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                         x1=y_test.max(), y1=y_test.max())
            st.plotly_chart(fig)
        
        with tab3:
            similar = df[
                (df['neighborhood_name'] == neighborhood) &
                (df['area'].between(area*0.8, area*1.2))
            ]
            if not similar.empty:
                st.dataframe(similar)
            else:
                st.info("No comparable properties found")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please check the data source and try again")
