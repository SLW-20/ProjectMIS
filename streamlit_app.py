import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with advanced styling
st.markdown("""
<style>
    :root {
        --primary: #2A9D8F;
        --secondary: #264653;
        --accent: #E9C46A;
    }

    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .stSidebar {
        background: var(--secondary) !important;
        color: white !important;
        border-right: 1px solid #dee2e6;
    }

    .sidebar .block-container {
        padding: 2rem 1rem;
    }

    .stSelectbox, .stSlider {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
    }

    .stHeader {
        color: var(--primary);
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .st-expander {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .plot-container {
        border-radius: 12px;
        overflow: hidden;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# App header with gradient
st.markdown("""
<div style="background: linear-gradient(135deg, var(--primary), var(--secondary));
            padding: 2rem;
            border-radius: 0 0 20px 20px;
            color: white;
            margin-bottom: 2rem;">
    <h1 style="margin:0; font-size: 2.5rem;">🏡 Smart Property Valuator</h1>
    <p style="opacity: 0.9; margin: 0.5rem 0 0;">AI-powered real estate valuation platform with market insights</p>
</div>
""", unsafe_allow_html=True)

# Data loading and preprocessing (unchanged from previous version)
@st.cache_data
def load_and_clean_data():
    # ... (same as before) ...

# Model training pipeline (unchanged from previous version)
@st.cache_resource
def train_price_model(_df):
    # ... (same as before) ...

# Main application
try:
    df = load_and_clean_data()
    
    # Sidebar with enhanced styling
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1rem 0;">
            <h2 style="color: white; border-bottom: 2px solid var(--accent); 
                padding-bottom: 0.5rem;">🏠 Property Details</h2>
        </div>
        """, unsafe_allow_html=True)
        
        neighborhood = st.selectbox(
            "Neighborhood District",
            options=sorted(df['neighborhood_name'].unique()),
            help="Select the property's neighborhood location"
        )

        classification = st.selectbox(
            "Property Class",
            options=sorted(df['classification_name'].unique()),
            help="Choose the property classification type"
        )

        property_type = st.selectbox(
            "Property Category",
            options=sorted(df['property_type_name'].unique()),
            help="Select the type of property"
        )

        area = st.slider(
            "Living Area (m²)",
            min_value=float(df['area'].quantile(0.05)),
            max_value=float(df['area'].quantile(0.95)),
            value=float(df['area'].median()),
            step=1.0,
            help="Adjust the total living area"
        )

    # Model training
    model, feature_names, metrics = train_price_model(df)

    # Prediction section
    input_data = pd.DataFrame([{
        'neighborhood_name': neighborhood,
        'classification_name': classification,
        'property_type_name': property_type,
        'area': area
    }])
    
    X_input = pd.get_dummies(input_data).reindex(columns=feature_names, fill_value=0)
    prediction = model.predict(X_input)[0]
    
    # Display results with animated cards
    col1, col2, col3 = st.columns([2,1,2])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--secondary); margin: 0 0 1rem;">Predicted Value</h3>
            <div style="font-size: 2.5rem; color: var(--primary); font-weight: 700;">
                ${prediction:,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        avg_price = df[df['neighborhood_name'] == neighborhood]['price'].mean()
        diff = prediction - avg_price
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: var(--secondary); margin: 0 0 1rem;">Market Comparison</h3>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 1.2rem;">Neighborhood Average</div>
                    <div style="font-size: 1.5rem; color: var(--primary);">${avg_price:,.0f}</div>
                </div>
                <div style="background: {'#2a9d8f20' if diff > 0 else '#e76f5120'}; 
                    padding: 0.5rem 1rem; border-radius: 8px;">
                    <span style="color: {'var(--primary)' if diff > 0 else '#e76f51'}; 
                        font-weight: 700;">{diff:+,.0f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Market insights with enhanced visualizations
    with st.expander("📈 Advanced Market Analytics", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Price Analysis", "Model Performance", "Comparable Properties"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Price Distribution Analysis")
                fig = px.histogram(df, x='price', 
                                 nbins=50, 
                                 color_discrete_sequence=['var(--primary)'],
                                 template='plotly_white')
                fig.update_layout(
                    hoverlabel=dict(bgcolor="white", font_size=12),
                    xaxis_title="Price (USD)",
                    yaxis_title="Number of Properties"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Price vs Area Correlation")
                fig = px.scatter(df, x='area', y='price',
                               color='neighborhood_name',
                               trendline="lowess",
                               color_discrete_sequence=px.colors.qualitative.Pastel,
                               template='plotly_white')
                fig.update_traces(marker=dict(size=8, opacity=0.6))
                fig.update_layout(
                    hovermode='closest',
                    xaxis_title="Area (m²)",
                    yaxis_title="Price (USD)",
                    legend_title="Neighborhood"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Model Accuracy Metrics")
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=metrics['r2'],
                    number={'font': {'color': 'var(--primary)'}, 'valueformat': '.0%'},
                    title={'text': "R² Score"},
                    domain={'row': 0, 'column': 0}
                ))
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=metrics['mae'],
                    number={'prefix': "$", 'font': {'color': 'var(--primary)'}},
                    title={'text': "MAE"},
                    domain={'row': 0, 'column': 1}
                ))
                fig.add_trace(go.Indicator(
                    mode="number",
                    value=metrics['rmse'],
                    number={'prefix': "$", 'font': {'color': 'var(--primary)'}},
                    title={'text': "RMSE"},
                    domain={'row': 0, 'column': 2}
                ))
                fig.update_layout(
                    grid={'rows': 1, 'columns': 3, 'pattern': "independent"},
                    template='plotly_white',
                    height=200
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### Prediction Accuracy")
                fig = px.scatter(x=y_test, y=y_pred, 
                                labels={'x': 'Actual Prices', 'y': 'Predicted Prices'},
                                trendline="lowess",
                                color_discrete_sequence=['var(--primary)'],
                                template='plotly_white')
                fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                            x1=y_test.max(), y1=y_test.max(),
                            line=dict(color="var(--accent)", dash="dash"))
                fig.update_layout(
                    hoverlabel=dict(bgcolor="white"),
                    xaxis_title="Actual Price (USD)",
                    yaxis_title="Predicted Price (USD)"
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            similar = df[
                (df['neighborhood_name'] == neighborhood) &
                (df['area'].between(area*0.8, area*1.2))
            ].sort_values('price', ascending=False)
            
            if not similar.empty:
                st.markdown(f"**Found {len(similar)} Comparable Properties**")
                styled_df = similar.style \
                    .background_gradient(subset=['price'], cmap='YlGnBu') \
                    .format({'price': '${:,.0f}', 'area': '{:,.0f}m²'}) \
                    .set_properties(**{'text-align': 'left', 'padding': '12px'}) \
                    .set_table_styles([{
                        'selector': 'thead',
                        'props': [('background', 'var(--primary)'), ('color', 'white')]
                    }])
                st.write(styled_df.to_html(), unsafe_allow_html=True)
            else:
                st.info("No comparable properties found in this area range")

except Exception as e:
    st.error(f"Application error: {str(e)}")
    st.info("Please check the data source and try again")
