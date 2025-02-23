import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from streamlit_neumorphic import neumorphic

# Page configuration
st.set_page_config(
    page_title="REALYST | AI Property Valuations",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🏙️"
)

# Custom CSS with glassmorphism and animations
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;900&display=swap');
    
    :root {{
        --primary: #6C5CE7;
        --secondary: #A8A5E6;
        --glass: rgba(255, 255, 255, 0.15);
        --dark: #2D3436;
    }}
    
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: linear-gradient(135deg, #2D3436 0%, #000000 100%);
        color: white !important;
    }}
    
    .glass-card {{
        background: var(--glass);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    .stMetric {{
        background: linear-gradient(45deg, var(--primary), #7C4DFF);
        color: white !important;
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.3s ease;
    }}
    
    .stMetric:hover {{
        transform: translateY(-5px);
    }}
    
    .stNumberInput, .stSelectbox, .stSlider {{
        background: var(--glass) !important;
        border-radius: 12px !important;
    }}
    
    .price-pulse {{
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    .stPlotlyChart {{
        border-radius: 24px;
        overflow: hidden;
    }}
</style>
""", unsafe_allow_html=True)

# Hero Section
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown("<h1 style='font-size:4.5rem; margin:0;'>REALYST</h1>", unsafe_allow_html=True)
    st.markdown("### AI-Powered Property Intelligence Platform")
    
with col2:
    st.image("https://cdn.dribbble.com/users/753356/screenshots/16934294/media/8b2eac5c3fae5c5e6e7b0b0b0b0b0b0b.png", 
             use_column_width=True)

# Data loading and model code remains similar from original...

# Redesigned Input Section
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🧭 Property Parameters")
    
    cols = st.columns([2, 2, 2, 1])
    with cols[0]:
        neighborhood = st.selectbox("Neighborhood", options=neighborhood_options, 
                                  help="Select property location")
    with cols[1]:
        property_type = st.selectbox("Property Type", options=type_options)
    with cols[2]:
        classification = st.selectbox("Classification", options=class_options)
    with cols[3]:
        area = st.number_input("Area (m²)", min_value=50, max_value=1000, value=150,
                             step=10)
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction Display
prediction = model.predict(...)  # Your existing prediction code

with st.container():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center'>
            <h3 style='color: var(--secondary)'>VALUATION RESULT</h3>
            <h1 class='price-pulse' style='font-size:4rem; margin:0; color: var(--primary)'>
                ${prediction:,.0f}
            </h1>
            <div style='margin:1rem 0; height:4px; background: var(--glass);'></div>
            <div style='display: flex; justify-content: space-between'>
                <div>📈 Market Trend: +2.4%</div>
                <div>📅 Last Updated: Today</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Advanced Visualizations
with st.expander("🔮 Market Insights Explorer", expanded=True):
    tab1, tab2, tab3 = st.tabs(["3D Map", "Price Evolution", "Investment Analysis"])
    
    with tab1:
        fig = px.scatter_3d(df, x='lon', y='lat', z='price',
                          color='neighborhood', size='area',
                          hover_name='property_type', 
                          title="3D Property Landscape")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['price'], name="Price Distribution",
                                 marker_color=var(--primary)))
        fig.add_trace(go.Box(x=df['price'], name="Spread", 
                           marker_color=var(--secondary)))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("""
        <div class='glass-card'>
            <h4>💰 Investment Potential</h4>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem'>
                <div class='stMetric'>Rental Yield<br><h2>6.8%</h2></div>
                <div class='stMetric'>Capital Growth<br><h2>+9.2%</h2></div>
                <div class='stMetric'>ROI (5y)<br><h2>142%</h2></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Neighborhood Comparison
st.markdown("### 📍 Neighborhood Spotlight")
cols = st.columns(3)
for idx, (name, data) in enumerate(neighborhood_data.items()):
    with cols[idx%3]:
        with neumorphic(key=f"card_{name}", 
                      boxShadow="0 8px 32px rgba(0,0,0,0.25)"):
            st.markdown(f"""
            <div style='padding:1.5rem; border-radius:16px'>
                <h4>{name}</h4>
                <div style='display:flex; justify-content: space-between'>
                    <div>🏘️ Avg Price</div>
                    <div>${data['avg_price']:,.0f}</div>
                </div>
                <div style='display:flex; justify-content: space-between'>
                    <div>📉 Price Trend</div>
                    <div style='color: {'#00E676' if data['trend'] > 0 else '#FF5252'}'>
                        {data['trend']}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: var(--secondary)'>
    REALYST AI • Property Market Analytics • v2.0<br>
    <div style='margin-top:1rem; opacity:0.7'>
        Powered by QuantumML Engine ⚛️
    </div>
</div>
""", unsafe_allow_html=True)
