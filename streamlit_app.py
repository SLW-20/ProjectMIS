import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title('🏠 Real Estate Price Prediction App')
st.info('This app predicts real estate prices in Abha!')

# Load and prepare data
with st.expander('Data'):
    st.write('**Raw data**')
    df = pd.read_csv('https://raw.githubusercontent.com/1Hani-77/TEST/refs/heads/main/abha%20real%20estate.csv')
    # Drop any rows with missing values
    df = df.dropna()
    st.dataframe(df)
    
    st.write('**Features (X)**')
    X_raw = df.drop(['price', 'description'], axis=1)
    st.dataframe(X_raw)
    
    st.write('**Target (y)**')
    y_raw = df.price
    st.dataframe(y_raw)

# Data visualization
with st.expander('Data visualization'):
    st.scatter_chart(
        data=df,
        x='size',
        y='price',
        color='property_type'
    )

# Input features
with st.sidebar:
    st.header('Property Features')
    
    property_type = st.selectbox('Property Type', 
                                df['property_type'].unique().tolist())
    
    size = st.slider('Size (sq meters)', 
                     float(df['size'].min()), 
                     float(df['size'].max()),
                     float(df['size'].mean()))
    
    rooms = st.slider('Number of Rooms',
                      int(df['rooms'].min()),
                      int(df['rooms'].max()),
                      int(df['rooms'].median()))
    
    bathrooms = st.slider('Number of Bathrooms',
                         int(df['bathrooms'].min()),
                         int(df['bathrooms'].max()),
                         int(df['bathrooms'].median()))
    
    location = st.selectbox('Location',
                           df['location'].unique().tolist())
    
    # Create DataFrame for input features
    input_data = {
        'property_type': property_type,
        'size': size,
        'rooms': rooms,
        'bathrooms': bathrooms,
        'location': location
    }
    input_df = pd.DataFrame(input_data, index=[0])

# Show input data
with st.expander('Input features'):
    st.write('**Selected Property Features**')
    st.dataframe(input_df)

# Data preparation
# Combine input with training data for consistent encoding
input_properties = pd.concat([input_df, X_raw], axis=0)

# Encode categorical variables
le = LabelEncoder()
categorical_cols = ['property_type', 'location']

for col in categorical_cols:
    input_properties[col] = le.fit_transform(input_properties[col])

# Separate back into input and training data
X = input_properties[1:]
input_row = input_properties[:1]

# Model training
# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_raw)

# Make prediction
prediction = model.predict(input_row)

# Display prediction
st.subheader('Predicted Price')
predicted_price = float(prediction[0])
formatted_price = "{:,.2f}".format(predicted_price)
st.success(f"Estimated Price: SAR {formatted_price}")

# Feature importance
with st.expander('Feature Importance'):
    feature_importance = pd.DataFrame({
        'Feature': input_properties.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    st.bar_chart(feature_importance.set_index('Feature'))
