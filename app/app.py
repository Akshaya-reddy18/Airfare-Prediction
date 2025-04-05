import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
from streamlit.components.v1 import html

# Set page configuration first
st.set_page_config(
    page_title="AirFare Predictor",
    page_icon="💰",
    layout="centered"
)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.utils import format_currency

# Custom CSS for animations and styling
st.markdown("""
<style>
    /* Main heading style */
    h2 {
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 2rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        animation: slideIn 1s ease-out;
        letter-spacing: 1.5px;
        background: linear-gradient(135deg, #83a4d4 0%, #b6fbff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        padding-bottom: 0.8rem;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 160px;
        height: 4px;
        background: linear-gradient(90deg, #83a4d4, #b6fbff);
        border-radius: 3px;
    }
    
    /* Subheading style */
    h3 {
        color: #34495e;
        font-weight: 600 !important;
        margin-bottom: 2rem !important;
        font-size: 1.8rem !important;
    }
    
    /* Form label styling */
    .stSelectbox label, .stNumberInput label {
        color: #000000 !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Animation keyframes */
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Price display styling */
    .price-display {
        font-size: 3.2rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 3rem 0;
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #e6f3ff 0%, #d1e9ff 100%);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        animation: fadeIn 0.8s ease-out, pulse 2s infinite;
    }
    
    /* Success message animation */
    .stSuccess {
        animation: slideIn 0.5s ease-out;
        margin-bottom: 1rem !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #83a4d4 0%, #b6fbff 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Form elements styling */
    .stSelectbox, .stNumberInput {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    /* Override Streamlit's default label color */
    div[data-baseweb="select"] label, div[class*="stNumberInput"] label {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h2>AirFare Predictor</h2>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    
    # Load model and preprocessor
    if st.session_state.model is None or st.session_state.preprocessor is None:
        try:
            model_path = os.path.join('models', 'trained_models', 'random_forest.joblib')
            preprocessor_path = os.path.join('models', 'trained_models', 'preprocessor.joblib')
            
            if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
                st.error("Model files not found. Please train the model first.")
                return
                
            st.session_state.model = joblib.load(model_path)
            preprocessor_state = joblib.load(preprocessor_path)
            st.session_state.preprocessor = DataPreprocessor()
            st.session_state.preprocessor.scaler = preprocessor_state['scaler']
            st.session_state.preprocessor.label_encoders = preprocessor_state['label_encoders']
            st.session_state.preprocessor.categorical_categories = preprocessor_state['categorical_categories']
            st.session_state.preprocessor.feature_names = preprocessor_state['feature_names']
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Create input form
    st.subheader("Enter Flight Details")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        airline = st.selectbox(
            "Airline",
            options=['Air India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara', 'AirAsia']
        )
        source_city = st.selectbox(
            "Source City",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
        )
        departure_time = st.selectbox(
            "Departure Time",
            options=['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
        )
        stops = st.selectbox(
            "Number of Stops",
            options=['zero', 'one', 'two_or_more']
        )
    
    with col2:
        destination_city = st.selectbox(
            "Destination City",
            options=['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
        )
        arrival_time = st.selectbox(
            "Arrival Time",
            options=['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night']
        )
        class_type = st.selectbox(
            "Class",
            options=['Economy', 'Business']
        )
        days_left = st.number_input(
            "Days Until Departure",
            min_value=1,
            max_value=50,
            value=7
        )
    
    # Center the predict button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button("Predict Price", use_container_width=True)
    
    # Make prediction
    if predict_button:
        try:
            # Create input data with all required features
            input_data = pd.DataFrame(columns=st.session_state.preprocessor.feature_names)
            
            # Fill in the values we have
            input_data.loc[0] = {
                'airline': airline,
                'source_city': source_city,
                'departure_time': departure_time,
                'stops': stops,
                'destination_city': destination_city,
                'arrival_time': arrival_time,
                'class': class_type,
                'days_left': days_left,
                'flight': f"{airline}-{source_city}-{destination_city}",  # Create a flight identifier
                'duration': 2.0  # Default duration
            }
            
            # Ensure columns are in the correct order
            input_data = input_data[st.session_state.preprocessor.feature_names]
            
            # Preprocess input data
            # Identify numeric and categorical columns
            numeric_columns = input_data.select_dtypes(include=['int64', 'float64']).columns
            categorical_columns = input_data.select_dtypes(include=['object']).columns
            
            # Handle unseen labels in categorical features
            input_data = st.session_state.preprocessor.handle_unseen_labels(input_data, categorical_columns)
            
            # Scale numeric features
            if len(numeric_columns) > 0:
                input_data[numeric_columns] = st.session_state.preprocessor.scaler.transform(input_data[numeric_columns])
            
            # Make prediction
            prediction = st.session_state.model.predict(input_data)[0]
            
            # Ensure prediction is realistic
            min_price = 1000  # Minimum price of ₹1000
            if prediction < min_price:
                prediction = min_price
            
            # Display results with animation
            st.success("Prediction Complete!")
            
            # Create a container for the price display
            price_container = st.container()
            
            # Center the price display with animation
            with price_container:
                st.markdown(f"""
                <div class="success-animation">
                    <div class="price-display">
                        {format_currency(prediction)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Scroll to the price display immediately
            st.markdown("""
            <script>
                setTimeout(function() {
                    const priceElement = document.querySelector('.price-display');
                    if (priceElement) {
                        priceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }, 100);
            </script>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please ensure the model and preprocessor are properly trained with the same features.")

if __name__ == "__main__":
    main() 