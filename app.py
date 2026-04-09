import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_lottie import st_lottie
import requests

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Impact Predictor", page_icon="🎓", layout="wide")

# --- CUSTOM CSS FOR ANIMATION & STYLE ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #357ABD;
        transform: scale(1.02);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.8);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_prevwabt.json")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- APP LAYOUT ---
st.title("🎓 Student AI Impact Analyzer")
st.write("Predict the impact of AI tools on academic performance using Machine Learning.")

with st.container():
    left_col, right_col = st.columns([1, 1.2])
    
    with left_col:
        st_lottie(lottie_ai, height=300, key="coding")

    with right_col:
        st.subheader("User Demographics & Habits")
        
        # Creating inputs based on model feature names
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            city = st.text_input("City", "New York")
            daily_hours = st.slider("Daily Usage Hours", 0.0, 24.0, 2.0)
        
        with col2:
            edu_level = st.selectbox("Education Level", ["High School", "Undergraduate", "Postgraduate", "PhD"])
            ai_tool = st.selectbox("Primary AI Tool", ["ChatGPT", "Gemini", "Claude", "Notion AI", "Other"])
            purpose = st.selectbox("Primary Purpose", ["Study", "Research", "Coding", "General Inquiry"])
            impact_grades = st.selectbox("Perceived Impact on Grades", ["Positive", "Neutral", "Negative"])

# --- PREDICTION LOGIC ---
if st.button("Analyze Impact"):
    # Note: Categorical variables usually need encoding (LabelEncoding or OneHot) 
    # to match the model's training state. 
    # This example assumes numerical/label input; adjust mapping if needed.
    
    input_data = pd.DataFrame([[
        age, gender, edu_level, city, ai_tool, daily_hours, purpose, impact_grades
    ]], columns=['Age', 'Gender', 'Education_Level', 'City', 'AI_Tool_Used', 
                 'Daily_Usage_Hours', 'Purpose', 'Impact_on_Grades'])
    
    # Placeholder for encoding if your model was trained on numbers:
    # for col in input_data.select_dtypes(include=['object']).columns:
    #     input_data[col] = label_encoder.transform(input_data[col])

    try:
        prediction = model.predict(input_data)
        
        st.balloons()
        st.markdown("---")
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Prediction Result</h3>
                <h1 style="color: #4A90E2;">{prediction[0]}</h1>
                <p>Based on a KNeighbors Classification analysis.</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.warning("Prediction Error: Ensure your categorical inputs (like 'City') match the encoding used during training.")
        st.error(str(e))

st.markdown("---")
st.caption("Model Version: 1.6.1 | Powered by Scikit-Learn & Streamlit")
