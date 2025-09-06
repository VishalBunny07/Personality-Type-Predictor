import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('personality_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature columns
with open('feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

# Personality mapping
personality_map = {0: 'Introvert', 1: 'Extrovert', 2: 'Ambivert'}

st.title("🧠 Personality Type Predictor")
st.write("Answer these questions to discover your personality type!")

# Create input fields for each feature
inputs = {}
col1, col2 = st.columns(2)

with col1:
    inputs['social_energy'] = st.slider("Social Energy (0-10)", 0.0, 10.0, 5.0)
    inputs['alone_time_preference'] = st.slider("Alone Time Preference (0-10)", 0.0, 10.0, 5.0)
    inputs['talkativeness'] = st.slider("Talkativeness (0-10)", 0.0, 10.0, 5.0)
    inputs['deep_reflection'] = st.slider("Deep Reflection (0-10)", 0.0, 10.0, 5.0)
    inputs['group_comfort'] = st.slider("Group Comfort (0-10)", 0.0, 10.0, 5.0)
    inputs['party_liking'] = st.slider("Party Liking (0-10)", 0.0, 10.0, 5.0)
    inputs['listening_skill'] = st.slider("Listening Skill (0-10)", 0.0, 10.0, 5.0)
    inputs['empathy'] = st.slider("Empathy (0-10)", 0.0, 10.0, 5.0)
    inputs['creativity'] = st.slider("Creativity (0-10)", 0.0, 10.0, 5.0)
    inputs['organization'] = st.slider("Organization (0-10)", 0.0, 10.0, 5.0)

with col2:
    inputs['leadership'] = st.slider("Leadership (0-10)", 0.0, 10.0, 5.0)
    inputs['risk_taking'] = st.slider("Risk Taking (0-10)", 0.0, 10.0, 5.0)
    inputs['public_speaking_comfort'] = st.slider("Public Speaking Comfort (0-10)", 0.0, 10.0, 5.0)
    inputs['curiosity'] = st.slider("Curiosity (0-10)", 0.0, 10.0, 5.0)
    inputs['routine_preference'] = st.slider("Routine Preference (0-10)", 0.0, 10.0, 5.0)
    inputs['excitement_seeking'] = st.slider("Excitement Seeking (0-10)", 0.0, 10.0, 5.0)
    inputs['friendliness'] = st.slider("Friendliness (0-10)", 0.0, 10.0, 5.0)
    inputs['emotional_stability'] = st.slider("Emotional Stability (0-10)", 0.0, 10.0, 5.0)
    inputs['planning'] = st.slider("Planning (0-10)", 0.0, 10.0, 5.0)
    inputs['spontaneity'] = st.slider("Spontaneity (0-10)", 0.0, 10.0, 5.0)

# Additional features
inputs['adventurousness'] = st.slider("Adventurousness (0-10)", 0.0, 10.0, 5.0)
inputs['reading_habit'] = st.slider("Reading Habit (0-10)", 0.0, 10.0, 5.0)
inputs['sports_interest'] = st.slider("Sports Interest (0-10)", 0.0, 10.0, 5.0)
inputs['online_social_usage'] = st.slider("Online Social Usage (0-10)", 0.0, 10.0, 5.0)
inputs['travel_desire'] = st.slider("Travel Desire (0-10)", 0.0, 10.0, 5.0)
inputs['gadget_usage'] = st.slider("Gadget Usage (0-10)", 0.0, 10.0, 5.0)
inputs['work_style_collaborative'] = st.slider("Work Style Collaborative (0-10)", 0.0, 10.0, 5.0)
inputs['decision_speed'] = st.slider("Decision Speed (0-10)", 0.0, 10.0, 5.0)
inputs['stress_handling'] = st.slider("Stress Handling (0-10)", 0.0, 10.0, 5.0)

if st.button("Predict Personality Type"):
    # Create input dataframe
    input_df = pd.DataFrame([inputs])
    
    # Ensure columns are in correct order
    input_df = input_df[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    personality = personality_map[prediction[0]]
    confidence = np.max(probability[0]) * 100
    
    st.success(f"### Your predicted personality type: **{personality}**")
    st.info(f"Confidence: {confidence:.2f}%")
    
    # Show personality description
    if personality == 'Introvert':
        st.write("""
        **Introvert Traits:**
        - Prefer solitude and quiet environments
        - Recharge energy through alone time
        - Thoughtful and reflective
        - Prefer deep conversations over small talk
        """)
    elif personality == 'Extrovert':
        st.write("""
        **Extrovert Traits:**
        - Gain energy from social interactions
        - Enjoy being around people
        - Outgoing and expressive
        - Think while speaking
        """)
    else:
        st.write("""
        **Ambivert Traits:**
        - Balance between introversion and extroversion
        - Adaptable to different social situations
        - Enjoy both social time and alone time
        - Flexible in communication style
        """)