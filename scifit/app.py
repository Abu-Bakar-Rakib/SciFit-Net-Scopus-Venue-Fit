import streamlit as st
import torch
import os

# Import the Predictor class from our model file
from model import Predictor, format_result

st.set_page_config(page_title="SCIFIT-Net Recommender", page_icon="📚", layout="centered")

st.title("📚 SCIFIT-Net: Scientific Venue Recommender")
st.write("Upload your title and abstract to get journal recommendations, scope compatibility, and consistency checks!")

# Sidebar for Model Configuration
st.sidebar.header("Model Configuration")
model_dir = st.sidebar.text_input("Path to Model Directory (containing best.pt, meta.json, etc.)", value="./scifit_out")

@st.cache_resource(show_spinner=False)
def load_model(directory):
    if not os.path.exists(directory) or not os.path.exists(os.path.join(directory, "best.pt")):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Predictor.load(directory, device)

predictor = None
if os.path.exists(model_dir):
    with st.spinner("Loading SCIFIT-Net Model (this may take a moment)..."):
        predictor = load_model(model_dir)
        if predictor:
            st.sidebar.success(f"Model loaded successfully on {predictor.device}!")
        else:
            st.sidebar.error("Could not load model. Ensure 'best.pt' and 'meta.json' are in the directory.")
else:
    st.sidebar.warning("Please provide a valid path to the trained model output directory.")

# Main Input UI
st.subheader("Enter Manuscript Details")
title_input = st.text_input("Manuscript Title", placeholder="e.g. Attention Is All You Need")
abstract_input = st.text_area("Manuscript Abstract", height=200, placeholder="Enter the full abstract here...")

if st.button("Predict Fit", type="primary"):
    if not title_input or not abstract_input:
        st.error("Please enter both a Title and an Abstract!")
    elif predictor is None:
        st.error("Model is not loaded! Please check the model directory in the sidebar.")
    else:
        with st.spinner("Analyzing manuscript and searching semantic space..."):
            # Run prediction
            result = predictor.predict(title_input, abstract_input)
            
            st.divider()
            st.header("🎯 Prediction Results")
            
            # --- 1. Best Venue & Scope Mismatch ---
            st.subheader("Best Venue Recommendation")
            st.success(f"**{result['best']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Scope-Mismatch Probability", f"{result['mismatch'] * 100:.1f}%")
            with col2:
                st.metric("Scope Distance", f"{result['scope_distance']:.2f}")
                
            # --- 2. Title-Abstract Consistency ---
            st.subheader("Title-Abstract Consistency")
            cons_score = result["consistency"]
            
            # Determine assessment based on default thresholds (0.85, 0.65, 0.45)
            if cons_score >= 0.85:
                cons_text = "Highly Consistent"
                cons_color = "green"
            elif cons_score >= 0.65:
                cons_text = "Consistent"
                cons_color = "blue"
            elif cons_score >= 0.45:
                cons_text = "Partially Consistent"
                cons_color = "orange"
            else:
                cons_text = "Inconsistent"
                cons_color = "red"
                
            st.markdown(f"**Score:** {cons_score * 100:.1f}%  |  **Assessment:** :{cons_color}[{cons_text}]")
            st.progress(float(cons_score))
            
            # --- 3. Top 5 Venues ---
            st.subheader("Top 5 Alternative Venues")
            for i, (venue, score) in enumerate(result['venues']):
                st.write(f"**{i+1}. {venue}** — Match Score: {score * 100:.1f}%")
                st.progress(float(score))
                
            # --- 4. Explainability ---
            st.subheader("🧠 Explainability")
            with st.expander("Key Concepts Extracted"):
                st.write(", ".join(result['concepts']))
                
            with st.expander(f"Similar Papers in {result['best']}"):
                for paper in result['similar']:
                    st.markdown(f"- **{paper['title']}**")
                    st.caption(f"DOI: {paper['doi']} | Similarity: {paper['sim']:.2f}")
