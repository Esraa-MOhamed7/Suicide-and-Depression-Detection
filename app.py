import streamlit as st
import joblib
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tfidf_svm.pkl")
st.set_page_config(page_title="Suicide Detection NLP", page_icon="🧠", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>AI-Powered Suicide Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #555;'>Using Natural Language Processing to identify risk in text</h4>",
    unsafe_allow_html=True
)

st.image(
    "https://img.freepik.com/premium-vector/mental-health-awareness-concept_23-2148533323.jpg?w=740",
    use_container_width=True
)

st.markdown("---")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found: tfidf_svm.pkl")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------
# Helpers
# --------------------------------
def clean_text(t: str) -> str:
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t

st.sidebar.title("About this project")
st.sidebar.info(
    "This app demonstrates how NLP can be applied to detect suicidal intent in text. "
    "It uses a TF‑IDF + SVM model trained on labeled data. "
    "Type or paste text below and click **Predict** to see the classification."
)

st.sidebar.markdown("### Tips")
st.sidebar.markdown("- Keep inputs short and meaningful.\n- Try both positive and negative examples.\n- Use this app for **educational purposes only**.")

st.header("Try the model")

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Example: Suicidal text"):
        st.session_state["example_text"] = "I feel hopeless and I want to end my life."
with col2:
    if st.button("Example: Non-suicidal text"):
        st.session_state["example_text"] = "I am excited to start my new job tomorrow!"

# Text area
user_input = st.text_area(
    "Enter text to analyze:",
    st.session_state.get("example_text", ""),
    height=100
)

# Prediction
if st.button("Predict"):
    text = clean_text(user_input)
    if len(text) < 3:
        st.warning("Please enter meaningful text.")
    else:
        pred = model.predict([text])[0]
        st.success(f"Prediction: {pred}")

        # Optional confidence margin
        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]
            margin = clf.decision_function(tfidf.transform([text]))[0]
            st.caption(f"Confidence margin: {margin:.3f}")
        except Exception:
            st.caption("Confidence margin not available for this classifier.")

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.caption("Developed for educational purposes — demonstrating NLP in mental health applications.")