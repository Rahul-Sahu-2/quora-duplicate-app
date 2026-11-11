import joblib
import streamlit as st
from helper import query_point_creator

st.set_page_config(page_title="Quora Duplicate Checker", page_icon="🔍")

@st.cache_resource
def load():
    model = joblib.load("model.pkl")
    cv = joblib.load("cv.pkl")
    return model, cv

model, cv = load()

st.title("🔍 Quora Duplicate Question Checker")
st.caption("Do questions daalo, model batayega ki dono ek hi matlab ke hain ya nahi.")

q1 = st.text_input("Question 1", placeholder="How do I learn Python?")
q2 = st.text_input("Question 2", placeholder="What is the best way to learn Python?")

if st.button("Check karo", type="primary"):
    if not q1.strip() or not q2.strip():
        st.warning("Pehle dono questions likho.")
    else:
        x = query_point_creator(q1, q2, cv)
        pred = model.predict(x)[0]
        prob = float(model.predict_proba(x)[0][1])
        if pred == 1:
            st.success(f"✅ Duplicate hain (confidence: {prob:.0%})")
        else:
            st.error(f"❌ Duplicate nahi hain (duplicate probability: {prob:.0%})")
        st.progress(prob)
