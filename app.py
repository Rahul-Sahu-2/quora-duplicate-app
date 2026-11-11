import joblib
import streamlit as st
from helper import query_point_creator

st.set_page_config(page_title="Duplicate Question Checker", page_icon="🔍", layout="centered")

THRESHOLD = 0.40  # lower = the model says "Duplicate" more often

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

    :root {
        --ink: #14213D;
        --muted: #5B6578;
        --paper: #F6F4EF;
        --card: #FFFFFF;
        --line: #D9D5CB;
        --accent: #14213D;
        --yes: #1E7B4F;
        --yes-bg: #E4F3EA;
        --no: #B4381F;
        --no-bg: #FBE9E4;
    }

    html, body, .stApp, [class*="css"] {
        font-family: 'Manrope', system-ui, -apple-system, sans-serif;
    }
    .stApp { background: var(--paper); }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 720px; padding-top: 3rem; padding-bottom: 4rem; }

    h1.title {
        color: var(--ink);
        font-weight: 800;
        font-size: 2.4rem;
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin: 0 0 0.5rem 0;
    }
    p.subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        margin: 0 0 2rem 0;
    }

    /* inputs */
    .stTextArea label p {
        color: var(--ink) !important;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .stTextArea textarea {
        background: var(--card) !important;
        color: var(--ink) !important;
        border: 1.5px solid var(--line) !important;
        border-radius: 10px !important;
        font-size: 1rem;
        padding: 0.8rem 0.9rem;
    }
    .stTextArea textarea::placeholder { color: #98A0AE !important; }
    .stTextArea textarea:focus {
        border-color: var(--ink) !important;
        box-shadow: 0 0 0 3px rgba(20, 33, 61, 0.15) !important;
    }

    /* button */
    div.stButton > button {
        background: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.6rem;
        font-weight: 700;
        font-size: 1rem;
        font-family: inherit;
    }
    div.stButton > button:hover { background: #22345E; color: #FFFFFF; }
    div.stButton > button:focus-visible { outline: 3px solid #8FA6D8; outline-offset: 2px; }

    /* verdict */
    .verdict {
        margin-top: 1.5rem;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        border-left: 8px solid;
    }
    .verdict .label { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.01em; margin: 0; }
    .verdict .note { font-size: 1rem; margin: 0.3rem 0 0 0; }
    .verdict.yes { background: var(--yes-bg); border-color: var(--yes); color: var(--yes); }
    .verdict.no  { background: var(--no-bg);  border-color: var(--no);  color: var(--no); }
    .verdict.yes .note, .verdict.no .note { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    model = joblib.load("model.pkl")
    cv = joblib.load("cv.pkl")
    return model, cv


model, cv = load()

st.markdown('<h1 class="title">Duplicate Question Checker</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter two questions and find out if they ask the same thing.</p>',
    unsafe_allow_html=True,
)

q1 = st.text_area("Question 1", placeholder="How can I learn Python programming?", height=90)
q2 = st.text_area("Question 2", placeholder="What is the best way to learn Python?", height=90)

if st.button("Check questions"):
    if not q1.strip() or not q2.strip():
        st.warning("Enter both questions to continue.")
    else:
        x = query_point_creator(q1, q2, cv)
        prob = float(model.predict_proba(x)[0][1])
        if prob >= THRESHOLD:
            st.markdown(
                '<div class="verdict yes"><p class="label">Duplicate</p>'
                '<p class="note">These questions ask the same thing.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict no"><p class="label">Not duplicate</p>'
                '<p class="note">These questions ask different things.</p></div>',
                unsafe_allow_html=True,
            )
