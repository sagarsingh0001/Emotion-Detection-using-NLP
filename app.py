import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="centered",
)

MODEL_PATH = "emotion_classifier.joblib"

# If your model predicts NUMBERS, edit this mapping to match your training labels.
# (If it already predicts the names, this is ignored.)
LABEL_MAP = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy",
}

EMOTION_STYLE = {
    "joy":      {"emoji": "😄", "color": "#F5B700", "desc": "Happiness, delight and positivity."},
    "sadness":  {"emoji": "😢", "color": "#4C78A8", "desc": "Sorrow, grief or disappointment."},
    "anger":    {"emoji": "😠", "color": "#E4572E", "desc": "Frustration, irritation or rage."},
    "fear":     {"emoji": "😨", "color": "#7B5EA7", "desc": "Anxiety, worry or being scared."},
    "love":     {"emoji": "❤️", "color": "#E84A8A", "desc": "Affection, care and warmth."},
    "surprise": {"emoji": "😲", "color": "#1FB5A8", "desc": "Astonishment or something unexpected."},
}

EXAMPLES = {
    "😄 Joy": "I just got the job offer and I can't stop smiling, today is the best day ever!",
    "😢 Sadness": "I miss my old friends so much, everything feels empty and quiet without them.",
    "😠 Anger": "I can't believe they cancelled my order again, this is absolutely infuriating!",
    "😨 Fear": "My heart is pounding, I heard footsteps outside and I'm terrified to look.",
    "❤️ Love": "I feel so grateful to have you in my life, you mean the world to me.",
    "😲 Surprise": "Wow, I never expected to see you here, what a shock!",
}

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 820px;}
    .hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 55%, #f093fb 100%);
        padding: 2.2rem 1.5rem; border-radius: 22px; text-align: center;
        color: white; margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(118, 75, 162, 0.35);
    }
    .hero h1 {margin: 0; font-size: 2.3rem; font-weight: 800;}
    .hero p {margin: .5rem 0 0; opacity: .92; font-size: 1.05rem;}
    .result-card {
        border-radius: 22px; padding: 1.8rem 1.2rem; text-align: center;
        color: white; margin: 1rem 0 1.2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
        animation: pop .45s ease;
    }
    .result-card .emoji {font-size: 4.2rem; line-height: 1;}
    .result-card .label {font-size: 2rem; font-weight: 800; text-transform: capitalize; margin-top: .4rem;}
    .result-card .conf {font-size: 1.05rem; opacity: .95;}
    .result-card .desc {font-size: .95rem; opacity: .85; margin-top: .3rem;}
    @keyframes pop {from {transform: scale(.92); opacity: 0;} to {transform: scale(1); opacity: 1;}}
    .stButton>button {
        border-radius: 12px; font-weight: 600; padding: .55rem 1.2rem;
    }
    div[data-testid="stTextArea"] textarea {border-radius: 14px; font-size: 1.05rem;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


def to_name(label):
    """Convert a raw model label (number or string) to an emotion name."""
    if isinstance(label, (int, np.integer)):
        return LABEL_MAP.get(int(label), str(label))
    try:
        return LABEL_MAP.get(int(label), str(label))
    except (ValueError, TypeError):
        return str(label).lower()


def predict(model, text):
    """Return (emotion_name, probabilities DataFrame)."""
    pred = model.predict([text])[0]
    classes = list(getattr(model, "classes_", []))

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba([text])[0]
    elif hasattr(model, "decision_function"):
        d = np.atleast_1d(model.decision_function([text])[0])
        e = np.exp(d - d.max())
        scores = e / e.sum()
    else:
        scores = None

    if scores is not None and classes:
        df = pd.DataFrame({
            "Emotion": [to_name(c) for c in classes],
            "Probability": scores,
        }).sort_values("Probability", ascending=False)
    else:
        df = None
    return to_name(pred), df


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app uses an NLP model trained on text to detect the "
        "emotion expressed in a sentence."
    )
    st.subheader("Emotions it detects")
    for name, s in EMOTION_STYLE.items():
        st.markdown(f"{s['emoji']} **{name.capitalize()}**  \n<small>{s['desc']}</small>",
                    unsafe_allow_html=True)
    st.divider()
    show_chart = st.toggle("Show probability chart", value=True)

# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Emotion Detection</h1>
        <p>Type a sentence and let NLP read the feeling behind it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load `{MODEL_PATH}`. Keep it in the same folder as app.py.\n\n{e}")
    st.stop()

if "text" not in st.session_state:
    st.session_state.text = ""
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("##### Try an example")
cols = st.columns(3)
for i, (label, sample) in enumerate(EXAMPLES.items()):
    if cols[i % 3].button(label, use_container_width=True, key=f"ex_{i}"):
        st.session_state.text = sample

text = st.text_area(
    "Your text",
    key="text",
    height=130,
    placeholder="e.g. I'm so excited for the trip tomorrow!",
)

analyze = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

if analyze:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            emotion, probs = predict(model, text.strip())

        style = EMOTION_STYLE.get(
            emotion, {"emoji": "🤔", "color": "#555", "desc": ""}
        )
        conf_text = ""
        if probs is not None:
            conf_text = f"Confidence: {probs.iloc[0]['Probability'] * 100:.1f}%"

        st.markdown(
            f"""
            <div class="result-card" style="background: linear-gradient(135deg, {style['color']}, {style['color']}cc);">
                <div class="emoji">{style['emoji']}</div>
                <div class="label">{emotion}</div>
                <div class="conf">{conf_text}</div>
                <div class="desc">{style['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probs is not None and show_chart:
            chart_df = probs.copy()
            chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, size=26)
                .encode(
                    x=alt.X("Probability:Q", axis=alt.Axis(format="%", title=None), scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Emotion:N", sort="-x", title=None),
                    color=alt.Color(
                        "Emotion:N",
                        scale=alt.Scale(
                            domain=list(EMOTION_STYLE.keys()),
                            range=[v["color"] for v in EMOTION_STYLE.values()],
                        ),
                        legend=None,
                    ),
                    tooltip=["Emotion", alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=250)
            )
            st.markdown("##### Emotion probabilities")
            st.altair_chart(chart, use_container_width=True)

        st.session_state.history.insert(
            0, {"Text": text.strip(), "Emotion": f"{style['emoji']} {emotion}"}
        )

if st.session_state.history:
    with st.expander("🕘 Recent predictions", expanded=False):
        st.dataframe(
            pd.DataFrame(st.session_state.history[:10]),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

st.caption("Built with Streamlit · scikit-learn · NLP")