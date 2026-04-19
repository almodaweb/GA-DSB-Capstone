import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
from openai import OpenAI

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DermAI Scan Platform", layout="wide")
    
# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "step" not in st.session_state:
    st.session_state.step = 1

if "results" not in st.session_state:
    st.session_state.results = None

if "explanation" not in st.session_state:
    st.session_state.explanation = None

# =========================
# GLOBAL THEME 
# =========================
st.markdown("""
<style>

.stApp { background-color: #ffffff; }

h1, h2, h3 { color: #4c1d95 !important; }

p, span, label { color: #2e1065 !important; }

.stButton>button {
    background-color: #6d28d9 !important;
    color: white !important;
    border-radius: 10px;
    height: 45px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #4c1d95 !important;
}

.stButton>button * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Fix ALL text rendering consistently */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: #3e2f84 !important;
}

/* Prevent white text from Streamlit overrides */
section.main * {
    color: inherit;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../Model/SCIN_model_DenseNet.keras")

@st.cache_resource
def load_labels():
    with open("../Model/SCIN_labels_DenseNet.json", "r") as f:
        labels = json.load(f)
    return {int(k): v for k, v in labels.items()}

model = load_model()
labels = load_labels()

# =========================
# PREPROCESS + PREDICT
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def predict(image):
    preds = model.predict(preprocess_image(image), verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:3]

    return [
        {"Condition": labels[i], "Confidence": float(preds[i])}
        for i in top_idx
    ]


# =========================
# OPENAI
# =========================
client = OpenAI(
    api_key=os.getenv("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"),
    base_url="https://openrouter.ai/api/v1"
)

def explain(condition):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a dermatology AI assistant."},
            {"role": "user", "content": f"Explain {condition} with causes and care."}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content


# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":

    # ========================
    # BRAND HERO 
    # =========================
    st.markdown("""
    <style>

    /* TEXT BOX */
    .text-box {
        background: #ede9fe;
        padding: 35px;
        border-radius: 25px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* TITLE */
    .hero-title {
        font-size: 42px;
        font-weight: 700;
        color: #4c1d95;
    }

    /* SUBTEXT */
    .hero-sub {
        font-size: 18px;
        margin-top: 10px;
        color: #5b21b6;
    }

    /* STORE BUTTONS */
    .store-container img {
        height: 40px;
        margin-right: 10px;
        margin-top: 15px;
    }

    /* CARD WRAPPER */
    .card {
        height: 100%;
        border-radius: 25px;
        overflow: hidden;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
        animation: float 5s ease-in-out infinite;
    }

    /* IMAGE */
    .card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* FLOAT ANIMATION */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-12px); }
        100% { transform: translateY(0px); }
    }

    </style>
    """, unsafe_allow_html=True)


    # COLUMNS
    col1, col2 = st.columns([1.5, 0.8])


    # LEFT SIDE
    with col1:
        st.markdown("""
    <div class="text-box">

    <div class="hero-title">
    Skin Examination Scanner
    </div>

    <div class="hero-sub">
    It’s easy to use. Using your phone's camera, you can recognize skin conditions of all shapes and sizes.
    </div>

    </div>
    """, unsafe_allow_html=True)


    # RIGHT SIDE
    with col2:
        st.markdown("""
    <div class="card">
    <img src="https://cdn.dribbble.com/userupload/44262450/file/f66d0c2c15d4ebee13094e6c8e8957b3.jpg?resize=1600x1200">
    </div>
    """, unsafe_allow_html=True)


    st.markdown("# DermAI Scan Platform")
    st.markdown("""
    <div class="hero-text" style="font-size:18px;">
                
    DermAI is an AI-powered skin analysis platform built using deep learning and dermatology expertise.
    With our application, you can analyze the condition of the skin for signs of disease in seconds.

    Assess and track skin concerns such as acne, rashes, pigmentation, and other dermatological patterns.
                
    By using our application on a regular basis, you will be able to contact your doctor in time and avoid morbidities.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # COLUMNS
    col1, col2 = st.columns([1, 1.2])

    # LEFT → IMAGE
    with col1:
        st.markdown("""
        <img src="https://www.cetaphil.in/dw/image/v2/BGGN_PRD/on/demandware.static/-/Sites-Galderma-IN-Library/default/dw67e1cc0a/AI-Tool/AI_SKIN_ANALYSIS_BANNER_497x568_1b.jpg?sw=500"
        style="
            width: 100%;
            max-width: 320px;
            height: auto;
            border-radius: 30px;
            display: block;
            margin: auto;
            box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
        ">
        """, unsafe_allow_html=True)

    # RIGHT → TEXT
    with col2:
        st.markdown("""
        <div style="
            font-size:34px;
            font-weight:800;
            color:#4c1d95;
            margin-bottom:5px;
        ">
        Monitor Your Skin Health in 3 Simple Steps
        </div>

        <div style="
            font-size:18px;
            color:#6b7280;
            margin-bottom:20px;
        ">
        </div>
        """, unsafe_allow_html=True)


        st.markdown("""
        <style>
        .step-row {
            display: flex;
            align-items: flex-start;
            margin-bottom: 14px;
        }

        .step-num {
            font-size: 36px;
            font-weight: 800;
            color: #4c1d95;
            width: 40px;
            line-height: 1;
        }

        .step-text {
            font-size: 20px;
            color: #374151;
            line-height: 1.4;
        }
        </style>
        """, unsafe_allow_html=True)


        # STEP 1
        st.markdown("""
        <div class="step-row">
            <div class="step-num">1</div>
            <div class="step-text">
                Take a clear photo of the affected skin area (5–10 cm)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # STEP 2
        st.markdown("""
        <div class="step-row">
            <div class="step-num">2</div>
            <div class="step-text">
                AI analyzes skin patterns and features
            </div>
        </div>
        """, unsafe_allow_html=True)

        # STEP 3
        st.markdown("""
        <div class="step-row">
            <div class="step-num">3</div>
            <div class="step-text">
                Get predicted conditions + confidence instantly
            </div>
        </div>
        """, unsafe_allow_html=True)

        # DISCLAIMER
        st.markdown("""
        <p style="font-size:13px; color:#6b7280; margin-top:10px;">
        DermAI does not replace professional medical advice. Always consult a dermatologist when needed.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:40px;"></div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
                st.markdown("""
                <h2 style='color:#4c1d95;'>
                Healthy skin starts with regular self-diagnosis
                </h2>
                """, unsafe_allow_html=True)

                st.markdown("""
                <p style='color:#6d28d9; font-size:16px;'>
                Skin diseases can be easily overlooked in the early stages.
                Performing weekly self-examinations will help you identify
                issues early and treat them faster.
                </p>
                """, unsafe_allow_html=True)

                # Try now button
                st.markdown("""
                <style>
                div[data-testid="stButton"] button[kind="secondary"] {
                    background-color: #3e2f84 !important;
                    color: white !important;
                    border-radius: 10px !important;
                    font-weight: 600 !important;
                }

                /* force text white */
                div[data-testid="stButton"] button[kind="secondary"] * {
                    color: white !important;
                }
                </style>
                """, unsafe_allow_html=True)

                if st.button("Try Now!"):
                    st.session_state.page = "scan"
                    st.rerun() 

    with col2:
                st.image(
                    "https://static.wixstatic.com/media/11062b_1338128c87624aa8b14a7773620d37e1~mv2.jpg",
                    use_container_width=True
                )

# =========================
# SCAN PAGE
# =========================
elif st.session_state.page == "scan":

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.session_state.step = 1
        st.rerun()

    # =========================
    # STEP INDICATOR 
    # =========================
    st.markdown("## Scan Your Skin, Get Instant Insights")
    st.markdown("""
    <p style="font-size:25px; color:#6b7280;">
    3 simple steps to identify potential skin conditions.
    </p>
    """, unsafe_allow_html=True)

    steps = ["Upload Image", "AI Analysis", "Results"]

    for i, name in enumerate(steps, 1):

        if st.session_state.step == i:
            st.markdown(f"""
            <div style="
                padding: 10px 14px;
                margin-bottom: 10px;
                border-radius: 12px;
                background-color: #ede9fe;
                border: 2px solid #4c1d95;
                font-weight: 600;
                color: #1f2937;
            ">
                {i}. {name}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="
                padding: 10px 14px;
                margin-bottom: 10px;
                border-radius: 12px;
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                color: #6b7280;
            ">
                {i}. {name}
            </div>
            """, unsafe_allow_html=True)


    # =========================
    # STEP 1 — UPLOAD 
    # =========================
    if st.session_state.step == 1:

        st.markdown("""
        <h3 style="color:#4c1d95;">Take a Photo</h3>
        <p style="color:#4b5563;">Upload a photo of the area of concern.</p>
        """, unsafe_allow_html=True)

        # UPLOADER VISIBILITY
        st.markdown("""
        <style>

        /* Upload Button ONLY */
        div[data-testid="stFileUploader"] button {
        background-color: #3e2f84 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
        }

        /* Force upload button text white */
        div[data-testid="stFileUploader"] button * {
        color: #ffffff !important;
        }
                    
        /* Upload button hover */
        div[data-testid="stFileUploader"] button:hover {
        background-color: #3e2f84 !important;
        color: white !important;
        }

        /* Fix uploader TEXT */
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small {
        color: white !important;
        }

        /* ===== START SCAN BUTTON ===== */
        div[data-testid="stButton"] button {
        background-color: #3e2f84 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: none !important;
      }

        /* Force inner text */
        div[data-testid="stButton"] button * {
        color: #ffffff !important;
        }

        * Make sure ALL text inside button stays white */
        div[data-testid="stButton"] > button * {
        color: #ffffff !important;
        }

        /* Button hover */
        div[data-testid="stButton"] > button:hover {
        background-color: #2e2266 !important;
        color: white !important;
        }

        </style>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

            if st.button("Start Scan"):
                st.session_state.image = image
                st.session_state.step = 2
                st.rerun()


    # =========================
    # STEP 2 — ANALYSIS
    # =========================
    if st.session_state.step == 2:

        st.subheader("Image Analysis")

        with st.spinner("DermAI is processing the image"):
            st.session_state.results = predict(st.session_state.image)
            top = st.session_state.results[0]
            st.session_state.explanation = explain(top["Condition"])

        st.session_state.step = 3
        st.rerun()

    # =========================
    # STEP 3 — RESULTS
    # =========================
    if st.session_state.step == 3:

        st.subheader("Scan Results")

        top = st.session_state.results[0]

        st.success(f"Most likely condition: {top['Condition']}")

        st.markdown("---")

        # =========================
        # CLINICAL INSIGHT
        # =========================
        st.subheader("Clinical Insights")
        st.write(st.session_state.explanation)

        # =========================
        # DOWNLOAD REPORT
        # =========================
        report_text = f"""
        DermAI Scan Report

        Prediction: {top['Condition']}
        Confidence: {top['Confidence']:.2%}

        Clinical Insights:
        {st.session_state.explanation}
        """
        st.markdown("""
            <div style="margin-top:40px;"></div>
            """, unsafe_allow_html=True)
        
        st.download_button(
            label="Download Report",
            data=report_text,
            file_name="dermai_report.txt",
            mime="text/plain"
        )

        st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

        # =========================
        # NEW SCAN BUTTON
        # =========================
        if st.button("New Scan"):
            st.session_state.step = 1
            st.session_state.results = None
            st.session_state.explanation = None
            st.rerun()

        st.markdown("""
            <style>

            /* NEW SCAN BUTTON */
            div[data-testid="stButton"] button {
                background-color: #3e2f84 !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }
            div[data-testid="stButton"] > button * {
                color: #ffffff !important;
            }
                    
            /* DOWNLOAD BUTTON */
            div[data-testid="stDownloadButton"] button {
                background-color: #3e2f84 !important;
                color: white !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }

            div[data-testid="stDownloadButton"] button * {
                color: white !important;
            }

            </style>
            """, unsafe_allow_html=True)
        
        