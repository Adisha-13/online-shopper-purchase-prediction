"""
Streamlit app — Online Shopper Purchase Intention Prediction
Assignment 2 (Machine Learning, M.Tech AIML)

Features:
  - Upload a CSV of test data (or use the bundled sample)
  - Choose which trained model to evaluate
  - View accuracy / AUC / precision / recall / F1 / MCC
  - View confusion matrix + full classification report
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from textwrap import dedent
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="Online Shopper Purchase Predictor", layout="wide")
# --------------------------------------------------------- Custom styling --
st.markdown(
    """
    <style>
    /* Main application background */
    .stApp {
        background: linear-gradient(
            135deg,
            #fff8fc 0%,
            #f7f0ff 50%,
            #eef7ff 100%
        );
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8eaff 0%, #fff4fa 100%);
        border-right: 2px solid #ead7f7;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #ead7f7;
        border-radius: 16px;
        padding: 14px;
        box-shadow: 0 4px 12px rgba(126, 87, 194, 0.10);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 18px rgba(126, 87, 194, 0.18);
    }

    /* Main heading card */
    .cute-header {
        background: linear-gradient(90deg, #f6d5f7, #d9e7ff);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(126, 87, 194, 0.15);
    }

    .cute-header h1 {
        color: #5b3a78;
        margin: 0;
        font-size: 2.4rem;
    }

    .cute-header p {
        color: #694f79;
        margin: 8px 0 0 0;
        font-size: 1.05rem;
    }

    /* Animated header icons */
    .shopping-icon {
        display: inline-block;
        margin: 0 5px;
        animation: iconFloat 2.2s ease-in-out infinite;
    }

    .shopping-icon:nth-child(2) {
        animation-delay: 0.3s;
    }

    .shopping-icon:nth-child(3) {
        animation-delay: 0.6s;
    }

    @keyframes iconFloat {
        0%, 100% {
            transform: translateY(0) rotate(0deg);
        }
        50% {
            transform: translateY(-8px) rotate(5deg);
        }
    }

    /* Mascot card */
    .mascot-card {
        margin-top: 25px;
        padding: 15px 8px;
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid #e6d0f2;
        border-radius: 18px;
        text-align: center;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(126, 87, 194, 0.10);
    }

    .mascot-stage {
        position: relative;
        height: 65px;
    }

    .pet {
        display: inline-block;
        font-size: 34px;
        animation: petBounce 1.2s ease-in-out infinite;
    }

    .cart {
        display: inline-block;
        font-size: 34px;
        margin-left: 2px;
        animation: cartWiggle 1.2s ease-in-out infinite;
    }

    .sparkle {
        display: inline-block;
        font-size: 17px;
        animation: sparkleGlow 1s ease-in-out infinite alternate;
    }

    .mascot-message {
        color: #6f4c7d;
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 2px;
    }

    @keyframes petBounce {
        0%, 100% {
            transform: translateY(5px);
        }
        50% {
            transform: translateY(-5px);
        }
    }

    @keyframes cartWiggle {
        0%, 100% {
            transform: translateX(0) rotate(0deg);
        }
        50% {
            transform: translateX(5px) rotate(3deg);
        }
    }

    @keyframes sparkleGlow {
        from {
            opacity: 0.4;
            transform: scale(0.8) rotate(0deg);
        }
        to {
            opacity: 1;
            transform: scale(1.2) rotate(15deg);
        }
    }

    /* Rounded dataframe and upload sections */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 14px;
        padding: 8px;
    }

    /* Style Streamlit buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid #cfaee7;
        background: linear-gradient(90deg, #efd5ff, #d7e8ff);
        color: #513466;
        font-weight: 600;
    }
    /* ------------------------------------------------ Animated shop scene */
.shop-scene {
    position: relative;
    height: 320px;
    margin-bottom: 25px;
    overflow: hidden;
    border: 2px solid #ead7f7;
    border-radius: 26px;
    background: linear-gradient(
        180deg,
        #dff4ff 0%,
        #f6eaff 58%,
        #fff5df 58%,
        #fff5df 100%
    );
    box-shadow: 0 8px 22px rgba(126, 87, 194, 0.16);
}

.shop-title {
    position: absolute;
    top: 22px;
    width: 100%;
    z-index: 5;
    color: #5d3a73;
    font-size: 2.3rem;
    font-weight: 800;
    text-align: center;
    text-shadow: 2px 2px 0 #ffffff;
}

.shop-subtitle {
    position: absolute;
    top: 72px;
    width: 100%;
    z-index: 5;
    color: #765684;
    font-size: 1rem;
    font-weight: 500;
    text-align: center;
}

/* Moving clouds */
.cloud {
    position: absolute;
    font-size: 42px;
    opacity: 0.8;
    animation: cloudMove 17s linear infinite;
}

.cloud-one {
    top: 35px;
    left: -80px;
}

.cloud-two {
    top: 95px;
    left: -120px;
    font-size: 30px;
    animation-delay: 7s;
    animation-duration: 22s;
}

@keyframes cloudMove {
    from {
        transform: translateX(0);
    }

    to {
        transform: translateX(1200px);
    }
}

/* Store */
.store {
    position: absolute;
    bottom: 20px;
    left: 50%;
    width: 245px;
    height: 150px;
    transform: translateX(-50%);
    border: 4px solid #865f95;
    border-radius: 12px 12px 5px 5px;
    background: #ffdff0;
    box-shadow: 0 8px 0 rgba(115, 78, 125, 0.14);
    z-index: 6;
}

.store-sign {
    position: absolute;
    top: -38px;
    left: 25px;
    width: 190px;
    padding: 7px 4px;
    box-sizing: border-box;
    white-space: nowrap;
    border: 3px solid #865f95;
    border-radius: 12px;
    background: #fff8cf;
    color: #704879;
    font-size: 0.9rem;
    font-weight: 800;
    text-align: center;
    animation: signGlow 1.8s ease-in-out infinite alternate;
}

@keyframes signGlow {
    from {
        box-shadow: 0 0 4px #ffffff;
    }

    to {
        box-shadow: 0 0 16px #ffcaef;
    }
}

/* Striped shop awning */
.awning {
    position: absolute;
    top: 0;
    left: -4px;
    width: 245px;
    height: 32px;
    border: 4px solid #865f95;
    background: repeating-linear-gradient(
        90deg,
        #cba8f5 0,
        #cba8f5 30px,
        #fff7fc 30px,
        #fff7fc 60px
    );
}

/* Store windows and door */
.window {
    position: absolute;
    top: 58px;
    width: 58px;
    height: 58px;
    border: 4px solid #865f95;
    border-radius: 7px;
    background: linear-gradient(135deg, #dff7ff, #fff9dd);
    text-align: center;
    font-size: 28px;
    line-height: 58px;
}

.window-left {
    left: 18px;
}

.window-right {
    right: 18px;
}

.door {
    position: absolute;
    bottom: 0;
    left: 94px;
    width: 49px;
    height: 78px;
    border: 4px solid #865f95;
    border-bottom: 0;
    border-radius: 7px 7px 0 0;
    background: #efe2ff;
}

.door-handle {
    position: absolute;
    top: 38px;
    right: 6px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #865f95;
}

/* Cat and shopping cart */
.pet-cart {
    position: absolute;
    bottom: 12px;
    left: -130px;
    z-index: 10;
    white-space: nowrap;
    animation: shoppingTrip 9s linear infinite;
}

.pet-character {
    display: inline-block;
    font-size: 43px;
    animation: petWalk 0.5s ease-in-out infinite alternate;
}

.moving-cart {
    display: inline-block;
    margin-left: -4px;
    font-size: 43px;
    animation: cartBounce 0.5s ease-in-out infinite alternate;
}

.cart-with-gift {
    position: relative;
    display: inline-block;
    width: 52px;
    height: 48px;
    margin-left: -4px;
    vertical-align: bottom;
    animation: cartBounce 0.5s ease-in-out infinite alternate;
}

.cart-with-gift .moving-cart {
    position: absolute;
    left: 0;
    bottom: 0;
    z-index: 3;
    margin: 0;
    font-size: 43px;
    line-height: 1;
    animation: none;
}

.cart-items {
    position: absolute;
    left: 3px;
    top: 3px;
    z-index: 2;
    display: flex;
    align-items: flex-end;
    font-size: 14px;
    line-height: 1;
    white-space: nowrap;
}

.cart-items span + span {
    margin-left: -6px;
}



/* Cat approaches the door with an empty cart */
.shopper-enter {
    position: absolute;
    bottom: 12px;
    left: -130px;
    z-index: 10;
    white-space: nowrap;
    transform-origin: center bottom;
    animation: enterStore 12s linear infinite;
}

/* Cat exits from the same door with a gift */
.shopper-leave {
    position: absolute;
    bottom: 12px;
    left: calc(50% - 70px);
    z-index: 10;
    opacity: 0;
    white-space: nowrap;
    transform-origin: center bottom;
    animation: leaveStore 12s linear infinite;
}

@keyframes enterStore {
    /* Start outside the scene */
    0% {
        left: -130px;
        opacity: 0;
        transform: scale(1);
    }

    5% {
        opacity: 1;
    }

    /* Travel toward the shop */
    32% {
        left: calc(50% - 70px);
        opacity: 1;
        transform: scale(1);
    }

    /* Pause in front of the door */
    42% {
        left: calc(50% - 70px);
        opacity: 1;
        transform: scale(1);
    }

    /* Enter through the door */
    48% {
        left: calc(50% - 70px);
        opacity: 0;
        transform: scale(0.55);
    }

    100% {
        left: calc(50% - 70px);
        opacity: 0;
        transform: scale(0.55);
    }
}

@keyframes leaveStore {
    /* Wait while the cat is inside */
    0%, 52% {
        left: calc(50% - 70px);
        opacity: 0;
        transform: scale(0.55);
    }

    /* Emerge from the door with the gift */
    58% {
        left: calc(50% - 70px);
        opacity: 1;
        transform: scale(1);
    }

    /* Pause briefly outside the door */
    64% {
        left: calc(50% - 70px);
        opacity: 1;
        transform: scale(1);
    }

    /* Leave the shop */
    94% {
        left: calc(100% + 30px);
        opacity: 1;
        transform: scale(1);
    }

    98%, 100% {
        left: calc(100% + 60px);
        opacity: 0;
        transform: scale(1);
    }
}



@keyframes petWalk {
    from {
        transform: translateY(0) rotate(-2deg);
    }

    to {
        transform: translateY(-5px) rotate(2deg);
    }
}

@keyframes cartBounce {
    from {
        transform: translateY(0) rotate(-1deg);
    }

    to {
        transform: translateY(-3px) rotate(2deg);
    }
}

/* Floating shopping items */
.floating-item {
    position: absolute;
    z-index: 3;
    font-size: 25px;
    animation: itemFloat 2.2s ease-in-out infinite;
}

.item-one {
    top: 115px;
    left: 12%;
}

.item-two {
    top: 135px;
    right: 13%;
    animation-delay: 0.7s;
}

.item-three {
    top: 65px;
    right: 5%;
    animation-delay: 1.2s;
}

@keyframes itemFloat {
    0%, 100% {
        transform: translateY(0) rotate(-5deg);
    }

    50% {
        transform: translateY(-12px) rotate(8deg);
    }
}

/* Sparkling stars */
.scene-sparkle {
    position: absolute;
    z-index: 4;
    font-size: 18px;
    animation: sceneSparkle 1.2s ease-in-out infinite alternate;
}

.sparkle-one {
    top: 110px;
    left: 25%;
}

.sparkle-two {
    top: 90px;
    right: 26%;
    animation-delay: 0.5s;
}

@keyframes sceneSparkle {
    from {
        opacity: 0.25;
        transform: scale(0.7) rotate(0deg);
    }

    to {
        opacity: 1;
        transform: scale(1.3) rotate(25deg);
    }
}
    </style>
    """,
    unsafe_allow_html=True,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(HERE, "model", "saved_models")
SAMPLE_TEST_PATH = os.path.join(HERE, "test_data.csv")
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_model(model_name: str):
    path = os.path.join(SAVED_MODELS_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


@st.cache_data
def load_sample_data():
    return pd.read_csv(SAMPLE_TEST_PATH)


def to_binary(series):
    """Handle Revenue arriving as True/False, TRUE/FALSE, or 1/0."""
    if series.dtype == bool:
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------------------------------------------------------------- Sidebar --
st.sidebar.markdown(
    """
    <div class="mascot-card">
        <div class="mascot-stage">
            <span class="sparkle">✨</span>
            <span class="pet">🐰</span>
            <span class="cart">🛒</span>
            <span class="sparkle">✨</span>
        </div>
        <div class="mascot-message">
            Let’s predict who’s ready to shop!
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("⚙️ Controls")

model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV)", type=["csv"], help="Must include the 'Revenue' column."
)

use_sample = st.sidebar.checkbox("Use bundled sample test_data.csv", value=uploaded_file is None)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Assignment 2 — Machine Learning\nM.Tech (AIML), BITS Pilani WILP"
)

# ------------------------------------------------------------------ Main --
shop_html = dedent(
    """
    <div class="shop-scene">
        <div class="cloud cloud-one">☁️</div>
        <div class="cloud cloud-two">☁️</div>

        <div class="shop-title">🛍️ ShopperSense ✨</div>

        <div class="shop-subtitle">
            Online Shopper Purchase Intention Predictor
        
        </div>

        <div class="floating-item item-one">🎁</div>
        <div class="floating-item item-two">🛍️</div>
        <div class="floating-item item-three">💖</div>

        <div class="scene-sparkle sparkle-one">✨</div>
        <div class="scene-sparkle sparkle-two">✨</div>

        <div class="store">
            <div class="store-sign">OPEN FOR SHOPPING</div>
            <div class="awning"></div>
            <div class="window window-left">👜</div>

            <div class="door">
                <div class="door-handle"></div>
            </div>

            <div class="window window-right">👗</div>
        </div>

        <!-- Cat enters with an empty cart -->
        <div class="shopper-enter">
            <span class="pet-character">🐱</span>
            <span class="moving-cart">🛒</span>
        </div>

        <!-- Cat leaves with a gift -->
        <div class="shopper-leave">
            <span class="pet-character">🐱</span>

            <span class="cart-with-gift">
                <span class="cart-items">
                    <span>🎁</span>
                    <span>💝</span>
                    <span>🧸</span>
                </span>
                <span class="moving-cart">🛒</span>
            </span>
        </div>
    </div>
    """
).replace("\n", "")

st.markdown(shop_html, unsafe_allow_html=True)
st.write(
    "Interactive demo of 5 classification models trained on the "
    "**UCI Online Shoppers Purchasing Intention** dataset — predicting "
    "whether a website visitor's session will end in a purchase "
    "(`Revenue = True`). Upload test data, pick a model, and inspect "
    "its performance."
)

if uploaded_file is not None and not use_sample:
    data = pd.read_csv(uploaded_file)
    st.success(f"Loaded uploaded file with {data.shape[0]} rows and {data.shape[1]} columns.")
elif use_sample:
    data = load_sample_data()
    st.info(f"Using bundled sample test_data.csv ({data.shape[0]} rows).")
else:
    st.warning("Upload a CSV file or check 'Use bundled sample test_data.csv' to continue.")
    st.stop()

st.subheader("📄 Preview of test data")
st.dataframe(data.head(10), use_container_width=True)

if TARGET not in data.columns:
    st.error(
        f"The uploaded file must contain the ground-truth '{TARGET}' column "
        "(True/False) to compute evaluation metrics."
    )
    st.stop()

# Ensure Weekend (if present) matches the numeric format the pipeline was
# trained on.
X = data.drop(columns=[TARGET]).copy()
if "Weekend" in X.columns:
    X["Weekend"] = to_binary(X["Weekend"]).fillna(X["Weekend"]).astype(int)

y_true = to_binary(data[TARGET])

pipe = load_model(model_name)
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)[:, 1]

# --------------------------------------------------------------- Metrics --
st.subheader(f"📊 Evaluation metrics — {model_name}")
metrics = compute_metrics(y_true, y_pred, y_proba)
cols = st.columns(len(metrics))
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.3f}")

# ------------------------------------------------------- Confusion matrix --
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(4, 3.2))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="RdPu",
    xticklabels=["No Purchase", "Purchase"],
    yticklabels=["No Purchase", "Purchase"],
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig, use_container_width=False)

# ---------------------------------------------------- Classification report
st.subheader("📋 Classification Report")
report_dict = classification_report(
    y_true, y_pred, target_names=["No Purchase", "Purchase"], output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).transpose().round(3)
st.dataframe(report_df, use_container_width=True)

# ---------------------------------------------------------- Predictions --
st.subheader("🗂️ Row-level predictions")
pred_view = X.copy()
pred_view["Actual"] = y_true.map({1: "Purchase", 0: "No Purchase"}).values
pred_view["Predicted"] = pd.Series(y_pred, index=X.index).map({1: "Purchase", 0: "No Purchase"}).values
pred_view["Purchase Probability"] = y_proba.round(3)
st.dataframe(pred_view.head(50), use_container_width=True)

st.markdown(
    """
    <div style="
        text-align: center;
        margin-top: 30px;
        padding: 15px;
        color: #765684;
        font-size: 0.9rem;
    ">
        🛍️ Built by <b>Aditi Vithob Shanbhag</b> using Streamlit and
        scikit-learn<br>
        M.Tech AIML — Machine Learning Assignment 2
    </div>
    """,
    unsafe_allow_html=True,
)
