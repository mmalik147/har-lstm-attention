
import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from model import HARNet

st.set_page_config(
    page_title="HARNet | Activity Recognition",
    page_icon="📱",
    layout="wide"
)

CLASS_NAMES = [
    "Walking", "Upstairs", "Downstairs",
    "Sitting", "Standing", "Laying"
]

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HARNet().to(device)

    checkpoint = torch.load(
        PROJECT / "results/checkpoints/best_harnet.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device

@st.cache_data
def load_data():
    data = torch.load(
        PROJECT / "data/prepared_tensors.pt",
        map_location="cpu",
        weights_only=False
    )
    return data

model, device = load_model()
data = load_data()

st.title("📱 HARNet")
st.subheader("Human Activity Recognition from Smartphone Sensors")
st.caption("Custom BiLSTM + Temporal Attention | Built with PyTorch")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Test Accuracy", "91.11%")
col2.metric("Best Validation Accuracy", "96.51%")
col3.metric("Training Epochs", "30")

st.divider()

st.header("Activity Prediction")

index = st.slider(
    "Select a test window",
    min_value=0,
    max_value=len(data["X_test"]) - 1,
    value=0
)

if st.button("Predict Activity", type="primary"):
    x = data["X_test"][index:index+1].to(device)

    with torch.no_grad():
        logits, attention = model(x)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted = int(np.argmax(probabilities))
    actual = int(data["y_test"][index])

    result_col1, result_col2 = st.columns(2)

    result_col1.success(
        f"Predicted Activity: {CLASS_NAMES[predicted]}"
    )

    result_col2.info(
        f"Actual Activity: {CLASS_NAMES[actual]}"
    )

    st.subheader("Activity Probabilities")

    probability_data = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(6)
    }

    st.bar_chart(probability_data)

    st.divider()
    st.subheader("Smartphone Sensor Signals")

    signal = data["X_test"][index].numpy()
    time_axis = np.arange(128) / 50

    channels = [
        ("Body Acceleration", 0, 3),
        ("Body Gyroscope", 3, 6),
        ("Total Acceleration", 6, 9)
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    for ax, (title, start, end) in zip(axes, channels):
        for channel in range(start, end):
            ax.plot(
                time_axis,
                signal[:, channel],
                label=f"Channel {channel+1}"
            )

        ax.set_title(title)
        ax.set_xlabel("Time (seconds)")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Temporal Attention Visualization")

    attention_weights = attention[0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_axis, attention_weights)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Attention Weight")
    ax.set_title("HARNet Attention Across 128 Time Steps")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"Test window: {index} | "
        f"Prediction confidence: {probabilities[predicted]*100:.2f}%"
    )

st.divider()
st.caption("AML Assignment 4 | Custom PyTorch Implementation")
