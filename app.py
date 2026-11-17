from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.inference import (
    CLASS_NAMES,
    get_device,
    load_resnet_model,
    load_yolo_model,
    predict_resnet,
    predict_yolo,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
YOLO_CHECKPOINT = MODEL_DIR / "best.pt"
RESNET_CHECKPOINT = MODEL_DIR / "resnet18_best.pth"

CLASS_LABELS_VI = {
    "Blast": "B\u1ec7nh \u0111\u1ea1o \u00f4n",
    "Brown spot": "B\u1ec7nh \u0111\u1ed1m n\u00e2u",
    "Healthy": "L\u00e1 kh\u1ecfe m\u1ea1nh",
    "Leaf smut": "B\u1ec7nh than l\u00e1",
    "Rice Tungro": "B\u1ec7nh v\u00e0ng l\u00f9n / l\u00f9n xo\u1eafn l\u00e1",
    "Sheath blight": "B\u1ec7nh kh\u00f4 v\u1eb1n",
}

st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading YOLO11n-cls...")
def cached_yolo(checkpoint: str):
    return load_yolo_model(checkpoint)


@st.cache_resource(show_spinner="Loading ResNet18...")
def cached_resnet(checkpoint: str, device_name: str):
    return load_resnet_model(checkpoint, device=torch.device(device_name))


def result_table(probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Class": [CLASS_LABELS_VI[name] for name in CLASS_NAMES],
            "Probability (%)": probabilities * 100,
        }
    )
    return frame.sort_values("Probability (%)", ascending=False).reset_index(drop=True)


def show_result(title: str, probabilities: np.ndarray):
    best_index = int(np.argmax(probabilities))
    best_class = CLASS_NAMES[best_index]
    confidence = float(probabilities[best_index])

    st.subheader(title)
    st.metric("K\u1ebft qu\u1ea3 d\u1ef1 \u0111o\u00e1n", CLASS_LABELS_VI[best_class])
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    st.dataframe(result_table(probabilities), use_container_width=True, hide_index=True)
    chart = pd.DataFrame(
        {"Probability (%)": probabilities},
        index=[CLASS_LABELS_VI[name] for name in CLASS_NAMES],
    )
    st.bar_chart(chart)


def main():
    st.title("🌾 Ph\u00e2n lo\u1ea1i b\u1ec7nh tr\u00ean l\u00e1 l\u00faa")
    st.caption("So s\u00e1nh m\u00f4 h\u00ecnh YOLO11n-cls v\u00e0 ResNet18")
    st.info(
        "C\u00f4ng c\u1ee5 h\u1ed7 tr\u1ee3 ph\u00e2n lo\u1ea1i h\u00ecnh \u1ea3nh, kh\u00f4ng thay th\u1ebf ch\u1ea9n \u0111o\u00e1n c\u1ee7a chuy\u00ean gia n\u00f4ng nghi\u1ec7p."
    )

    with st.sidebar:
        st.header("C\u1ea5u h\u00ecnh")
        mode = st.radio(
            "Ch\u1ecdn m\u00f4 h\u00ecnh",
            ["So sanh ca hai", "YOLO11n-cls", "ResNet18"],
        )
        st.caption(f"Thiet bi: {get_device()}")
        st.caption("Input size: 224 x 224")

    uploaded = st.file_uploader(
        "T\u1ea3i \u1ea3nh l\u00e1 l\u00faa (JPG, JPEG, PNG, BMP, WEBP)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is None:
        st.markdown(
            "### H\u01b0\u1edbng d\u1eabn\n"
            "1. Ch\u1ecdn m\u00f4 h\u00ecnh trong thanh b\u00ean tr\u00e1i.\n"
            "2. T\u1ea3i l\u00ean m\u1ed9t \u1ea3nh l\u00e1 l\u00faa r\u00f5 n\u00e9t.\n"
            "3. B\u1ea5m n\u00fat B\u1eaft \u0111\u1ea7u d\u1ef1 \u0111o\u00e1n."
        )
        return

    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"Khong doc duoc anh: {exc}")
        return

    st.image(image, caption="\u1ea2nh \u0111\u1ea7u v\u00e0o", use_container_width=True)

    if not st.button("Bat dau du doan", type="primary"):
        return

    device = get_device()
    try:
        if mode in ("So sanh ca hai", "YOLO11n-cls") and not YOLO_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Thieu {YOLO_CHECKPOINT}. Dat file best.pt vao thu muc models/."
            )
        if mode in ("So sanh ca hai", "ResNet18") and not RESNET_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Thieu {RESNET_CHECKPOINT}. Dat file resnet18_best.pth vao thu muc models/."
            )

        if mode == "So sanh ca hai":
            left, right = st.columns(2)
            with st.spinner("Dang chay ca YOLO11n-cls va ResNet18..."):
                yolo = cached_yolo(str(YOLO_CHECKPOINT))
                resnet = cached_resnet(str(RESNET_CHECKPOINT), str(device))
                yolo_probs = predict_yolo(yolo, image, device=device)
                resnet_probs = predict_resnet(resnet, image, device=device)
            with left:
                show_result("YOLO11n-cls", yolo_probs)
            with right:
                show_result("ResNet18", resnet_probs)
        elif mode == "YOLO11n-cls":
            with st.spinner("Dang chay YOLO11n-cls..."):
                yolo = cached_yolo(str(YOLO_CHECKPOINT))
                probabilities = predict_yolo(yolo, image, device=device)
            show_result("YOLO11n-cls", probabilities)
        else:
            with st.spinner("Dang chay ResNet18..."):
                resnet = cached_resnet(str(RESNET_CHECKPOINT), str(device))
                probabilities = predict_resnet(resnet, image, device=device)
            show_result("ResNet18", probabilities)

    except Exception as exc:
        st.error(f"Khong the chay du doan: {exc}")
        st.exception(exc)

    st.caption(
        "L\u01b0u \u00fd: Confidence l\u00e0 \u0111i\u1ec3m tin c\u1eady do m\u00f4 h\u00ecnh \u0111\u01b0a ra, kh\u00f4ng ph\u1ea3i b\u1ea3o \u0111\u1ea3m d\u1ef1 \u0111o\u00e1n \u0111\u00fang. "
        "M\u00f4 h\u00ecnh ch\u1ec9 \u0111\u01b0\u1ee3c hu\u1ea5n luy\u1ec7n tr\u00ean 6 l\u1edbp v\u00e0 ch\u01b0a \u0111\u01b0\u1ee3c x\u00e1c nh\u1eadn l\u00e0 c\u00f3 th\u1ec3 ph\u00e1t hi\u1ec7n m\u1ecdi \u1ea3nh ngo\u00e0i ph\u00e2n ph\u1ed1i."
    )


if __name__ == "__main__":
    main()
