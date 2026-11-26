from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Skin Disease Image Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    APP_DIR / "models" / "best.pt",
    APP_DIR / "best.pt",
]
IMAGE_SIZE = 224  # Keep consistent with the training notebook.
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


@st.cache_resource(show_spinner="Loading the trained YOLOv8 model...")
def load_model(model_path: str) -> YOLO:
    """Load and cache the trained YOLOv8 classification checkpoint."""
    return YOLO(model_path)


def find_model_file() -> Path | None:
    """Find best.pt in the recommended models folder or project root."""
    for candidate in MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def class_name_for(names, class_index: int) -> str:
    """Handle Ultralytics class names whether represented as a dict or list."""
    if isinstance(names, dict):
        return str(names.get(class_index, names.get(str(class_index), class_index)))
    if isinstance(names, (list, tuple)) and 0 <= class_index < len(names):
        return str(names[class_index])
    return str(class_index)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🔬 Skin Image Classifier")
    st.caption("YOLOv8 · Image classification")
    st.markdown("---")
    st.markdown("**Supported categories**")
    st.markdown(
        "- Chickenpox\n- Measles\n- Monkeypox\n- Normal"
    )
    st.markdown("---")
    st.caption("Educational project only. Not a medical diagnostic tool.")


# -----------------------------------------------------------------------------
# Main page
# -----------------------------------------------------------------------------
st.title("Skin Disease Image Classification")
st.write(
    "Upload a skin image to see the class predicted by the trained YOLOv8 model."
)
st.info(
    "This application provides an experimental image-classification result only. "
    "It cannot confirm a disease or replace assessment by a qualified healthcare professional."
)

model_path = find_model_file()
if model_path is None:
    st.error("The trained model file `best.pt` was not found.")
    st.markdown(
        "Place your trained checkpoint in **`models/best.pt`** inside this project. "
        "Alternatively, place it beside `app.py` as `best.pt`, then restart the app."
    )
    st.stop()

try:
    model = load_model(str(model_path))
except Exception as exc:
    st.error("The model could not be loaded. Check that `best.pt` is the trained YOLOv8 classification checkpoint.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()

st.caption(f"Model loaded: `{model_path.relative_to(APP_DIR)}`")

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=SUPPORTED_TYPES,
    help="Upload one JPG, JPEG, PNG, WEBP, or BMP image.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error("The uploaded file could not be opened as an image. Please choose a valid image file.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.stop()

    preview_col, result_col = st.columns([1, 1], gap="large")
    with preview_col:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with result_col:
        st.subheader("Classification result")
        if st.button("Analyze image", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing image..."):
                    predictions = model.predict(
                        source=image,
                        imgsz=IMAGE_SIZE,
                        verbose=False,
                    )

                if not predictions or predictions[0].probs is None:
                    st.error("The loaded model did not return classification probabilities. Please check the model file.")
                else:
                    result = predictions[0]
                    probabilities = result.probs.data.detach().cpu().numpy()
                    top_index = int(result.probs.top1)
                    top_confidence = float(result.probs.top1conf)
                    predicted_class = class_name_for(model.names, top_index)

                    st.success(f"Predicted class: **{predicted_class}**")
                    st.metric("Model confidence", f"{top_confidence:.2%}")

                    class_scores = {
                        class_name_for(model.names, index): float(score)
                        for index, score in enumerate(probabilities)
                    }
                    class_scores = dict(
                        sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
                    )

                    st.markdown("**Class probability scores**")
                    for class_label, score in class_scores.items():
                        score_col, value_col = st.columns([5, 1])
                        score_col.write(class_label)
                        value_col.write(f"{score:.2%}")
                        st.progress(max(0.0, min(1.0, score)))
                    st.caption(
                        "Confidence scores describe the model's output, not the probability "
                        "that a person medically has a disease."
                    )
            except Exception as exc:
                st.error("Prediction failed. Please try another image or verify the model checkpoint.")
                with st.expander("Technical details"):
                    st.code(str(exc))
else:
    st.markdown("### Get started")
    st.write("1. Upload an image using the uploader above.\n2. Select **Analyze image**.\n3. Review the predicted class and model scores.")

st.markdown("---")
st.caption(
    "Research/educational demonstration · Chickenpox, Measles, Monkeypox, Normal · "
    "Not intended for clinical use."
)
