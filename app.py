import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import base64
import io
import zipfile

from generator_model import Generator


@st.cache_resource(show_spinner=True)
def load_generator_model(ckpt_dir):
    model = Generator()
    dummy_input = tf.random.normal([1, 256, 256, 3])
    model(dummy_input, training=False)
    checkpoint = tf.train.Checkpoint(generator=model)
    checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    return model


def preprocess_image(img_pil, target_size=(256, 256)):
    img = img_pil.resize(target_size)
    img = np.array(img).astype(np.float32)
    img = img / 127.5 - 1  # Normalize to [-1, 1]
    return np.expand_dims(img, axis=0)


def postprocess_image(output_tensor, target_size):
    output = output_tensor[0]
    output = (output + 1) * 127.5
    output = tf.clip_by_value(output, 0, 255)
    output = tf.cast(output, tf.uint8).numpy()
    return Image.fromarray(output).resize(target_size)


@st.cache_data
def cartoonify_image(img_bgr):
    for _ in range(5):
        img_bgr = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    data = np.float32(img_bgr).reshape((-1, 3))
    _, label, center = cv2.kmeans(data, 8, None,
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                  10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()].reshape(img_bgr.shape)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(result, edges_colored)
    return cartoon


def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    st.set_page_config(page_title="AnimeGANv2 Stylizer", layout="wide")
    st.markdown("""
    <style>
    [data-testid="stHeader"] { background: transparent; }
    .main-header {
        background: linear-gradient(to right, #4e54c8, #8f94fb);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin: 0;
    }
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🎨 AnimeGANv2 Image Stylizer</h1>
        <p>Transform your image with stunning anime effects using <b>AnimeGANv2</b> and <b>OpenCV Cartoonifier</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        orig_w, orig_h = image.size

        with st.spinner("🚀 Loading models..."):
            hayao_model = load_generator_model("AnimeGANv2/checkpoint/generator_Hayao_weight")
            shinkai_model = load_generator_model("AnimeGANv2/checkpoint/generator_Shinkai_weight")
            paprika_model = load_generator_model("AnimeGANv2/checkpoint/generator_Paprika_weight")

        with st.spinner("🎨 Cartoonifying using OpenCV..."):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cartoon_img = cartoonify_image(img_bgr)
            cartoon_pil = Image.fromarray(cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB)).resize((orig_w, orig_h))

        with st.spinner("🧠 Applying AnimeGANv2 styles..."):
            input_tensor = preprocess_image(image)
            hayao_out = postprocess_image(hayao_model(input_tensor, training=False), (orig_w, orig_h))
            shinkai_out = postprocess_image(shinkai_model(input_tensor, training=False), (orig_w, orig_h))
            paprika_out = postprocess_image(paprika_model(input_tensor, training=False), (orig_w, orig_h))

        st.markdown('<h2 style="text-align:center; color:#4e54c8;">🖼 Output Gallery</h2>', unsafe_allow_html=True)
        titles = ["Original", "Cartoonified", "Hayao", "Shinkai", "Paprika"]
        images = [image, cartoon_pil, hayao_out, shinkai_out, paprika_out]
        cols = st.columns(len(titles))

        for i, col in enumerate(cols):
            with col:
                st.image(images[i], caption=titles[i], use_column_width=True)
                b64 = base64.b64encode(image_to_bytes(images[i])).decode()
                st.markdown(f'<a href="data:image/png;base64,{b64}" download="{titles[i].lower()}.png">⬇ Download</a>',
                            unsafe_allow_html=True)

        # Download all as ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for title, img_obj in zip(titles, images):
                zip_file.writestr(f"{title.lower()}.png", image_to_bytes(img_obj))
        zip_buffer.seek(0)
        b64_zip = base64.b64encode(zip_buffer.read()).decode()
        st.markdown(f"""
        <div style="text-align:center; margin-top:20px;">
            <a href="data:application/zip;base64,{b64_zip}" download="stylized_outputs.zip"
               style="background-color:#4e54c8; color:white; padding:10px 20px;
               border-radius:8px; font-weight:600; text-decoration:none;">
               ⬇ Download All as ZIP
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.success("✅ Stylization complete!")

    st.markdown('<div class="footer">&copy; 2025 Rudraksh Zodage</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
