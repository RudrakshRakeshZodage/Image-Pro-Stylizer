import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import cv2
import base64
import io
import zipfile

from generator_model import Generator  # AnimeGANv2 Generator class


@st.cache_resource(show_spinner=True)
def load_generator_model(ckpt_dir):
    model = Generator()
    dummy_input = tf.random.normal([1, 256, 256, 3])
    model(dummy_input)
    checkpoint = tf.train.Checkpoint(generator=model)
    checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()
    return model


def preprocess_image(img, target_size=(256, 256)):
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized).astype(np.float32)
    img_array = img_array / 127.5 - 1
    return np.expand_dims(img_array, axis=0)


def postprocess_image(output, original_size):
    output = (output + 1) * 127.5
    output = tf.cast(output, tf.uint8)
    image_np = output[0].numpy()
    return Image.fromarray(image_np).resize(original_size)


@st.cache_data(show_spinner=False)
def cartoonify_image(img_bgr):
    for _ in range(5):
        img_bgr = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    data = np.float32(img_bgr).reshape((-1, 3))
    _, label, center = cv2.kmeans(
        data,
        8,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
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
    st.set_page_config(
        page_title="Image Stylizer", layout="wide", initial_sidebar_state="collapsed"
    )

    # ✨ CSS Styling
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] { background-color: transparent; }
        .main-header {
            background: linear-gradient(to right, #4e54c8, #8f94fb);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 1.5rem;
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
        .reportview-container .main .block-container { padding-top: 0rem; }
        a {
            text-decoration: none;
            color: #4e54c8;
            font-weight: 600;
        }
        a:hover {
            text-decoration: underline;
            color: #3b3f99;
        }
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.9rem;
            margin-top: 40px;
            margin-bottom: 20px;
            font-family: 'Segoe UI', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="main-header">
            <h1>🎨 Image Pro Stylizer</h1>
            <p>Convert your image into amazing Anime styles and cartoons using <b>OpenCV</b> and <b>AnimeGANv2</b> (Hayao, Shinkai, Paprika).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        orig_w, orig_h = image.size

        with st.spinner("🚀 Loading AnimeGANv2 models..."):
            hayao_model = load_generator_model("AnimeGANv2/checkpoint/generator_Hayao_weight")
            shinkai_model = load_generator_model("AnimeGANv2/checkpoint/generator_Shinkai_weight")
            paprika_model = load_generator_model("AnimeGANv2/checkpoint/generator_Paprika_weight")

        with st.spinner("🎨 Cartoonifying using OpenCV..."):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cartoon_img = cartoonify_image(img_bgr)
            cartoon_pil = Image.fromarray(cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2RGB)).resize(
                (orig_w, orig_h)
            )

        with st.spinner("🧠 Stylizing ..."):
            input_tensor = preprocess_image(image)
            hayao_out = postprocess_image(hayao_model(input_tensor, training=False), (orig_w, orig_h))
            shinkai_out = postprocess_image(shinkai_model(input_tensor, training=False), (orig_w, orig_h))
            paprika_out = postprocess_image(paprika_model(input_tensor, training=False), (orig_w, orig_h))

        st.markdown('<h2 style="text-align:center; color:#4e54c8;">🖼 Output Gallery</h2>', unsafe_allow_html=True)

        titles = ["Original", "Cartoonified", "Hayao", "Shinkai", "Paprika"]
        images = [image, cartoon_pil, hayao_out, shinkai_out, paprika_out]
        cols = st.columns(5)

        for idx, col in enumerate(cols):
            with col:
                st.image(images[idx], caption=titles[idx], use_column_width=True)
                img_bytes = image_to_bytes(images[idx])
                b64_img = base64.b64encode(img_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64_img}" download="{titles[idx].lower()}.png">⬇ Download</a>'
                st.markdown(href, unsafe_allow_html=True)

        # ZIP Download button for all images
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            for title, img_obj in zip(titles, images):
                zip_file.writestr(f"{title.lower()}.png", image_to_bytes(img_obj))
        zip_buffer.seek(0)
        b64_zip = base64.b64encode(zip_buffer.read()).decode()

        st.markdown(
            f"""
            <div style="text-align:center; margin-top:20px;">
                <a href="data:application/zip;base64,{b64_zip}" download="all_outputs.zip" style="
                    background-color: #4e54c8;
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                ">⬇ Download All as ZIP</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success("✅ All outputs generated successfully!")

    # Footer with copyright
    st.markdown(
        """
        <div class="footer">
            &copy; 2025 Rudraksh Zodage
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
