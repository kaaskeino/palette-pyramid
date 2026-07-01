"""
app.py  —  Palette Pyramid Generator
Streamlit UI: 画像 + タイトル/サブタイトルを入力するとピラミッド配色図(PNG)が生成されダウンロードできる
"""

import io
import streamlit as st
from PIL import Image
from image_to_palette_pyramid import extract_palette, draw_pyramid_image

# ページ設定
st.set_page_config(
    page_title="Palette Pyramid",
    page_icon="🎨",
    layout="centered",
)

# タイトル
st.title("🎨 Palette Pyramid")
st.caption("Upload a painting or artwork to extract its color palette as a triangle pyramid.")

# --- 入力フォーム ---
uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "webp"])

col1, col2 = st.columns(2)
with col1:
    title = st.text_input("Title", placeholder="Skrik")
with col2:
    subtitle = st.text_input("Subtitle", placeholder="Edvard Munch, 1893")

n = st.select_slider(
    "Number of colors",
    options=[3, 4, 5],
    value=3,
    format_func=lambda x: f"n={x}  →  {x*x} colors",
)

# --- 生成ボタン ---
if uploaded:
    st.image(uploaded, caption="Input image", use_container_width=True)

generate = st.button("Generate pyramid", type="primary", disabled=uploaded is None)

if generate and uploaded:
    with st.spinner("Extracting palette…"):
        img_pil = Image.open(uploaded).convert("RGB")

        # 一時ファイルを使わずメモリ上で処理
        buf_in = io.BytesIO()
        img_pil.save(buf_in, format="PNG")
        buf_in.seek(0)

        palette = extract_palette(buf_in, n_colors=n * n)
        result = draw_pyramid_image(palette, n=n, title=title, subtitle=subtitle)

    st.success("Done!")
    st.image(result, caption="Palette Pyramid", use_container_width=True)

    # ダウンロードボタン
    buf_out = io.BytesIO()
    result.save(buf_out, format="PNG")
    buf_out.seek(0)

    filename = (title.lower().replace(" ", "_") or "palette_pyramid") + ".png"
    st.download_button(
        label="⬇️  Download PNG",
        data=buf_out,
        file_name=filename,
        mime="image/png",
    )
