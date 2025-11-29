import streamlit as st
from generator import generate_images

st.title("Local Text-to-Image Generator (Windows)")

prompt = st.text_area("Prompt", "a futuristic city at sunset, 4k, highly detailed")
negative = st.text_area("Negative prompt", "low quality, blurry")
model = st.selectbox("Model", ["runwayml/stable-diffusion-v1-5"])
num = st.slider("Number of images", 1, 4, 1)
gs = st.slider("Guidance scale", 1.0, 15.0, 7.5)
width = st.selectbox("Width", [512, 768])
height = st.selectbox("Height", [512, 768])

if st.button("Generate"):
    st.write("Generating… please wait.")
    images = generate_images(prompt, negative, num, gs, height, width, model)
    for img in images:
        st.image(img)
