import streamlit as st
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import os
import numpy as np

# Set up the page configuration
st.set_page_config(
    page_title="MusicGen App",
    page_icon="🎵",  # You can also use a local path to an image or an emoji
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for background image and other styles
background_image_url = "https://i.imgur.com/T47F1kl.jpeg"  # Direct URL to the background image on Imgur

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stTitle {{
        color: #FF0000
;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Set up the app title
st.title("MusicGen App")

# Load the model and processor
@st.cache_resource
def load_model_and_processor():
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key is None:
        st.error("Hugging Face API key not found. Please set the HUGGINGFACE_API_KEY environment variable.")
        return None, None, None

    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", use_auth_token=api_key)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", use_auth_token=api_key)
    return model, processor, device

model, processor, device = load_model_and_processor()

if model is not None and processor is not None:
    # Get the sampling rate from the model configuration
    sampling_rate = model.config.audio_encoder.sampling_rate

    # User input
    st.markdown("**Enter text to generate music:**")
    text_input = st.text_input("", "a soulful piano track")
    #text_input = st.text_input("Enter text to generate music:", "a soulful piano track")

    # Generate music button
    if st.button("Generate Music"):
        inputs = processor(
            text=[text_input],
            padding=True,
            return_tensors="pt"
        )

        with st.spinner("Generating music..."):
            audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=512)
            audio_data = audio_values[0].cpu().numpy()

        st.audio(audio_data, format="audio/wav", sample_rate=sampling_rate)
        st.success("Music generated successfully!")
else:
    st.error("Failed to load model and processor. Please check the Hugging Face API key and try again.")
