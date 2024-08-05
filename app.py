import streamlit as st
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import os
import numpy as np

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
    text_input = st.text_input("Enter text to generate music:", "a soulful piano track")

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
