import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from extract_features import extract_features
from generate_caption import generate_caption

# Paths
model_path = 'models/model.h5'
tokenizer_path = 'data/tokenizer.pkl'
max_length = 34  # Set according to your dataset preprocessing

# Load model and tokenizer
model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

st.title("Image Caption Generator")
uploaded_file = st.file_uploader("Upload an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Generating caption...")
    photo_feature = extract_features(uploaded_file)
    caption = generate_caption(photo_feature, model, tokenizer, max_length)
    st.write("Caption:", caption)
