# Streamlit app

import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer

# where we saved the trained model
MODEL_PATH = './outputs/finetuned-en-el/best-model'
# base model to use if we havent trained yet
FALLBACK_MODEL = 'Helsinki-NLP/opus-mt-en-el'


@st.cache_resource
def load_model():
    # load the model once and cache it so it doesnt reload every time
    if __import__('os').path.isdir(MODEL_PATH):
        path = MODEL_PATH
    else:
        path = FALLBACK_MODEL

    tok = MarianTokenizer.from_pretrained(path)
    model = MarianMTModel.from_pretrained(path)
    model.eval()
    return model, tok


def translate(text, model, tok):
    inputs = tok(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        # num_beams=4 means the model tries 4 different translations and picks the best one
        out = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)


st.set_page_config(page_title="English to Greek Translator", menu_items={})

# hide the top toolbar and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# FRONTEND
st.title("English to Greek Translator")
st.write("Type an English sentence below and click Translate.")

model, tok = load_model()

user_input = st.text_area("English text", height=150, placeholder="e.g. The sun was setting over the mountains.")

if st.button("Translate"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Translating..."):
            result = translate(user_input, model, tok)
        st.subheader("Greek translation")
        st.write(result)
