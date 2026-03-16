# English to Greek Translation

## Project Overview

A Deep Learning course project that fine-tunes a pre-trained translation model to translate text from English to Greek. We take a model that already knows how to translate (MarianMT by Helsinki-NLP) and make it better by training it on a book dataset. We then measure how good the translations are using two metrics: BERTScore and SacreBLEU.

---

## Files

- **dataset.py** — downloads the dataset from HuggingFace and prepares it for training
- **train.py** — fine-tunes the translation model and saves the best version
- **eval_model.py** — loads the saved model and prints the evaluation scores
- **app.py** — Streamlit web app to translate sentences interactively in the browser
- **requirements.txt** — all the Python packages you need to install
- **README.md** — this file

---

## Setup

1. Clone the repo
```bash
git clone https://github.com/CptMike/text_translation_dl
cd text_translation_dl
```

2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## How to run

**Step 1 — Train the model** (this takes a while)
```bash
python3 train.py
```
The model will be saved in `outputs/finetuned-en-el/best-model/`

**Step 2 — Evaluate the model**
```bash
python3 eval_model.py
```
This prints BERTScore and SacreBLEU scores side by side.

**Step 3 — Run the web app**
```bash
streamlit run app.py
```
Opens a browser where you can type English text and see the Greek translation.

---

## Results

| Metric | Score |
|---|---|
| SacreBLEU | 12.31 |
| BERTScore F1 | 0.785 |

BERTScore is the main metric because it measures meaning similarity, not just word overlap. This matters more for Greek which has complex grammar.

---
