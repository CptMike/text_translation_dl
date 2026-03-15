# Project Report — English to Greek Translation

## 1. Project Description

This project is about building a translation system from English to Greek using Deep Learning. We start with a pre-trained model on a real dataset and and fine tune it. In the end we evaluate how good the translations are.

---

## 2. Language Choice

The translation chosen was English to Greek. It is also a more challenging language pair compared to something like English to French, because Greek has a different alphabet and more complex grammar rules.

---

## 3. Dataset

The dataset used was  `opus_books`  from HuggingFace. It contains parallel sentences in English and Greek, taken from books. The dataset was split into 90% training and 10% validation.

The data was tokenized using the MarianMT tokenizer and capped at 128 tokens per sentence to keep training manageable.

---

## 4. Model Choice

MarianMT was chosen (`Helsinki-NLP/opus-mt-en-el`) for the following reasons:

- It is a model specifically built for machine translation.
- Helsinki-NLP provides a pre-trained version for English-Greek, meaning the model already has a foundation in both languages.
- It is fully supported by the HuggingFace `transformers` library.


---

## 5. Fine-Tuning

The model was fine-tuned using `Seq2SeqTrainer` from HuggingFace with the following hyperparameteres and values:



-Learning rate | 2e-5 
-Batch size | 16 
-Epochs | 3 
-Weight decay | 0.01 

These hyperparameters follow the standard recommendations for fine-tuning pre-trained transformer models on small datasets, as suggested in the HuggingFace documentation.

---

## 6. Evaluation Metrics

The default metric for translation tasks is BLEU, but it has known problems. It only measures word overlap between the prediction and the reference, so it can miss translations that are correct but use different words.

Later, BERTScore as the primary metric. BERTScore uses a pre-trained language model to compare the meaning of sentences and not just the words. This makes it a better fit for Greek, where the same meaning can be expressed in many different word forms.

SacreBLEU was kept as a secondary baseline for comparison.

---

## 7. Results

Metric:SacreBLEU  Score: 12.31 (out of 100)
Metric: BERTScore F1 Score:  0.785 (out of 1)

The BERTScore F1 of 0.785 means the model captures around 78% of the semantic meaning of the correct translation. For a 3-epoch fine-tune, this is a decent result.

---

## 8. Demo Application

A  web interface was built with Streamlit. After training, the model is loaded in the browser where the user can type any English sentence and get an instant Greek translation.

---

## 9. Notes

During testing on a second machine, training failed because the available GPU (NVIDIA GTX 960) was too old to be supported by the current version of PyTorch. PyTorch detected the GPU but could not run on it, causing a CUDA error. The code was updated to automatically detect this situation and fall back to CPU training, so the project works on any machine regardless of GPU compatibility.

