# Evaluate the translation model using BERTScore and SacreBLEU

import os
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
import evaluate as hf_evaluate

from dataset import load_opus_books

MODEL_PATH = './outputs/finetuned-en-el/best-model'
NUM_SAMPLES = 100


def cuda_works():
    try:
        a = torch.zeros(1).cuda()
        b = torch.ones(1).cuda()
        _ = (a + b).item()  # forces a compute kernel — fails on sm_52, caught by except
        return True
    except Exception:
        return False


def load_model_and_tokenizer(model_path):
    print("loading model...")
    tok = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.eval()

    # use GPU if available and compute-capable, otherwise CPU
    device = 'cuda' if (torch.cuda.is_available() and cuda_works()) else 'cpu'
    model.to(device)
    print(f"using device: {device}")
    return model, tok


def get_translations(src_sentences, model, tokenizer):
    device = next(model.parameters()).device.type
    results = []
    batch_size = 32

    for i in tqdm(range(0, len(src_sentences), batch_size), desc="Translating"):
        batch = src_sentences[i : i + batch_size]

        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        results.extend(decoded)

    print(f"translated {len(results)} sentences")
    return results


def main():
    if not os.path.isdir(MODEL_PATH):
        print("no fine-tuned model found, using base model")
        model_path = 'Helsinki-NLP/opus-mt-en-el'
    else:
        model_path = MODEL_PATH

    model, tokenizer = load_model_and_tokenizer(model_path)

    print("Loading data...")
    raw = load_opus_books()
    val_data = raw['validation'].select(range(NUM_SAMPLES))

    en_sentences = [ex['en'] for ex in val_data['translation']]
    el_refs = [ex['el'] for ex in val_data['translation']]

    preds = get_translations(en_sentences, model, tokenizer)

    # SacreBLEU
    sacrebleu = hf_evaluate.load('sacrebleu')
    bleu_result = sacrebleu.compute(predictions=preds, references=[[r] for r in el_refs])
    bleu_score = round(bleu_result['score'], 4)

    # BERTScore - runs on cpu to avoid cuda timeout
    bertscore = hf_evaluate.load('bertscore')
    bert_result = bertscore.compute(predictions=preds, references=el_refs, lang='el', device='cpu')

    bert_p = round(sum(bert_result['precision']) / len(bert_result['precision']), 4)
    bert_r = round(sum(bert_result['recall']) / len(bert_result['recall']), 4)
    bert_f1 = round(sum(bert_result['f1']) / len(bert_result['f1']), 4)

    print(f"\nSacreBLEU: {bleu_score}")
    print(f"BERTScore F1: {bert_f1} (precision: {bert_p}, recall: {bert_r})")

    print("\nExample translations:")
    for i in range(5):
        print(f"\nEN : {en_sentences[i]}")
        print(f"EL : {preds[i]}")
        print(f"REF: {el_refs[i]}")


if __name__ == "__main__":
    main()
