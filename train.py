# Fine-tune MarianMT on English to Greek translation

import os
import numpy as np
import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate as hf_evaluate

from dataset import get_tokenized_datasets

MODEL_NAME = 'Helsinki-NLP/opus-mt-en-el'
OUTPUT_DIR = './outputs/finetuned-en-el'


def compute_metrics(eval_preds, tokenizer):
    # track bleu during training to see if its improving
    sacrebleu = hf_evaluate.load('sacrebleu')
    preds, labels = eval_preds

    # put the padding back so we can decode the labels into text
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = sacrebleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
    return {'bleu': round(result['score'], 4)}


def cuda_works():
    # some old GPUs are detected but not actually compatible with the pytorch version
    try:
        torch.zeros(1).cuda()
        return True
    except Exception:
        return False

use_cpu = not (torch.cuda.is_available() and cuda_works())
if use_cpu:
    print("no compatible GPU found, training on CPU")


def train():
    data, tokenizer = get_tokenized_datasets()

    print("Loading model...")
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('./outputs/logs', exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        predict_with_generate=True,
        generation_max_length=128,
        load_best_model_at_end=True,
        metric_for_best_model='bleu',
        greater_is_better=True,
        logging_dir='./outputs/logs',
        logging_steps=50,
        save_total_limit=2,
        fp16=False,
        use_cpu=use_cpu,
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        # stop early if model stops improving
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    save_path = os.path.join(OUTPUT_DIR, 'best-model')
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"saved to {save_path}")

    return trainer


if __name__ == "__main__":
    trainer = train()
    print("Final evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v}")
