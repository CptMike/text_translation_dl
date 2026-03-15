# Load and preprocess the opus_books dataset for English to Greek translation

from datasets import load_dataset
from transformers import MarianTokenizer

MODEL_NAME = 'Helsinki-NLP/opus-mt-en-el'


def load_opus_books():
    print("Loading dataset...")
    dataset = load_dataset('Helsinki-NLP/opus_books', 'el-en')

    # the dataset only has train so we split it ourselves
    split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    split['validation'] = split.pop('test')

    print(f"train: {len(split['train'])}, val: {len(split['validation'])}")
    return split


def preprocess_function(examples, tokenizer, max_length=128):
    en_sentences = [ex['en'] for ex in examples['translation']]
    el_sentences = [ex['el'] for ex in examples['translation']]

    enc = tokenizer(en_sentences, max_length=max_length, truncation=True, padding='max_length')

    # tokenize the greek side
    greek_enc = tokenizer(text_target=el_sentences, max_length=max_length, truncation=True, padding='max_length')

    # replace padding with -100 so the model doesnt learn from empty spaces
    fixed_labels = []
    for ids in greek_enc['input_ids']:
        new_ids = []
        for token_id in ids:
            if token_id == tokenizer.pad_token_id:
                new_ids.append(-100)
            else:
                new_ids.append(token_id)
        fixed_labels.append(new_ids)

    enc['labels'] = fixed_labels
    return enc


def get_tokenized_datasets():
    raw_datasets = load_opus_books()

    # load tokenizer here instead of separate function
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    print("tokenizer loaded")

    tokenized = raw_datasets.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets['train'].column_names,
    )

    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print("Done tokenizing!")
    return tokenized, tokenizer


if __name__ == "__main__":
    datasets, tok = get_tokenized_datasets()
    print(datasets)
    sample = datasets['train'][0]
    print("input_ids shape:", sample['input_ids'].shape)
    print("labels shape:", sample['labels'].shape)
