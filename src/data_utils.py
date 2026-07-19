"""
data_utils.py
-------------
Utilities for loading captions, cleaning text, building a
vocabulary/tokenizer, and generating (image_features, input_seq) ->
next_word training batches for the captioning model.

Expected dataset format (Flickr8k-style captions.txt):
    image,caption
    1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs.
    1000268201_693b08cb0e.jpg,A girl going into a wooden building.
    ...
(each image usually has 5 captions)

Download the Flickr8k dataset (images + captions.txt) from Kaggle:
https://www.kaggle.com/datasets/adityajn105/flickr8k
"""

import re
import pickle
import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

START_TOKEN = "startseq"
END_TOKEN = "endseq"


# ------------------------------------------------------------------
# Loading + cleaning captions
# ------------------------------------------------------------------
def load_captions(captions_path):
    """
    Reads a Flickr8k-style captions.txt file and returns:
        { image_filename: [caption1, caption2, ...] }
    """
    mapping = defaultdict(list)
    with open(captions_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header row if present (e.g. "image,caption")
    if lines and lines[0].strip().lower().startswith("image"):
        lines = lines[1:]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Handle both "image.jpg,caption" and "image.jpg#0,caption" styles
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        image_id, caption = parts
        image_id = image_id.split("#")[0].strip().replace(".jpg", "")
        mapping[image_id].append(caption.strip())

    return dict(mapping)


def clean_caption(caption):
    """Lowercase, strip punctuation/numbers, and wrap with start/end tokens."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    words = [w for w in caption.split() if len(w) > 1]
    return f"{START_TOKEN} {' '.join(words)} {END_TOKEN}"


def clean_all_captions(mapping):
    """Applies clean_caption() to every caption in the mapping (in place-safe copy)."""
    cleaned = {}
    for image_id, captions in mapping.items():
        cleaned[image_id] = [clean_caption(c) for c in captions]
    return cleaned


# ------------------------------------------------------------------
# Tokenizer / vocabulary
# ------------------------------------------------------------------
def build_tokenizer(cleaned_mapping, num_words=None):
    """
    Builds a Keras Tokenizer over every cleaned caption.
    num_words: optionally cap vocabulary size to the N most frequent words.
    """
    all_captions = [c for caps in cleaned_mapping.values() for c in caps]
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)
    return tokenizer


def max_caption_length(cleaned_mapping):
    all_captions = [c for caps in cleaned_mapping.values() for c in caps]
    return max(len(c.split()) for c in all_captions)


def save_tokenizer(tokenizer, path):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------
# Train/test split
# ------------------------------------------------------------------
def train_test_split(image_ids, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    split_idx = int(len(ids) * (1 - test_ratio))
    return ids[:split_idx], ids[split_idx:]


# ------------------------------------------------------------------
# Data generator (keeps memory usage low: one image's captions per batch group)
# ------------------------------------------------------------------
def data_generator(image_ids, cleaned_mapping, features, tokenizer, max_length, vocab_size, batch_size=32):
    """
    Yields batches of ([image_feature_batch, input_seq_batch], output_word_batch)
    for training the decoder with teacher forcing.

    For every caption "startseq A cat sits endseq", this expands into
    training pairs like:
        (image, "startseq")              -> "a"
        (image, "startseq a")            -> "cat"
        (image, "startseq a cat")        -> "sits"
        (image, "startseq a cat sits")   -> "endseq"
    """
    X_img, X_seq, y = [], [], []
    n = 0
    
    # Normalize feature keys to bare IDs right before the loop
    cleaned_features = {k.replace('\\', '/').split('/')[-1].split('.')[0]: v for k, v in features.items()}
    
    while True:
        for image_id in image_ids:
            # Normalize the current image_id to a bare ID
            img_key = image_id.replace('\\', '/').split('/')[-1].split('.')[0]
            
            if img_key not in cleaned_features:
                continue
            feature = cleaned_features[img_key]
            for caption in cleaned_mapping[image_id]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_word = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_word = tf.keras.utils.to_categorical([out_word], num_classes=vocab_size)[0]
                    X_img.append(feature)
                    X_seq.append(in_seq)
                    y.append(out_word)
                    n += 1
                    if n == batch_size:
                        yield (np.array(X_img), np.array(X_seq)), np.array(y)
                        X_img, X_seq, y = [], [], []
                        n = 0
