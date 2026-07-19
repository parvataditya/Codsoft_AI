"""
train.py
--------
Trains the CNN-RNN (or CNN-Transformer) image captioning model.

Prerequisites:
1. Download the Flickr8k dataset (images + captions.txt) into dataset/
2. Run feature_extractor.py first to precompute and cache image features — this avoids re-running the CNN on every epoch.

Usage:
    python train.py \
        --captions dataset/captions.txt \
        --features models/features.pkl \
        --arch lstm \
        --epochs 20 \
        --batch_size 32
"""

import argparse
import os
import pickle
import math
from pyexpat import features

import tensorflow as tf

from data_utils import (
    load_captions, clean_all_captions, build_tokenizer, max_caption_length,
    save_tokenizer, train_test_split, data_generator,
)
from model import build_lstm_captioning_model, build_transformer_captioning_model


def count_training_pairs(image_ids, cleaned_mapping, features):
    """Counts total (input_seq -> next_word) training pairs, needed for steps_per_epoch."""
    total = 0
    for image_id in image_ids:
        if image_id not in features:
            continue
        for caption in cleaned_mapping[image_id]:
            total += len(caption.split()) - 1
    return total


def main():
    parser = argparse.ArgumentParser(description="Train the image captioning model.")
    parser.add_argument("--captions", required=True, help="Path to captions.txt")
    parser.add_argument("--features", required=True, help="Path to precomputed features.pkl")
    parser.add_argument("--arch", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size_cap", type=int, default=None,
    help="Optionally cap vocabulary to N most frequent words")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading and cleaning captions...")
    raw_mapping = load_captions(args.captions)
    cleaned_mapping = clean_all_captions(raw_mapping)

    print("Loading precomputed image features...")
    with open(args.features, "rb") as f:
        features = pickle.load(f)

    # Keep only images we actually have both a caption and a feature for
    # Strip paths and extensions from feature keys for a clean match
    feature_ids = {k.replace('\\', '/').split('/')[-1].split('.')[0] for k in features.keys()}
    image_ids = [img_id for img_id in cleaned_mapping if img_id.replace('\\', '/').split('/')[-1].split('.')[0] in feature_ids]
    print(f"Usable images: {len(image_ids)}")

    train_ids, val_ids = train_test_split(image_ids, test_ratio=0.1)
    print(f"Train: {len(train_ids)} | Validation: {len(val_ids)}")

    print("Building tokenizer...")
    tokenizer = build_tokenizer(cleaned_mapping, num_words=args.vocab_size_cap)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max_caption_length(cleaned_mapping)
    print(f"Vocabulary size: {vocab_size} | Max caption length: {max_length}")

    save_tokenizer(tokenizer, os.path.join(args.out_dir, "tokenizer.pkl"))
    with open(os.path.join(args.out_dir, "config.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "max_length": max_length, "arch": args.arch}, f)

    feature_dim = next(iter(features.values())).shape[0]
    print(f"Feature vector dimension: {feature_dim}")

    if args.arch == "lstm":
        model = build_lstm_captioning_model(vocab_size, max_length, feature_dim=feature_dim)
    else:
        model = build_transformer_captioning_model(vocab_size, max_length, feature_dim=feature_dim)

    model.summary()

    # Calculate robust steps per epoch using path-blind matching
    flat_features = {k.replace('\\', '/').split('/')[-1].split('.')[0] for k in features.keys()}
    
    train_pairs = sum(len(c.split()) - 1 for img_id in train_ids for c in cleaned_mapping.get(img_id, []) if img_id.replace('\\', '/').split('/')[-1].split('.')[0] in flat_features)
    val_pairs = sum(len(c.split()) - 1 for img_id in val_ids for c in cleaned_mapping.get(img_id, []) if img_id.replace('\\', '/').split('/')[-1].split('.')[0] in flat_features)
    
    train_steps = math.ceil(train_pairs / args.batch_size)
    val_steps = math.ceil(val_pairs / args.batch_size)

    # Repaired data generators
    train_gen = data_generator(train_ids, cleaned_mapping, features, tokenizer, max_length, vocab_size, batch_size=args.batch_size)
    val_gen = data_generator(val_ids, cleaned_mapping, features, tokenizer, max_length, vocab_size, batch_size=args.batch_size)

    checkpoint_path = os.path.join(args.out_dir, "caption_model_epoch{epoch:02d}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=False),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    final_path = os.path.join(args.out_dir, "caption_model_final.keras")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
