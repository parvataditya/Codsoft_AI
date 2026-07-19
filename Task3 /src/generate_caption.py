"""
generate_caption.py
--------------------
Generates a caption for a new image using the trained model.
Supports two decoding strategies:
  - Greedy search: always pick the most likely next word (fast).
  - Beam search: keep the top-k most likely partial sequences at
    each step, giving generally higher-quality captions.

Usage:
    python generate_caption.py --image path/to/photo.jpg \
        --model models/caption_model_final.keras \
        --tokenizer models/tokenizer.pkl \
        --config models/config.pkl \
        --search beam --beam_width 3
"""

import argparse
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import START_TOKEN, END_TOKEN
from feature_extractor import build_resnet50_extractor, build_vgg16_extractor
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def extract_single_image_feature(image_path, backbone="resnet50"):
    if backbone == "resnet50":
        model, preprocess_fn, target_size = build_resnet50_extractor()
    else:
        model, preprocess_fn, target_size = build_vgg16_extractor()

    image = load_img(image_path, target_size=target_size)
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_fn(arr)
    feature = model.predict(arr, verbose=0)
    return feature.squeeze()


def greedy_search(model, feature, tokenizer, max_length):
    in_text = START_TOKEN
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        pred = model.predict([np.expand_dims(feature, 0), seq], verbose=0)
        next_id = np.argmax(pred)
        next_word = tokenizer.index_word.get(next_id)
        if next_word is None:
            break
        in_text += " " + next_word
        if next_word == END_TOKEN:
            break
    return clean_output(in_text)


def beam_search(model, feature, tokenizer, max_length, beam_width=3):
    start_seq = tokenizer.texts_to_sequences([START_TOKEN])[0]
    sequences = [(start_seq, 0.0)]  # (token_ids, cumulative_log_prob)

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            # If sequence already ended, keep it as-is (don't expand further)
            if tokenizer.index_word.get(seq[-1]) == END_TOKEN:
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([np.expand_dims(feature, 0), padded], verbose=0)[0]
            top_ids = np.argsort(preds)[-beam_width:]

            for word_id in top_ids:
                prob = preds[word_id]
                if prob <= 0:
                    continue
                candidate = (seq + [word_id], score + np.log(prob))
                all_candidates.append(candidate)

        # Keep only the top `beam_width` sequences overall
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        sequences = all_candidates[:beam_width]

        # Stop early if every beam has ended
        if all(tokenizer.index_word.get(seq[-1]) == END_TOKEN for seq, _ in sequences):
            break

    best_seq = sequences[0][0]
    words = [tokenizer.index_word.get(i, "") for i in best_seq]
    return clean_output(" ".join(words))


def clean_output(text):
    text = text.replace(START_TOKEN, "").replace(END_TOKEN, "")
    return " ".join(text.split()).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", required=True, help="Path to the trained .keras model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.pkl")
    parser.add_argument("--config", required=True, help="Path to config.pkl")
    parser.add_argument("--backbone", choices=["resnet50", "vgg16"], default="resnet50")
    parser.add_argument("--search", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_width", type=int, default=3)
    args = parser.parse_args()

    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    with open(args.config, "rb") as f:
        config = pickle.load(f)

    max_length = config["max_length"]
    model = tf.keras.models.load_model(args.model)

    print("Extracting image features...")
    feature = extract_single_image_feature(args.image, backbone=args.backbone)

    print(f"Generating caption using {args.search} search...")
    if args.search == "greedy":
        caption = greedy_search(model, feature, tokenizer, max_length)
    else:
        caption = beam_search(model, feature, tokenizer, max_length, beam_width=args.beam_width)

    print("\nCaption:", caption)


if __name__ == "__main__":
    main()
