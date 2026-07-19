"""
Image Captioning Inference Script
Designed for internship repository demonstration.
Processes a pre-extracted image feature and generates a caption using the trained model.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- Configuration Paths ---
MODEL_PATH = 'models/caption_model_final.keras'
CAPTIONS_PATH = 'dataset/captions.txt'
FEATURES_PATH = 'models/features.pkl'
MAX_LENGTH = 34

def load_tokenizer(captions_path):
    """Rebuilds the tokenizer to match the exact word-to-index vocabulary mapping."""
    with open(captions_path, 'r') as f:
        next(f)  # Skip CSV header
        lines = f.readlines()

    captions_list = []
    for line in lines:
        tokens = line.split(',')
        if len(tokens) < 2:
            continue
        caption = tokens[1].strip().lower()
        captions_list.append(f"startseq {caption} endseq")

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions_list)
    return tokenizer

def generate_caption(model, feature, tokenizer, max_length):
    """Generates a caption word-by-word using greedy search decoding."""
    in_text = 'startseq'
    
    # Ensure feature has a batch dimension (1, feature_dim)
    if len(feature.shape) == 1:
        feature = np.expand_dims(feature, axis=0)
        
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict the next token probabilities
        predictions = model.predict([feature, sequence], verbose=0)
        idx = np.argmax(predictions)
        
        # Map index back to word
        word = None
        for w, index in tokenizer.word_index.items():
            if index == idx:
                word = w
                break
                
        if word is None or word == 'endseq':
            break
            
        in_text += ' ' + word
        
    # Format final output for clean display
    return in_text.replace('startseq', '').strip()

def main():
    print("--- Initializing Evaluation Pipeline ---")
    
    # 1. Load the text tokenizer
    tokenizer = load_tokenizer(CAPTIONS_PATH)
    print(f"[SUCCESS] Tokenizer configured with vocabulary size: {len(tokenizer.word_index) + 1}")
    
    # 2. Load the optimized model weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model weights not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[SUCCESS] Trained captioning model loaded.")

    # 3. Fetch a sample feature from the dataset
    with open(FEATURES_PATH, 'rb') as f:
        features_dict = pickle.load(f)
    
    # Select a test sample key from the dictionary
    sample_key = list(features_dict.keys())[0]
    feature = features_dict[sample_key]
    
    # 4. Execute prediction
    print("\nRunning inference...")
    generated_text = generate_caption(model, feature, tokenizer, MAX_LENGTH)
    
    print("\n=============================================")
    print(f"Target Image Key : {sample_key}")
    print(f"Generated Caption: {generated_text}")
    print("=============================================\n")

if __name__ == '__main__':
    main()