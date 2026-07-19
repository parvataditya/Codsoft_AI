# Image Captioning AI (CNN + RNN/Transformer)

An image captioning system that combines **computer vision** and
**natural language processing**: a pre-trained CNN (ResNet50 or
VGG16) extracts visual features from an image, and a
recurrent (LSTM) or Transformer-based decoder generates a natural
language caption describing it.

## Objective

Combine computer vision and NLP to build an image captioning AI.
Use pre-trained image recognition models (VGG/ResNet) to extract
image features, then use an RNN or Transformer to generate captions
for those images.

## Architecture

```
   Image ──► ResNet50 / VGG16 (pretrained, frozen) ──► feature vector
                                                             │
                                                             ▼
"startseq a dog runs" ──► Embedding ──► LSTM / Transformer ──► Dense(softmax) ──► next word
```

**1. Encoder (Computer Vision)** — `feature_extractor.py`
A pre-trained ResNet50 (or VGG16), trained on ImageNet, is used with
its classification head removed. Every image is passed through once
to get a fixed-length feature vector (2048-d for ResNet50, 4096-d
for VGG16) that summarizes what's in the image. These weights are
**frozen** — we reuse ImageNet's learned visual features rather than
training a CNN from scratch, which would need millions of images.

**2. Decoder (NLP)** — `model.py`
Two interchangeable decoder options are implemented, as allowed by
the task:

- **`build_lstm_captioning_model()`** *(default)* — a classic
  CNN+LSTM "merge" architecture. The image feature and the
  partially-generated caption (embedded + passed through an LSTM)
  are merged and passed through a final softmax layer that predicts
  the next word. This works well on small-to-medium datasets like
  Flickr8k.

- **`build_transformer_captioning_model()`** *(alternative)* — a
  lightweight Transformer decoder block with masked self-attention
  (over previously generated words) and cross-attention (over the
  image feature), following the same encoder-decoder idea used in
  modern captioning/translation models.

**3. Training** — `train.py`
Trains the decoder with **teacher forcing**: at each step, the model
is given the correct previous words and learns to predict the next
one. The CNN encoder is not retrained (its weights stay frozen) —
only the decoder learns.

**4. Inference** — `generate_caption.py`
Generates a caption for a brand-new image using either:
- **Greedy search** — always pick the single most likely next word (fast).
- **Beam search** — track the top-k most probable partial captions at
  each step, producing generally more fluent/accurate results.

## Project Structure

```
image-captioning/
├── requirements.txt
├── README.md
├── dataset/                    # you provide this (see Dataset section)
│   ├── Images/
│   └── captions.txt
├── models/                     # generated after training
│   ├── features.pkl
│   ├── tokenizer.pkl
│   ├── config.pkl
│   └── caption_model_final.keras
└── src/
    ├── data_utils.py           # caption loading/cleaning, tokenizer, data generator
    ├── feature_extractor.py    # ResNet50 / VGG16 feature extraction
    ├── model.py                # LSTM and Transformer decoder architectures
    ├── train.py                # training script
    └── generate_caption.py     # inference (greedy + beam search)
```

## Dataset

This project is designed around **Flickr8k** (8,000 images, 5
captions each — a good size for a laptop/single-GPU setup):

- Download from Kaggle:
  https://www.kaggle.com/datasets/adityajn105/flickr8k
- Extract it so you have:
  ```
  dataset/Images/*.jpg
  dataset/captions.txt      # format: image,caption
  ```

(Flickr30k or MS-COCO can also be used the same way if you want a
larger dataset — just point the scripts at the new paths. Larger
datasets need significantly more training time.)

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and works on CPU (slow) or GPU (recommended for
training; feature extraction is a one-time cost and fine on CPU).

## Usage

### Step 1 — Extract image features (run once)

```bash
python src/feature_extractor.py \
    --images_dir dataset/Images \
    --output models/features.pkl \
    --backbone resnet50
```

This downloads ImageNet-pretrained ResNet50 weights (first run only)
and saves a `{filename: feature_vector}` dictionary — so training
doesn't need to re-run the CNN on every epoch.

### Step 2 — Train the captioning model

```bash
python src/train.py \
    --captions dataset/captions.txt \
    --features models/features.pkl \
    --arch lstm \
    --epochs 20 \
    --batch_size 32
```

Use `--arch transformer` to train the Transformer decoder instead.
Checkpoints are saved to `models/` after every epoch, plus a final
model, tokenizer, and config file needed for inference.

> **Note on training time:** Flickr8k with the LSTM model typically
> takes 15-30 minutes per epoch on CPU, much faster on GPU. 15-20
> epochs is usually enough to get sensible (if imperfect) captions.

### Step 3 — Generate a caption for a new image

```bash
python src/generate_caption.py \
    --image path/to/your/photo.jpg \
    --model models/caption_model_final.keras \
    --tokenizer models/tokenizer.pkl \
    --config models/config.pkl \
    --search beam --beam_width 3
```

Example output:
```
Extracting image features...
Generating caption using beam search...

Caption: a dog is running through the grass
```

## Design Notes / Why This Approach

- **Transfer learning for the encoder**: training a CNN from scratch
  needs millions of labeled images; reusing ImageNet-pretrained
  ResNet50/VGG16 features is standard practice and lets the decoder
  focus purely on language generation.
- **Precomputing features**: extracting CNN features is the
  expensive part. Doing it once and caching to `features.pkl` (rather
  than inside the training loop) makes iterating on the decoder much
  faster.
- **Teacher forcing**: during training, the decoder always sees the
  *true* previous word, not its own (possibly wrong) prediction —
  this stabilizes training. At inference time there's no ground
  truth, so greedy/beam search is used instead.
- **Two decoder options**: the LSTM merge-model is simpler and
  trains well on small datasets; the Transformer decoder is included
  to show the more modern attention-based alternative mentioned in
  the task, and scales better with more data/compute.

## Possible Future Improvements

- Add **visual attention** (e.g., Bahdanau/Luong-style attention over
  spatial CNN feature maps instead of a single pooled vector) — this
  is what the original "Show, Attend and Tell" paper does, and
  typically improves caption quality noticeably.
- Evaluate with **BLEU score** against reference captions
  (`nltk.translate.bleu_score`) to quantitatively track model quality.
- Fine-tune the CNN encoder's later layers on the target dataset
  instead of keeping it fully frozen.
- Serve the trained model behind a simple Flask/FastAPI endpoint or
  Streamlit demo for interactive use.
- Train on a larger dataset (Flickr30k, MS-COCO) for more diverse,
  robust captions.

## What This Demonstrates

- Transfer learning with pre-trained CNNs (ResNet50/VGG16) for
  feature extraction
- Sequence modeling with LSTMs and Transformer decoders
- Combining vision and language models into a single multimodal
  pipeline
- Teacher forcing during training vs. greedy/beam search at inference
- Practical considerations for training on limited compute (caching
  features, batching, early stopping)
