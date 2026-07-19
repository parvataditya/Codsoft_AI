"""
feature_extractor.py
---------------------
Extracts image features using a pre-trained ResNet50 (trained on
ImageNet). This is the "computer vision" half of the pipeline: we
strip off ResNet50's final classification layer and use its
2048-dimensional pooled output as a compact visual representation
of each image, which the decoder (RNN) will later condition on.

Usage:
    python feature_extractor.py --images_dir dataset/Images \
        --output models/features.pkl

Notes:
- You can swap ResNet50 for VGG16 by using
tensorflow.keras.applications.vgg16 instead (see the
`build_vgg16_extractor` function below) — both are valid choices
per the task description.
- Feature extraction only needs to be run once; the resulting
features.pkl is then reused by both train.py and
generate_caption.py.
"""

import os
import argparse
import pickle
from tqdm import tqdm

import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


def build_resnet50_extractor():
    """ResNet50 pretrained on ImageNet, with the classification head removed.
    Output: (batch, 2048) feature vector via global average pooling."""
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return base_model, resnet_preprocess, (224, 224)


def build_vgg16_extractor():
    """Alternative: VGG16 pretrained on ImageNet, second-to-last FC layer.
    Output: (batch, 4096) feature vector."""
    base_model = VGG16(weights="imagenet")
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    return model, vgg_preprocess, (224, 224)


def extract_features(images_dir, model, preprocess_fn, target_size):
    features = {}
    filenames = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for filename in tqdm(filenames, desc="Extracting features"):
        path = os.path.join(images_dir, filename)
        try:
            image = load_img(path, target_size=target_size)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        arr = img_to_array(image)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_fn(arr)

        feature = model.predict(arr, verbose=0)
        features[filename] = feature.squeeze()

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract CNN image features for captioning.")
    parser.add_argument("--images_dir", required=True, help="Folder containing dataset images")
    parser.add_argument("--output", default="models/features.pkl", help="Where to save extracted features")
    parser.add_argument("--backbone", choices=["resnet50", "vgg16"], default="resnet50")
    args = parser.parse_args()

    if args.backbone == "resnet50":
        model, preprocess_fn, target_size = build_resnet50_extractor()
    else:
        model, preprocess_fn, target_size = build_vgg16_extractor()

    print(f"Using {args.backbone} as the feature extractor (ImageNet pretrained weights).")
    features = extract_features(args.images_dir, model, preprocess_fn, target_size)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved {len(features)} feature vectors to {args.output}")


if __name__ == "__main__":
    main()
