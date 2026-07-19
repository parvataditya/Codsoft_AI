"""
model.py
--------
Defines the caption-generation model(s).

Two architectures are provided:

1. build_lstm_captioning_model()  [DEFAULT]
   A classic CNN-RNN "merge" architecture (similar in spirit to the
   Show-and-Tell / "merge-model" captioning papers):
     - Image features (from ResNet50/VGG16) are projected into a
       dense embedding.
     - The partial caption so far is embedded and passed through an
       LSTM.
     - The two representations are merged (added) and passed through
       a final Dense+softmax layer to predict the next word.

2. build_transformer_captioning_model()  [ALTERNATIVE]
   A lightweight Transformer decoder that attends over the image
   feature (treated as a single "memory" token) and the previously
   generated words, using a causal self-attention mask. This
   satisfies the "transformer-based model" option mentioned in the
   task.

Both take the same inputs — (image_feature_vector, input_sequence)
— and output a softmax distribution over the vocabulary for the next
word, so they are interchangeable in train.py / generate_caption.py.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ------------------------------------------------------------------
# Option 1: CNN + LSTM "merge" model (default, recommended for
# smaller datasets like Flickr8k)
# ------------------------------------------------------------------
def build_lstm_captioning_model(vocab_size, max_length, feature_dim=2048, embedding_dim=256, units=256):
    # --- Image feature branch ---
    image_input = layers.Input(shape=(feature_dim,), name="image_features")
    img = layers.Dropout(0.5)(image_input)
    img = layers.Dense(units, activation="relu")(img)

    # --- Caption sequence branch ---
    seq_input = layers.Input(shape=(max_length,), name="input_sequence")
    seq = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(seq_input)
    seq = layers.Dropout(0.5)(seq)
    seq = layers.LSTM(units)(seq)

    # --- Merge + decode ---
    merged = layers.Add()([img, seq])
    merged = layers.Dense(units, activation="relu")(merged)
    output = layers.Dense(vocab_size, activation="softmax")(merged)

    model = Model(inputs=[image_input, seq_input], outputs=output, name="cnn_lstm_captioner")
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


# ------------------------------------------------------------------
# Option 2: Lightweight Transformer decoder
# ------------------------------------------------------------------
class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)
        self.drop3 = layers.Dropout(rate)

    def call(self, x, memory, training=False):
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        attn1 = self.self_attn(x, x, attention_mask=causal_mask, training=training)
        x = self.norm1(x + self.drop1(attn1, training=training))

        attn2 = self.cross_attn(x, memory, training=training)
        x = self.norm2(x + self.drop2(attn2, training=training))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.drop3(ffn_out, training=training))
        return x


def build_transformer_captioning_model(vocab_size, max_length, feature_dim=2048,
                                        embed_dim=256, num_heads=4, ff_dim=512):
    # Image feature acts as a single "memory" token the decoder attends to
    image_input = layers.Input(shape=(feature_dim,), name="image_features")
    memory = layers.Dense(embed_dim, activation="relu")(image_input)
    memory = layers.Reshape((1, embed_dim))(memory)  # (batch, 1, embed_dim)

    seq_input = layers.Input(shape=(max_length,), name="input_sequence")
    token_embed = layers.Embedding(vocab_size, embed_dim, mask_zero=False)(seq_input)
    positions = tf.range(start=0, limit=max_length, delta=1)
    pos_embed = layers.Embedding(max_length, embed_dim)(positions)
    x = token_embed + pos_embed

    x = TransformerDecoderBlock(embed_dim, num_heads, ff_dim)(x, memory)

    # Predict next word using the representation at the last real token position.
    # For simplicity in this reference implementation we pool over the sequence;
    # for production use, gather the last non-padded timestep instead.
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=[image_input, seq_input], outputs=output, name="cnn_transformer_captioner")
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model
