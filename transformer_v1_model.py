from keras import layers, models
import tensorflow as tf

# Function to create positional encoding
def positional_encoding(seq_len, d_model):
    pos = tf.range(seq_len)[:, tf.newaxis]
    i = tf.range(d_model)[tf.newaxis, :]
    angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    angle_rads = pos * angle_rates
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, ...]

# Function to build the Transformer model
def build_model(input_shape=(2500, 8), num_classes=3, d_model=128, num_heads=8, ff_dim=512, num_layers=4, dropout_rate=0.1):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Embedding layer (projects input features to d_model dimensions)
    x = layers.Dense(d_model, dtype=tf.float32)(inputs)

    # Add positional encoding
    pos_enc = tf.cast(positional_encoding(input_shape[0], d_model), dtype=tf.float32)
    x = tf.add(x, pos_enc)

    # Transformer Encoder layers
    for _ in range(num_layers):
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout_rate)(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization()(x + attn_output)

        # Feed-Forward Network
        ff_output = layers.Dense(ff_dim, activation='relu')(x)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        x = layers.LayerNormalization()(x + ff_output)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

