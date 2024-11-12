from keras import layers, models

def build_model(input_shape=(2500, 8)):
    def conv_block(x, filters, kernel_size=3, strides=1):
        shortcut = x

        # First 1x1 Conv layer (Bottleneck)
        x = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Second Conv layer (Main convolution)
        x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Third 1x1 Conv layer
        x = layers.Conv1D(filters * 4, kernel_size=1, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)

        # Adjust the shortcut to match dimensions if necessary
        if shortcut.shape[-1] != filters * 4:
            shortcut = layers.Conv1D(filters * 4, kernel_size=1, strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add skip connection and activation
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # Initial Conv layer
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Stage 1: 3 blocks with 64 filters (9 layers)
    for _ in range(3):
        x = conv_block(x, 64)

    # Stage 2: 14 blocks with 128 filters (42 layers)
    for _ in range(14):
        x = conv_block(x, 128)

    # Stage 3: 30 blocks with 256 filters (90 layers)
    for _ in range(30):
        x = conv_block(x, 256)

    # Stage 4: 3 blocks with 512 filters (9 layers)
    for _ in range(3):
        x = conv_block(x, 512)

    # Global Average Pooling and Output
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(3, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
