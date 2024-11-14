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

    def identity_block(x, filters, kernel_size=3):
        shortcut = x

        # First 1x1 Conv layer
        x = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Second Conv layer
        x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Third 1x1 Conv layer
        x = layers.Conv1D(filters * 4, kernel_size=1, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)

        # Add skip connection and activation
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # Initial Conv layer
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # ResNet Blocks
    for filters, num_blocks in zip([64, 128, 256, 512], [3, 4, 6, 3]):
        # First block uses conv_block with strides of 2
        x = conv_block(x, filters, strides=2)
        for _ in range(num_blocks - 1):
            x = identity_block(x, filters)

    # Global Average Pooling and Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(3, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs, x)

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
