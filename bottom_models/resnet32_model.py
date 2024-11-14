from keras import layers, models

def residual_block(x, filters, kernel_size=3, stride=1):
    """Defines a residual block with two convolutional layers and a skip connection."""
    shortcut = x

    # First Conv Layer
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Second Conv Layer
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)

    # Skip Connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection and activation
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

def build_model(input_shape=(2500, 8), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Define ResNet-32 structure with (3, 4, 6, 3) blocks
    num_blocks = [3, 4, 6, 3]
    filters = [64, 128, 256, 512]

    for i, num_block in enumerate(num_blocks):
        for j in range(num_block):
            stride = 2 if j == 0 and i != 0 else 1  # Downsample at the first block of each stage
            x = residual_block(x, filters[i], stride=stride)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected Layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
