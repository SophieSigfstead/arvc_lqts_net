from keras import layers, models

def build_model(input_shape=(2500, 8)):
    inputs = layers.Input(shape=input_shape)

    # First Conv Layer
    x = layers.Conv1D(96, kernel_size=11, strides=4, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual Block
    shortcut = layers.Conv1D(256, kernel_size=1, padding='same')(x)  # Match the filter dimension
    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])

    # Spatial Attention Mechanism
    attention = layers.Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')(x)
    attention_output = layers.Multiply()([x, attention])

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(attention_output)

    # Fully Connected Layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    outputs = layers.Dense(3, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
