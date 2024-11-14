from keras import layers, models

def build_model(input_shape=(2500, 8)):
    inputs = layers.Input(shape=input_shape)

    # First Conv Layer
    x = layers.Conv1D(96, kernel_size=11, strides=4, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Inception Module
    branch1 = layers.Conv1D(64, kernel_size=1, activation='relu', padding='same')(x)
    branch2 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    branch3 = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.Concatenate()([branch1, branch2, branch3])
    x = layers.BatchNormalization()(x)

    # Squeeze-and-Excitation Block
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(units=64, activation='relu')(se)
    se = layers.Dense(units=192, activation='sigmoid')(se)
    x = layers.Multiply()([x, se])

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

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
