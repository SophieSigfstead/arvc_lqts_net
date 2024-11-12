from keras import layers, models, metrics

def build_model(input_shape=(2500, 8)):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(96, kernel_size=11, strides=4, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=2, padding='same'))

    model.add(layers.Conv1D(256, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=2, padding='same'))

    model.add(layers.Conv1D(384, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Conv1D(384, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=3, strides=2, padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model