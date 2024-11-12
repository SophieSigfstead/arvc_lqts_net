from keras import layers, models, metrics

def build_model(input_shape=(2500, 8)):
    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    for _ in range(2):
        model.add(layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    for _ in range(2):
        model.add(layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    for _ in range(3):
        model.add(layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

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