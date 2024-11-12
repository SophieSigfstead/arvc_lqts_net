from keras import layers, models, metrics

def build_model(input_shape=(2500, 8)):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model