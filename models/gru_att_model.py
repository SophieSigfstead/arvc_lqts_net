from keras import layers, models, metrics, backend as K

def build_model(input_shape=(2500, 8)):
    def attention_layer(inputs):
        attention = layers.Dense(1, activation='tanh')(inputs)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(inputs.shape[-1])(attention)
        attention = layers.Permute([2, 1])(attention)
        output = layers.Multiply()([inputs, attention])
        return layers.Lambda(lambda x: K.sum(x, axis=1))(output)

    model = models.Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.GRU(64, return_sequences=True))
    model.add(layers.GRU(64, return_sequences=True))

    # Attention mechanism
    model.add(layers.Lambda(attention_layer))

    # Fully connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model