from keras import layers, models, metrics

def build_model(input_shape=(2500, 8)):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    conv1d_params = {'activation':'relu', 'padding':'same', 'strides':1}
    maxpool1d_params = {'padding':'same'}

    # Temporal analysis
    model.add(layers.Conv1D(filters=16, kernel_size=5, **conv1d_params, input_shape=(2500, 8)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=16, kernel_size=5, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=32, kernel_size=5, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=4, **maxpool1d_params))
    model.add(layers.Conv1D(filters=32, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=64, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2, **maxpool1d_params))
    model.add(layers.Conv1D(filters=64, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=4, **maxpool1d_params))

    # Spatial analysis
    model.add(layers.Conv1D(filters=12, kernel_size=3, **conv1d_params))
    model.add(layers.BatchNormalization())

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(rate=0.2))

    # Output
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model