from keras import layers, models, metrics

def build_model(input_shape=(2500, 8)):
    def residual_block(x, filters, kernel_size=3):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    for filters in [64, 128, 256, 512]:
        x = residual_block(x, filters)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(5, activation='softmax')(x)

    model = models.Model(inputs, x)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', metrics.AUC(name='auc')])
    return model