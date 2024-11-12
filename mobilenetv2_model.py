from keras import layers, models, metrics, applications

def build_model(input_shape=(2500, 8)):
    input_shape_add_channels = (2500, 8, 1)
    base_model = applications.MobileNetV2(input_shape=input_shape_add_channels, include_top=False, weights=None)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling1D(),
        layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model