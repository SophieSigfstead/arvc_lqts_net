from keras import layers, models, backend as K

def build_model(input_shape=(2500, 8)):
    # Define the attention mechanism
    def attention_layer(inputs):
        attention_scores = layers.Dense(1, activation='tanh')(inputs)
        attention_weights = layers.Activation('softmax')(attention_scores)
        attention_output = layers.Multiply()([inputs, attention_weights])
        output = layers.Lambda(lambda x: K.sum(x, axis=1))(attention_output)
        return output

    inputs = layers.Input(shape=input_shape)

    # GRU Layers
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.GRU(64, return_sequences=True)(x)

    # Attention Layer
    x = attention_layer(x)

    # Fully Connected Layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer for 3 classes
    outputs = layers.Dense(3, activation='softmax')(x)

    # Create Model
    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
