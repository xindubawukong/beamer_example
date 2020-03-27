def create_costco(shape, rank, nc):
    inputs = [keras.Input(shape=(1,), dtype="int32") for i in range(len(shape))]
    embeds = [
        keras.layers.Embedding(output_dim=rank, input_dim=shape[i])(inputs[i])
        for i in range(len(shape))]
    x = keras.layers.Concatenate(axis=1)(embeds)
    x = keras.layers.Reshape(target_shape=(rank, len(shape), 1))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(nc, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="relu")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model