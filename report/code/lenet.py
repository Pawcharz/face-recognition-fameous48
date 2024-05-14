conv_regularizer = regularizers.l2(0.0009096443481619992)
dense_regularizer = regularizers.l2(0.011905583599301073)

dropout_base = 0.09439855997376015
dropout_inc = 0.14131761625994724
dropout_1 = dropout_base
dropout_2 = dropout_base + dropout_inc
dropout_3 = dropout_base + 2*dropout_inc

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=5,
                        padding="same", activation="tanh",
                        kernel_regularizer=conv_regularizer)

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    DefaultConv2D(6),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Dropout(dropout_1),
    DefaultConv2D(16),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Dropout(dropout_2),
    DefaultConv2D(120),

    layers.Flatten(),
    layers.Dropout(dropout_3),
    DefaultConv2D(84),
    layers.Dense(CLASSES, activation='softmax')
])