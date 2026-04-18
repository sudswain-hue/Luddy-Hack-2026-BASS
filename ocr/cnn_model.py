# ocr/cnn_model.py
"""
CNN architecture for character recognition.
Trained on EMNIST dataset (62 classes: 0-9, A-Z, a-z).
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_cnn(num_classes: int = 10, input_shape: tuple = (28, 28, 1)) -> Model:
    """
    3-block CNN for character recognition.

    Architecture justification:
    - 28x28 input: matches EMNIST image size, no resize needed
    - 3 conv blocks with increasing filters (32→64→128):
        lets the network learn hierarchical features from
        edges → strokes → full character shapes
    - 3x3 kernels: standard for character recognition,
        captures local pixel relationships without over-smoothing
    - MaxPooling after each block: reduces spatial dims,
        provides translation invariance for shifted characters
    - BatchNorm: stabilizes training, allows higher learning rates
    - ReLU activation: prevents vanishing gradients, fast to compute
    - Dropout before final dense: prevents overfitting on 62-class problem
    - Softmax output: probability distribution over 62 character classes
    """
    inputs = tf.keras.Input(shape=input_shape, name="image")

    # Block 1: 28x28 -> 14x14
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2: 14x14 -> 7x7
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3: 7x7 -> 3x3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification")(x)

    model = Model(inputs=inputs, outputs=outputs, name="character_cnn")
    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """Compile with Adam optimizer and categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    # sanity check — build and print summary
    model = build_cnn()
    model = compile_model(model)
    model.summary()
    print("\nTotal parameters:", model.count_params())