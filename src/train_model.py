"""
This is the demo code that uses hy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      dra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
import tensorflow as tf
from omegaconf import DictConfig
from process import process_data


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""

    train_ds, val_ds, class_names = process_data(config)

    num_classes = len(class_names)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3)
    model.save("../models/" + config.model.name)

    print(f"Save the output to {config.model.name}")


if __name__ == "__main__":
    train_model()
