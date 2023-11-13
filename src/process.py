"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import tensorflow as tf
from omegaconf import DictConfig


def process_data(config: DictConfig):
    """Function to process the data"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.data_train.raw_train,
        validation_split=0.2,
        subset="training",
        seed=config.process.seed,
        image_size=(config.process.img_height, config.process.img_width),
        batch_size=config.process.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.data_train.raw_train,
        validation_split=0.2,
        subset="validation",
        seed=config.process.seed,
        image_size=(config.process.img_height, config.process.img_width),
        batch_size=config.process.batch_size,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
