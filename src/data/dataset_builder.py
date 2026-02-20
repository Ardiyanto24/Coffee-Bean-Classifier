import logging
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import BaselineConfig


class DatasetBuilder:
    """
    Builds a tf.data.Dataset from a CSV file containing image paths and labels.

    The dataset is rebuilt per backbone because each architecture requires
    a different preprocess_input function and image_size.

    Required Config Fields
    ----------------------
    - config.batch_size : int
        Number of samples per batch.
    - config.path_col : str
        Column name for image file paths.
    - config.label_col : str
        Column name for integer encoded labels.

    Parameters
    ----------
    config : BaselineConfig
        Configuration object.
    """

    def __init__(self, config: BaselineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_and_preprocess(
        self,
        path: tf.Tensor,
        label: tf.Tensor,
        preprocess_fn: Callable,
        image_size: Tuple[int, int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Loads a single image from disk, resizes it, and applies backbone-specific
        preprocessing.

        Parameters
        ----------
        path : tf.Tensor
            String tensor containing the absolute path to the image file.
        label : tf.Tensor
            Integer tensor for the class label.
        preprocess_fn : Callable
            Backbone-specific preprocess_input function (e.g. resnet50.preprocess_input).
        image_size : Tuple[int, int]
            Target (height, width) to resize the image.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            Preprocessed image tensor of shape (H, W, 3) and its label.
        """
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, label

    def build(
        self,
        csv_path: str,
        preprocess_fn: Callable,
        image_size: Tuple[int, int],
        shuffle: bool = False
    ) -> tf.data.Dataset:
        """
        Reads a CSV and returns a batched, prefetched tf.data.Dataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file with 'filepath' and 'encoded_label' columns.
        preprocess_fn : Callable
            Backbone-specific preprocessing function.
        image_size : Tuple[int, int]
            Target image resolution (height, width).
        shuffle : bool
            If True, shuffles the dataset (use True for training set only).

        Returns
        -------
        tf.data.Dataset
            Batched and prefetched dataset ready for model.fit() or model.predict().
        """
        df     = pd.read_csv(csv_path)
        paths  = df[self.config.path_col].values
        labels = df[self.config.label_col].values.astype(np.int32)

        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(df), seed=42)

        dataset = dataset.map(
            lambda p, l: self._load_and_preprocess(p, l, preprocess_fn, image_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        self.logger.info(
            f"Dataset built from {csv_path} — "
            f"{len(df)} samples | image_size={image_size} | shuffle={shuffle}"
        )
        return dataset