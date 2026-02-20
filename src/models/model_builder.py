import logging
from typing import Callable, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    EfficientNetB3,
    MobileNetV3Small,
    ResNet50,
)
from tensorflow.keras.applications import (
    densenet     as densenet_preprocess,
    efficientnet as efficientnet_preprocess,
    resnet50     as resnet50_preprocess,
)
from tensorflow.keras.applications.mobilenet_v3 import (
    preprocess_input as mobilenetv3_preprocess_input,
)

from src.config import BaselineConfig


class ModelBuilder:
    """
    Builds classification models using frozen pretrained backbones.

    Architecture: Frozen Backbone → GlobalAveragePooling2D → Dense(softmax)

    Supports 5 built-in backbones and user-registered custom models.
    Each backbone has its own image_size and preprocess_input function
    stored in BACKBONE_CONFIG.

    Required Config Fields
    ----------------------
    - config.num_classes : int
        Number of output classes.
    - config.learning_rate : float
        Learning rate for the Adam optimizer.

    Parameters
    ----------
    config : BaselineConfig
        Configuration object.
    """

    # Built-in backbone registry:
    # key → (backbone_class, preprocess_fn, image_size)
    BACKBONE_CONFIG: Dict = {
        'ResNet50': (
            ResNet50,
            resnet50_preprocess.preprocess_input,
            (224, 224)
        ),
        'EfficientNetB0': (
            EfficientNetB0,
            efficientnet_preprocess.preprocess_input,
            (224, 224)
        ),
        'EfficientNetB3': (
            EfficientNetB3,
            efficientnet_preprocess.preprocess_input,
            (300, 300)
        ),
        'MobileNetV3Small': (
            MobileNetV3Small,
            mobilenetv3_preprocess_input,
            (224, 224)
        ),
        'DenseNet121': (
            DenseNet121,
            densenet_preprocess.preprocess_input,
            (224, 224)
        ),
    }

    def __init__(self, config: BaselineConfig):
        self.config  = config
        self.logger  = logging.getLogger(self.__class__.__name__)
        # Registry for custom models: key → (model, preprocess_fn, image_size)
        self._custom_registry: Dict = {}

    def register_custom_model(
        self,
        name: str,
        model: keras.Model,
        image_size: Tuple[int, int],
        preprocess_fn: Optional[Callable] = None
    ) -> None:
        """
        Registers a user-provided tf.keras.Model for training.

        The registered model must already include its classification head.
        It will be compiled and trained as-is without modification.
        If preprocess_fn is None, images are normalized to [0, 1] by default.

        Parameters
        ----------
        name : str
            Unique identifier for this custom model.
        model : tf.keras.Model
            A fully defined Keras model including output layer.
        image_size : Tuple[int, int]
            Input image resolution (height, width) expected by the model.
        preprocess_fn : Callable or None
            Preprocessing function. If None, defaults to dividing by 255.0.

        Raises
        ------
        ValueError
            If the name conflicts with a built-in backbone key.
        """
        if name in self.BACKBONE_CONFIG:
            raise ValueError(
                f"'{name}' conflicts with a built-in backbone name. "
                f"Please choose a different name for your custom model."
            )
        if preprocess_fn is None:
            preprocess_fn = lambda x: x / 255.0
            self.logger.warning(
                f"No preprocess_fn provided for '{name}'. "
                f"Defaulting to [0, 1] normalization (divide by 255)."
            )
        self._custom_registry[name] = (model, preprocess_fn, image_size)
        self.logger.info(f"Custom model '{name}' registered | image_size={image_size}")

    def get_backbone_config(self, model_name: str) -> Tuple[Callable, Tuple[int, int]]:
        """
        Returns the preprocess_fn and image_size for a given model name.

        Parameters
        ----------
        model_name : str
            Name of the backbone or registered custom model.

        Returns
        -------
        Tuple[Callable, Tuple[int, int]]
            (preprocess_fn, image_size)

        Raises
        ------
        ValueError
            If model_name is not found in either registry.
        """
        if model_name in self.BACKBONE_CONFIG:
            _, preprocess_fn, image_size = self.BACKBONE_CONFIG[model_name]
            return preprocess_fn, image_size
        if model_name in self._custom_registry:
            _, preprocess_fn, image_size = self._custom_registry[model_name]
            return preprocess_fn, image_size
        raise ValueError(
            f"Model '{model_name}' not found. "
            f"Available built-in: {list(self.BACKBONE_CONFIG.keys())}. "
            f"Registered custom: {list(self._custom_registry.keys())}."
        )

    def build(self, model_name: str) -> keras.Model:
        """
        Builds a classification model for the given backbone name.

        For built-in backbones:
            - Loads pretrained ImageNet weights
            - Freezes all backbone layers (trainable=False)
            - Adds GlobalAveragePooling2D + Dense(num_classes, softmax) head

        For custom models:
            - Returns the registered model as-is

        Parameters
        ----------
        model_name : str
            Name of the backbone or registered custom model.

        Returns
        -------
        tf.keras.Model
            Compiled-ready model.
        """
        if model_name in self._custom_registry:
            model, _, _ = self._custom_registry[model_name]
            self.logger.info(f"Using custom model '{model_name}'")
            return model

        backbone_class, _, image_size = self.BACKBONE_CONFIG[model_name]
        input_shape = (*image_size, 3)

        backbone = backbone_class(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        backbone.trainable = False

        inputs  = keras.Input(shape=input_shape)
        x       = backbone(inputs, training=False)
        x       = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(self.config.num_classes, activation='softmax')(x)
        model   = keras.Model(inputs, outputs, name=model_name)

        trainable_params   = sum([tf.size(w).numpy() for w in model.trainable_weights])
        untrainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        self.logger.info(
            f"Model '{model_name}' built | input_shape={input_shape} | "
            f"trainable_params={trainable_params:,} | frozen_params={untrainable_params:,}"
        )
        return model

    def compile(self, model: keras.Model) -> keras.Model:
        """
        Compiles the model with Adam optimizer and sparse categorical crossentropy.

        Parameters
        ----------
        model : tf.keras.Model
            Uncompiled Keras model.

        Returns
        -------
        tf.keras.Model
            Compiled model ready for training.
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model