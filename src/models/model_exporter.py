import logging
import os
from typing import Dict, Tuple

import tensorflow as tf
import tf2onnx
from tensorflow import keras

from src.config import BaselineConfig


class ModelExporter:
    """
    Exports a trained Keras model to .keras and .onnx formats.

    Both formats are saved after every model finishes training:
    - .keras  : for further training, tuning, or XAI treatment
    - .onnx   : for deployment and inference pipeline development

    Required Config Fields
    ----------------------
    - config.output_dir : str
        Root directory. Models are saved under output_dir/models/.

    Parameters
    ----------
    config : BaselineConfig
        Configuration object.
    """

    def __init__(self, config: BaselineConfig):
        self.config     = config
        self.logger     = logging.getLogger(self.__class__.__name__)
        self.models_dir = os.path.join(config.output_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    def save_keras(self, model: keras.Model, model_name: str) -> str:
        """
        Saves the model in native Keras format (.keras).

        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model.
        model_name : str
            Used as the filename prefix.

        Returns
        -------
        str
            Absolute path to the saved .keras file.
        """
        save_path = os.path.join(self.models_dir, f"{model_name}.keras")
        model.save(save_path)
        self.logger.info(f"Keras model saved → {save_path}")
        return save_path

    def save_onnx(
        self,
        model: keras.Model,
        model_name: str,
        image_size: Tuple[int, int]
    ) -> str:
        """
        Converts the Keras model to ONNX format and saves it.

        Uses tf2onnx with opset=13.

        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model.
        model_name : str
            Used as the filename prefix.
        image_size : Tuple[int, int]
            Model input resolution (height, width).

        Returns
        -------
        str
            Absolute path to the saved .onnx file.
        """
        save_path = os.path.join(self.models_dir, f"{model_name}.onnx")
        input_sig = [tf.TensorSpec(
            shape=(None, *image_size, 3),
            dtype=tf.float32,
            name='input'
        )]
        tf2onnx.convert.from_keras(
            model,
            input_signature=input_sig,
            opset=13,
            output_path=save_path
        )
        self.logger.info(f"ONNX model saved  → {save_path}")
        return save_path

    def export(
        self,
        model: keras.Model,
        model_name: str,
        image_size: Tuple[int, int]
    ) -> Dict[str, str]:
        """
        Exports the model to both .keras and .onnx formats.

        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model.
        model_name : str
            Used as the filename prefix for both outputs.
        image_size : Tuple[int, int]
            Model input resolution (height, width).

        Returns
        -------
        Dict[str, str]
            {'keras': path, 'onnx': path}
        """
        keras_path = self.save_keras(model, model_name)
        onnx_path  = self.save_onnx(model, model_name, image_size)
        return {'keras': keras_path, 'onnx': onnx_path}