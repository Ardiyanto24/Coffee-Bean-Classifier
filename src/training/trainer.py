import logging
import os
from typing import Dict, List

from tensorflow import keras

from src.config import BaselineConfig
from src.data.dataset_builder import DatasetBuilder
from src.models.model_builder import ModelBuilder
from src.models.model_exporter import ModelExporter


class Trainer:
    """
    Manages the training loop for one or multiple baseline models.

    Internally uses DatasetBuilder, ModelBuilder, and ModelExporter.
    After training each model, results are stored in self.results for
    downstream use by Evaluator and ComparisonTable.

    Required Config Fields
    ----------------------
    - config.models_to_train : List[str]
        List of model names to train.
    - config.epochs : int
        Maximum training epochs.
    - config.output_dir : str
        Root directory for checkpoints and exported models.
    - config.train_csv / config.val_csv : str
        Paths to training and validation CSVs.

    Parameters
    ----------
    config : BaselineConfig
        Configuration object.
    model_builder : ModelBuilder
        Shared ModelBuilder instance (carries custom model registry).
    """

    def __init__(self, config: BaselineConfig, model_builder: ModelBuilder):
        self.config          = config
        self.model_builder   = model_builder
        self.dataset_builder = DatasetBuilder(config)
        self.exporter        = ModelExporter(config)
        self.logger          = logging.getLogger(self.__class__.__name__)
        self.results: Dict   = {}

    def _get_callbacks(self, model_name: str) -> List:
        """
        Returns the list of Keras callbacks for training.

        Callbacks:
        - EarlyStopping     : patience=5 on val_loss, restores best weights
        - ModelCheckpoint   : saves best weights to output_dir/checkpoints/
        - ReduceLROnPlateau : factor=0.5, patience=3 on val_loss

        Parameters
        ----------
        model_name : str
            Used to name the checkpoint file.

        Returns
        -------
        List
            List of tf.keras.callbacks instances.
        """
        ckpt_dir  = os.path.join(self.config.output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.weights.h5")

        return [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

    def train_one(self, model_name: str) -> Dict:
        """
        Runs the full training pipeline for a single model.

        Steps:
            1. Resolve preprocess_fn and image_size via ModelBuilder
            2. Build train and val tf.data.Dataset
            3. Build and compile the model
            4. Train with callbacks
            5. Export .keras and .onnx
            6. Store result in self.results

        Parameters
        ----------
        model_name : str
            Name of the model to train.

        Returns
        -------
        Dict
            {'model': model, 'history': history, 'image_size': image_size,
             'preprocess_fn': preprocess_fn}
        """
        self.logger.info(f"{'='*50}")
        self.logger.info(f" Training: {model_name}")
        self.logger.info(f"{'='*50}")

        preprocess_fn, image_size = self.model_builder.get_backbone_config(model_name)

        train_ds = self.dataset_builder.build(
            self.config.train_csv, preprocess_fn, image_size, shuffle=True
        )
        val_ds = self.dataset_builder.build(
            self.config.val_csv, preprocess_fn, image_size, shuffle=False
        )

        model = self.model_builder.build(model_name)
        model = self.model_builder.compile(model)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=self._get_callbacks(model_name),
            verbose=1
        )

        self.exporter.export(model, model_name, image_size)

        result = {
            'model':        model,
            'history':      history,
            'image_size':   image_size,
            'preprocess_fn': preprocess_fn
        }
        self.results[model_name] = result
        self.logger.info(f"Training complete for '{model_name}'")
        return result

    def train_all(self) -> Dict:
        """
        Trains all models listed in config.models_to_train sequentially.

        Returns
        -------
        Dict
            Dictionary of all results keyed by model_name.
        """
        for model_name in self.config.models_to_train:
            try:
                self.train_one(model_name)
            except Exception as e:
                self.logger.error(f"Training failed for '{model_name}': {e}")
        self.logger.info(f"All models trained. Total: {len(self.results)}")
        return self.results