import logging
import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tensorflow import keras

from src.config import BaselineConfig
from src.data.dataset_builder import DatasetBuilder


class Evaluator:
    """
    Evaluates trained models on the test set.

    Produces:
    - Numerical metrics : Accuracy, F1, Precision, Recall, ROC-AUC (macro OvR),
                          and per-class breakdown
    - Confusion matrix plot
    - Training curve plot (loss and accuracy)

    Required Config Fields
    ----------------------
    - config.test_csv : str
        Path to test CSV.
    - config.class_names : List[str]
        Ordered list of class names.
    - config.num_classes : int
        Number of output classes.
    - config.output_dir : str
        Root directory. Plots are saved under output_dir/plots/.

    Parameters
    ----------
    config : BaselineConfig
        Configuration object.
    """

    def __init__(self, config: BaselineConfig):
        self.config          = config
        self.logger          = logging.getLogger(self.__class__.__name__)
        self.plots_dir       = os.path.join(config.output_dir, 'plots')
        self.dataset_builder = DatasetBuilder(config)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _get_predictions(
        self,
        model: keras.Model,
        preprocess_fn: Callable,
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs inference on the test set and returns predictions and true labels.

        Parameters
        ----------
        model : tf.keras.Model
            Trained model.
        preprocess_fn : Callable
            Backbone-specific preprocessing function.
        image_size : Tuple[int, int]
            Model input resolution.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (y_true, y_pred_classes, y_pred_proba)
        """
        test_ds        = self.dataset_builder.build(
            self.config.test_csv, preprocess_fn, image_size, shuffle=False
        )
        y_pred_proba   = model.predict(test_ds, verbose=0)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)

        test_df = pd.read_csv(self.config.test_csv)
        y_true  = test_df[self.config.label_col].values

        return y_true, y_pred_classes, y_pred_proba

    def evaluate(
        self,
        model: keras.Model,
        model_name: str,
        preprocess_fn: Callable,
        image_size: Tuple[int, int]
    ) -> Dict:
        """
        Computes all evaluation metrics for a trained model on the test set.

        Metrics:
        - Overall  : Accuracy, F1 (macro), Precision (macro), Recall (macro),
                     ROC-AUC (OvR macro)
        - Per-class: F1, Precision, Recall for each class

        Parameters
        ----------
        model : tf.keras.Model
            Trained model.
        model_name : str
            Used for logging and output labeling.
        preprocess_fn : Callable
            Backbone-specific preprocessing function.
        image_size : Tuple[int, int]
            Model input resolution.

        Returns
        -------
        Dict
            Flat dictionary of all metrics (overall + per-class).
        """
        y_true, y_pred, y_proba = self._get_predictions(model, preprocess_fn, image_size)

        y_true_bin = label_binarize(y_true, classes=list(range(self.config.num_classes)))

        metrics = {
            'accuracy':        round(accuracy_score(y_true, y_pred), 4),
            'f1_macro':        round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'precision_macro': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'recall_macro':    round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
            'roc_auc_macro':   round(roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro'), 4),
        }

        f1_per_class        = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class    = recall_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.config.class_names):
            metrics[f'f1_{class_name}']        = round(f1_per_class[i], 4)
            metrics[f'precision_{class_name}'] = round(precision_per_class[i], 4)
            metrics[f'recall_{class_name}']    = round(recall_per_class[i], 4)

        self.logger.info(
            f"[{model_name}] Accuracy={metrics['accuracy']} | "
            f"F1={metrics['f1_macro']} | ROC-AUC={metrics['roc_auc_macro']}"
        )

        self._last_y_true  = y_true
        self._last_y_pred  = y_pred
        self._last_y_proba = y_proba

        return metrics

    def plot_confusion_matrix(self, model_name: str) -> None:
        """
        Plots and saves the confusion matrix for the last evaluated model.
        Must be called after evaluate().

        Parameters
        ----------
        model_name : str
            Used for the plot title and saved filename.
        """
        cm = confusion_matrix(self._last_y_true, self._last_y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.config.class_names,
            yticklabels=self.config.class_names,
            ax=ax
        )
        ax.set_title(f'Confusion Matrix — {model_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout()

        save_path = os.path.join(self.plots_dir, f"{model_name}_confusion_matrix.png")
        plt.savefig(save_path, dpi=150)
        plt.show()
        self.logger.info(f"Confusion matrix saved → {save_path}")

    def plot_training_curve(
        self,
        history: keras.callbacks.History,
        model_name: str
    ) -> None:
        """
        Plots and saves the training and validation loss and accuracy curves.

        Parameters
        ----------
        history : tf.keras.callbacks.History
            History object returned by model.fit().
        model_name : str
            Used for the plot title and saved filename.
        """
        hist       = history.history
        epochs_ran = range(1, len(hist['loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Training Curves — {model_name}', fontsize=13, fontweight='bold')

        axes[0].plot(epochs_ran, hist['loss'],     label='Train Loss', linewidth=2)
        axes[0].plot(epochs_ran, hist['val_loss'], label='Val Loss',   linewidth=2, linestyle='--')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_ran, hist['accuracy'],     label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs_ran, hist['val_accuracy'], label='Val Accuracy',   linewidth=2, linestyle='--')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f"{model_name}_training_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.show()
        self.logger.info(f"Training curve saved → {save_path}")


class ComparisonTable:
    """
    Collects and displays evaluation metrics across all trained baseline models.

    The resulting table is the primary decision artifact for selecting which
    model to carry forward to XAI and Hyperparameter Tuning.
    Best model selection is left to the user.

    Required Config Fields
    ----------------------
    - config.output_dir : str
        Directory where comparison_table.csv will be saved.

    Parameters
    ----------
    config : BaselineConfig
        Configuration object.
    """

    def __init__(self, config: BaselineConfig):
        self.config   = config
        self.logger   = logging.getLogger(self.__class__.__name__)
        self._records: List[Dict] = []

    def add(self, model_name: str, metrics_dict: Dict) -> None:
        """
        Adds a model's evaluation metrics to the comparison table.

        Parameters
        ----------
        model_name : str
            Identifier for the model.
        metrics_dict : Dict
            Dictionary of metrics returned by Evaluator.evaluate().
        """
        record = {'model': model_name, **metrics_dict}
        self._records.append(record)
        self.logger.info(f"Added '{model_name}' to comparison table.")

    def show(self) -> pd.DataFrame:
        """
        Returns and displays the comparison DataFrame sorted by accuracy (desc).

        Returns
        -------
        pd.DataFrame
            Comparison table with all models and their metrics.
        """
        df = pd.DataFrame(self._records)
        df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)

        overall_cols = [
            'model', 'accuracy', 'f1_macro',
            'precision_macro', 'recall_macro', 'roc_auc_macro'
        ]
        print("\n" + "=" * 60)
        print("  BASELINE MODEL COMPARISON TABLE")
        print("=" * 60)
        try:
            from IPython.display import display
            display(df[overall_cols].style.highlight_max(
                subset=overall_cols[1:], color='#d4edda', axis=0
            ))
            print("\n  Per-Class Metrics:")
            display(df)
        except ImportError:
            print(df[overall_cols].to_string(index=False))

        return df

    def save(self) -> str:
        """
        Saves the comparison table as a CSV file.

        Returns
        -------
        str
            Absolute path to the saved CSV file.
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, 'comparison_table.csv')
        df = pd.DataFrame(self._records)
        df.to_csv(save_path, index=False)
        self.logger.info(f"Comparison table saved → {save_path}")
        return save_path