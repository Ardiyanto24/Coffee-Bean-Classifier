import logging
import warnings
from dataclasses import dataclass, field
from typing import List

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


@dataclass
class BaselineConfig:
    """
    Configuration dataclass for the baseline modeling pipeline.
    All pipeline classes read their parameters from this object.

    Notes
    -----
    - image_size and preprocess_input are NOT stored here.
      They are resolved automatically per backbone by ModelBuilder.
    - models_to_train entries must match keys in ModelBuilder.BACKBONE_CONFIG
      or be registered via ModelBuilder.register_custom_model().
    - train_csv, val_csv, test_csv, dan output_dir tidak di-hardcode.
      Isi saat membuat instance config di notebook atau script.
    """

    # --- Data paths (wajib diisi, tidak ada default path Kaggle) ---
    train_csv:  str = ""
    val_csv:    str = ""
    test_csv:   str = ""

    # --- Dataset settings ---
    num_classes:  int       = 4
    class_names:  List[str] = field(default_factory=lambda: [
        'defect', 'longberry', 'peaberry', 'premium'
    ])
    path_col:     str       = 'filepath'
    label_col:    str       = 'encoded_label'

    # --- Model selection ---
    models_to_train: List[str] = field(default_factory=lambda: [
        'ResNet50', 'EfficientNetB0', 'EfficientNetB3',
        'MobileNetV3Small', 'DenseNet121'
    ])

    # --- Training hyperparameters ---
    batch_size:    int   = 32
    epochs:        int   = 30
    learning_rate: float = 1e-3

    # --- Output (wajib diisi, tidak ada default path Kaggle) ---
    output_dir: str = ""