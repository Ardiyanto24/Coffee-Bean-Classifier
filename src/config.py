import logging
import warnings
from dataclasses import dataclass, field
from typing import List

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ==============================================================================
# PreprocessingConfig
# Digunakan oleh: notebooks/01_preprocessing.ipynb
#                 src/data/preprocessing.py
# ==============================================================================

@dataclass
class PreprocessingConfig:
    """
    Configuration dataclass for the preprocessing pipeline.
    All pipeline classes read their parameters from this object.

    Notes
    -----
    - phash_hash_size=8 produces a 64-bit hash (8x8 DCT grid), which is the
      standard for perceptual hashing. Increasing this value raises sensitivity
      but also computation time.
    - phash_threshold=4 is the standard Hamming distance cutoff for 64-bit pHash.
      Images with distance <= 4 are considered near-duplicates.
    """

    # --- Data paths (wajib diisi, tidak ada default path Kaggle) ---
    metadata_path:  str = ""
    image_base_dir: str = ""

    # --- Duplicate detection settings ---
    phash_hash_size: int = 8    # 8x8 DCT grid → 64-bit hash
    phash_threshold: int = 4    # Hamming distance <= 4 → near-duplicate

    # --- Split settings ---
    val_size:     float = 0.15
    test_size:    float = 0.15
    random_state: int   = 42

    # --- Column names ---
    path_col:  str = 'filepath'
    label_col: str = 'label'

    # --- Output settings ---
    save_splits: bool = True
    output_dir:  str  = ""


# ==============================================================================
# BaselineConfig
# Digunakan oleh: notebooks/02_modeling_baseline.ipynb
#                 src/training/trainer.py
#                 src/training/evaluator.py
# ==============================================================================

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
    train_csv: str = ""
    val_csv:   str = ""
    test_csv:  str = ""

    # --- Dataset settings ---
    num_classes:  int       = 4
    class_names:  List[str] = field(default_factory=lambda: [
        'defect', 'longberry', 'peaberry', 'premium'
    ])
    path_col:  str = 'filepath'
    label_col: str = 'encoded_label'

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