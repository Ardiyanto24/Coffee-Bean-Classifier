"""
preprocessing.py — Pipeline preprocessing untuk Coffee Bean Classification.

Berisi:
- DuplicateDetector     : deteksi & hapus exact + near-duplicate via MD5 dan pHash
- LabelProcessor        : encoding string label ke integer via sklearn LabelEncoder
- DataSplitter          : group-aware stratified split (train/val/test)
- PreprocessingPipeline : orchestrator yang menjalankan semua step secara berurutan
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.config import PreprocessingConfig


# ==============================================================================
# DuplicateDetector
# ==============================================================================

class DuplicateDetector:
    """
    Detects and removes exact and near-duplicate images from a DataFrame.

    Strategy
    --------
    - Exact duplicates  : identified via MD5 hash of raw image bytes.
    - Near-duplicates   : identified via 64-bit DCT-based perceptual hash (pHash).
                          All image pairs are compared using Hamming distance.
                          Pairs with distance <= phash_threshold are near-duplicates.

    Required Config Fields
    ----------------------
    - config.phash_hash_size : int
        DCT grid size for pHash. Default 8 produces a 64-bit hash (8x8).
    - config.phash_threshold : int
        Maximum Hamming distance to flag a pair as near-duplicate.
    - config.path_col : str
        Column name containing image file paths.

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _compute_md5(self, filepath: str) -> Optional[str]:
        """
        Computes MD5 hash of an image file's raw bytes.

        Parameters
        ----------
        filepath : str
            Absolute path to the image file.

        Returns
        -------
        str or None
            MD5 hex digest string, or None if the file cannot be read.
        """
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.warning(f"MD5 failed for {filepath}: {e}")
            return None

    def _compute_phash(self, filepath: str) -> Optional[imagehash.ImageHash]:
        """
        Computes a 64-bit DCT-based perceptual hash (pHash) for an image.

        Parameters
        ----------
        filepath : str
            Absolute path to the image file.

        Returns
        -------
        imagehash.ImageHash or None
            64-bit perceptual hash object, or None if the image cannot be opened.
        """
        try:
            img = Image.open(filepath)
            return imagehash.phash(img, hash_size=self.config.phash_hash_size)
        except Exception as e:
            self.logger.warning(f"pHash failed for {filepath}: {e}")
            return None

    def remove_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes exact duplicate images using MD5 hashing.
        Keeps the first occurrence of each unique MD5 hash.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a column specified by config.path_col.

        Returns
        -------
        pd.DataFrame
            DataFrame with exact duplicates removed. Adds an 'md5' column.
        """
        self.logger.info("[1/2] Computing MD5 hashes for exact duplicate detection...")
        df       = df.copy()
        df['md5'] = df[self.config.path_col].apply(self._compute_md5)

        n_before  = len(df)
        df        = df.dropna(subset=['md5'])
        df        = df.drop_duplicates(subset='md5', keep='first').reset_index(drop=True)
        n_removed = n_before - len(df)

        self.logger.info(f"    Exact duplicates removed : {n_removed}")
        self.logger.info(f"    Remaining records        : {len(df)}")
        return df

    def remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes near-duplicate images using 64-bit DCT-based pHash.

        Process:
            1. Compute 64-bit pHash for every image.
            2. Compare all pairs using Hamming distance.
            3. Flag pairs with distance <= phash_threshold as near-duplicates.
            4. Keep only the first image in each near-duplicate group.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (should be run after remove_exact_duplicates).

        Returns
        -------
        pd.DataFrame
            DataFrame with near-duplicates removed.
            Adds 'phash' and 'phash_group' columns for traceability.
        """
        self.logger.info("[2/2] Computing 64-bit pHash (DCT) for near-duplicate detection...")
        self.logger.info(f"    Hash size  : {self.config.phash_hash_size}x{self.config.phash_hash_size} = {self.config.phash_hash_size**2}-bit")
        self.logger.info(f"    Threshold  : Hamming distance <= {self.config.phash_threshold}")

        df         = df.copy()
        df['phash'] = df[self.config.path_col].apply(self._compute_phash)
        df          = df.dropna(subset=['phash']).reset_index(drop=True)

        hashes    = df['phash'].tolist()
        threshold = self.config.phash_threshold
        keep_mask = [True] * len(hashes)
        group_ids = list(range(len(hashes)))

        for i in range(len(hashes)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(hashes)):
                if not keep_mask[j]:
                    continue
                if hashes[i] - hashes[j] <= threshold:
                    keep_mask[j] = False
                    group_ids[j] = i

        df['phash_group'] = group_ids
        n_before  = len(df)
        df        = df[keep_mask].reset_index(drop=True)
        n_removed = n_before - len(df)

        self.logger.info(f"    Near-duplicates removed : {n_removed}")
        self.logger.info(f"    Remaining records       : {len(df)}")
        return df

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the full duplicate removal pipeline: exact first, then near-duplicates.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with both exact and near-duplicates removed.
        """
        self.logger.info("=== Duplicate Detection Started ===")
        df = self.remove_exact_duplicates(df)
        df = self.remove_near_duplicates(df)
        self.logger.info("=== Duplicate Detection Completed ===")
        return df


# ==============================================================================
# LabelProcessor
# ==============================================================================

class LabelProcessor:
    """
    Handles categorical (integer) label encoding for class labels.

    Uses sklearn's LabelEncoder to convert string labels to integer indices.
    Stores class name mappings for use in modeling and evaluation stages.

    Required Config Fields
    ----------------------
    - config.label_col : str
        Column name in the DataFrame containing string class labels.

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config       = config
        self.encoder      = LabelEncoder()
        self.class_names:  List[str]      = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.logger       = logging.getLogger(self.__class__.__name__)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits LabelEncoder on the label column and transforms it to integers.
        Adds an 'encoded_label' column to the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the label column.

        Returns
        -------
        pd.DataFrame
            DataFrame with an added 'encoded_label' integer column.
        """
        df                 = df.copy()
        df['encoded_label'] = self.encoder.fit_transform(df[self.config.label_col])
        self.class_names   = list(self.encoder.classes_)
        self.class_to_idx  = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.idx_to_class  = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.logger.info(f"Classes found ({len(self.class_names)}): {self.class_names}")
        self.logger.info(f"Mapping: {self.class_to_idx}")
        return df

    def get_class_info(self) -> Dict:
        """
        Returns a dictionary with class mapping information.

        Returns
        -------
        dict
            Keys: 'class_names', 'class_to_idx', 'idx_to_class', 'num_classes'.
        """
        return {
            'class_names' : self.class_names,
            'class_to_idx': self.class_to_idx,
            'idx_to_class': self.idx_to_class,
            'num_classes' : len(self.class_names)
        }


# ==============================================================================
# DataSplitter
# ==============================================================================

class DataSplitter:
    """
    Performs Group-Aware Stratified Split to prevent data leakage.

    Groups are constructed from pHash group IDs so that visually similar
    images are guaranteed to fall in the same split. Class distribution
    is preserved via stratification.

    Required Config Fields
    ----------------------
    - config.val_size      : float  — fraction for validation
    - config.test_size     : float  — fraction for testing
    - config.random_state  : int    — random seed
    - config.label_col     : str    — column name for class labels
    - config.save_splits   : bool   — whether to save split CSVs
    - config.output_dir    : str    — directory to save output CSVs

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_groups(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts group IDs from 'phash_group' column if available,
        otherwise falls back to unique index-based groups.
        """
        if 'phash_group' in df.columns:
            return df['phash_group'].values
        self.logger.warning("'phash_group' not found. Using row index as group (no group-awareness).")
        return df.index.values

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame into train, validation, and test sets
        using Group-Aware Stratified K-Fold.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'encoded_label' (from LabelProcessor)
            and optionally 'phash_group' (from DuplicateDetector).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            (train_df, val_df, test_df)
        """
        df     = df.copy().reset_index(drop=True)
        labels = df['encoded_label'].values
        groups = self._get_groups(df)

        # Step 1: Split test from the rest
        n_splits_test = max(2, round(1 / self.config.test_size))
        sgkf_test     = StratifiedGroupKFold(
            n_splits=n_splits_test, shuffle=True,
            random_state=self.config.random_state
        )
        train_val_idx, test_idx = next(sgkf_test.split(df, labels, groups))

        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test      = df.iloc[test_idx].reset_index(drop=True)

        # Step 2: Split val from train+val
        val_ratio_adjusted = self.config.val_size / (1 - self.config.test_size)
        n_splits_val       = max(2, round(1 / val_ratio_adjusted))
        labels_tv          = df_train_val['encoded_label'].values
        groups_tv          = self._get_groups(df_train_val)

        sgkf_val           = StratifiedGroupKFold(
            n_splits=n_splits_val, shuffle=True,
            random_state=self.config.random_state
        )
        train_idx, val_idx = next(sgkf_val.split(df_train_val, labels_tv, groups_tv))

        df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
        df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)

        total = len(df)
        self.logger.info(
            f"Split complete — "
            f"Train: {len(df_train)} ({len(df_train)/total:.1%}) | "
            f"Val: {len(df_val)} ({len(df_val)/total:.1%}) | "
            f"Test: {len(df_test)} ({len(df_test)/total:.1%})"
        )
        self._log_class_distribution(df_train, df_val, df_test)

        if self.config.save_splits:
            self._save_splits(df_train, df_val, df_test)

        return df_train, df_val, df_test

    def _log_class_distribution(
        self,
        train: pd.DataFrame,
        val:   pd.DataFrame,
        test:  pd.DataFrame
    ) -> None:
        self.logger.info("Class distribution per split:")
        dist = pd.DataFrame({
            'train': train[self.config.label_col].value_counts(),
            'val'  : val[self.config.label_col].value_counts(),
            'test' : test[self.config.label_col].value_counts()
        }).fillna(0).astype(int)
        print(dist)

    def _save_splits(
        self,
        train: pd.DataFrame,
        val:   pd.DataFrame,
        test:  pd.DataFrame
    ) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        train.to_csv(f"{self.config.output_dir}/train.csv", index=False)
        val.to_csv(f"{self.config.output_dir}/val.csv",     index=False)
        test.to_csv(f"{self.config.output_dir}/test.csv",   index=False)
        self.logger.info(f"Split CSVs saved to: {self.config.output_dir}")


# ==============================================================================
# PreprocessingPipeline
# ==============================================================================

class PreprocessingPipeline:
    """
    High-level pipeline orchestrating the full preprocessing workflow.

    Steps executed in order:
        1. Load metadata CSV and resolve full image paths
        2. Remove exact duplicates (MD5)
        3. Remove near-duplicates (64-bit DCT pHash, Hamming distance <= 4)
        4. Encode labels (categorical integer encoding)
        5. Group-Aware Stratified Split (train / val / test)

    Parameters
    ----------
    config : PreprocessingConfig
        Configuration object.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config             = config
        self.duplicate_detector = DuplicateDetector(config)
        self.label_processor    = LabelProcessor(config)
        self.data_splitter      = DataSplitter(config)
        self.logger             = logging.getLogger(self.__class__.__name__)

    def _load_metadata(self) -> pd.DataFrame:
        """
        Loads the metadata CSV and resolves full image paths.
        Drops rows where the resolved image path does not exist on disk.

        Returns
        -------
        pd.DataFrame
            DataFrame with resolved absolute paths in the path column.
        """
        self.logger.info(f"Loading metadata from: {self.config.metadata_path}")
        df = pd.read_csv(self.config.metadata_path)

        df[self.config.path_col] = df[self.config.path_col].apply(
            lambda p: str(Path(self.config.image_base_dir) / Path(p).name)
            if not os.path.isabs(p) else p
        )

        missing = df[~df[self.config.path_col].apply(os.path.exists)]
        if len(missing) > 0:
            self.logger.warning(f"{len(missing)} image paths not found and will be dropped.")
            df = df[df[self.config.path_col].apply(os.path.exists)].reset_index(drop=True)

        self.logger.info(f"Metadata loaded: {len(df)} valid records.")
        return df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Executes the full preprocessing pipeline end-to-end.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]
            - train_df   : Training split DataFrame
            - val_df     : Validation split DataFrame
            - test_df    : Test split DataFrame
            - class_info : Dict with class_names, class_to_idx,
                           idx_to_class, num_classes
        """
        df         = self._load_metadata()
        df         = self.duplicate_detector.run(df)
        df         = self.label_processor.fit_transform(df)
        class_info = self.label_processor.get_class_info()
        train_df, val_df, test_df = self.data_splitter.split(df)

        self.logger.info("✅ Preprocessing pipeline completed successfully.")
        return train_df, val_df, test_df, class_info