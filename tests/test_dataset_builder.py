"""
test_dataset_builder.py — Unit tests untuk DatasetBuilder.

Strategi: REAL test tanpa mock — kita buat CSV dan gambar
kecil secara programatik di tmp_path, lalu jalankan
DatasetBuilder sungguhan. Yang ditest adalah:
- Dataset bisa di-build dari CSV tanpa error
- Shape batch tensor sesuai (batch_size, H, W, 3)
- Dtype tensor float32
- Label di-load dengan benar dari CSV
- Shuffle tidak merusak integritas data
- Error handling jika CSV tidak ada
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

# Pastikan root project ada di sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import TensorFlow hanya jika tersedia
# Jika tidak ada, semua test di file ini di-skip otomatis
tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow tidak terinstall — test DatasetBuilder di-skip."
)

from src.config import BaselineConfig
from src.data.dataset_builder import DatasetBuilder


# ── Fixtures ───────────────────────────────────────────────────────────────────

CLASS_NAMES  = ['defect', 'longberry', 'peaberry', 'premium']
NUM_CLASSES  = 4
SAMPLES_PER_CLASS = 3   # Total: 12 gambar — cukup untuk test, cepat di-generate
IMAGE_SIZE   = (32, 32) # Kecil agar test cepat


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory):
    """
    Buat dataset kecil yang lengkap: folder gambar + CSV.

    scope="module" berarti fixture ini dibuat sekali per file test,
    bukan sekali per test function. Ini penting karena generate gambar
    dan tulis CSV memakan waktu — tidak perlu diulang tiap test.

    Struktur yang dibuat:
        tmp/
        ├── images/
        │   ├── defect_0.jpg ... defect_2.jpg
        │   ├── longberry_0.jpg ...
        │   └── ...
        └── splits/
            ├── train.csv
            └── val.csv
    """
    tmp_path   = tmp_path_factory.mktemp("dataset")
    images_dir = tmp_path / "images"
    splits_dir = tmp_path / "splits"
    images_dir.mkdir()
    splits_dir.mkdir()

    # Warna per kelas agar gambar berbeda
    class_colors = {
        'defect':    (180, 80,  80),
        'longberry': (160, 120, 60),
        'peaberry':  (80,  120, 180),
        'premium':   (80,  160, 80),
    }

    rows = []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        for i in range(SAMPLES_PER_CLASS):
            # Generate gambar kecil
            color    = class_colors[class_name]
            img      = Image.new('RGB', IMAGE_SIZE, color=color)
            img_path = images_dir / f"{class_name}_{i}.jpg"
            img.save(str(img_path))

            rows.append({
                'filepath':      str(img_path),
                'encoded_label': label_idx,
            })

    # Buat CSV
    df         = pd.DataFrame(rows)
    train_csv  = splits_dir / "train.csv"
    val_csv    = splits_dir / "val.csv"
    df.to_csv(str(train_csv), index=False)
    df.iloc[:4].to_csv(str(val_csv), index=False)  # val pakai subset kecil

    return {
        "train_csv": str(train_csv),
        "val_csv":   str(val_csv),
        "df":        df,
        "tmp_path":  tmp_path,
    }


@pytest.fixture(scope="module")
def config(sample_dataset):
    """Buat BaselineConfig yang mengarah ke dataset kecil."""
    return BaselineConfig(
        train_csv   = sample_dataset["train_csv"],
        val_csv     = sample_dataset["val_csv"],
        test_csv    = sample_dataset["val_csv"],  # pakai val sebagai test
        num_classes = NUM_CLASSES,
        class_names = CLASS_NAMES,
        batch_size  = 4,
        output_dir  = str(sample_dataset["tmp_path"] / "output"),
    )


@pytest.fixture(scope="module")
def builder(config):
    """Buat DatasetBuilder dari config."""
    return DatasetBuilder(config)


@pytest.fixture(scope="module")
def preprocess_fn():
    """Preprocess function sederhana: normalisasi ke [0, 1]."""
    return lambda x: x / 255.0


# ── Tests: build() basic ───────────────────────────────────────────────────────

class TestDatasetBuilderBuild:
    """Test bahwa build() bisa menghasilkan dataset yang valid."""

    def test_build_returns_tf_dataset(self, builder, sample_dataset, preprocess_fn):
        """Output build() harus berupa tf.data.Dataset."""
        ds = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        assert isinstance(ds, tf.data.Dataset), \
            f"Expected tf.data.Dataset, got {type(ds)}"

    def test_dataset_is_iterable(self, builder, sample_dataset, preprocess_fn):
        """Dataset harus bisa di-iterate tanpa error."""
        ds = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        # Ambil satu batch — jika gagal, test akan raise exception
        batch = next(iter(ds))
        assert batch is not None

    def test_batch_returns_tuple(self, builder, sample_dataset, preprocess_fn):
        """Setiap batch harus berupa tuple (images, labels)."""
        ds    = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        batch = next(iter(ds))
        assert isinstance(batch, tuple) and len(batch) == 2, \
            f"Expected tuple of length 2, got {type(batch)} of length {len(batch)}"


# ── Tests: batch shape dan dtype ──────────────────────────────────────────────

class TestDatasetBatchShape:
    """Test bahwa shape dan dtype tensor dalam batch sesuai yang diharapkan."""

    def test_image_batch_shape(self, builder, sample_dataset, preprocess_fn):
        """
        Shape image batch harus (batch_size, H, W, 3).
        batch_size dari config = 4, IMAGE_SIZE = (32, 32), channels = 3.
        """
        ds           = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        images, _    = next(iter(ds))
        expected_shape = (4, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        assert tuple(images.shape) == expected_shape, \
            f"Expected shape {expected_shape}, got {tuple(images.shape)}"

    def test_label_batch_shape(self, builder, sample_dataset, preprocess_fn):
        """Shape label batch harus (batch_size,) — satu label per gambar."""
        ds        = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        _, labels = next(iter(ds))
        assert tuple(labels.shape) == (4,), \
            f"Expected shape (4,), got {tuple(labels.shape)}"

    def test_image_dtype_is_float(self, builder, sample_dataset, preprocess_fn):
        """Image tensor harus bertipe float (hasil preprocessing)."""
        ds        = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        images, _ = next(iter(ds))
        assert images.dtype in (tf.float32, tf.float64), \
            f"Expected float dtype, got {images.dtype}"

    def test_label_dtype_is_int(self, builder, sample_dataset, preprocess_fn):
        """Label tensor harus bertipe integer."""
        ds        = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        _, labels = next(iter(ds))
        assert labels.dtype in (tf.int32, tf.int64), \
            f"Expected int dtype, got {labels.dtype}"

    def test_image_resize_applied(self, builder, sample_dataset, preprocess_fn):
        """
        DatasetBuilder harus resize gambar ke target image_size.
        Kita gunakan target berbeda dari ukuran asli (32x32) untuk membuktikan
        resize benar-benar terjadi.
        """
        target_size = (64, 64)  # Berbeda dari IMAGE_SIZE (32, 32)
        ds          = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            target_size,
            shuffle=False
        )
        images, _   = next(iter(ds))
        assert tuple(images.shape)[1:3] == target_size, \
            f"Expected spatial dims {target_size}, got {tuple(images.shape)[1:3]}"


# ── Tests: label integrity ────────────────────────────────────────────────────

class TestDatasetLabelIntegrity:
    """Test bahwa label di-load dengan benar dari CSV."""

    def test_label_values_in_valid_range(self, builder, sample_dataset, preprocess_fn):
        """Semua label harus dalam range [0, num_classes-1]."""
        ds = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        for _, labels in ds:
            label_vals = labels.numpy()
            assert np.all(label_vals >= 0), \
                f"Ditemukan label negatif: {label_vals}"
            assert np.all(label_vals < NUM_CLASSES), \
                f"Label melebihi num_classes: {label_vals}"

    def test_all_classes_represented(self, builder, sample_dataset, preprocess_fn):
        """Semua kelas harus muncul minimal sekali di dataset."""
        ds          = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        all_labels = []
        for _, labels in ds:
            all_labels.extend(labels.numpy().tolist())

        unique_labels = set(all_labels)
        expected      = set(range(NUM_CLASSES))
        assert unique_labels == expected, \
            f"Kelas yang muncul: {unique_labels}, expected: {expected}"

    def test_total_samples_match_csv(self, builder, sample_dataset, preprocess_fn):
        """Total sampel dari dataset harus sama dengan jumlah baris di CSV."""
        ds = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        total_samples = sum(len(labels) for _, labels in ds)
        csv_rows      = len(sample_dataset["df"])
        assert total_samples == csv_rows, \
            f"Dataset punya {total_samples} sampel, CSV punya {csv_rows} baris"


# ── Tests: shuffle ────────────────────────────────────────────────────────────

class TestDatasetShuffle:
    """Test bahwa shuffle bisa diaktifkan tanpa merusak integritas data."""

    def test_shuffle_does_not_change_total_samples(
        self, builder, sample_dataset, preprocess_fn
    ):
        """Shuffle tidak boleh mengurangi atau menambah jumlah sampel."""
        ds_no_shuffle = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        ds_shuffle = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=True
        )
        total_no_shuffle = sum(len(l) for _, l in ds_no_shuffle)
        total_shuffle    = sum(len(l) for _, l in ds_shuffle)
        assert total_no_shuffle == total_shuffle, \
            f"Shuffle mengubah jumlah sampel: {total_no_shuffle} → {total_shuffle}"

    def test_shuffle_does_not_change_label_distribution(
        self, builder, sample_dataset, preprocess_fn
    ):
        """Distribusi label harus sama sebelum dan sesudah shuffle."""
        def get_label_counts(ds):
            counts = np.zeros(NUM_CLASSES, dtype=int)
            for _, labels in ds:
                for lbl in labels.numpy():
                    counts[lbl] += 1
            return counts

        ds_no_shuffle = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=False
        )
        ds_shuffle = builder.build(
            sample_dataset["train_csv"],
            preprocess_fn,
            IMAGE_SIZE,
            shuffle=True
        )
        counts_no_shuffle = get_label_counts(ds_no_shuffle)
        counts_shuffle    = get_label_counts(ds_shuffle)

        np.testing.assert_array_equal(
            counts_no_shuffle, counts_shuffle,
            err_msg="Distribusi label berubah setelah shuffle"
        )


# ── Tests: Error Handling ─────────────────────────────────────────────────────

class TestDatasetBuilderErrorHandling:
    """Test bahwa error handling DatasetBuilder berjalan dengan benar."""

    def test_build_with_nonexistent_csv_raises_error(self, builder, preprocess_fn):
        """Harus raise error jika CSV tidak ditemukan."""
        with pytest.raises(Exception):
            ds = builder.build(
                "/path/yang/tidak/ada/data.csv",
                preprocess_fn,
                IMAGE_SIZE,
                shuffle=False
            )
            # Consume dataset untuk trigger error
            list(ds)