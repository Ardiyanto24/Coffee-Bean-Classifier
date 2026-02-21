"""
test_predictor.py — Unit tests untuk CoffeeBeanPredictor.

Strategi: MOCK ONNX session sehingga test bisa berjalan tanpa
file .onnx sungguhan dan tanpa GPU. Yang ditest adalah:
- Format output predict() sudah benar
- Semua key wajib ada di output dict
- Tipe data setiap field sesuai
- Nilai confidence dan probabilitas dalam range yang valid
- get_model_info() mengembalikan struktur yang benar
- preprocess() menghasilkan shape tensor yang benar
- Error handling saat registry tidak ditemukan
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Pastikan root project ada di sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.predictor import BACKBONE_PREPROCESS, CoffeeBeanPredictor


# ── Fixtures ───────────────────────────────────────────────────────────────────

CLASS_NAMES = ['defect', 'longberry', 'peaberry', 'premium']

@pytest.fixture
def dummy_registry(tmp_path):
    """
    Buat registry.json sementara di tmp_path.

    Menggunakan tmp_path bawaan pytest — folder temporary yang otomatis
    dibuat dan dihapus setelah test selesai. Tidak mengotori folder project.
    """
    # Dummy .onnx file (kosong, hanya agar path exists)
    onnx_path = tmp_path / "EfficientNetB0.onnx"
    onnx_path.write_bytes(b"dummy")

    registry = {
        "active_model": "baseline/EfficientNetB0",
        "models": {
            "baseline/EfficientNetB0": {
                "phase":        "baseline",
                "backbone":     "EfficientNetB0",
                "onnx_path":    str(onnx_path),
                "input_size":   [224, 224],
                "class_names":  CLASS_NAMES,
                "class_to_idx": {c: i for i, c in enumerate(CLASS_NAMES)},
                "metrics":      {},
                "version":      "baseline-v1.0",
            }
        }
    }
    registry_file = tmp_path / "registry.json"
    registry_file.write_text(json.dumps(registry))
    return str(registry_file)


@pytest.fixture
def mock_predictor(dummy_registry):
    """
    Buat CoffeeBeanPredictor dengan ONNX session di-mock.

    patch('onnxruntime.InferenceSession') mengganti InferenceSession
    dengan MagicMock — object tiruan yang bisa dikonfigurasi
    untuk mengembalikan nilai apapun tanpa benar-benar load model.
    """
    with patch('onnxruntime.InferenceSession') as mock_session_cls:
        # Konfigurasi mock session
        mock_session = MagicMock()

        # Mock get_inputs() — predictor butuh nama input pertama
        mock_input      = MagicMock()
        mock_input.name = "input"
        mock_session.get_inputs.return_value = [mock_input]

        # Mock run() — kembalikan probabilitas dummy yang valid
        # Shape: (1, 4) → batch 1, 4 kelas
        # Kelas 'premium' (index 3) punya probabilitas tertinggi: 0.94
        dummy_proba = np.array([[0.01, 0.02, 0.03, 0.94]], dtype=np.float32)
        mock_session.run.return_value = [dummy_proba]

        mock_session_cls.return_value = mock_session

        predictor = CoffeeBeanPredictor(dummy_registry)
        yield predictor


@pytest.fixture
def sample_image():
    """Buat PIL Image kecil (32x32) untuk dipakai sebagai input test."""
    return Image.new('RGB', (32, 32), color=(160, 120, 60))


# ── Tests: predict() output format ────────────────────────────────────────────

class TestPredictOutputFormat:
    """
    Kelompok test untuk memastikan format output predict() benar.

    Semua test di class ini berbagi fixture mock_predictor dan sample_image.
    Dikelompokkan dalam class agar mudah dijalankan bersama:
        pytest tests/test_predictor.py::TestPredictOutputFormat
    """

    def test_predict_returns_dict(self, mock_predictor, sample_image):
        """Output predict() harus berupa dict."""
        result = mock_predictor.predict(sample_image)
        assert isinstance(result, dict), \
            f"Expected dict, got {type(result)}"

    def test_predict_has_required_keys(self, mock_predictor, sample_image):
        """Output dict harus punya semua key wajib."""
        result   = mock_predictor.predict(sample_image)
        required = {"class", "confidence", "probabilities", "class_id"}
        missing  = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_class_is_string(self, mock_predictor, sample_image):
        """Field 'class' harus bertipe str."""
        result = mock_predictor.predict(sample_image)
        assert isinstance(result["class"], str), \
            f"Expected str, got {type(result['class'])}"

    def test_class_is_valid_label(self, mock_predictor, sample_image):
        """Field 'class' harus salah satu dari class_names yang valid."""
        result = mock_predictor.predict(sample_image)
        assert result["class"] in CLASS_NAMES, \
            f"'{result['class']}' bukan class name yang valid. Expected one of {CLASS_NAMES}"

    def test_confidence_is_float(self, mock_predictor, sample_image):
        """Field 'confidence' harus bertipe float."""
        result = mock_predictor.predict(sample_image)
        assert isinstance(result["confidence"], float), \
            f"Expected float, got {type(result['confidence'])}"

    def test_confidence_in_valid_range(self, mock_predictor, sample_image):
        """Field 'confidence' harus dalam range [0.0, 1.0]."""
        result = mock_predictor.predict(sample_image)
        assert 0.0 <= result["confidence"] <= 1.0, \
            f"Confidence {result['confidence']} out of range [0, 1]"

    def test_class_id_is_int(self, mock_predictor, sample_image):
        """Field 'class_id' harus bertipe int."""
        result = mock_predictor.predict(sample_image)
        assert isinstance(result["class_id"], int), \
            f"Expected int, got {type(result['class_id'])}"

    def test_class_id_in_valid_range(self, mock_predictor, sample_image):
        """Field 'class_id' harus dalam range [0, num_classes-1]."""
        result = mock_predictor.predict(sample_image)
        assert 0 <= result["class_id"] < len(CLASS_NAMES), \
            f"class_id {result['class_id']} out of range [0, {len(CLASS_NAMES)-1}]"

    def test_probabilities_is_dict(self, mock_predictor, sample_image):
        """Field 'probabilities' harus bertipe dict."""
        result = mock_predictor.predict(sample_image)
        assert isinstance(result["probabilities"], dict), \
            f"Expected dict, got {type(result['probabilities'])}"

    def test_probabilities_has_all_classes(self, mock_predictor, sample_image):
        """Field 'probabilities' harus punya entry untuk semua kelas."""
        result  = mock_predictor.predict(sample_image)
        missing = set(CLASS_NAMES) - set(result["probabilities"].keys())
        assert not missing, f"Missing classes in probabilities: {missing}"

    def test_probabilities_sum_to_one(self, mock_predictor, sample_image):
        """Jumlah semua probabilitas harus mendekati 1.0 (toleransi ±0.01)."""
        result = mock_predictor.predict(sample_image)
        total  = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01, \
            f"Probabilities sum to {total}, expected ~1.0"

    def test_probabilities_all_in_valid_range(self, mock_predictor, sample_image):
        """Setiap nilai probabilitas harus dalam range [0.0, 1.0]."""
        result = mock_predictor.predict(sample_image)
        for cls, prob in result["probabilities"].items():
            assert 0.0 <= prob <= 1.0, \
                f"Probability for '{cls}' = {prob} out of range [0, 1]"

    def test_class_matches_highest_probability(self, mock_predictor, sample_image):
        """Field 'class' harus sesuai dengan kelas yang punya probabilitas tertinggi."""
        result       = mock_predictor.predict(sample_image)
        expected_cls = max(result["probabilities"], key=result["probabilities"].get)
        assert result["class"] == expected_cls, \
            f"class='{result['class']}' tidak sesuai dengan argmax='{expected_cls}'"

    def test_confidence_matches_class_probability(self, mock_predictor, sample_image):
        """Field 'confidence' harus sama dengan probabilitas kelas yang diprediksi."""
        result     = mock_predictor.predict(sample_image)
        class_prob = result["probabilities"][result["class"]]
        assert abs(result["confidence"] - class_prob) < 1e-4, \
            f"confidence={result['confidence']} tidak sesuai dengan prob={class_prob}"


# ── Tests: predict() dengan berbagai format gambar ────────────────────────────

class TestPredictImageFormats:
    """Test bahwa predict() bisa handle berbagai format gambar input."""

    def test_predict_with_rgb_image(self, mock_predictor):
        """Gambar RGB harus bisa diproses."""
        image  = Image.new('RGB', (100, 100), color=(100, 150, 200))
        result = mock_predictor.predict(image)
        assert "class" in result

    def test_predict_with_rgba_image(self, mock_predictor):
        """Gambar RGBA (dengan alpha channel) harus diconvert ke RGB dan diproses."""
        image  = Image.new('RGBA', (100, 100), color=(100, 150, 200, 255))
        result = mock_predictor.predict(image)
        assert "class" in result

    def test_predict_with_grayscale_image(self, mock_predictor):
        """Gambar grayscale harus diconvert ke RGB dan diproses."""
        image  = Image.new('L', (100, 100), color=128)
        result = mock_predictor.predict(image)
        assert "class" in result

    def test_predict_with_small_image(self, mock_predictor):
        """Gambar kecil (1x1) harus di-resize dan diproses tanpa error."""
        image  = Image.new('RGB', (1, 1), color=(100, 100, 100))
        result = mock_predictor.predict(image)
        assert "class" in result

    def test_predict_with_large_image(self, mock_predictor):
        """Gambar besar harus di-resize dan diproses tanpa error."""
        image  = Image.new('RGB', (2000, 2000), color=(100, 100, 100))
        result = mock_predictor.predict(image)
        assert "class" in result


# ── Tests: preprocess() ───────────────────────────────────────────────────────

class TestPreprocess:
    """Test bahwa preprocess() menghasilkan tensor dengan shape dan dtype benar."""

    def test_preprocess_output_shape(self, mock_predictor, sample_image):
        """
        Output preprocess() harus punya shape (1, H, W, 3).
        1 = batch dimension, H dan W = input_size backbone, 3 = RGB channels.
        """
        tensor = mock_predictor.preprocess(sample_image)
        h, w   = mock_predictor._input_size
        assert tensor.shape == (1, h, w, 3), \
            f"Expected shape (1, {h}, {w}, 3), got {tensor.shape}"

    def test_preprocess_output_dtype(self, mock_predictor, sample_image):
        """Output preprocess() harus bertipe float32."""
        tensor = mock_predictor.preprocess(sample_image)
        assert tensor.dtype == np.float32, \
            f"Expected float32, got {tensor.dtype}"

    def test_preprocess_output_is_numpy(self, mock_predictor, sample_image):
        """Output preprocess() harus berupa numpy array."""
        tensor = mock_predictor.preprocess(sample_image)
        assert isinstance(tensor, np.ndarray), \
            f"Expected np.ndarray, got {type(tensor)}"


# ── Tests: get_model_info() ───────────────────────────────────────────────────

class TestGetModelInfo:
    """Test bahwa get_model_info() mengembalikan struktur yang benar."""

    def test_get_model_info_returns_dict(self, mock_predictor):
        """Output get_model_info() harus berupa dict."""
        info = mock_predictor.get_model_info()
        assert isinstance(info, dict)

    def test_get_model_info_has_required_keys(self, mock_predictor):
        """Output get_model_info() harus punya key penting."""
        info     = mock_predictor.get_model_info()
        required = {"active_model", "backbone", "input_size", "class_names"}
        missing  = required - set(info.keys())
        assert not missing, f"Missing keys in model_info: {missing}"

    def test_get_model_info_backbone_matches(self, mock_predictor):
        """Backbone di model_info harus sesuai dengan yang ada di registry."""
        info = mock_predictor.get_model_info()
        assert info["backbone"] == "EfficientNetB0"

    def test_get_model_info_class_names_correct(self, mock_predictor):
        """class_names di model_info harus sesuai dengan CLASS_NAMES."""
        info = mock_predictor.get_model_info()
        assert info["class_names"] == CLASS_NAMES


# ── Tests: Error Handling ─────────────────────────────────────────────────────

class TestErrorHandling:
    """Test bahwa error handling berjalan dengan benar."""

    def test_registry_not_found_raises_error(self):
        """Harus raise FileNotFoundError jika registry.json tidak ada."""
        with pytest.raises(FileNotFoundError):
            CoffeeBeanPredictor("/path/yang/tidak/ada/registry.json")

    def test_active_model_not_in_registry_raises_error(self, tmp_path):
        """Harus raise KeyError jika active_model tidak ada di daftar models."""
        registry = {
            "active_model": "baseline/ModelYangTidakAda",
            "models": {}
        }
        registry_file = tmp_path / "registry.json"
        registry_file.write_text(json.dumps(registry))

        with patch('onnxruntime.InferenceSession'):
            with pytest.raises(KeyError):
                CoffeeBeanPredictor(str(registry_file))


# ── Tests: BACKBONE_PREPROCESS ────────────────────────────────────────────────

class TestBackbonePreprocess:
    """Test bahwa semua fungsi normalisasi di BACKBONE_PREPROCESS bekerja benar."""

    @pytest.mark.parametrize("backbone", [
        'EfficientNetB0', 'EfficientNetB3',
        'ResNet50', 'DenseNet121', 'MobileNetV3Small'
    ])
    def test_preprocess_fn_output_shape(self, backbone):
        """
        Setiap fungsi normalisasi harus mempertahankan shape input.

        @pytest.mark.parametrize menjalankan test ini 5 kali —
        sekali untuk setiap backbone — tanpa perlu tulis 5 fungsi terpisah.
        """
        fn     = BACKBONE_PREPROCESS[backbone]
        dummy  = np.ones((224, 224, 3), dtype=np.float32) * 128.0
        output = fn(dummy)
        assert output.shape == (224, 224, 3), \
            f"[{backbone}] Shape berubah: {dummy.shape} → {output.shape}"

    @pytest.mark.parametrize("backbone", [
        'EfficientNetB0', 'EfficientNetB3',
        'ResNet50', 'DenseNet121', 'MobileNetV3Small'
    ])
    def test_preprocess_fn_output_dtype(self, backbone):
        """Setiap fungsi normalisasi harus mengembalikan float32."""
        fn     = BACKBONE_PREPROCESS[backbone]
        dummy  = np.ones((224, 224, 3), dtype=np.float32) * 128.0
        output = fn(dummy)
        assert output.dtype == np.float32, \
            f"[{backbone}] Expected float32, got {output.dtype}"