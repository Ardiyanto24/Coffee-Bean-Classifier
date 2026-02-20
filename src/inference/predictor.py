"""
predictor.py — Inference engine untuk Coffee Bean Quality Classifier.

Berisi:
- BACKBONE_PREPROCESS : mapping backbone → fungsi normalisasi numpy
- CoffeeBeanPredictor : class utama untuk inference via ONNX Runtime

Digunakan oleh: app/pages/1_Prediction.py (Streamlit)

Tidak ada dependency ke TensorFlow — hanya butuh onnxruntime, numpy, Pillow.
Ini yang membuat Docker image tetap ringan.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)


# ==============================================================================
# BACKBONE_PREPROCESS
# Mapping backbone name → fungsi normalisasi numpy.
#
# Harus konsisten dengan ModelBuilder.BACKBONE_CONFIG di src/models/model_builder.py
# karena model ditraining dengan normalisasi tersebut.
#
# Cara menambah backbone baru:
#   Tambahkan entry baru di dict ini dengan fungsi normalisasi yang sesuai.
# ==============================================================================

def _efficientnet_normalize(x: np.ndarray) -> np.ndarray:
    """EfficientNet: scale pixel values to [-1, 1]."""
    return (x / 127.5) - 1.0

def _resnet_normalize(x: np.ndarray) -> np.ndarray:
    """ResNet50: subtract ImageNet channel means (BGR order, Keras convention)."""
    x = x.astype(np.float32)
    x[..., 0] -= 103.939  # B
    x[..., 1] -= 116.779  # G
    x[..., 2] -= 123.680  # R
    return x

def _densenet_normalize(x: np.ndarray) -> np.ndarray:
    """DenseNet: scale to [0, 1] then subtract ImageNet mean, divide by std."""
    x = x / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (x - mean) / std

def _mobilenetv3_normalize(x: np.ndarray) -> np.ndarray:
    """MobileNetV3: scale pixel values to [-1, 1]."""
    return (x / 127.5) - 1.0


BACKBONE_PREPROCESS: Dict[str, Callable] = {
    'EfficientNetB0':   _efficientnet_normalize,
    'EfficientNetB3':   _efficientnet_normalize,
    'ResNet50':         _resnet_normalize,
    'DenseNet121':      _densenet_normalize,
    'MobileNetV3Small': _mobilenetv3_normalize,
}


# ==============================================================================
# CoffeeBeanPredictor
# ==============================================================================

class CoffeeBeanPredictor:
    """
    Inference engine untuk Coffee Bean Quality Classifier.

    Load model ONNX berdasarkan konfigurasi di registry.json,
    lakukan preprocessing gambar, dan kembalikan hasil prediksi.

    Model hanya di-load sekali saat __init__ (singleton pattern)
    sehingga inference berikutnya tidak perlu load ulang.

    Parameters
    ----------
    registry_path : str or Path
        Path ke file registry.json yang berisi daftar model dan model aktif.

    Raises
    ------
    FileNotFoundError
        Jika registry.json atau file ONNX tidak ditemukan.
    KeyError
        Jika active_model tidak ditemukan di registry.
    ValueError
        Jika backbone model aktif tidak ada di BACKBONE_PREPROCESS.

    Examples
    --------
    >>> predictor = CoffeeBeanPredictor("models/registry.json")
    >>> image = Image.open("sample.jpg")
    >>> result = predictor.predict(image)
    >>> print(result["class"])       # "premium"
    >>> print(result["confidence"])  # 0.94
    """

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.logger        = logging.getLogger(self.__class__.__name__)

        # Load registry
        self._registry    = self._load_registry()
        self._model_config = self._get_active_model_config()

        # Resolve ONNX path relatif terhadap lokasi registry.json
        onnx_path = self.registry_path.parent / self._model_config['onnx_path']
        onnx_path = Path(self._model_config['onnx_path'])

        # Load ONNX session — dilakukan sekali di sini
        self._session     = self._load_onnx_session(onnx_path)
        self._input_name  = self._session.get_inputs()[0].name

        # Resolve preprocessing function berdasarkan backbone
        backbone = self._model_config['backbone']
        if backbone not in BACKBONE_PREPROCESS:
            raise ValueError(
                f"Backbone '{backbone}' tidak ada di BACKBONE_PREPROCESS. "
                f"Tambahkan entry baru di predictor.py untuk backbone ini."
            )
        self._preprocess_fn = BACKBONE_PREPROCESS[backbone]
        self._input_size    = tuple(self._model_config['input_size'])  # (H, W)
        self._class_names   = self._model_config['class_names']

        self.logger.info(
            f"CoffeeBeanPredictor ready | "
            f"backbone={backbone} | "
            f"input_size={self._input_size} | "
            f"classes={self._class_names}"
        )

    # ── Private Methods ────────────────────────────────────────────────────────

    def _load_registry(self) -> Dict:
        """
        Membaca dan memvalidasi file registry.json.

        Returns
        -------
        Dict
            Isi registry.json sebagai dictionary.

        Raises
        ------
        FileNotFoundError
            Jika registry.json tidak ditemukan.
        """
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"registry.json tidak ditemukan: {self.registry_path}\n"
                f"Pastikan file ada di path yang benar."
            )
        with open(self.registry_path, 'r') as f:
            registry = json.load(f)

        self.logger.info(f"Registry loaded dari: {self.registry_path}")
        return registry

    def _get_active_model_config(self) -> Dict:
        """
        Mengambil konfigurasi model yang aktif dari registry.

        Returns
        -------
        Dict
            Config model aktif (backbone, onnx_path, input_size, class_names, dll).

        Raises
        ------
        KeyError
            Jika active_model tidak ditemukan di registry.
        """
        active_key = self._registry.get('active_model')
        if not active_key:
            raise KeyError("'active_model' tidak ditemukan di registry.json.")

        models = self._registry.get('models', {})
        if active_key not in models:
            raise KeyError(
                f"Model '{active_key}' tidak ditemukan di registry.json. "
                f"Model yang tersedia: {list(models.keys())}"
            )

        config = models[active_key]
        self.logger.info(f"Active model: {active_key}")
        return config

    def _load_onnx_session(self, onnx_path: Path) -> ort.InferenceSession:
        """
        Load ONNX model ke InferenceSession.

        Parameters
        ----------
        onnx_path : Path
            Path ke file .onnx.

        Returns
        -------
        ort.InferenceSession
            ONNX Runtime inference session siap pakai.

        Raises
        ------
        FileNotFoundError
            Jika file .onnx tidak ditemukan.
        """
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"File ONNX tidak ditemukan: {onnx_path}\n"
                f"Pastikan file model sudah diletakkan di path yang benar."
            )

        # Gunakan CPU provider — tidak butuh GPU untuk inference
        session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        self.logger.info(f"ONNX model loaded: {onnx_path}")
        return session

    # ── Public Methods ─────────────────────────────────────────────────────────

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Memproses PIL Image menjadi input tensor yang siap untuk ONNX model.

        Langkah:
            1. Convert ke RGB (handle grayscale atau RGBA)
            2. Resize ke input_size sesuai backbone
            3. Convert ke numpy array float32
            4. Normalisasi sesuai backbone (via BACKBONE_PREPROCESS)
            5. Tambah batch dimension: (H, W, C) → (1, H, W, C)

        Parameters
        ----------
        image : PIL.Image.Image
            Gambar input dalam format PIL Image.

        Returns
        -------
        np.ndarray
            Tensor dengan shape (1, H, W, 3) dan dtype float32,
            siap dimasukkan ke ONNX session.
        """
        # Step 1: Pastikan RGB
        image = image.convert('RGB')

        # Step 2: Resize sesuai backbone
        h, w  = self._input_size
        image = image.resize((w, h), Image.Resampling.BILINEAR)

        # Step 3: Convert ke numpy float32
        x = np.array(image, dtype=np.float32)  # shape: (H, W, 3)

        # Step 4: Normalisasi sesuai backbone
        x = self._preprocess_fn(x)

        # Step 5: Tambah batch dimension
        x = np.expand_dims(x, axis=0)          # shape: (1, H, W, 3)

        return x

    def predict(self, image: Image.Image) -> Dict:
        """
        Menjalankan inference pada satu gambar dan mengembalikan hasil prediksi.

        Parameters
        ----------
        image : PIL.Image.Image
            Gambar input dalam format PIL Image.
            Bisa dalam format apapun (RGB, RGBA, grayscale) —
            akan otomatis diconvert ke RGB di dalam preprocess().

        Returns
        -------
        Dict
            Dictionary berisi:
            - "class"         : str  — nama kelas prediksi tertinggi
            - "confidence"    : float — probabilitas kelas prediksi (0.0–1.0)
            - "probabilities" : Dict[str, float] — probabilitas semua kelas
            - "class_id"      : int  — index kelas prediksi

        Examples
        --------
        >>> result = predictor.predict(image)
        >>> # result = {
        >>> #   "class": "premium",
        >>> #   "confidence": 0.94,
        >>> #   "class_id": 3,
        >>> #   "probabilities": {
        >>> #     "defect": 0.01,
        >>> #     "longberry": 0.02,
        >>> #     "peaberry": 0.03,
        >>> #     "premium": 0.94
        >>> #   }
        >>> # }
        """
        # Preprocess
        input_tensor = self.preprocess(image)

        # Inference
        outputs = self._session.run(None, {self._input_name: input_tensor})
        proba   = outputs[0][0]  # shape: (num_classes,)

        # Ambil kelas dengan probabilitas tertinggi
        class_id   = int(np.argmax(proba))
        confidence = float(proba[class_id])
        class_name = self._class_names[class_id]

        # Build output dict
        probabilities = {
            name: round(float(prob), 4)
            for name, prob in zip(self._class_names, proba)
        }

        return {
            "class":         class_name,
            "confidence":    round(confidence, 4),
            "class_id":      class_id,
            "probabilities": probabilities,
        }

    def get_model_info(self) -> Dict:
        """
        Mengembalikan informasi lengkap tentang model yang sedang aktif.

        Berguna untuk ditampilkan di halaman "About" Streamlit app.

        Returns
        -------
        Dict
            Semua field dari config model aktif di registry.json,
            ditambah key 'active_model' untuk referensi.
        """
        return {
            'active_model': self._registry.get('active_model'),
            **self._model_config
        }