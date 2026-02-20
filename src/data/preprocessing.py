"""
preprocessing.py — Utilities untuk deduplication pipeline.

Berisi:
- md5_file()   : menghitung MD5 hash file (exact duplicate detection)
- phash_64()   : menghitung 64-bit perceptual hash (near-duplicate detection)
- hamming64()  : menghitung Hamming distance antara dua 64-bit integer
- UnionFind    : data structure untuk near-duplicate clustering
"""

import hashlib
from typing import Optional

import numpy as np
from PIL import Image


# ── MD5 ───────────────────────────────────────────────────────────────────────

def md5_file(path: str, chunk_size: int = 1 << 20) -> str:
    """
    Menghitung MD5 hash dari sebuah file.

    Digunakan untuk mendeteksi exact duplicate — dua file dengan MD5
    yang sama dijamin memiliki konten yang identik byte-per-byte.

    Parameters
    ----------
    path : str
        Path ke file yang akan di-hash.
    chunk_size : int
        Ukuran chunk saat membaca file (default 1MB).
        Menggunakan chunked read agar aman untuk file besar.

    Returns
    -------
    str
        MD5 hash dalam format hex string (32 karakter).
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


# ── pHASH ─────────────────────────────────────────────────────────────────────

def _get_dct_backend() -> Optional[str]:
    """
    Mendeteksi backend DCT yang tersedia di environment.
    Prioritas: scipy → cv2 → None (raise error).
    """
    try:
        from scipy.fftpack import dct  # noqa: F401
        return "scipy"
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401
        return "cv2"
    except ImportError:
        pass
    return None


def _dct2(a: np.ndarray, backend: str) -> np.ndarray:
    """
    Menghitung 2D DCT type-II dengan backend yang diberikan.

    Parameters
    ----------
    a : np.ndarray
        2D array float32.
    backend : str
        "scipy" atau "cv2".

    Returns
    -------
    np.ndarray
        Hasil DCT 2D.
    """
    if backend == "scipy":
        from scipy.fftpack import dct
        return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    else:
        import cv2
        return cv2.dct(a.astype(np.float32))


def phash_64(path: str, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """
    Menghitung 64-bit perceptual hash (pHash) dari sebuah gambar.

    pHash menggunakan DCT untuk mengekstrak fitur frekuensi rendah gambar,
    sehingga dua gambar yang sangat mirip (near-duplicate) akan menghasilkan
    hash dengan Hamming distance yang kecil.

    Parameters
    ----------
    path : str
        Path ke file gambar.
    hash_size : int
        Ukuran hash grid (default 8 → 64-bit hash).
    highfreq_factor : int
        Faktor resize sebelum DCT (default 4 → resize ke 32x32).

    Returns
    -------
    int
        64-bit perceptual hash sebagai Python int.

    Raises
    ------
    ImportError
        Jika scipy maupun cv2 tidak tersedia di environment.
    """
    backend = _get_dct_backend()
    if backend is None:
        raise ImportError(
            "Butuh scipy atau opencv-python untuk DCT. "
            "Install salah satunya: pip install scipy  atau  pip install opencv-python"
        )

    img_size = hash_size * highfreq_factor  # default: 32
    img = (
        Image.open(path)
        .convert("L")
        .resize((img_size, img_size), Image.Resampling.BILINEAR)
    )
    pixels = np.asarray(img, dtype=np.float32)

    dct = _dct2(pixels, backend)
    dct_low  = dct[:hash_size, :hash_size]
    dct_flat = dct_low.flatten()

    # Threshold: median dari semua bit kecuali DC term
    median = np.median(dct_flat[1:])
    bits   = (dct_flat > median).astype(np.uint8)

    # Pack bits menjadi 64-bit integer
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


# ── HAMMING ───────────────────────────────────────────────────────────────────

def hamming64(a: int, b: int) -> int:
    """
    Menghitung Hamming distance antara dua 64-bit integer.

    Digunakan untuk mengukur kemiripan dua pHash — semakin kecil
    Hamming distance, semakin mirip kedua gambar.

    Parameters
    ----------
    a : int
        Hash pertama (64-bit integer).
    b : int
        Hash kedua (64-bit integer).

    Returns
    -------
    int
        Jumlah bit yang berbeda (0–64).
    """
    return (a ^ b).bit_count()


# ── UNION-FIND ────────────────────────────────────────────────────────────────

class UnionFind:
    """
    Data structure Union-Find (Disjoint Set Union) dengan path compression
    dan union by rank.

    Digunakan untuk mengelompokkan near-duplicate images menjadi cluster
    berdasarkan Hamming distance pHash mereka.

    Parameters
    ----------
    n : int
        Jumlah elemen (indeks 0 hingga n-1).
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        """
        Mencari root dari elemen x dengan path compression.

        Parameters
        ----------
        x : int
            Indeks elemen.

        Returns
        -------
        int
            Root dari elemen x.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """
        Menggabungkan dua set yang berisi elemen a dan b.

        Parameters
        ----------
        a : int
            Indeks elemen pertama.
        b : int
            Indeks elemen kedua.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1