"""
dataloader.py — DataLoader builder untuk Coffee Bean Classification.

Menyediakan:
- IMAGENET_MEAN, IMAGENET_STD : konstanta normalisasi ImageNet
- get_default_transforms()    : transforms standar tanpa augmentasi
- build_dataloaders()         : builder DataLoader dari split DataFrame
"""

from typing import Optional, Tuple

import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.torch.dataset import CoffeeBeanDataset


# ── Konstanta ImageNet ─────────────────────────────────────────────────────────

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_default_transforms() -> Tuple[T.Compose, T.Compose]:
    """
    Mengembalikan transforms standar tanpa augmentasi.

    Digunakan untuk sanity check preprocessing pipeline —
    bukan untuk training (training menggunakan TF/Keras pipeline).

    Returns
    -------
    Tuple[T.Compose, T.Compose]
        (train_transform, val_transform)
        Keduanya identik karena tidak ada augmentasi.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform, transform


# ── DataLoader Builder ────────────────────────────────────────────────────────

def build_dataloaders(
    train_df:    pd.DataFrame,
    val_df:      pd.DataFrame,
    test_df:     Optional[pd.DataFrame] = None,
    batch_size:  int  = 32,
    num_workers: int  = 2,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Membangun DataLoader untuk train, val, dan opsional test split.

    Transforms yang digunakan adalah default ImageNet normalization
    tanpa augmentasi (cocok untuk sanity check, bukan training).

    Parameters
    ----------
    train_df : pd.DataFrame
        DataFrame split training.
    val_df : pd.DataFrame
        DataFrame split validasi.
    test_df : pd.DataFrame, optional
        DataFrame split test. Jika None, dl_test akan bernilai None.
    batch_size : int
        Jumlah sampel per batch (default 32).
    num_workers : int
        Jumlah worker untuk parallel data loading (default 2).
    pin_memory : bool
        Jika True, data di-pin ke memory untuk transfer GPU lebih cepat.

    Returns
    -------
    Tuple[DataLoader, DataLoader, Optional[DataLoader]]
        (dl_train, dl_val, dl_test)
    """
    train_tf, val_tf = get_default_transforms()

    ds_train = CoffeeBeanDataset(train_df, transform=train_tf)
    ds_val   = CoffeeBeanDataset(val_df,   transform=val_tf)
    ds_test  = CoffeeBeanDataset(test_df,  transform=val_tf) if test_df is not None else None

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    ) if ds_test is not None else None

    return dl_train, dl_val, dl_test