"""
dataset.py — PyTorch Dataset untuk Coffee Bean Classification.

Digunakan di pipeline preprocessing untuk sanity check
bahwa split CSV bisa di-load dan gambar bisa dibaca dengan benar.
"""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class CoffeeBeanDataset(Dataset):
    """
    PyTorch Dataset yang membaca gambar dari CSV split.

    CSV harus memiliki kolom:
    - filepath  : path absolut ke file gambar
    - class_id  : integer label kelas (0-indexed)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dari split CSV (train/val/test).
    transform : callable, optional
        Transformasi torchvision yang diaplikasikan ke setiap gambar.
        Jika None, gambar dikembalikan sebagai PIL Image.
    """

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

        required = {"filepath", "class_id"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Split DataFrame missing required columns: {missing}. "
                f"Columns found: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        """
        Mengambil satu sample dari dataset.

        Parameters
        ----------
        idx : int
            Indeks sample.

        Returns
        -------
        Tuple[Tensor, int]
            (image_tensor, class_id) jika transform diberikan,
            (PIL.Image, class_id) jika transform adalah None.
        """
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        y   = int(row["class_id"])

        if self.transform:
            img = self.transform(img)

        return img, y