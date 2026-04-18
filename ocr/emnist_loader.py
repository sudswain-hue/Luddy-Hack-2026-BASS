# ocr/emnist_loader.py
"""
Manual EMNIST loader — reads the NIST matlab file directly,
no tensorflow-datasets needed.
"""
import numpy as np
from pathlib import Path
from scipy.io import loadmat


EMNIST_PATH = Path(__file__).parent / "data" / "matlab" / "emnist-balanced.mat"


def load_emnist_balanced():
    """
    EMNIST Balanced — 47 classes, 112,800 train + 18,800 test.
    Returns (x_train, y_train), (x_test, y_test).
    """
    if not EMNIST_PATH.exists():
        raise FileNotFoundError(
            f"EMNIST not found at {EMNIST_PATH}. "
            f"Run: curl -L -o data/emnist.zip https://biometrics.nist.gov/cs_links/EMNIST/matlab.zip"
        )

    print(f"Loading EMNIST from {EMNIST_PATH}...")
    mat = loadmat(str(EMNIST_PATH))
    data = mat['dataset']

    # matlab struct is nested — extract train and test
    x_train = data['train'][0, 0]['images'][0, 0]
    y_train = data['train'][0, 0]['labels'][0, 0]
    x_test  = data['test'][0, 0]['images'][0, 0]
    y_test  = data['test'][0, 0]['labels'][0, 0]

    # reshape: each row is a flattened 28x28 image
    # EMNIST stores them transposed — we rotate to correct orientation
    x_train = x_train.reshape(-1, 28, 28, order='F').astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 28, 28, order='F').astype(np.float32) / 255.0

    # add channel dimension (N, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test  = x_test[..., np.newaxis]

    # squeeze label arrays to 1D
    y_train = y_train.flatten().astype(np.int64)
    y_test  = y_test.flatten().astype(np.int64)

    print(f"Train size: {len(x_train)}")
    print(f"Test size:  {len(x_test)}")
    print(f"Classes:    {len(np.unique(y_train))}")

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (xtr, ytr), (xte, yte) = load_emnist_balanced()
    print(f"\nTrain shape: {xtr.shape}, labels shape: {ytr.shape}")
    print(f"Label range: {ytr.min()} to {ytr.max()}")
    print(f"Pixel range: {xtr.min():.3f} to {xtr.max():.3f}")