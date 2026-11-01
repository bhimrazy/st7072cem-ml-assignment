"""Build and freeze the full feature matrix (sets A, B and NOWX) for every panel row.

    uv run python -m src.freeze

Writes data/processed/feature-matrix.parquet. Fitted quantities use the training seasons 2017 to
2022; training rows carry the leave-one-season-out target encoding. The notebooks read this file.
"""

import time

from . import config as C
from . import features as Fe

MATRIX_PATH = C.PROCESSED / "feature-matrix.parquet"


def main():
    t0 = time.time()
    F = Fe.load_all()
    F.to_parquet(MATRIX_PATH, index=False)
    nan = F[Fe.FEATURES_B].isna().sum().sum()
    print(
        f"feature matrix {F.shape}; features A {len(Fe.FEATURES_A)}, B {len(Fe.FEATURES_B)}, NOWX {len(Fe.FEATURES_NOWX)}; "
        f"NaN cells in B {nan}; {time.time() - t0:.0f} s"
    )
    print("split rows:", F["split"].value_counts().to_dict())


if __name__ == "__main__":
    main()
