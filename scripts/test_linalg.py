"""Sanity check for BLAS stability on Windows."""

from __future__ import annotations

import numpy as np


def main() -> None:
    print("rand_start", flush=True)
    a = np.random.rand(200, 200)
    print("matmul_start", flush=True)
    b = a @ a
    print("svd_start", flush=True)
    u, s, vt = np.linalg.svd(a)
    print("b", b.shape, flush=True)
    print("u", u.shape, flush=True)
    print("s", s.shape, flush=True)
    print("vt", vt.shape, flush=True)
    print("LINALG_OK", flush=True)


if __name__ == "__main__":
    main()
