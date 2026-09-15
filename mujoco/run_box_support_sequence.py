"""Run the box-support sequence with shared AVX2 inference kernels."""

import os
from pathlib import Path
import sys


def main(entrypoint_name="check_box_support_sequence.py"):
    environment = os.environ.copy()
    environment.update(
        ATEN_CPU_CAPABILITY="avx2",
        MKL_CBWR="AVX2",
        DNNL_MAX_CPU_ISA="AVX2",
    )
    entrypoint = str(Path(__file__).resolve().with_name(entrypoint_name))
    os.execve(sys.executable, [sys.executable, entrypoint, *sys.argv[1:]], environment)


if __name__ == "__main__":
    main()