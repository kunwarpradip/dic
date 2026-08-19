from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    miller_dir = root / "hcp_slip_twin_miller_indices"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        "DIC_Event_App",
        "--windowed",
        "--onedir",
        "--clean",
        "--collect-submodules",
        "skimage",
        "--collect-submodules",
        "scipy",
        "--collect-data",
        "skimage",
        "--collect-data",
        "scipy",
        "--add-data",
        f"{miller_dir}{os.pathsep}hcp_slip_twin_miller_indices",
        "dic_qt_launcher.py",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
