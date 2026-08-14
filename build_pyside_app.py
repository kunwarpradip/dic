from __future__ import annotations

import subprocess
import sys


def main() -> int:
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
        "dic_qt_launcher.py",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
