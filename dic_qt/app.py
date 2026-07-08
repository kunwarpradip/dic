from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def _has_gui_display() -> bool:
    platform = os.environ.get("QT_QPA_PLATFORM", "").split(":", 1)[0]
    if platform in {"offscreen", "minimal", "minimalegl", "vnc"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> int:
    if sys.platform.startswith("linux") and not _has_gui_display():
        print(
            "No GUI display is available for the Qt app.\n\n"
            "Run it from a desktop session, an SSH session with X11 forwarding, "
            "or a VNC/remote-desktop session.\n\n"
            "Examples:\n"
            "  ssh -Y <host>\n"
            "  conda activate dicqt\n"
            "  cd /home/pkunwar/DIC_event_detection\n"
            "  python -m dic_qt.app\n\n"
            "For a non-interactive smoke test only:\n"
            "  QT_QPA_PLATFORM=offscreen python -m dic_qt.app\n",
            file=sys.stderr,
        )
        return 2

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
