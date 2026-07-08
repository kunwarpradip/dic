# DIC Qt Python Tool

Fresh Python implementation of the Java Swing DIC event detection tool.

This project is based on the decompiled Java feature inventory in:

```text
docs/DIC_JAVA_TOOL_FEATURE_INVENTORY.md
```

The target is a faithful desktop workflow first:

- open DIC image
- load/save the matching SQLite DB under `app_data`
- zoom and pan image
- click a high-intensity seed pixel to create an event
- automatically generate DIC-only seed pixels and grow events from them
- toggle event visibility
- merge selected events
- delete selected events
- cut a selected/visible event by dragging a line
- show a cursor-centered preview panel

## Install

PySide6 is not installed in the current `dicevent` environment yet.

The current `dicevent` environment uses Python 3.14, which is too new for
PySide6 wheels. Use the dedicated Qt environment:

```bash
cd /home/pkunwar/DIC_event_detection
source /home/pkunwar/miniconda3/etc/profile.d/conda.sh
conda env create -f dic_qt/environment.yml
conda activate dicqt
```

## Run

The UI is a real desktop Qt application, so it needs a display server. Run it
from a Linux desktop session, an SSH session with X11 forwarding, or a VNC /
remote-desktop session:

```bash
python -m dic_qt.app
```

If you are connected over SSH, use X11 forwarding:

```bash
ssh -Y <host>
conda activate dicqt
cd /home/pkunwar/DIC_event_detection
python -m dic_qt.app
```

If the session has no display, this command only validates startup code and
does not provide an interactive UI:

```bash
QT_QPA_PLATFORM=offscreen python -m dic_qt.app
```

If `xcb` is selected and Qt reports `xcb-cursor0` or `libxcb-cursor0` is
missing, install that system package on the host or use a VNC/desktop session
that already provides a working Qt platform plugin.

## Current Status

Implemented in this scaffold:

- Java-compatible data model and SQLite repository.
- Java-faithful seed-growing algorithm.
- Separate raw, display, and detection image buffers for scalar DIC maps.
- DIC-only auto-seed generator using robust normalization, CLAHE, ridge
  filtering, skeletonization, and duplicate rejection.
- Java-style merge/delete/cut core operations.
- PySide6 main window, image canvas, event list, settings panel, preview panel.

This is the first desktop baseline. It is intentionally not a Streamlit app.
