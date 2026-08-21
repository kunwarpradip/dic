# DIC Event Detection App

This repository contains a Streamlit workflow for DIC event detection, EBSD boundary cuts, and EBSD/DIC trace-alignment analysis.

## Run Locally

1. Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd dic
```

2. Create and activate the conda environment:

```bash
conda env create -f dic_qt/environment.yml
conda activate dicqt
```

3. At the root folder:

```bash
python -m dic_qt.app
```

This runs the software. 

4. To create an executable build (Use mac for macOS build and Windows machine for WindowsOS build - both follow the same process of building)
```bash
pip install pyinstaller
python build_pyside_app.py
```

This creates a dist folder which contains the executable and all the dependencies required for the run. 

## Streamlit as Debugging and Playground App

3.1. If you want to test it in a debugging way: Start the Streamlit app from the repository root:

```bash
streamlit run tests/streamlit_seed_method_compare.py
```

3.2. Open the local URL shown in the terminal, usually:

It gives you localhost address and port to launch the streamlit app in the browser. For example:
```text
http://localhost:8501
```



