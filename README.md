# DIC Event Detection Streamlit App

This repository contains a Streamlit workflow for DIC event detection, EBSD boundary cuts, and EBSD/DIC trace-alignment analysis.

## Run Locally

1. Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd DIC_Local_App
```

2. Create and activate the conda environment:

```bash
conda env create -f dic_qt/environment.yml
conda activate dicqt
```

<!-- 3. Make sure the required data folders are present in the project root:

```text
z_share_DIC_data_for_hv_mvu_pk/
z_share_EBSD_DIC_alignment_data/
z_share_EBSD_data_for_hv_mvu_pk/
hcp_slip_twin_miller_indices/
```

Large `.tif`, `.ang`, and output `.csv` files may be provided separately through Git LFS or a shared drive. Keep them in the same folder paths expected by the app. -->

3. Start the Streamlit app:

```bash
streamlit run tests/streamlit_seed_method_compare.py
```

5. Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Notes

- The first tab, `Crystal Setup`, controls which crystal structure and slip/twin trace modes are used during alignment scoring.
- HCP pyramidal and twin modes require the CSV files inside `hcp_slip_twin_miller_indices/`.
- If the app cannot find a default image or CSV, use the file upload/path controls inside the relevant tab.
