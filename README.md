# AI/ML Nowcasting of Thunderstorms & Lightning (India) — End-to-End Architecture (Phase 0 – Phase 3)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Domain](https://img.shields.io/badge/Domain-India%20Subcontinent%20(68--98%C2%B0E%2C%206--38%C2%B0N)-green.svg)]()
[![Data Sources](https://img.shields.io/badge/Public%20Datasets-IMD%20%7C%20MOSDAC%20%7C%20ILLN%20%7C%20NCMRWF%20%7C%20BharatBench-purple.svg)]()

## 📌 Executive Summary & Goals

This codebase provides the complete, working end-to-end implementation for **0–6 hour probabilistic nowcasting of convective initiation, thunderstorm evolution, and cloud-to-ground lightning over India**, built entirely on publicly available Indian atmospheric datasets. 

Instead of relying on generic weather APIs or proprietary foreign data, every layer in this architecture is specifically tailored to real, accessible Indian data streams:
1. **IMD Doppler Weather Radar (DWR) Network** (39+ stations via `pyiwr` & `Py-ART`)
2. **ISRO MOSDAC INSAT-3D/3DR/3DS** Imager & Sounder (TIR1, TIR2, WV, MIR, VIS channels via `Satpy`)
3. **Indian Lightning Location Network (ILLN)** / **GPM-LIS / WWLLN** lightning stroke density
4. **NCMRWF IMDAA / NGFS / NCUM Reanalysis & Forecasts** (12 km resolution CAPE, CIN, Bulk Shear, PWAT)
5. **MoES BharatBench** Kaggle Benchmark (ERA5-derived India-gridded NetCDF archive for pretraining)

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TB
    subgraph SRC["Public Indian Data Sources"]
        R[IMD DWR Network<br/>39+ radars via pyiwr/Py-ART]
        S[ISRO MOSDAC<br/>INSAT-3DR/3DS Imager - IR/VIS/WV]
        L[ILLN (IITM) + GPM-LIS/WWLLN<br/>lightning strikes/flashes]
        M[NCMRWF IMDAA/NGFS/NCUM<br/>CAPE, shear, moisture]
        BB[BharatBench Kaggle<br/>MoES ERA5-derived India benchmark]
    end

    subgraph ING["Ingestion & Quality Control Layer (Phase 0)"]
        DWR_IN[IMD DWR QC & Polar-to-Cartesian]
        INS_IN[MOSDAC INSAT BT Calibration & Reprojection]
        LGT_IN[Lightning Flash-Density Rasterizer]
        NWP_IN[NCMRWF Covariate Extraction]
    end

    subgraph PROC["Multi-Sensor Fusion & Feature Layer (Phase 1)"]
        REG[Spatial & Temporal Regridder 0.02°]
        EXT[Convective Feature Extractor<br/>VIL, Echo Top, BT Gradient, Motion Vectors]
        ZARR[(Zarr Cloud-Native Data Lake)]
    end

    subgraph ML["AI/ML Nowcasting Core (Phase 2 & Phase 3)"]
        P2_BASE[Phase 2: Baseline Nowcaster<br/>PySTEPS Optical Flow + tobac Cell Tracking]
        P3_PRE[Phase 3: Pretraining Module<br/>Masked Spatiotemporal Autoencoder on BharatBench]
        P3_TRANS[Phase 3: Earthformer Transformer Core<br/>Spatiotemporal Cuboid Attention Model]
        VERIF[Meteorological Evaluator<br/>CSI, POD, FAR, FSS Metrics]
    end

    R --> DWR_IN
    S --> INS_IN
    L --> LGT_IN
    M --> NWP_IN
    BB -. Offline Pretraining .-> P3_PRE

    DWR_IN --> REG
    INS_IN --> REG
    LGT_IN --> REG
    NWP_IN --> REG

    REG --> EXT
    EXT --> ZARR
    
    ZARR --> P2_BASE
    P3_PRE --> P3_TRANS
    ZARR --> P3_TRANS

    P2_BASE --> VERIF
    P3_TRANS --> VERIF
```

---

## 📊 Public Datasets & Technical Specifications

| Data Type | Source / Dataset | Open Access Details | Processing Stack |
| :--- | :--- | :--- | :--- |
| **Radar** | IMD Doppler Weather Radar (DWR) Network (39+ stations) | Raw volume scans via IMD Mausam / Zenodo (`10.6084/m9.figshare.22704910`) | `pyiwr` -> `CfRadial` -> `Py-ART` gridding onto Cartesian grid |
| **Satellite** | ISRO INSAT-3D / 3DR / 3DS Imager & Sounder | MOSDAC (`mosdac.gov.in`) Open Data API (TIR1, TIR2, WV, MIR, VIS) | `Satpy` reprojection + Brightness Temp (BT) calibration |
| **Lightning** | ILLN (IITM Pune) + NASA GPM-LIS / WWLLN | IITM research access / NASA GES DISC public fallback | Kernel density estimation to 0.02° flash-density rasters |
| **NWP / Reanalysis** | NCMRWF IMDAA (12km) + NGFS (25km) | NCMRWF Data Portal (`rds.ncmrwf.gov.in`) | `cfgrib` / `xarray` extraction of CAPE, CIN, PWAT, 0-6km Shear |
| **ML Benchmark** | MoES BharatBench | Kaggle (`kaggle.com/datasets/maslab/bharatbench`) | NetCDF benchmark pretraining loader |

---

## 🗺️ Build Roadmap: Implementation Status (Phases 0–3)

### ✅ Phase 0 — Data Access & Ingestion Setup
- [x] Standard India spatial grid configuration (`68.0°E - 98.0°E`, `6.0°N - 38.0°N`, resolution `0.02°` ~2 km).
- [x] Ingestion connector for IMD DWR network (`src/ingestion/imd_dwr_ingest.py`) with polar-to-Cartesian conversion and clutter filtering.
- [x] Ingestion connector for ISRO MOSDAC INSAT-3DR/3DS (`src/ingestion/mosdac_insat_ingest.py`) with Brightness Temperature (BT) calibration and Overshooting Top detection.
- [x] Lightning point strike ingestion & 10-minute flash-density rasterizer (`src/ingestion/lightning_ingest.py`).
- [x] NCMRWF IMDAA 12 km reanalysis extractor for CAPE, CIN, PWAT, and 0-6 km Bulk Shear (`src/ingestion/ncmrwf_ingest.py`).
- [x] Kaggle MoES BharatBench benchmark archive loader (`src/ingestion/bharatbench_ingest.py`).

### ✅ Phase 1 — Data Foundation & Multi-Sensor Fusion
- [x] Spatial & temporal regridder (`src/fusion/regridder.py`) co-registering heterogenous sources into a 12-channel tensor.
- [x] Convective feature engineering (`src/fusion/feature_extractor.py`):
  - Vertically Integrated Liquid (VIL)
  - Echo Top Height (18 dBZ proxy)
  - Spatial Brightness Temperature Gradients
  - Lucas-Kanade Optical Flow Storm Motion Vectors ($u, v$)
  - Convective Potential Index (CPI)
- [x] Cloud-native Zarr data lake builder (`src/fusion/zarr_lake_builder.py`) with Dask chunking optimized for PyTorch sequence loading (`(time, lat, lon)`).

### ✅ Phase 2 — Baseline Operational Nowcast
- [x] PySTEPS Lucas-Kanade optical flow reflectivity extrapolation (`src/models/baseline_pysteps.py`) up to 6 hours (36 lead steps of 10 min).
- [x] `tobac` storm-cell tracking module identifying severe convective cell centroids, intensities, and trajectories.

### ✅ Phase 3 — Pretraining & Deep Learning Core
- [x] Earthformer / MetNet-3 style PyTorch Transformer model (`src/models/earthformer_spatiotemporal.py`) featuring Spatiotemporal Cuboid Attention.
- [x] Self-supervised Masked Autoencoder (MAE) pretraining module (`src/models/pretrainer.py`) on BharatBench ERA5 India benchmark.
- [x] End-to-end multi-loss fine-tuning trainer (`src/models/trainer.py`) combining MSE for reflectivity with Focal Loss for lightning rare events.
- [x] Standardized meteorological evaluation suite computing **Critical Success Index (CSI)**, **Probability of Detection (POD)**, **False Alarm Ratio (FAR)**, and **Fractions Skill Score (FSS)**.

---

## 📂 Project Directory Structure

```
sihsecond/
├── config/
│   ├── settings.yaml              # Domain bounding box, spatial grid, model hyperparameters
│   └── grid_config.py             # India domain grid loader (68-98°E, 6-38°N at 0.02°)
├── src/
│   ├── ingestion/                 # PHASE 0: Public Dataset Connectors
│   │   ├── imd_dwr_ingest.py      # IMD DWR network reader & quality control
│   │   ├── mosdac_insat_ingest.py # ISRO MOSDAC INSAT-3DR/3DS reader & BT calibration
│   │   ├── lightning_ingest.py    # ILLN / GPM-LIS / WWLLN flash-density rasterizer
│   │   ├── ncmrwf_ingest.py       # NCMRWF IMDAA 12km NWP reanalysis reader
│   │   └── bharatbench_ingest.py  # Kaggle MoES BharatBench benchmark loader
│   ├── fusion/                    # PHASE 1: Fusion & Feature Store
│   │   ├── regridder.py           # Multi-sensor spatial/temporal co-registration
│   │   ├── feature_extractor.py   # VIL, Echo Top, BT Gradient, Optical Flow vectors
│   │   └── zarr_lake_builder.py   # Zarr data lake persistence with Dask chunking
│   ├── models/                    # PHASE 2 & 3: AI/ML Modeling Core
│   │   ├── baseline_pysteps.py    # Phase 2 PySTEPS optical flow & tobac cell tracker
│   │   ├── earthformer_spatiotemporal.py # Phase 3 Deep Transformer with Cuboid Attention
│   │   ├── pretrainer.py          # Self-supervised MAE pretraining on BharatBench
│   │   └── trainer.py             # Fine-tuning & CSI / POD / FAR verification metrics
│   └── pipeline.py                # Master Orchestrator executing Phase 0 to Phase 3
├── requirements.txt               # Dependencies (PyTorch, Xarray, Py-ART, PySTEPS, Zarr)
└── README.md                      # Architecture documentation & execution guide
```

---

## ⚡ Execution & Verification Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Individual Modules (Self-Tests)

- **Test IMD Radar Ingestion:**
  ```bash
  python -m src.ingestion.imd_dwr_ingest
  ```
- **Test Multi-Sensor Fusion & Regridding:**
  ```bash
  python -m src.fusion.regridder
  ```
- **Test Phase 2 PySTEPS Baseline Nowcast:**
  ```bash
  python -m src.models.baseline_pysteps
  ```
- **Test Phase 3 Deep Spatiotemporal Transformer Architecture:**
  ```bash
  python -m src.models.earthformer_spatiotemporal
  ```

### 3. Run End-to-End Pipeline (Phases 0 through 3)
```bash
python -m src.pipeline
```

---

## ⚡ Key India-Specific Challenges & Mitigations

1. **ILLN Ground Network Access:**
   - *Challenge:* ILLN network data is managed by IITM Pune and requires research permissions.
   - *Mitigation:* The pipeline seamlessly uses public **NASA GPM-LIS** and **WWLLN** as operational substitutes during prototyping.
2. **Research Portal Latencies (MOSDAC & NCMRWF):**
   - *Challenge:* MOSDAC and NCMRWF RDS are research portals without commercial 24/7 SLAs.
   - *Mitigation:* The pipeline incorporates graceful degradation (falling back to the last-known NWP field and spatially interpolating satellite gaps).
3. **Radar Coverage Blindspots:**
   - *Challenge:* Certain regions (mountainous terrain or un-covered inland gaps) lack dense DWR radar sweeps.
   - *Mitigation:* Multi-sensor fusion weights **INSAT satellite TIR channels and NWP convective indices (CAPE/PWAT)** heavily over radar-blind zones.
4. **Extreme Rare-Event Imbalance (Lightning Strikes):**
   - *Challenge:* Cloud-to-ground lightning flashes occur in < 1% of grid cells.
   - *Mitigation:* The fine-tuning loss utilizes **Focal Loss ($\gamma=2.0, \alpha=0.25$)** to prevent under-prediction.
