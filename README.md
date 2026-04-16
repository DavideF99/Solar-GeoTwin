# ☀️ Solar-GeoTwin: AI-Powered Site Selection & Yield Forecasting

**An Industrial Engineering approach to renewable energy infrastructure using Computer Vision and Geospatial Data Science.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://solar-geotwin-fvkfsar3p5a5arko38hjka.streamlit.app/)

## 📌 Project Overview

Solar-GeoTwin is a "Digital Twin" predictive engine designed to automate the identification of optimal solar farm locations. By fusing multi-spectral satellite imagery with topographic physics, the system identifies high-suitability land and provides a regressive estimation of annual energy potential (GWh).

This project demonstrates a full MLOps pipeline: from raw planetary data ingestion via **Google Earth Engine** to a deployed **Deep Learning (U-Net)** segmentation model.

---

## 🚀 Key Features

- **Multi-spectral Data Fusion:** Ingests Sentinel-2 (Top of Atmosphere) and SRTM (Digital Elevation Model) data to analyze surface reflectance and terrain slope.
- **Semantic Segmentation:** Utilizes a custom **U-Net** architecture (PyTorch) trained on diverse global biomes to generate pixel-level suitability probability maps.
- **Physics-Informed Regression:** Calculates annual energy yield ($kWh/year$) by correlating identified suitable areas with historical Global Horizontal Irradiance (GHI).
- **Dynamic Decision Support:** A real-time dashboard featuring adjustable AI confidence thresholds for risk-adjusted site prospecting.

---

## 🛠️ Technical Stack

- **Core AI:** PyTorch (U-Net), Torchvision.
- **Geospatial Engine:** Google Earth Engine (EE) API, `leafmap`, `geemap`.
- **Backend & Data:** `numpy`, `matplotlib`, `pandas`.
- **Deployment:** Streamlit Community Cloud, GitHub Actions, GCP Service Accounts.

---

## 🧪 Methodology & Implementation

### 1. The Spatial Engineering Pipeline

Instead of manual labeling, "Ground Truth" masks were generated using a custom **Spatial Processor**. This engine filters pixels based on:

1.  **Topographic Slope:** $\le 5\%$ for utility-scale viability.
2.  **Land Cover:** NDVI (Normalized Difference Vegetation Index) masking to avoid protected biomass and water bodies.
3.  **Solar Resource:** Minimum irradiance thresholds based on NASA POWER meteorological data.

### 2. AI Model Architecture

The engine employs a **U-Net** encoder-decoder architecture with 4 input channels (Red, Green, Blue, Near-Infrared).

- **Activation:** Sigmoid output for granular probability mapping.
- **Loss Function:** Combined BCE (Binary Cross Entropy) + Dice Loss to handle class imbalance in rural vs. urban terrain.
- **Inference:** Threshold-calibrated at $0.70$ for high-precision site identification.

### 3. Power Regression Formula

The estimated yield is calculated using the following performance model:
$$E = A \times \eta \times H \times PR$$
Where:

- $A$ = Total identified suitable area ($m^2$)
- $\eta$ = Module efficiency (Fixed at $20\%$)
- $H$ = Annual average solar radiation ($kWh/m^2/y$)
- $PR$ = Performance Ratio (Default $0.75$ for system losses)

---

## 📂 Repository Structure

```text
├── .streamlit/          # Cloud deployment configuration & secrets
├── models/              # Trained PyTorch weights (.pth)
├── modules/             # Core logic
│   ├── ai_engine.py     # U-Net Architecture
│   ├── data_pipeline.py # GEE Data Ingestion & Service Account Auth
│   └── spatial_eng.py   # Physics-based filtering & Regression
├── app.py               # Streamlit Dashboard UI
├── requirements.txt     # Production dependencies
└── README.md
```
