import streamlit as st
import leafmap.foliumap as leafmap
import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.data_pipeline import GeoDataFetcher
from modules.ai_engine import SolarUNet
from modules.spatial_eng import SpatialProcessor

# --- 1. Page Configuration ---
st.set_page_config(page_title="Solar-GeoTwin | AI Site Selection", layout="wide")
st.title("☀️ Solar-GeoTwin: Global Site Selection")
st.markdown("""
    **Industrial Engineering Dashboard** | Transitioning from traditional site surveys to AI-driven spatial twins for renewable energy.
""")

# --- 2. Setup & Model Loading ---
PROJECT_ID = 'solar-geoai-dferreri45'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Loads the U-Net model and caches it to prevent reloading on every interaction."""
    model = SolarUNet(in_channels=4, out_channels=1).to(device)
    try:
        model.load_state_dict(torch.load("models/solar_unet_v1.pth", map_location=device))
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Weights not found. Using uninitialized model.")
    model.eval()
    return model

# Initialize our core classes
model = load_model()
fetcher = GeoDataFetcher(PROJECT_ID)
processor = SpatialProcessor()

# --- 3. Sidebar Controls ---
st.sidebar.header("📍 Location Parameters")
lon = st.sidebar.number_input("Longitude (East)", value=23.3, format="%.4f")
lat = st.sidebar.number_input("Latitude (North)", value=-28.3, format="%.4f")

st.sidebar.header("🧠 AI Configuration")
threshold = st.sidebar.slider("AI Confidence Cutoff", 0.0, 1.0, 0.70) 

# --- 4. Main Analysis Logic ---
if st.sidebar.button("Run Global Analysis"):
    with st.spinner("Fetching Satellite Data & Running AI..."):
        
        # A. Data Ingestion: Pull Sentinel-2 satellite and SRTM terrain data
        raw = fetcher.get_sentinel_composite(lon, lat, 2000, "2024-01-01", "2024-12-31")
        processed = processor.calculate_metrics(raw)
        
        # B. Image Preparation: Export the 256x256 patch for the U-Net
        img_array = fetcher.export_single_patch(processed, lon, lat) 
        
        # C. Inference: Generate the probability map
        # Convert (H,W,C) -> (C,H,W) and move to GPU/MPS
        input_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # D. Regression: Energy Yield Calculation
        binary_mask = prob_map > threshold
        
        # Dynamic Irradiance logic: Northern Cape (~2280) vs Global Average (~1800)
        # In a production app, we would pull this live from NASA POWER API
        site_ghi = 2282 if (20 < lon < 25 and -30 < lat < -25) else 1800
        
        # Calculate yield using our new SpatialProcessor function
        # Formula: Area * Efficiency * Irradiance * Performance Ratio
        annual_kwh, total_m2 = processor.estimate_yield(
            binary_mask, 
            avg_irradiance=site_ghi, 
            resolution=10, # Sentinel-2 pixels are 10m
            efficiency=0.20 # 20% panel efficiency
        )
        
        # E. UI Visualization
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentinel-2 RGB")
            st.image(np.clip(img_array[:,:,:3] * 3, 0, 1), use_container_width=True)
            
        with col2:
            st.subheader("AI Suitability Overlay")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.clip(img_array[:,:,:3] * 3, 0, 1)) # Background
            ax.imshow(binary_mask, alpha=0.5, cmap='Wistia') # Suitability Mask
            ax.axis('off')
            st.pyplot(fig)
            
        # F. Sidebar Metrics (The "Business Case")
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Estimated Yield")
        st.sidebar.metric("Suitable Area", f"{total_m2/10000:.2f} Hectares")
        st.sidebar.metric("Annual Generation", f"{annual_kwh/1e6:.2f} GWh/yr")
        st.sidebar.caption(f"Based on {site_ghi} kWh/m²/yr solar resource.")
            
        st.success(f"Site analysis complete for {lat}, {lon}")

# --- 5. Global Context Map ---
st.divider()
st.subheader("Interactive Context Map")
m = leafmap.Map(center=[lat, lon], zoom=12)
m.add_basemap("SATELLITE")
m.to_streamlit(height=500)