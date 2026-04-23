import streamlit as st
import leafmap.foliumap as leafmap
import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from modules.data_pipeline import GeoDataFetcher
from modules.ai_engine import SolarUNet
from modules.spatial_eng import SpatialProcessor

# --- 1. Page Configuration & UI Theme ---
st.set_page_config(page_title="Solar-GeoTwin | AI Site Selection", layout="wide")

# Professional CSS for sidebar contrast
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #12141d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Setup & Model Loading ---
PROJECT_ID = 'solar-geoai-dferreri45'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = SolarUNet(in_channels=4, out_channels=1).to(device)
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    weights_path = current_dir / "models" / "solar_unet_v1.pth"
    
    try:
        model.load_state_dict(torch.load(str(weights_path), map_location=device))
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ Weights not found at: {weights_path}")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        
    model.eval()
    return model

model = load_model()
fetcher = GeoDataFetcher(PROJECT_ID)
processor = SpatialProcessor()

# --- 3. Sidebar: Configuration & Methodology ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/solar-panel.png", width=80)
    st.title("Solar-GeoTwin")
    st.caption("AI-Driven Spatial Intelligence for Renewables")
    
    with st.expander("📍 Location Parameters", expanded=True):
        lon = st.number_input("Longitude (East)", value=23.3, format="%.4f")
        lat = st.number_input("Latitude (North)", value=-28.3, format="%.4f")

    with st.expander("⚙️ Engineering Settings", expanded=True):
        st.write("🧠 **Regional Model Calibration**")
        
        # Regional Calibration Logic
        is_trained_area = (-35.0 <= lat <= -22.0 and 15.0 <= lon <= 33.0)

        if is_trained_area:
            st.success("📍 **Trained Region Detected**")
            st.caption("The AI is highly calibrated for this geography. Optimal range: **0.70 — 0.85**.")
        else:
            st.warning("🌐 **Out-of-Distribution Area**")
            st.caption("AI not trained on this terrain. Try a lower confidence range: **0.40 — 0.55**.")

        threshold = st.slider(
            "AI Confidence Cutoff", 
            0.0, 1.0, 0.70, 
            help="Adjust this to match the regional calibration guidance above."
        )
        
        st.divider()
        efficiency = st.number_input("Panel Efficiency", value=0.20)
        perf_ratio = st.number_input("Performance Ratio", value=0.75)

    run_btn = st.button("🚀 Run Global Analysis", use_container_width=True)
    
    st.divider()
    st.info("""
    **Methodology:**
    1. **Data:** Sentinel-2 & NASA POWER.
    2. **AI:** Custom U-Net Segmentation.
    3. **Logic:** Slope < 5° & Sparse Vegetation.
    """)

# --- 4. Main Dashboard Header ---
st.title("☀️ Global Site Selection Engine")
st.markdown("Transitioning from traditional site surveys to **AI-driven spatial twins**.")

# --- 5. Analysis Logic ---
if run_btn:
    with st.status("Initializing Spatial Analysis...", expanded=True) as status:
        st.write("📡 Querying Google Earth Engine...")
        raw = fetcher.get_sentinel_composite(lon, lat, 2000, "2024-01-01", "2024-12-31")
        processed = processor.calculate_metrics(raw)
        
        st.write("🧠 Executing U-Net Inference...")
        img_array = fetcher.export_single_patch(processed, lon, lat) 
        input_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        st.write("🌦️ Fetching NASA Data...")
        daily_ghi = fetcher.get_nasa_irradiance(lat, lon)
        annual_ghi = daily_ghi * 365.25
        
        binary_mask = prob_map > threshold
        annual_kwh, total_m2 = processor.estimate_yield(
            binary_mask, 
            avg_irradiance=annual_ghi, 
            resolution=10, 
            efficiency=efficiency 
        )
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # --- 6. Results Visualization ---
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Solar Resource", f"{annual_ghi:.1f} kWh/m²/y")
    kpi2.metric("Identified Area", f"{total_m2/10000:.2f} Hectares")
    kpi3.metric("Est. Generation", f"{(annual_kwh * perf_ratio) / 1e6:.2f} GWh/yr")

    tab_vis, tab_heat, tab_data = st.tabs(["🗺️ Spatial Suitability", "🔥 Confidence Heatmap", "📊 Technical Deep-Dive"])

    with tab_heat:
        st.subheader("Raw AI Probability Map")
        st.markdown("Actual raw values (0.0 to 1.0) assigned to the terrain.")
        
        col_h1, col_h2 = st.columns([2, 1])
        with col_h1:
            fig_heat, ax_heat = plt.subplots(figsize=(10, 10))
            im = ax_heat.imshow(prob_map, cmap='magma') 
            ax_heat.axis('off')
            plt.colorbar(im, ax=ax_heat, label='Confidence Level')
            st.pyplot(fig_heat)
        
        with col_h2:
            st.write("**Confidence Distribution**")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 8))
            ax_hist.hist(prob_map.ravel(), bins=50, color='#39FF14', alpha=0.7)
            ax_hist.set_xlabel("Confidence Score")
            ax_hist.set_ylabel("Pixel Count")
            st.pyplot(fig_hist)
            st.info("Use this distribution to find the natural 'break' in AI confidence for this specific site.")
        
    with tab_vis:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Sentinel-2 RGB")
            st.image(np.clip(img_array[:,:,:3] * 3, 0, 1), use_container_width=True)

        neon_green_cmap = ListedColormap(["#00000000", "#39FF14"])
            
        with col_img2:
            st.subheader("AI Suitability Mask")
            fig_mask, ax_mask = plt.subplots(figsize=(10, 10))
            ax_mask.imshow(np.clip(img_array[:,:,:3] * 3, 0, 1)) 
            ax_mask.imshow(binary_mask, alpha=0.7, cmap=neon_green_cmap) 
            ax_mask.axis('off')
            st.pyplot(fig_mask)

        with st.expander("📍 View Suitable Plot Coordinates"):
            y_indices, x_indices = np.where(binary_mask == True)
            lat_per_m = 1.0 / 111320.0 
            lon_per_m = 1.0 / (111320.0 * np.cos(np.radians(lat)))
            
            suitable_coords = []
            for y, x in zip(y_indices, x_indices):
                offset_y = (127.5 - y) * 10 
                offset_x = (x - 127.5) * 10
                p_lat = lat + (offset_y * lat_per_m)
                p_lon = lon + (offset_x * lon_per_m)
                suitable_coords.append({"Latitude": round(p_lat, 6), "Longitude": round(p_lon, 6)})
            
            df_coords = pd.DataFrame(suitable_coords)
            st.write(f"Detected **{len(df_coords)}** suitable 10m x 10m plots.")
            st.dataframe(df_coords, use_container_width=True, height=300)
            csv = df_coords.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv, file_name="suitable_plots.csv")

    with tab_data:
        st.subheader("Site Performance Metrics")
        st.markdown(f"""
        - **Coordinates:** {lat}, {lon}
        - **Primary Constraint:** Slope < 5°
        - **Secondary Constraint:** NDVI < 0.4
        - **Calculated Yield:** {annual_kwh:,.0f} kWh per annum.
        """)

# --- 7. Interactive Background Map ---
st.divider()
st.subheader("Interactive Context Explorer")
m = leafmap.Map(center=[lat, lon], zoom=12)
m.add_basemap("SATELLITE")
m.add_marker(location=[lat, lon], tooltip="Analyzed Site Center")
m.to_streamlit(height=500)