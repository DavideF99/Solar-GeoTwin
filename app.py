import streamlit as st
import leafmap.foliumap as leafmap
import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.data_pipeline import GeoDataFetcher
from modules.ai_engine import SolarUNet
from modules.spatial_eng import SpatialProcessor

# --- 1. Page Configuration & UI Theme ---
st.set_page_config(page_title="Solar-GeoTwin | AI Site Selection", layout="wide")

# Add a professional CSS tweak to the sidebar for better contrast
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

# @st.cache_resource
# def load_model():
#     model = SolarUNet(in_channels=4, out_channels=1).to(device)
#     try:
#         model.load_state_dict(torch.load("models/solar_unet_v1.pth", map_location=device)) 
#     except FileNotFoundError:
#         st.sidebar.warning("⚠️ Weights not found. Using uninitialized model.")
#     model.eval()
#     return model

@st.cache_resource
def load_model():
    model = SolarUNet(in_channels=4, out_channels=1).to(device)
    
    # Get the directory that THIS file (app.py) is in
    base_path = os.path.dirname(__file__)
    # Create the absolute path to the weights
    weights_path = os.path.join(base_path, "models", "solar_unet_v1.pth")
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        # Optional: Add a success message to the sidebar to confirm it worked!
        # st.sidebar.success("✅ Model weights loaded.")
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ Weights not found at {weights_path}")
    
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
    
    with st.expander("⚙️ Engineering Settings", expanded=False):
        threshold = st.slider("AI Confidence Cutoff", 0.0, 1.0, 0.70, help="Higher values reduce false positives in site selection.")
        efficiency = st.number_input("Panel Efficiency", value=0.20, help="Standard commercial module efficiency (20%).")
        perf_ratio = st.number_input("Performance Ratio", value=0.75, help="System losses (inverter, wiring, soiling).")

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
st.markdown("Transitioning from traditional site surveys to **AI-driven spatial twins** for utility-scale solar prospecting.")

# --- 5. Analysis Logic ---
if run_btn:
    with st.status("Initializing Spatial Analysis...", expanded=True) as status:
        st.write("📡 Querying Google Earth Engine (Sentinel-2 + SRTM)...")
        raw = fetcher.get_sentinel_composite(lon, lat, 2000, "2024-01-01", "2024-12-31")
        processed = processor.calculate_metrics(raw)
        
        st.write("🧠 Executing U-Net Deep Learning Inference...")
        img_array = fetcher.export_single_patch(processed, lon, lat) 
        input_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        st.write("🌦️ Fetching NASA Climatology Data...")
        daily_ghi = fetcher.get_nasa_irradiance(lat, lon)
        
        # Convert daily average to annual total (using 365.25 for leap year accuracy)
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
    # Top Row: High-Level KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Solar Resource", f"{annual_ghi:.1f} kWh/m²/y", help="NASA POWER 30-year average GHI.")
    kpi2.metric("Identified Area", f"{total_m2/10000:.2f} Hectares", help="Total land meeting slope and AI suitability criteria.")
    kpi3.metric("Est. Generation", f"{annual_kwh/1e6:.2f} GWh/yr", help="Annual yield including performance losses.")

    # Main Tabs for Visuals
    tab_vis, tab_data = st.tabs(["🗺️ Spatial Suitability", "📊 Technical Deep-Dive"])
    
    with tab_vis:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.subheader("Sentinel-2 RGB")
            st.image(np.clip(img_array[:,:,:3] * 3, 0, 1), use_container_width=True)
            
        with col_img2:
            st.subheader("AI Suitability Mask")
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.clip(img_array[:,:,:3] * 3, 0, 1))
            ax.imshow(binary_mask, alpha=0.5, cmap='Wistia')
            ax.axis('off')
            st.pyplot(fig)
            
    with tab_data:
        st.subheader("Site Performance Metrics")
        st.markdown(f"""
        - **Coordinates:** {lat}, {lon}
        - **Primary Constraint:** Slope < 5° (Industrial Engineering Standard)
        - **Secondary Constraint:** NDVI < 0.4 (Minimized Environmental Impact)
        - **Calculated Yield:** {annual_kwh:,.0f} kWh per annum.
        """)

# --- 7. Interactive Background Map ---
st.divider()
st.subheader("Interactive Context Explorer")
m = leafmap.Map(center=[lat, lon], zoom=12)
m.add_basemap("SATELLITE")
m.to_streamlit(height=500)