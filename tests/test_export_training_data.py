# Updated tests/export_training_data.py
import sys
import os
import ee
from pathlib import Path

# Path setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from modules.data_pipeline import GeoDataFetcher
from modules.spatial_eng import SpatialProcessor

def main():
    PROJECT_ID = 'solar-geoai-dferreri45'
    
    # --- BIOME SELECTION ---
    # Define diverse environmental contexts for the AI to learn
    locations = {
        "Northern_Cape_Desert": [23.3, -28.3],    # Arid/Flat
        "Western_Cape_Mountains": [19.0, -34.0],  # Steep/Complex
        "Gauteng_Grassland": [28.2, -25.8],       # High Vegetation
        "Vietnam_Coastal": [108.2, 16.0],         # Humid/Tropical (Your local area!)
        "Sahara_Pure_Sand": [0.0, 20.0]           # Extreme Arid
    }
    
    fetcher = GeoDataFetcher(PROJECT_ID)
    processor = SpatialProcessor()

    # 2. LOOP & EXPORT
    patches_per_location = 40  # 5 locations * 40 = 200 total patches
    
    for name, coords in locations.items():
        print(f"🌍 Processing Biome: {name}...")
        lon, lat = coords
        
        # Fetch a 40km x 40km area to sample from
        raw = fetcher.get_sentinel_composite(lon, lat, 20000, "2024-01-01", "2024-12-31")
        processed = processor.calculate_metrics(raw)
        
        # KEY CHANGE: Pass the 'name' variable to the 'area_name' parameter
        fetcher.export_patches(
            processed, 
            lon, 
            lat, 
            area_name=name, 
            n_patches=patches_per_location
        )

    print(f"✅ Generalist Dataset Ready: 200 patches in /data")

if __name__ == "__main__":
    main()