import sys
import os
import ee

# 1. Path setup to ensure modules are discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_pipeline import GeoDataFetcher
from modules.spatial_eng import SpatialProcessor

def main():
    # --- Configuration ---
    PROJECT_ID = 'solar-geoai-dferreri45'
    # Center of the "Solar Corridor" in Northern Cape, SA
    LON, LAT = 23.3, -28.3 
    BUFFER = 5000  # 5km radius for a quick test
    
    print(f"--- Initializing Solar-GeoTwin Spatial Test ---")
    
    # 2. Initialize Engines
    try:
        fetcher = GeoDataFetcher(PROJECT_ID)
        processor = SpatialProcessor()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 3. Step 1: Fetch Raw Satellite Data (Module A)
    print("Step 1: Fetching Sentinel-2 Composite...")
    raw_composite = fetcher.get_sentinel_composite(
        LON, LAT, BUFFER, "2024-01-01", "2024-12-31"
    )

    # 4. Step 2: Calculate Spatial Metrics (Module B)
    print("Step 2: Calculating NDVI, Slope, and Aspect...")
    processed_image = processor.calculate_metrics(raw_composite)

    # 5. Step 3: Validation & Metadata Check
    print("\n--- Validation Results ---")
    bands = processed_image.bandNames().getInfo()
    print(f"Total Bands available: {len(bands)}")
    print(f"Band List: {bands}")

    # Check for our engineered features
    expected_features = ['NDVI', 'Slope', 'Aspect', 'Suitable_Mask']
    for feature in expected_features:
        if feature in bands:
            print(f"✅ Feature '{feature}' successfully generated.")
        else:
            print(f"❌ Feature '{feature}' is missing!")

    # 6. Sample Data Value (Pixel Check)
    # We'll sample the center pixel to see real-world engineering values
    sample = processed_image.sample(ee.Geometry.Point([LON, LAT]), 10).first().toDictionary().getInfo()
    
    print("\n--- Physical Data at Center Point ---")
    print(f"Vegetation Index (NDVI): {sample.get('NDVI', 'N/A'):.4f}")
    print(f"Terrain Slope: {sample.get('Slope', 'N/A'):.2f} degrees")
    print(f"Suitability Score: {'SUITABLE' if sample.get('Suitable_Mask') == 1 else 'UNSUITABLE'}")

if __name__ == "__main__":
    main()