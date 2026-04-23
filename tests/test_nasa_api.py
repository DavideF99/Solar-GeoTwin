import sys
import os
# Ensure the script can find your modules folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_pipeline import GeoDataFetcher
import streamlit as st

def test_nasa_integration():
    # 1. Initialize Fetcher (Mocking the project ID for a simple API test)
    # Note: This test only checks the NASA API method, so GEE initialization isn't strictly needed
    fetcher = GeoDataFetcher(project_id="test-project")
    
    # 2. Test Coordinates (Northern Cape, South Africa)
    lat, lon = -29.0, 21.0
    
    print(f"--- Testing NASA POWER API Integration ---")
    print(f"Target Coordinates: Lat {lat}, Lon {lon}")
    
    # 3. Call the new method
    daily_ghi = fetcher.get_nasa_irradiance(lat, lon)
    
    # 4. Validation Logic
    if daily_ghi != 5.0:  # 5.0 is our 'failure' fallback
        print(f"✅ SUCCESS: NASA API returned {daily_ghi:.2f} kWh/m2/day")
        
        annual_ghi = daily_ghi * 365.25
        print(f"📊 Calculated Annual GHI: {annual_ghi:.2f} kWh/m2/year")
        
        # Engineering Check: Most global GHI values fall between 800 and 2800
        if 800 < annual_ghi < 3000:
            print("✅ VALUE CHECK: Data is within realistic physical bounds.")
        else:
            print("⚠️ VALUE WARNING: Data seems outside typical solar ranges.")
    else:
        print("❌ FAILURE: API returned fallback value (5.0). Check your internet connection or API URL.")

if __name__ == "__main__":
    test_nasa_integration()