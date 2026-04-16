import sys
import os

# This adds the parent directory (Solar-GeoTwin) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_pipeline import GeoDataFetcher

# Your specific Project ID
PROJECT_ID = 'solar-geoai-dferreri45'
LON, LAT = 23.3, -28.3  # Postmasburg, Northern Cape

fetcher = GeoDataFetcher(PROJECT_ID)

print("--- Fetching NASA Weather Data ---")
weather_df = fetcher.fetch_nasa_power(LON, LAT, "2024-01-01", "2024-12-31")
print(weather_df.head())

print("\n--- Testing GEE Connection ---")
# This just verifies we can define the composite without errors
composite = fetcher.get_sentinel_composite(LON, LAT, 5000, "2024-01-01", "2024-12-31")
print("Satellite Composite Metadata:", composite.getInfo()['bands'])