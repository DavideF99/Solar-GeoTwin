import ee
import numpy as np
import requests

class SpatialProcessor:
    def __init__(self):
        pass

    def calculate_metrics(self, image):
        """
        Calculates NDVI, Slope, and Aspect from a Sentinel-2 + SRTM stack.
        """
        # 1. NDVI Calculation
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # 2. Terrain Analysis (Using SRTM Global 30m)
        srtm = ee.Image("USGS/SRTMGL1_003")
        terrain = ee.Terrain.products(srtm)
        slope = terrain.select('slope').rename('Slope')
        aspect = terrain.select('aspect').rename('Aspect')

        # 3. Create a Suitability Mask (The 'Industrial Engineer' Logic)
        # Criteria: Slope < 5 degrees AND NDVI < 0.4 (Bare land)
        mask = slope.lt(5).And(ndvi.lt(0.4))
        
        return image.addBands([ndvi, slope, aspect, mask.rename('Suitable_Mask')])
    
    def estimate_yield(self, binary_mask, avg_irradiance, resolution=10, efficiency=0.20):
        """
        Calculates the estimated annual energy yield.
        - binary_mask: The AI prediction (Boolean/Binary array)
        - avg_irradiance: Annual average GHI (kWh/m2/year)
        - resolution: Meters per pixel (Sentinel-2 is 10m)
        - efficiency: Standard system performance ratio (0.20 default)
        """
        # 1. Count suitable pixels
        suitable_pixel_count = np.sum(binary_mask)
        
        # 2. Convert to Area (m2)
        # Each Sentinel-2 pixel is 10m x 10m = 100m2
        total_area_m2 = suitable_pixel_count * (resolution ** 2)
        
        # 3. Energy Formula: E = A * r * H * PR
        # A = Area, r = efficiency, H = Irradiance
        annual_yield_kwh = total_area_m2 * efficiency * avg_irradiance
        
        return annual_yield_kwh, total_area_m2