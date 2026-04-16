# modules/data_pipeline.py
import ee
import requests
import pandas as pd
from datetime import datetime
import geemap
import numpy as np
import os
import streamlit as st
import json


class GeoDataFetcher:
    def __init__(self, project_id):
        self.project_id = project_id
        # Trigger the specialized authentication
        self._authenticate()

    def _authenticate(self):
        # 1. Check for Streamlit Secrets (Production)
        if "gcp_service_account" in st.secrets:
            try:
                # Access the TOML dictionary we created
                info = st.secrets["gcp_service_account"]
                
                # Create credentials from the secret dictionary
                # This is the "Key" that opens the Google Earth Engine door
                credentials = ee.ServiceAccountCredentials(
                    info['client_email'], 
                    key_data=info['private_key']
                )
                
                # Initialize with BOTH the credentials AND the project ID
                ee.Initialize(credentials=credentials, project=self.project_id)
            except Exception as e:
                st.error(f"Cloud Auth Failed: {e}")
        
        # 2. Local Fallback (For your Mac M3)
        else:
            try:
                ee.Initialize(project=self.project_id)
            except Exception:
                # This opens the browser window for local auth if needed
                ee.Authenticate()
                ee.Initialize(project=self.project_id)
        
    def get_sentinel_composite(self, lon, lat, buffer_m, start_date, end_date):
        """Fetches a cloud-free median composite for a circular ROI."""
        roi = ee.Geometry.Point([lon, lat]).buffer(buffer_m)
        
        def mask_s2_clouds(image):
            qa = image.select('QA60')
            cloud_bit = 1 << 10
            cirrus_bit = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
            return image.updateMask(mask).divide(10000)

        collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(mask_s2_clouds))

        return collection.median().clip(roi)

    def fetch_nasa_power(self, lon, lat, start, end):
        """
        Fetches GHI (Global Horizontal Irradiance) from NASA POWER API.
        Format: YYYYMMDD
        """
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN", # GHI
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "format": "JSON"
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()['properties']['parameter']['ALLSKY_SFC_SW_DWN']
            # Convert dictionary to DataFrame
            df = pd.DataFrame(list(data.items()), columns=['Date', 'GHI'])
            # Convert 'Date' string to actual datetime objects for easier plotting later
            df['Date'] = pd.to_datetime(df['Date']) 
            return df
        else:
            print(f"Failed to fetch NASA data. Status: {response.status_code}")
            return pd.DataFrame() # Return empty DF to avoid crashing the rest of the script
        
    def export_patches(self, image, lon, lat, area_name="region", patch_size=256, n_patches=40, output_dir='data'):
        """
        Extracts fixed-size patches by specifying scale and region.
        """
        import os
        import numpy as np
        import geemap
        
        os.makedirs(f"{output_dir}/images", exist_ok=True)
        os.makedirs(f"{output_dir}/masks", exist_ok=True)
        
        print(f"Exporting {n_patches} patches...")

        for i in range(n_patches):
            # Create a small random offset
            offset_lon = lon + (np.random.uniform(-0.02, 0.02))
            offset_lat = lat + (np.random.uniform(-0.02, 0.02))
            
            # Define the point and create a square buffer
            # 256 pixels * 10m = 2560m, but we use half-distance for center-to-edge
            half_dist = (patch_size * 10) / 2
            region = ee.Geometry.Point([offset_lon, offset_lat]).buffer(half_dist).bounds()
            
            try:
                # We use scale=10 to force 10-meter pixels
                img_patch = geemap.ee_to_numpy(
                    image.select(['B4', 'B3', 'B2', 'B8']), 
                    region=region, 
                    scale=10
                )
                
                mask_patch = geemap.ee_to_numpy(
                    image.select(['Suitable_Mask']), 
                    region=region, 
                    scale=10
                )

                # Trim to exactly 256x256 in case of small rounding errors from GEE
                if img_patch is not None and mask_patch is not None:
                    img_patch = img_patch[:patch_size, :patch_size, :]
                    mask_patch = mask_patch[:patch_size, :patch_size]
                    
                    # New naming convention: area_name + index
                    img_name = f"{area_name}_patch_{i}.npy"
                    mask_name = f"{area_name}_patch_{i}.npy"
                    
                    # Save logic
                    np.save(os.path.join(output_dir, 'images', img_name), img_patch)
                    np.save(os.path.join(output_dir, 'masks', mask_name), mask_patch)
                else:
                    print(f"Skipping patch {i}: Empty data.")
                    
            except Exception as e:
                print(f"Error exporting patch {i}: {e}")

        print(f"Successfully exported to {output_dir}")

    def export_single_patch(self, image, lon, lat):
        region = ee.Geometry.Point([lon, lat]).buffer(1280).bounds()
        patch = geemap.ee_to_numpy(image.select(['B4', 'B3', 'B2', 'B8']), region=region, scale=10)
        return patch[:256, :256, :]