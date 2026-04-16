import sys
import os
import torch
from torch.utils.data import DataLoader

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.ai_engine import SolarDataset, SolarUNet

def test_mps_flow():
    # 1. Detect M3 GPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    # 2. Initialize Dataset and Loader
    dataset = SolarDataset(data_dir='data')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 3. Initialize Model and Move to M3
    # in_channels=4 (RGB+NIR), out_channels=1 (Suitability Mask)
    model = SolarUNet(in_channels=4, out_channels=1).to(device)
    
    # 4. Fetch one batch
    images, masks = next(iter(loader))
    
    # Move tensors to M3
    images = images.to(device)
    masks = masks.to(device)

    print(f"Input Batch Shape: {images.shape}") # Should be [4, 4, 256, 256]
    print(f"Target Mask Shape: {masks.shape}") # Should be [4, 1, 256, 256]

    # 5. Forward Pass (Test Calculation on GPU)
    with torch.no_grad():
        output = model(images)
    
    print(f"Model Output Shape: {output.shape}")
    print("✅ Successfully ran inference on M3 GPU!")

if __name__ == "__main__":
    test_mps_flow()