import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from modules.ai_engine import SolarUNet, SolarDataset, SolarTrainer
import json

def main():
    # 1. Hardware Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🌞 Training Solar-GeoTwin on: {device}")

    # 2. Data Setup
    full_dataset = SolarDataset(data_dir='data')

    # 80% for training, 20% for validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)

    # 3. Model & Trainer
    model = SolarUNet(in_channels=4, out_channels=1)
    trainer = SolarTrainer(model, device)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=20, gamma=0.1)

    # 4. Initialize history tracking
    history = {
        "train_loss": [],
        "val_loss": []
    }

    # 5. Training Loop 
    EPOCHS = 70
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            loss = trainer.train_step(images, masks)
            train_loss += loss
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss_bce = trainer.criterion(outputs, masks)
                loss_dice = trainer.dice_loss(outputs, masks)
                val_loss += (loss_bce + loss_dice).item()
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 6. Save the Weights (Crucial for Phase 4)
    torch.save(model.state_dict(), "models/solar_unet_v2.pth")

    # 7. Save history to JSON
    with open("models/training_history_v2.json", "w") as f:
        json.dump(history, f)
        
    print("✅ Model and training history saved to /models")

if __name__ == "__main__":
    main()