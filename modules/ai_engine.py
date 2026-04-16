from torch.utils.data import Dataset
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SolarDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        data_dir: Path to the root 'data' folder
        """
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, 'images', '*.npy')))
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, 'masks', '*.npy')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load numpy arrays
        image = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.float32)

        # GEE export often results in (H, W, C). 
        # PyTorch needs (C, H, W), so we transpose.
        image = image.transpose(2, 0, 1)
        
        # Ensure mask is (1, H, W)
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        elif mask.shape[2] == 1:
            mask = mask.transpose(2, 0, 1)

        # Convert to Tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor

class SolarUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SolarUNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder (Downsampling)
        self.enc1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = double_conv(128, 256)
        
        # Decoder (Upsampling)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128) # 128 from up2 + 128 from enc2 (skip connection)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64) # 64 from up1 + 64 from enc1
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        
        # Bottleneck
        bn = self.bottleneck(p2)
        
        # Decoder with Skip Connections
        u2 = self.up2(bn)
        m2 = torch.cat([u2, c2], dim=1) # The Skip Connection
        d2 = self.dec2(m2)
        
        u1 = self.up1(d2)
        m1 = torch.cat([u1, c1], dim=1) # The Skip Connection
        d1 = self.dec1(m1)
        
        return torch.sigmoid(self.final_conv(d1))
    
class SolarTrainer:
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        # Use BCE with Logits for numerical stability
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.):
        """Dice loss is essential for imbalanced satellite imagery."""
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def train_step(self, images, masks):
        self.model.train()
        images, masks = images.to(self.device), masks.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        
        # Combined Loss: 50% Pixel-wise BCE + 50% Spatial Dice Loss
        loss_bce = self.criterion(outputs, masks)
        loss_dice = self.dice_loss(outputs, masks)
        total_loss = loss_bce + loss_dice
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()