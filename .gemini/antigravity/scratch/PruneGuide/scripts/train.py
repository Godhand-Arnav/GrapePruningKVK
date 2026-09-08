import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.hourglass import StackedHourglass
from pipeline.dataset import GrapevineDataset, TOTAL_HEATMAPS

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, targets_s1, targets_s2, _ in pbar:
        images = images.to(device)
        targets_s1 = targets_s1.to(device)
        targets_s2 = targets_s2.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: returns list of stage predictions [head_1, head_2]
        preds = model(images)
        
        # Multi-stage supervision loss: L = L1 + L2
        loss1 = criterion(preds[0], targets_s1)
        loss2 = criterion(preds[1], targets_s2)
        total_loss = loss1 + loss2
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
    return running_loss / max(len(dataloader), 1)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets_s1, targets_s2, _ in dataloader:
            images = images.to(device)
            targets_s1 = targets_s1.to(device)
            targets_s2 = targets_s2.to(device)
            
            preds = model(images)
            loss1 = criterion(preds[0], targets_s1)
            loss2 = criterion(preds[1], targets_s2)
            val_loss += (loss1 + loss2).item()
            
    return val_loss / max(len(dataloader), 1)

def main():
    parser = argparse.ArgumentParser(description='Train PruneGuide Stacked Hourglass Model')
    parser.add_argument('--data_dir', type=str, default=os.path.join(BASE_DIR, 'data', 'samples'),
                        help='Path to dataset directory containing train/ and test/')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (1 is recommended for SHG)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=512, help='Input resolution (512 or 1024)')
    parser.add_argument('--hg_channels', type=int, default=64, help='Hourglass channels (64 for fast/low-VRAM, 128/256 for full)')
    parser.add_argument('--save_dir', type=str, default=os.path.join(BASE_DIR, 'checkpoints'))
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*60)
    print(f'Training PruneGuide on device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Dataset: {args.data_dir}')
    print(f'Resolution: {args.img_size}x{args.img_size}, Epochs: {args.epochs}, Batch Size: {args.batch_size}')
    print('='*60)
    
    # Datasets and Loaders
    train_dataset = GrapevineDataset(args.data_dir, split='train', target_size=(args.img_size, args.img_size))
    test_dataset = GrapevineDataset(args.data_dir, split='test', target_size=(args.img_size, args.img_size))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize 2-stage Stacked Hourglass
    model = StackedHourglass(
        num_heatmaps=TOTAL_HEATMAPS, 
        in_channels=3, 
        base_channels=64, 
        hg_channels=args.hg_channels, 
        num_stacks=2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, test_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        print(f'Epoch {epoch:03d}/{args.epochs:03d} - Train Loss: {train_loss:.5f} - Val Loss: {val_loss:.5f} ({elapsed:.1f}s)')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'hg_channels': args.hg_channels,
                'num_heatmaps': TOTAL_HEATMAPS
            }, ckpt_path)
            print(f'  --> Saved best model checkpoint to {ckpt_path}')
            
    print('='*60)
    print(f'Training complete! Best validation loss: {best_loss:.5f}')
    print(f'Checkpoint saved at: {os.path.join(args.save_dir, "best_model.pth")}')
    print('='*60)

if __name__ == '__main__':
    main()
