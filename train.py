import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as cfg
from src.cnn_model import FullGestureModel

class GestureDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        self.sign_to_idx = {sign: i for i, sign in enumerate(cfg.SIGNS)}
        for sign in cfg.SIGNS:
            folder = os.path.join(cfg.DATA_FRAMES_DIR, sign)
            if not os.path.exists(folder):
                continue
            files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
            total_clips = len(files) // cfg.FRAMES_PER_CLIP
            for clip_idx in range(total_clips):
                clip_files = files[clip_idx * cfg.FRAMES_PER_CLIP:(clip_idx + 1) * cfg.FRAMES_PER_CLIP]
                if len(clip_files) == cfg.FRAMES_PER_CLIP:
                    self.samples.append((folder, clip_files, self.sign_to_idx[sign]))
        print(f"Dataset loaded: {len(self.samples)} clips across {len(cfg.SIGNS)} signs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, clip_files, label = self.samples[idx]
        frames = []
        prev_gray = None
        for fname in clip_files:
            img = Image.open(os.path.join(folder, fname)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            gray = torch.mean(img, dim=0, keepdim=True)
            if prev_gray is None:
                diff = torch.zeros_like(gray)
            else:
                diff = torch.abs(gray - prev_gray)
            prev_gray = gray
            frame = torch.cat([img, diff], dim=0)
            frames.append(frame)
        sequence = torch.stack(frames, dim=0)
        if sequence.shape[0] < cfg.SEQUENCE_LENGTH:
            pad = cfg.SEQUENCE_LENGTH - sequence.shape[0]
            sequence = torch.cat([sequence, sequence[-1:].repeat(pad, 1, 1, 1)], dim=0)
        else:
            sequence = sequence[:cfg.SEQUENCE_LENGTH]
        return sequence, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(cfg.FRAME_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(cfg.FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return train_transform, val_transform

def train():
    print("\n-- Phase 5: Training --")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    train_transform, val_transform = get_transforms()
    full_dataset = GestureDataset(transform=train_transform)
    if len(full_dataset) == 0:
        print("ERROR: No training data found. Complete Phase 2 first.")
        return
    train_size = int(cfg.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"   Train samples : {train_size}")
    print(f"   Val samples   : {val_size}")
    model = FullGestureModel(
        in_channels=cfg.CNN_IN_CHANNELS,
        feature_dim=cfg.CNN_FEATURE_DIM,
        hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_layers=cfg.LSTM_NUM_LAYERS,
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.LSTM_DROPOUT
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model params  : {total_params:,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.NUM_EPOCHS}"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        scheduler.step()
        print(f"   Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"   Best model saved (acc: {best_val_acc:.1f}%)")
    torch.save(model.state_dict(), cfg.FINAL_MODEL_PATH)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.LOG_DIR, "training_curves.png"))
    print(f"\n-- Training Complete --")
    print(f"   Best Val Accuracy : {best_val_acc:.1f}%")
    print(f"   Best model saved  : {cfg.BEST_MODEL_PATH}")
    print(f"   Training curves   : {cfg.LOG_DIR}/training_curves.png")
    print("   Ready for Phase 6!")

train()