import torch
import torch.nn as nn

class GestureCNN(nn.Module):
    def __init__(self, in_channels=4, feature_dim=128):
        super(GestureCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, feature_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.projector(x)
        return x

class GestureLSTM(nn.Module):
    def __init__(self, feature_dim=128, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.classifier(out[:, -1, :])
        return out

class FullGestureModel(nn.Module):
    def __init__(self, in_channels=4, feature_dim=128, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super(FullGestureModel, self).__init__()
        self.cnn = GestureCNN(in_channels, feature_dim)
        self.lstm = GestureLSTM(feature_dim, hidden_size, num_layers, num_classes, dropout)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        out = self.lstm(features)
        return out

if __name__ == "__main__":
    print("Testing model architecture...")
    model = FullGestureModel()
    dummy = torch.randn(2, 20, 4, 64, 64)
    output = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {output.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total:,}")
    print("Model test passed!")