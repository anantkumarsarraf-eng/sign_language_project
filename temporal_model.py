import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as cfg
from src.cnn_model import FullGestureModel

class FrameDifferencer(nn.Module):
    def __init__(self):
        super(FrameDifferencer, self).__init__()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        rgb = x[:, :, :3, :, :]
        diffs = []
        for i in range(seq_len):
            if i == 0:
                diff = torch.zeros(batch_size, 1, h, w, device=x.device)
            else:
                prev = rgb[:, i-1, :, :, :]
                curr = rgb[:, i,   :, :, :]
                diff = torch.mean(torch.abs(curr - prev), dim=1, keepdim=True)
            diffs.append(diff)
        diff_tensor = torch.stack(diffs, dim=1)
        out = torch.cat([x, diff_tensor], dim=2)
        return out

class SlidingWindowClassifier:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.buffer = []

    def add_frame(self, frame_tensor):
        self.buffer.append(frame_tensor)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def is_ready(self):
        return len(self.buffer) == self.window_size

    def get_window(self):
        return torch.stack(self.buffer, dim=0).unsqueeze(0)

    def clear(self):
        self.buffer = []

class TemporalSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions = []

    def add_prediction(self, pred):
        self.predictions.append(pred)
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)

    def get_smoothed(self):
        if not self.predictions:
            return None
        from collections import Counter
        counts = Counter(self.predictions)
        return counts.most_common(1)[0][0]

    def clear(self):
        self.predictions = []

class GesturePredictor:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.differencer = FrameDifferencer()
        self.slider = SlidingWindowClassifier(cfg.SEQUENCE_LENGTH)
        self.smoother = TemporalSmoother(cfg.SMOOTHING_WINDOW)
        self.suppression_counter = 0
        self.last_sign = None
        self.model.eval()

    def predict(self, frame_tensor):
        self.slider.add_frame(frame_tensor)
        if not self.slider.is_ready():
            return None, 0.0
        window = self.slider.get_window().to(self.device)
        rgb_window = window
        diff_window = self.differencer(rgb_window.unsqueeze(0).squeeze(0) if rgb_window.dim() == 4 else rgb_window)
        with torch.no_grad():
            output = self.model(diff_window)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
        if confidence < cfg.CONFIDENCE_THRESHOLD:
            return None, confidence
        predicted_sign = cfg.SIGNS[pred_idx]
        self.smoother.add_prediction(predicted_sign)
        smoothed_sign = self.smoother.get_smoothed()
        if self.suppression_counter > 0:
            self.suppression_counter -= 1
            return None, confidence
        if smoothed_sign == self.last_sign:
            return None, confidence
        self.last_sign = smoothed_sign
        self.suppression_counter = cfg.SUPPRESSION_FRAMES
        return smoothed_sign, confidence

if __name__ == "__main__":
    print("Testing temporal model...")
    differencer = FrameDifferencer()
    dummy_seq = torch.randn(1, 20, 3, 64, 64)
    out = differencer(dummy_seq)
    print(f"Differencer input  : {dummy_seq.shape}")
    print(f"Differencer output : {out.shape}")
    slider = SlidingWindowClassifier(window_size=20)
    for i in range(25):
        frame = torch.randn(3, 64, 64)
        slider.add_frame(frame)
    print(f"Slider ready       : {slider.is_ready()}")
    print(f"Window shape       : {slider.get_window().shape}")
    smoother = TemporalSmoother(window_size=5)
    for sign in ["HELLO","HELLO","YES","HELLO","HELLO"]:
        smoother.add_prediction(sign)
    print(f"Smoothed prediction: {smoother.get_smoothed()}")
    print("Temporal model test passed!")