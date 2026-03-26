import cv2
import torch
import numpy as np
import sys
import os
import time
import pyttsx3
from collections import deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import config as cfg
from src.cnn_model import FullGestureModel

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def load_model():
    model = FullGestureModel(
        in_channels=cfg.CNN_IN_CHANNELS,
        feature_dim=cfg.CNN_FEATURE_DIM,
        hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_layers=cfg.LSTM_NUM_LAYERS,
        num_classes=cfg.NUM_CLASSES,
        dropout=0.0
    )
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame(frame):
    frame = cv2.resize(frame, cfg.FRAME_SIZE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - 0.5) / 0.5
    frame = torch.tensor(frame).permute(2, 0, 1)
    return frame

def draw_ui(frame, sign, confidence, smoothed, suppression):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    if sign:
        cv2.putText(frame, f"Detected: {sign}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    else:
        cv2.putText(frame, "Performing sign...", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if smoothed:
        cv2.putText(frame, f"Speaking: {smoothed}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    if suppression > 0:
        cv2.rectangle(frame, (w - 20, h - 20), (w - 10, h - 10), (0, 0, 255), -1)
    return frame

def main():
    print("\n-- Phase 6: Real-Time Inference --")
    print("   Loading model...")
    model = load_model()
    print("   Model loaded!")
    print("   Press Q to quit\n")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("   Warming up camera...")
    for _ in range(30):
        cap.read()
    print("   Camera ready!\n")

    frame_buffer = deque(maxlen=cfg.SEQUENCE_LENGTH)
    prediction_buffer = deque(maxlen=cfg.SMOOTHING_WINDOW)
    prev_gray = None
    last_spoken = None
    suppression_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, cfg.FRAME_SIZE)
        tensor = torch.tensor(resized.astype(np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.permute(2, 0, 1)

        gray = torch.mean(tensor, dim=0, keepdim=True)
        if prev_gray is None:
            diff = torch.zeros_like(gray)
        else:
            diff = torch.abs(gray - prev_gray)
        prev_gray = gray

        frame_with_diff = torch.cat([tensor, diff], dim=0)
        frame_buffer.append(frame_with_diff)

        current_sign = None
        current_conf = 0.0
        smoothed_sign = None

        if len(frame_buffer) == cfg.SEQUENCE_LENGTH:
            sequence = torch.stack(list(frame_buffer), dim=0).unsqueeze(0)
            with torch.no_grad():
                output = model(sequence)
                probs = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
                current_conf = confidence.item()
                if current_conf >= cfg.CONFIDENCE_THRESHOLD:
                    current_sign = cfg.SIGNS[pred_idx.item()]
                    prediction_buffer.append(current_sign)

            if len(prediction_buffer) == cfg.SMOOTHING_WINDOW:
                from collections import Counter
                counts = Counter(prediction_buffer)
                smoothed_sign = counts.most_common(1)[0][0]

                if suppression_counter == 0 and smoothed_sign != last_spoken:
                    print(f"   Speaking: {smoothed_sign} (confidence: {current_conf:.0%})")
                    speak(smoothed_sign.replace("_", " "))
                    last_spoken = smoothed_sign
                    suppression_counter = cfg.SUPPRESSION_FRAMES

        if suppression_counter > 0:
            suppression_counter -= 1

        frame = draw_ui(frame, current_sign, current_conf, smoothed_sign, suppression_counter)
        cv2.imshow("Sign Language to Speech - Press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n   Inference stopped.")

main()
