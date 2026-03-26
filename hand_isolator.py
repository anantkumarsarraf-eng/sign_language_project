import cv2
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src import config as cfg

def get_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(cfg.SKIN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(cfg.SKIN_HSV_UPPER, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def get_hand_roi(frame):
    mask = get_skin_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < cfg.MIN_HAND_AREA:
        return None, None, mask
    x, y, w, h = cv2.boundingRect(largest)
    pad = cfg.ROI_PADDING
    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_frame, x + w + pad)
    y2 = min(h_frame, y + h + pad)
    roi = frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2, y2)
    return roi, bbox, mask

def is_hand_present(frame):
    mask = get_skin_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(largest) >= cfg.MIN_HAND_AREA

def draw_hand_bbox(frame, bbox, label=None, confidence=None):
    if bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if label:
        text = f"{label} {confidence:.0%}" if confidence else label
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame