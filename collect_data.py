import cv2
import os
import sys
import time

print("Starting data collection script...")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

from src import config as cfg
print(f"Config loaded. Signs: {cfg.SIGNS}")

for sign in cfg.SIGNS:
    path = os.path.join(PROJECT_ROOT, cfg.DATA_FRAMES_DIR, sign)
    os.makedirs(path, exist_ok=True)

print("Folders created. Opening webcam...")

def count_existing_clips(sign):
    folder = os.path.join(PROJECT_ROOT, cfg.DATA_FRAMES_DIR, sign)
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    return len(files) // cfg.FRAMES_PER_CLIP

def draw_text(frame, text, y, color=(255, 255, 255), scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (11, y + 1), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (10, y), font, scale, color, thickness)

def warmup_camera(cap, frames=30):
    print("Warming up camera...")
    for i in range(frames):
        cap.read()
    print("Camera ready!")

def collect_sign(cap, sign, clip_index):
    folder = os.path.join(PROJECT_ROOT, cfg.DATA_FRAMES_DIR, sign)
    countdown_start = time.time()
    countdown_secs = 3
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Cannot read from webcam.")
            return False
        frame = cv2.flip(frame, 1)
        elapsed = time.time() - countdown_start
        remaining = countdown_secs - int(elapsed)
        if remaining <= 0:
            break
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        draw_text(frame, f"Sign: {sign}", y=30, color=(0, 255, 255), scale=0.9)
        draw_text(frame, f"Clip {clip_index + 1} / {cfg.CLIPS_PER_SIGN}", y=60, color=(200, 200, 200), scale=0.7)
        cx = frame.shape[1] // 2
        cy = frame.shape[0] // 2
        cv2.putText(frame, str(remaining), (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
        cv2.putText(frame, "Get ready!", (cx - 80, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Data Collection - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    frames_saved = 0
    while frames_saved < cfg.FRAMES_PER_CLIP:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.flip(frame, 1)
        resized = cv2.resize(frame, cfg.FRAME_SIZE)
        filename = f"{clip_index:04d}_{frames_saved:04d}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, resized)
        frames_saved += 1
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        draw_text(frame, f"RECORDING: {sign}", y=30, color=(0, 0, 255), scale=0.9)
        draw_text(frame, f"Clip {clip_index + 1}/{cfg.CLIPS_PER_SIGN}  Frame {frames_saved}/{cfg.FRAMES_PER_CLIP}", y=60, color=(200, 200, 200), scale=0.65)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
        cv2.imshow("Data Collection - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    print(f"    Clip {clip_index + 1} saved ({frames_saved} frames)")
    return True

def main():
    print("\n-- Phase 2: Data Collection --")
    print(f"   Signs        : {cfg.SIGNS}")
    print(f"   Clips/sign   : {cfg.CLIPS_PER_SIGN}")
    print(f"   Frames/clip  : {cfg.FRAMES_PER_CLIP}")
    print(f"   Save folder  : {cfg.DATA_FRAMES_DIR}")
    print("\n   Press Q anytime to quit and resume later.\n")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Webcam not found.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    warmup_camera(cap, frames=30)
    print("Webcam opened successfully!\n")
    for sign in cfg.SIGNS:
        existing = count_existing_clips(sign)
        if existing >= cfg.CLIPS_PER_SIGN:
            print(f"  {sign:12s} - already complete ({existing} clips)")
            continue
        print(f"\n  --> Now collecting: {sign}")
        print(f"      Progress: {existing}/{cfg.CLIPS_PER_SIGN} clips already done")
        print(f"      Perform the sign clearly. Vary speed and angle each clip.\n")
        for clip_idx in range(existing, cfg.CLIPS_PER_SIGN):
            success = collect_sign(cap, sign, clip_idx)
            if not success:
                print(f"\n  Stopped at {sign} clip {clip_idx}. Run again to resume.")
                cap.release()
                cv2.destroyAllWindows()
                return
        print(f"  {sign} COMPLETE - {cfg.CLIPS_PER_SIGN} clips saved.")
    cap.release()
    cv2.destroyAllWindows()
    print("\n-- Collection Complete! --")
    total = 0
    for sign in cfg.SIGNS:
        folder = os.path.join(PROJECT_ROOT, cfg.DATA_FRAMES_DIR, sign)
        count = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
        total += count
        print(f"   {sign:12s} : {count} frames")
    print(f"\n   Total frames: {total}")
    print("   Ready for Phase 3!\n")

main()