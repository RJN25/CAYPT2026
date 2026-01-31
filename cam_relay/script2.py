import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("slomo.mov")

max_height_pixel = 0
frame_count = 0

# Read first frame as background reference
ret, bg = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    diff = cv2.absdiff(bg_gray, gray)

    # Threshold to isolate water jet
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Clean noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find jet pixels
    ys, xs = np.where(thresh > 0)

    if len(ys) > 0:
        highest_jet_pixel = frame.shape[0] - np.min(ys)
        max_height_pixel = max(max_height_pixel, highest_jet_pixel)

    frame_count += 1

cap.release()

print(f"Maximum jet height (pixels): {max_height_pixel}")
