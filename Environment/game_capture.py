import mss
import numpy as np
import cv2

def capture_screen(region=None):
    with mss.mss() as sct:
        screenshot = sct.grab(region) if region else sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale
        return img
