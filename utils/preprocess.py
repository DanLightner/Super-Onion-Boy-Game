import cv2
import numpy as np

def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame
