import cv2
import numpy as np
from collections import deque


def preprocess_frame(frame):
    """
    Preprocess a raw game frame for input to the DQN.

    Args:
        frame (numpy.ndarray): Raw grayscale frame from game capture

    Returns:
        numpy.ndarray: Preprocessed frame (84x84, normalized)
    """
    # Resize to 84x84 (standard size for DQN)
    resized = cv2.resize(frame, (84, 84))

    # Normalize pixel values to range [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def stack_frames(frame_stack, new_frame, is_new_episode=False):
    """
    Stack frames for temporal information.

    Args:
        frame_stack (deque): Current stack of frames
        new_frame (numpy.ndarray): New frame to add to stack
        is_new_episode (bool): Whether this is the start of a new episode

    Returns:
        deque: Updated frame stack
    """
    # For new episodes, create a stack with the same frame repeated
    if is_new_episode or frame_stack is None:
        frame_stack = deque([new_frame for _ in range(4)], maxlen=4)
    else:
        # Otherwise, add the new frame to the stack
        frame_stack.append(new_frame)

    return frame_stack


def frame_stack_to_tensor(frame_stack):
    """
    Convert a frame stack to a tensor suitable for network input.

    Args:
        frame_stack (deque): Stack of preprocessed frames

    Returns:
        numpy.ndarray: Stacked frames as a single array
    """
    # Stack the frames along the first dimension (channel dimension)
    return np.array(frame_stack)