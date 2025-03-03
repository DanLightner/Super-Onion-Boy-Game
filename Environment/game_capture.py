import mss
import numpy as np
import cv2
import pyautogui
import matplotlib.pyplot as plt


def capture_screen(region=None):
    """
    Capture a screenshot of the game window.

    Args:
        region (dict, optional): Dictionary with keys 'top', 'left', 'width', 'height' defining the region to capture.
                                If None, captures the primary monitor.

    Returns:
        numpy.ndarray: Grayscale image of the captured region.
    """
    with mss.mss() as sct:
        # Use the specified region or default to the primary monitor
        screenshot = sct.grab(region) if region else sct.grab(sct.monitors[2])
        # Convert screenshot to numpy array
        img = np.array(screenshot)
        # Convert from BGRA to grayscale for simpler processing
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        # Display the image using OpenCV
        cv2.imshow("Captured Screen", img)
        cv2.waitKey(0)  # Press any key to close the window
        cv2.destroyAllWindows()
        return img

# Call the function and verify it captures the screen
#captured_image = capture_screen()

# Check the shape and type of the output
#print("Image Type:", type(captured_image))
#print("Image Shape:", captured_image.shape)

def detect_game_window():
    """
    Automatically detect the Super Onion Boy game window.

    Returns:
        dict: Region dictionary with 'top', 'left', 'width', 'height' or None if not found
    """

    # Get the game window
    windows = pyautogui.getWindowsWithTitle("Super Onion Boy")

    if not windows:  # Check if the list is empty
        print("Error: Super Onion Boy window not found.")
        return None
    else:
        return "it worked"


    onionboy = windows[0]  # Get the first matching window

    onionboy.resizeTo(322, 221)
    onionboy.moveTo(
        779, -711)
    # Move the cursor to the top-left corner of the game window
    pyautogui.moveTo(onionboy.left + onionboy.width / 2, onionboy.top + onionboy.height / 2)
    onionboy.click()


    return {
        'top': onionboy.top,
        'left': onionboy.left,
        'width': onionboy.width,
        'height': onionboy.height
    }


game_window = detect_game_window()

if game_window:
    print("Game window positioned:", game_window)

# Test function
if game_window:
    print("Game window detected:", game_window)

def detect_game_over():
    """
    Detect if the game is in a 'game over' state.

    Returns:
        bool: True if game over screen is detected, False otherwise
    """
    # TODO: Implement game over detection
    # This could use template matching or specific pixel patterns
    return False


def detect_player(frame):
    """
    Locate the player character in the current frame.

    Args:
        frame (numpy.ndarray): The current game frame

    Returns:
        tuple: (x, y) coordinates of the player or None if not found
    """
    # TODO: Implement player detection
    # Could use template matching, contour detection, or other CV techniques
    return None