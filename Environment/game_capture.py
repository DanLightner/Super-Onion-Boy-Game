import mss
import numpy as np
import cv2
import pyautogui
import matplotlib.pyplot as plt

# Define the game window region
# Load the template images for "Game Over" and "You have Lost a Life!"
game_over_template = cv2.imread('game_over_template.png', cv2.IMREAD_GRAYSCALE)
lost_life_template = cv2.imread('lost_life_template.png', cv2.IMREAD_GRAYSCALE)


game_window = {'top': -690, 'left': 780, 'width': 340, 'height': 225}

# Amount to crop from each side
crop_pixels = 10  # Adjust as needed

# Adjusted window region
cropped_window = {
    'top': game_window['top'] + crop_pixels,
    'left': game_window['left'] + crop_pixels,
    'width': game_window['width'] - (2 * crop_pixels),
    'height': game_window['height'] - (5 * crop_pixels)
}

def capture_screen(region=cropped_window):
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
    onionboy = windows[0]  # Get the first matching window
    onionboy.resizeTo(340, 220)
    onionboy.moveTo(780, -710)
    onionboy.restore()
    pyautogui.moveTo(onionboy.left + onionboy.width / 2, onionboy.top + onionboy.height / 2)
    pyautogui.click()
    return {
        'top': onionboy.top,
        'left': onionboy.left,
        'width': onionboy.width,
        'height': onionboy.height
    }


game_window = detect_game_window()

#if game_window:
   #print("Game window positioned:", game_window)

# Test function
#if game_window:
    #print("Game window detected:", game_window)


def detect_game_over(cropped_window):
    # Define the region to capture (top portion of the game window)
    region = {
        'top': cropped_window['top'],  # Top of the window
        'left': cropped_window['left'],  # Left of the window
        'width': cropped_window['width'],  # Full width of the window
        'height': 50  # Limit height to capture just the top portion (adjust as needed)
    }

    #need to edit this later - kind of inefficient at the moment
    with mss.mss() as sct:
        screenshot = sct.grab(region)  # Capture only the top portion
        img = np.array(screenshot)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)  # Convert to grayscale for matching

        # Display the captured image for debugging purposes
        cv2.imshow("Captured Image", img)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()

        # Match the "Game Over" template
        result_game_over = cv2.matchTemplate(img_gray, game_over_template, cv2.TM_CCOEFF_NORMED)
        result_lost_life = cv2.matchTemplate(img_gray, lost_life_template, cv2.TM_CCOEFF_NORMED)

        # Define a threshold for detecting the templates
        threshold = 0.8  # Adjust this value based on your tests (higher = more strict)

        # Check if any region matches the template with a confidence higher than the threshold
        if np.any(result_game_over >= threshold) or np.any(result_lost_life >= threshold):
            return True
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