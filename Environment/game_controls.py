import pyautogui
import time


# Disable pyautogui's fail-safe (optional, but useful for development)
# pyautogui.FAILSAFE = False

def press_key(key, duration=0.1):
    """
    Press a key for the specified duration.

    Args:
        key (str): The key to press
        duration (float): How long to hold the key down in seconds
    """
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)


def move_left():
    """Move the character to the left."""
    press_key("left")


def move_right():
    """Move the character to the right."""
    press_key("right")


def jump():
    """Make the character jump."""
    press_key("space")


def attack():
    """Make the character attack."""
    press_key("x")


def do_nothing():
    """Do nothing for one time step."""
    time.sleep(0.1)


def perform_action(action_id):
    """
    Perform an action based on its ID.

    Args:
        action_id (int): ID of the action to perform
            0: Do nothing
            1: Move left
            2: Move right
            3: Jump
            4: Attack
    """
    actions = {
        0: do_nothing,
        1: move_left,
        2: move_right,
        3: jump,
        4: attack
    }

    if action_id in actions:
        actions[action_id]()
    else:
        print(f"Invalid action ID: {action_id}")


def reset_game():
    """
    Reset the game after a game over or when starting a new episode.
    """
    # Navigate game menus to restart
    # This is game-specific and might require a sequence of key presses
    press_key("esc")  # Example: press escape to open menu
    time.sleep(0.5)
    press_key("r")  # Example: press 'r' to restart