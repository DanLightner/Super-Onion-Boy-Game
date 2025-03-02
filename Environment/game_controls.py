import pyautogui
import time

def press_key(key, duration=0.1):
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

def move_left():
    press_key("left")

def move_right():
    press_key("right")

def jump():
    press_key("space")

def attack():
    press_key("x")
