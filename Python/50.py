
import pyautogui
import cv2
import numpy as np
import time

# Define target object image
TARGET_IMAGE = "target.png"  # Replace with actual object image

# Find target on screen
def locate_target():
    screen = pyautogui.screenshot()
    screen_np = np.array(screen)
    screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)
    target = cv2.imread(TARGET_IMAGE, cv2.IMREAD_GRAYSCALE)

    result = cv2.matchTemplate(screen_gray, target, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > 0.8:
        return max_loc
    return None

# Perform automated action
def perform_action():
    target_pos = locate_target()
    if target_pos:
        x, y = target_pos
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.click()
        print(f"Clicked at {x}, {y}")

if __name__ == "__main__":
    print("Starting game bot...")
    while True:
        perform_action()
        time.sleep(1)  # Adjust action frequency
