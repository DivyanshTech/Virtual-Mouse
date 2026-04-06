import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time  # For unique filename

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
smooth_factor = 0.4

# ================= Video Recording Setup =================
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Works on Windows
filename = f"virtual_mouse_{int(time.time())}.avi"
out = cv2.VideoWriter(filename, fourcc, 20.0, (640,480))
print(f"🎥 Recording will be saved as {filename}")
# =========================================================

cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)
cv2.resizeWindow("Virtual Mouse", 300, 200)
cv2.moveWindow("Virtual Mouse", 10, 10)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Record resized frame
    frame_to_save = cv2.resize(img, (640,480))
    out.write(frame_to_save)

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        draw.draw_landmarks(img, lm)

        x_index = int(lm.landmark[8].x * w)
        y_index = int(lm.landmark[8].y * h)

        screen_x = np.interp(x_index, (0, w), (0, screen_w))
        screen_y = np.interp(y_index, (0, h), (0, screen_h))

        curr_x = prev_x + (screen_x - prev_x) * smooth_factor
        curr_y = prev_y + (screen_y - prev_y) * smooth_factor

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        x_thumb = int(lm.landmark[4].x * w)
        y_thumb = int(lm.landmark[4].y * h)

        distance = np.hypot(x_index - x_thumb, y_index - y_thumb)

        if distance < 40:
            cv2.putText(img, "Click", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pyautogui.click()

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # ✅ Video save
cv2.destroyAllWindows()