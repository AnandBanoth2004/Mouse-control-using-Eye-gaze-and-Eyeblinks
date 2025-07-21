import cv2
import mediapipe as mp
import pyautogui
import time
import sys

# Starting the camera
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

cv2.namedWindow("Eye Controlled Mouse", cv2.WINDOW_NORMAL)

# Measuring response time 
gesture_stats = {
    "Cursor":      {"attempts": 0, "success": 0, "response": 0},
    "Left_Click":  {"attempts": 0, "success": 0, "response": 0},
    "Right_Click": {"attempts": 0, "success": 0, "response": 0},
    "Scroll_Up":   {"attempts": 0, "success": 0, "response": 0},
    "Scroll_Down": {"attempts": 0, "success": 0, "response": 0}
}
last_left_state  = False
last_right_state = False
last_click_time  = time.time()
last_scroll_time = time.time()
last_stat_print  = time.time()

#values for clicking and exiting 
SCROLL_THRESHOLD = 0.03      
DEBOUNCE_TIME    = 0.5       
MOUTH_OPEN_THRESHOLD = 0.03  

#function that updates the stats
def update_stats(gesture, success, start_ts):
    gesture_stats[gesture]["attempts"] += 1
    if success:
        gesture_stats[gesture]["success"] += 1
        gesture_stats[gesture]["response"] += (time.time() - start_ts) * 1000

#printing the stats
def print_stats():
    print("\nGesture Accuracy Report:")
    for key, s in gesture_stats.items():
        at, su = s["attempts"], s["success"]
        if at > 0:
            acc = su / at * 100
            avg = (s["response"] / su) if su > 0 else 0
            print(f"  {key:12s}  Acc: {acc:5.1f}% ({su}/{at})   Avg resp: {avg:.2f} ms")
        else:
            print(f"  {key:12s}  No attempts yet")
    print("-" * 50)

# Main loop
print("Eye Controlled Mouse is active")
print("Open your mouth wide to exit the program")
print("-" * 50)

start_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = face_mesh.process(rgb)
    h, w, _ = frame.shape

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        #checks if the mouth is open
        mouth_open = False
        try:
            if (lm[14].y - lm[13].y) > MOUTH_OPEN_THRESHOLD:
                mouth_open = True
                print("Mouth open detected - exiting program")
                break
        except:
            pass  #error handling for open mouth detection

        for idx in [145, 159, 374, 386]:
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

        # Moving cursor with respect to the position of the user's eye landmarks
        t0 = time.time()
        gaze = lm[477]
        screen_x, screen_y = gaze.x * screen_w, gaze.y * screen_h
        pyautogui.moveTo(screen_x, screen_y)
        update_stats("Cursor", True, t0)
        cv2.circle(frame, (int(gaze.x*w), int(gaze.y*h)), 3, (0,255,0), -1)

        # Left click (left eye wink)
        left_closed = (lm[145].y - lm[159].y) < 0.007
        if left_closed and not last_left_state and (time.time()-last_click_time)>DEBOUNCE_TIME:
            t0 = time.time()
            pyautogui.click()
            update_stats("Left_Click", True, t0)
            last_click_time = time.time()
        last_left_state = left_closed

        # Right click (right eye wink)
        right_closed = (lm[374].y - lm[386].y) < 0.008
        if right_closed and not last_right_state and (time.time()-last_click_time)>DEBOUNCE_TIME:
            t0 = time.time()
            pyautogui.rightClick()
            update_stats("Right_Click", True, t0)
            last_click_time = time.time()
        last_right_state = right_closed

        # Scroll up/down
        if (time.time() - last_scroll_time) > DEBOUNCE_TIME:
            tilt = lm[374].y - lm[145].y
            if abs(tilt) > SCROLL_THRESHOLD:
                t0 = time.time()
                if tilt < 0:
                    pyautogui.scroll(300)
                    update_stats("Scroll_Up", True, t0)
                else:
                    pyautogui.scroll(-300)
                    update_stats("Scroll_Down", True, t0)
                last_scroll_time = time.time()

    # Add instructions to frame
    cv2.putText(frame, "Open mouth to exit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Eye Controlled Mouse", frame)

    # Stats tracking
    if (time.time() - last_stat_print) > 15:
        print_stats()
        last_stat_print = time.time()

    # Try to handle key events
    key = cv2.waitKey(1) & 0xFF
    if key != 255:  # If any key is pressed (not just q)
        print(f"Key detected: {key}")
        break

    # Alternative exit method - if run for 5 minutes, automatically exit
    if time.time() - start_time > 300:  
        print("Session time limit reached - exiting")
        break

# Cleanup
print("\nFinal Accuracy Report:")
print_stats()
cam.release()
cv2.destroyAllWindows()