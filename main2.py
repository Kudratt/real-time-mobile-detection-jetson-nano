import cv2
import os
import csv
import time
from datetime import datetime

SAVE_DIR = "violations"
LOG_FILE = "logs.csv"
CAM_INDEX = 0
WINDOW_NAME = "AI Surveillance System"
ALERT_COOLDOWN = 3

os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Image"])

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(3, 960)
cap.set(4, 540)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
last_alert = 0

ZONE_X1, ZONE_Y1 = 220, 100
ZONE_X2, ZONE_Y2 = 740, 450

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected. Restarting...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(CAM_INDEX)
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    person_found = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if x > ZONE_X1 and y > ZONE_Y1 and x+w < ZONE_X2 and y+h < ZONE_Y2:
            person_found = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Person Detected", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.rectangle(frame, (ZONE_X1, ZONE_Y1), (ZONE_X2, ZONE_Y2), (255, 0, 0), 2)
    cv2.putText(frame, "Restricted Zone", (ZONE_X1, ZONE_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if person_found:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: UNAUTHORIZED MOVEMENT DETECTED",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)

        if time.time() - last_alert > ALERT_COOLDOWN:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img_name = os.path.join(SAVE_DIR, f"violation_{ts}.jpg")
            cv2.imwrite(img_name, frame)

            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, "Restricted zone violation", img_name])

            last_alert = time.time()

    cv2.rectangle(frame, (0, frame.shape[0]-35), (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
    cv2.putText(frame, "AI Surveillance Monitor", (15, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(frame, now, (720, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.circle(frame, (900, 20), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (915, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()