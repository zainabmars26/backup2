import cv2
import numpy as np
from ultralytics import YOLO

class KalmanFilter:
    def __init__(self):
        # Initialize Kalman Filter: 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                             [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                            [0, 1, 0, 1], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self):
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

# 1. Load your trained model
model = YOLO(r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\runs_300e_10_5\detect\train\weights\best.pt") 

# 2. Setup Video
cap = cv2.VideoCapture(r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\video_dataset\day\video4.mp4")
# Get video properties for the writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(r'C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset/video4.mp4', fourcc, fps, (frame_width, frame_height))

kf = KalmanFilter()

# Tracking state
last_w, last_h = 0, 0
lost_frames = 0
trajectory_yolo = []
trajectory_kalmn = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Kalman Predict Phase
    pred_x, pred_y = kf.predict()
    
    # YOLO Detect Phase
    results = model(frame, conf=0.5, verbose=False)
    detection_found = False
    
    for r in results[0].boxes:
        # We take the first detected object of interest
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        last_w, last_h = (x2 - x1), (y2 - y1)
        
        # Update Kalman with real detection
        kf.update(cx, cy)
        
        # Draw YOLO Box (Green)
        trajectory_yolo.append((pred_x, pred_y, (0, 255, 0)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, "YOLO", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detection_found = True
        lost_frames = 0
        break # Tracking one specific object

    if not detection_found:
        lost_frames += 1
        # Only predict if we haven't been lost for too long (e.g., 50 frames)
        if lost_frames < 50:
            # Calculate box based on Kalman Prediction
            px1 = int(pred_x - last_w/2)
            py1 = int(pred_y - last_h/2)
            px2 = int(pred_x + last_w/2)
            py2 = int(pred_y + last_h/2)
            
            # Draw Predicted Box (Red)
            trajectory_kalmn.append((pred_x, pred_y, (0, 0, 255)))

            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(frame, "KALMAN PREDICT", (px1, py1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- DRAW the trajectory ---
    for point in trajectory_yolo:
        cv2.circle(frame, (point[0], point[1]), 3, point[2], -1)
        
    for point in trajectory_kalmn:
        cv2.circle(frame, (point[0], point[1]), 3, point[2], -1)


    out.write(frame)
    #cv2.imshow("YOLO + Kalman Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release() # CRITICAL: If you don't release, the video file will be corrupted
cv2.destroyAllWindows()
