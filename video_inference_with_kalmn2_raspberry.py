import cv2
import numpy as np
from picamera2 import Picamera2
import hailo_platform
from draw_rectangle import  draw_boxes_with_nms



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
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        # Trust detections, but not completely
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5

        # Start with large uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100

        self.initialized = False # Track if the filter has a starting point

    def initialize_state(self, x, y):
        # Set initial position and zero velocity
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return 0, 0
        predicted = self.kf.predict()
        return int(predicted[0][0]), int(predicted[1][0])

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)

# 1. Load your trained model
model_path = "./training-11-5/best_fixed.hef"

# 2. Setup Video

video_path = "/home/zainab/m2-res_574p.mp4"
video_path = "/home/zainab/m2-res_480p_13.mp4"
video_path = "/home/zainab/m2-res_304p_1.mp4"

cap = cv2.VideoCapture(video_path)

# VideoWriter setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"fps = {fps}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("/home/zainab/m2-res_304p_1_inferenced.mp4", fourcc, fps, (frame_width, frame_height))
kf = KalmanFilter()
kalmn_threshold=40
# Tracking state
last_w, last_h = 0, 0
lost_frames = 0
all_trajectories = []
lost_frame_count=0

# --- Load HEF and configure device ---
hef = hailo_platform.HEF(model_path)

with hailo_platform.VDevice() as target:
    configure_params = hailo_platform.ConfigureParams.create_from_hef(
        hef,
        interface=hailo_platform.HailoStreamInterface.PCIe
    )
    network_group = target.configure(hef, configure_params)[0]

    from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams
    input_vstreams_params = InputVStreamParams.make_from_network_group(network_group)
    output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group)

    input_name = hef.get_input_vstream_infos()[0].name

    with network_group.activate():
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            print("Processing video...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 1. ONLY predict if the tracker has actually started tracking something
                if kf.initialized:
                    pred_x, pred_y = kf.predict()
                    

                # Resize to network input size
                resized_img = cv2.resize(frame, (640, 640))
                input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}

                # Run inference
                raw_result = infer_pipeline.infer(input_data)

                # Draw bounding boxes
                frame_with_boxes, filtered_boxes, final_detections = draw_boxes_with_nms(frame, raw_result)
                
                detection_found = False
                if len(filtered_boxes) > 0:
                    detection_found = True
                    lost_frame_count = 0
                    
                    best_det = max(final_detections, key=lambda x: x['confidence'])
                    confidence = best_det['confidence']
                    noise = 20 * (1.0 - confidence) + 1

                    kf.kf.measurementNoiseCov = np.array([
                        [noise, 0],
                        [0, noise]
                    ], dtype=np.float32)
                    
                    box = best_det['box']
                    obj_center_x = int(box[0] + (box[2] / 2))
                    obj_center_y = int(box[1] + (box[3] / 2))
                    
                    last_w = box[2] 
                    last_h = box[3]
                    
                    # If this is the first detection ever, initialize the filter state post
                    if not kf.initialized:
                        kf.initialize_state(obj_center_x, obj_center_y)
                    else:
                        kf.update(obj_center_x, obj_center_y)
                    
                    print(f"yolo = {obj_center_x}")

                # 2. Object Lost Phase (Only runs if we have a valid target initialized)
                if not detection_found: 
                    if kf.initialized:
                        lost_frame_count += 1
                        
                        if lost_frame_count < kalmn_threshold:
                           
                            
                            # Calculate the bounding box around the predicted center
                            px1 = int(pred_x - last_w / 2)
                            py1 = int(pred_y - last_h / 2)
                            px2 = int(pred_x + last_w / 2)
                            py2 = int(pred_y + last_h / 2)
                            
                            # Draw the prediction tracking rectangle (Red)
                            cv2.rectangle(frame_with_boxes, (px1, py1), (px2, py2), (0, 0, 255), 2)
                            cv2.putText(frame_with_boxes, f"LOST: {lost_frame_count}", (px1, py1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            # Reset filter tracking if object has been missing for too long
                            kf.initialized = False
                    else:
                        # Optional: Object hasn't been found yet since video started
                        pass

                # 3. SINGLE Write & Show Strategy
                # This ensures every frame is recorded exactly once regardless of whether it's a YOLO or Kalman frame
                out.write(frame_with_boxes)
                cv2.imshow("Video Detection", frame_with_boxes)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as output_video.mp4")


