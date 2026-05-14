from gpiozero import Servo
from time import sleep
import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2
import hailo_platform
from draw_rectangle import draw_boxes_with_nms
from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams
import math
import serial

class KalmanFilter:
    def __init__(self):
        # Initialize Kalman Filter: 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        '''
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                             [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                            [0, 1, 0, 1], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        '''
      
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                             [0, 1, 0, 0]], np.float32)
        
        # Use a smaller delta time (e.g., 0.1) instead of 1 to dampen the velocity impact
        dt = 0.5 
        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], 
                                             [0, 1, 0, dt], 
                                             [0, 0, 1, 0], 
                                             [0, 0, 0, 1]], np.float32)

        # DAMPENING: Give position some noise, but give velocity VERY low noise
        # This forces the filter to believe the velocity is constant and changes slowly.
        self.kf.processNoiseCov = np.array([[1e-3, 0, 0, 0],
                                            [0, 1e-3, 0, 0],
                                            [0, 0, 5e-3, 0],  # Velocity X noise
                                            [0, 0, 0, 5e-3]], # Velocity Y noise
                                           dtype=np.float32)
        
        # Add Measurement Noise: Tells the filter "YOLO isn't 100% perfect"
        # Increasing this (e.g., to 0.1) will make the filter even smoother.
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
       

    def predict(self):
        predicted = self.kf.predict()
        return int(predicted[0]), int(predicted[1])

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)




lost_frame_count = 0
LOST_THRESHOLD =50
model_path = "./second_training/best_fixed.hef"
model_path = "./fifth_training/best_5.hef"
model_path="./six_training/best_fixed.hef"
model_path = "./training-11-5/best_fixed.hef"

hef = hailo_platform.HEF(model_path)
    

stop = False
lock = threading.Lock()
latest_frame = None
latest_detection = None

state = "SEARCHING"

##Frame and interesting region
frame_width = 1280  # Default initial value

center_threshold = 0.15 # 15% margin on either side of center

left_bound = frame_width * (0.5 - center_threshold)
right_bound = frame_width * (0.5 + center_threshold)

#########################Arduino
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2) # Wait for Arduino to reset after connection

detection_found = False 
kalmn_threshold = 30 #frames
kf = KalmanFilter()
last_w = 0
last_h = 0


def send_objectX(pixel):
    command = f"pixel_x:{pixel}\n"
    ser.write(command.encode('utf-8'))
    #print(f"pixel: {pixel}")

#def send_command(command):
    #ser.write(command.encode('utf-8'))
    #print(f"commmand: {command} sent")
    
    
last_command_sent = None  # Track last sent command

def send_if_changed(cmd, pixel_x):
    global last_command_sent
    #if cmd != last_command_sent:
        #send_command(cmd)
    if cmd == "STOP":
        command = f"Stop:{pixel_x}\n"
        ser.write(command.encode('utf-8'))
        #print(f"{command}")
    elif cmd == "GO":
        command = f"GO\n"
        ser.write(command.encode('utf-8'))
        #print(f"{command}")
            
       # last_command_sent = cmd
        
def camera_task():
    global latest_frame

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        raw={"size": picam2.sensor_resolution}

    )

    picam2.configure(config)
    picam2.start()


    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


        with lock:
            latest_frame = frame


def detection_task():
    global stop, obj_center_x,lost_frame_count, kalmn_threshold, detection_found, kf, last_w, last_h

    global latest_frame, latest_detection, state
    detection_counter = 0  # How many frames have we seen the object?
    threshold = 1
    
    

    with hailo_platform.VDevice() as target:
        configure_params = hailo_platform.ConfigureParams.create_from_hef(
            hef,
            interface=hailo_platform.HailoStreamInterface.PCIe
        )

        network_group = target.configure(hef, configure_params)[0]


        input_params = InputVStreamParams.make_from_network_group(network_group)
        output_params = OutputVStreamParams.make_from_network_group(network_group)

        input_name = hef.get_input_vstream_infos()[0].name

        with network_group.activate():
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                #print("Inference started at 10 FPS...")

                target_fps = 30
                frame_duration = 1.0 / target_fps
                
                
                while True:
                    start_time = time.perf_counter()

                    # 1. Grab the latest frame
                    with lock:
                        if latest_frame is None:
                            time.sleep(0.01) # Short sleep to prevent CPU pegging
                            continue
                        frame = latest_frame.copy()
                    
                     

                    # 2. Process and Infer
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Corrected: Frame is BGR from cv2
                    resized = cv2.resize(rgb, (640, 640))
                    input_data = {input_name: np.expand_dims(resized, axis=0).astype(np.uint8)}
                    
                    result = pipeline.infer(input_data)
                    output_frame, filtered_box, final_detections = draw_boxes_with_nms(frame, result)
                    
                    frame_h, frame_w, _ = frame.shape
                    frame_center_x = int(frame_w / 2)
                    frame_center_y = int(frame_h / 2)
                    
                    cv2.circle(output_frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), -1)
                    cv2.putText(output_frame, f"Frame Center: {frame_center_x}, {frame_center_y}", 
                                (frame_center_x + 10, frame_center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    
                    
                    detection_found = False
                    if len(filtered_box) > 0:
                        detection_found = True
                        
                        lost_frame_count  = 0
                        best_det = max(final_detections, key=lambda x: x['confidence'])
                        box = best_det['box']
                        obj_center_x = box[0] + (box[2] / 2)
                        obj_center_y = int(box[1] + (box[3] / 2))
                        
                        #print(box[0], box[1], box[2], box[3])                       
                        
                        
                        # 1. Calculate and cast to int
                        c_x = int(obj_center_x)
                        c_y = int(obj_center_y)
                        
                        # Tracking state
                        last_w = box[2] 
                        last_h = box[3]
                        kf.update(c_x, c_y)
                        #kf.kf.statePost[2] = 0
                        #kf.kf.statePost[3] = 0

                        # 2. Draw the circle
                        cv2.circle(output_frame, (c_x, c_y), 5, (0, 0, 255), -1)

                        # 3. Draw the text (also requires integer coordinates)
                        cv2.putText(output_frame, f"Obj: {c_x}, {c_y}", (c_x + 10, c_y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        

                        # 2. Draw Object Center (Red)
                        #cv2.circle(output_frame, (obj_center_x, obj_center_y), 5, (0, 0, 255), -1)
                        #cv2.putText(output_frame, f"Obj Center: {obj_center_x}, {obj_center_y}", 
                                    #(obj_center_x + 10, obj_center_y + 20), 
                                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        cv2.line(output_frame, (frame_center_x, frame_center_y), (c_x, c_y), (255, 255, 0), 2)
    
                        stop = True
                        send_if_changed('STOP', obj_center_x)
                        print(f"yolo = {obj_center_x}")

                        #frame_center = frame.shape[1] / 2
                        #error = obj_center_x - frame_center

                        
                    if not detection_found and lost_frame_count < LOST_THRESHOLD:
                        pred_x, pred_y = kf.predict()
                        lost_frame_count += 1
                        if lost_frame_count < kalmn_threshold:
                            
                            '''
                            k_x1 = int(pred_x - last_w/2)
                            k_y1 =int(pred_y - last_h/2)
                            k_x2 =int(pred_x + last_w/2)
                            k_y2 =int(pred_y + last_h/2)
                            
                            
                            cv2.rectangle(output_frame, (k_x1, k_y1), (k_x2, k_y2), (0, 0, 255), 2)
                            cv2.putText(output_frame, "Kalmn", (k_x1, k_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            '''
                            safe_x = float(np.clip(pred_x, 0, frame_w))
        
                            # 3. Draw for debugging
                            k_x1 = int(safe_x - last_w/2)
                            k_y1 = int(pred_y - last_h/2)
                            cv2.rectangle(output_frame, (k_x1, k_y1), (k_x1 + int(last_w), k_y1 + int(last_h)), (0, 0, 255), 2)
        
        
                            stop = True
                            send_if_changed('STOP', safe_x)
                            print(f"safe_x = {safe_x}")
                        
                            
                    if lost_frame_count >= LOST_THRESHOLD:
                        send_if_changed('GO', None)
                        print("Go")
                             
                             
                    '''        
                    else:
                        lost_frame_count += 1
                        if lost_frame_count < kalmn_threshold and initilize_kalmn:
                            #predict based on kalmn filter:
                            pred_x, pred_y = kf.predict()
                            
                            #kalmn bounding box:
                            
                            
                            cv2.rectangle(output_frame, (k_x1, k_y1), (k_x2, k_y2), (0, 0, 255), 2)
                            cv2.putText(output_frame, "Kalmn", (k_x1, k_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            
                            stop = True
                            send_if_changed('STOP', pred_x)
                        
                        # Only tell Arduino to start scanning if we've lost it for a while
                        if lost_frame_count >= LOST_THRESHOLD:
                            send_if_changed('GO', None)
                            initilize_kalmn = False
                        else:
                            # Optional: Send the LAST known position to keep the servo steady
                            # This prevents the "jitter" of switching between scan/track
                            pass
                        '''
                    
                    #ok 
                    
                    '''
                    if len(filtered_box) > 0:
                        best_det = max(final_detections, key=lambda x: x['confidence'])
                        box = best_det['box']
                        obj_center_x = box[0] + (box[2] / 2)

                        frame_center = frame.shape[1] / 2
                        error = obj_center_x - frame_center

                        Kp = 0.05
                        servo_angle -= Kp * error
                        servo_angle = max(0, min(180, servo_angle))

                        send_angle(int(servo_angle))
                    else:
                        # optional: sweep or do nothing
                        pass

                    '''
                    cv2.imshow("Detection", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    processing_time = time.perf_counter() - start_time
                    sleep_time = frame_duration - processing_time
                    if sleep_time > 0:
                        time.sleep(sleep_time)





t1 = threading.Thread(target=camera_task, daemon=True)
t2 = threading.Thread(target=detection_task, daemon=True)

t1.start()
t2.start()

t2.join()

