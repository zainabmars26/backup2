import cv2
import numpy as np
from picamera2 import Picamera2
import hailo_platform
from draw_rectangle import draw_boxes_with_nms
import time

model_path="./second_training/best_fixed.hef"
model_path="./fifth_training/best_5.hef"
model_path="./six_training/best_fixed.hef"

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

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        raw={"size": picam2.sensor_resolution}

    )

    picam2.configure(config)
    picam2.start()

    with network_group.activate():
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_name = hef.get_input_vstream_infos()[0].name
            
            # --- FPS Control Setup ---
            target_fps = 10
            frame_duration = 1.0 / target_fps  # 0.1 seconds per frame
            # -------------------------

            print(f"Starting real-time camera inference at {target_fps} FPS...")
            while True:
                start_time = time.time()  # Record start of loop

                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                resized_img = cv2.resize(frame, (640, 640))
                input_data = {input_name: np.expand_dims(resized_img, axis=0).astype(np.uint8)}             
                
                result = infer_pipeline.infer(input_data)
                output_frame, filtered_box, final_detections = draw_boxes_with_nms(frame, result)
                
                # ... [Keep your drawing/logic code here] ...
                frame_h, frame_w, _ = frame.shape
                frame_center_x = int(frame_w / 2)
                frame_center_y = int(frame_h / 2)
                cv2.circle(output_frame, (frame_center_x, frame_center_y), 5, (0, 255, 0), -1)
                cv2.putText(output_frame, f"Frame Center: {frame_center_x}, {frame_center_y}", 
                            (frame_center_x + 10, frame_center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if len(filtered_box) > 0:
                    best_det = max(final_detections, key=lambda x: x['confidence'])
                    box = best_det['box']
                    obj_center_x = box[0] + (box[2] / 2)
                    obj_center_y = int(box[1] + (box[3] / 2))
                    
                    
                    
                    # 1. Calculate and cast to int
                    c_x = int(obj_center_x)
                    c_y = int(obj_center_y)

                    # 2. Draw the circle
                    cv2.circle(output_frame, (c_x, c_y), 5, (0, 0, 255), -1)

                    # 3. Draw the text (also requires integer coordinates)
                    cv2.putText(output_frame, f"Obj: {c_x}, {c_y}", (c_x + 10, c_y + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    

                  
                    
                    cv2.line(output_frame, (frame_center_x, frame_center_y), (c_x, c_y), (255, 255, 0), 2)


                cv2.imshow("Real-time Detection", output_frame)

                # --- FPS Control Logic ---
                elapsed_time = time.time() - start_time
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                # -------------------------

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()


