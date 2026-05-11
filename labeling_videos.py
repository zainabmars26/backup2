import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
model_path = r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\best.pt"
input_folder = r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\video_dataset\day"
#input_folder = r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\New folder"
output_images_path = r'C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\video_dataset\output_/images'
output_labels_path = r'C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\video_dataset\output_/labels'
frame_stride = 5  # Change this to save every Nth frame (e.g., 1 for every frame)

# Create output directories
os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

# Load your custom model
model = YOLO(model_path)
print(model.names)
# Process folder
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]

for video_name in video_files:
    video_path = os.path.join(input_folder, video_name)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    video_base_name = os.path.splitext(video_name)[0]

    print(f"Processing: {video_name}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process frame based on the stride
        if frame_count % frame_stride == 0:
            # Run inference
            #results = model(frame, conf=0.25)  # Adjust confidence threshold as needed
            # 'classes=[0]' tells the model to ONLY report detections for your first class
            results = model(frame, conf=0.35)
            # Define naming convention: videoName_frame0001
            file_id = f"{video_base_name}_frame{saved_count:04d}"
            img_filename = f"{file_id}.jpg"
            txt_filename = f"{file_id}.txt"

            # Save the frame as an image
            cv2.imwrite(os.path.join(output_images_path, img_filename), frame)

            # Save predictions in YOLO .txt format (class x_center y_center width height)
           # results[0].save_txt(os.path.join(output_labels_path, txt_filename))
            txt_path = os.path.join(output_labels_path, f"{file_id}.txt")

            # Check if there are any detections
            if len(results[0].boxes) > 0:
                # Detections found: Save standard YOLO format
                results[0].save_txt(txt_path)
            else:
                # No detections: Create an empty text file
                with open(txt_path, 'w') as f:
                    pass  # Just create an empty file
                        
            saved_count += 1
        
        frame_count += 1

    cap.release()

print("Labeling complete.")
