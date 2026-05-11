import os
from ultralytics import YOLO

# --- Configuration ---
model_path = r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\best.pt"     
input_folder = r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\output\split_data_random\folder_3\images"
output_labels_path = r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\video_dataset\output\split_data_random\folder_3\labels"
conf_threshold = 0.35

# Ensure output directory exists
os.makedirs(output_labels_path, exist_ok=True)

# Load your custom model
model = YOLO(model_path)

# Supported image extensions
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(img_extensions)]

print(f"Found {len(image_files)} images. Starting prediction...")

for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    
    # Run prediction
    results = model.predict(img_path, conf=conf_threshold, save=False)
    
    # Define the output text file path (same name as image, but .txt)
    file_id = os.path.splitext(img_name)[0]
    txt_path = os.path.join(output_labels_path, f"{file_id}.txt")

    # Access the first result (since we are passing one image at a time)
    result = results[0]

    if len(result.boxes) > 0:
        # Save standard YOLO format labels
        # Note: save_txt creates a folder named 'labels' by default, 
        # so we manually handle the path for better control.
        result.save_txt(txt_path)
    else:
        # No detections: Create an empty text file for negative samples
        with open(txt_path, 'w') as f:
            pass 

print(f"Processing complete. Labels saved to: {output_labels_path}")
