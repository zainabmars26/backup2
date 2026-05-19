import cv2
import numpy as np

calsses = ['Drone']

'''def draw_boxes(image, raw_results, threshold=0.5):
    h, w, _ = image.shape
    
    # 1. Focus on the 20x20 layer as a starting point
    # We MUST convert to float32 to avoid OverflowErrors
    boxes_raw = raw_results['best/conv91'][0].astype(np.float32) 
    scores_raw = raw_results['best/conv94'][0].astype(np.float32)

    # 2. De-quantize if the values look like large integers (0-255)
    # If your values are already 0.0 to 1.0, you can skip this.
    # If they are 0-255, we normalize them:
    if boxes_raw.max() > 1.0:
        boxes_raw /= 255.0
    if scores_raw.max() > 1.0:
        scores_raw /= 255.0

    for y in range(20):
        for x in range(20):
            class_scores = scores_raw[y, x]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence > threshold:
                box = boxes_raw[y, x]
                
                # YOLO coordinates are often: [center_x, center_y, width, height]
                # relative to the cell. 
                # This is a standard transformation:
                center_x = int(((x + box[0]) / 20) * w)
                center_y = int(((y + box[1]) / 20) * h)
                
                # We use float math first, then cast to int at the very end
                width  = int(box[2] * w)
                height = int(box[3] * h)

                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)

                # Draw with safe integer coordinates
                cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
                
    return image'''

def draw_boxes_with_nms(image, raw_results, conf_threshold=0.98, iou_threshold=0.3):
    h, w, _ = image.shape
    all_boxes = []
    all_scores = []
    all_class_ids = []

    # Iterate through your 3 scales
    # Scales: (20, 20), (40, 40), (80, 80)
    scales = [
        ('best_fixed/conv91', 'best_fixed/conv94', 20),
        ('best_fixed/conv77', 'best_fixed/conv80', 40),
        ('best_fixed/conv61', 'best_fixed/conv64', 80)

    ]

    #for box_layer, score_layer, grid_size in scales:
    for box_layer, score_layer, grid_size in scales:
        boxes_raw = raw_results[box_layer][0].astype(np.float32) / 255.0
        scores_raw = raw_results[score_layer][0].astype(np.float32) / 255.0
        #print(f"Layer {score_layer} shape: {scores_raw.shape}")
        #print(f"Sample value at [0,0]: {scores_raw[0, 0]}")
        for y in range(grid_size):
            for x in range(grid_size):
                val = scores_raw[y, x]
                
                # Handle single-class vs multi-class shape
                if isinstance(val, np.ndarray):
                    class_id = int(np.argmax(val))
                    conf = float(val[class_id])
                else:
                    class_id = 0
                    conf = float(val)

                if conf > conf_threshold:
                    box = boxes_raw[y, x]
                    
                    # 1. Decode to pixel coordinates (Adjust math to your YOLO version)
                    center_x = int(((x + box[0]) / grid_size) * w)
                    center_y = int(((y + box[1]) / grid_size) * h)
                    box_w    = int(box[2] * w)
                    box_h    = int(box[3] * h)
                    
                    # OpenCV NMS expects [x_top_left, y_top_left, width, height]
                    left = int(center_x - box_w / 2)
                    top  = int(center_y - box_h / 2)

                    all_boxes.append([left, top, box_w, box_h])
                    all_scores.append(float(conf))
                    all_class_ids.append(class_id)

    # 2. RUN THE NMS FILTER
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_threshold, iou_threshold)

    # 3. DRAW ONLY THE SURVIVORS
    final_detections = []

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = all_boxes[i]
            label = f" {calsses[all_class_ids[i]]}: {all_scores[i]:.2f}"
            
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # store everything together
            final_detections.append({
                "box": [x, y, bw, bh],
                "confidence": all_scores[i],
                "class_id": class_id,
                "label": calsses[class_id]
            })

    return image, all_boxes, final_detections
'''
if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = all_boxes[i]
            
            # --- CUSTOM DRAWING LOGIC ---
            color = (0, 255, 0)  # Green color (BGR)
            thickness = 2
            
            # 1. Define corner line lengths (e.g., 20% of the width/height or a fixed pixel length)
            line_len_x = int(bw * 0.2)
            line_len_y = int(bh * 0.2)
            
            # Top-Left Corner
            cv2.line(image, (x, y), (x + line_len_x, y), color, thickness)
            cv2.line(image, (x, y), (x, y + line_len_y), color, thickness)
            
            # Top-Right Corner
            cv2.line(image, (x + bw, y), (x + bw - line_len_x, y), color, thickness)
            cv2.line(image, (x + bw, y), (x + bw, y + line_len_y), color, thickness)
            
            # Bottom-Left Corner
            cv2.line(image, (x, y + bh), (x + line_len_x, y + bh), color, thickness)
            cv2.line(image, (x, y + bh), (x, y + bh - line_len_y), color, thickness)
            
            # Bottom-Right Corner
            cv2.line(image, (x + bw, y + bh), (x + bw - line_len_x, y + bh), color, thickness)
            cv2.line(image, (x + bw, y + bh), (x + bw, y + bh - line_len_y), color, thickness)
            
            # 2. Draw Crosshair Ticks (Top, Bottom, Left, Right)
            center_x = x + int(bw / 2)
            center_y = y + int(bh / 2)
            tick_len = int(min(bw, bh) * 0.15) # Length of each crosshair line
            gap = int(min(bw, bh) * 0.05)       # Small center gap so lines don't touch in the middle
            
            # Top tick
            cv2.line(image, (center_x, y), (center_x, center_y - gap), color, thickness)
            # Bottom tick
            cv2.line(image, (center_x, y + bh), (center_x, center_y + gap), color, thickness)
            # Left tick
            cv2.line(image, (x, center_y), (center_x - gap, center_y), color, thickness)
            # Right tick
            cv2.line(image, (x + bw, center_y), (center_x + gap, center_y), color, thickness)
            # ----------------------------

            # Typo fix from your original snippet: changed 'calsses' to 'classes' (assuming you have a classes list)
            # If your variable is actually spelled 'calsses', change it back below!
            current_class_id = all_class_ids[i]
            label = f" {classes[current_class_id]}: {all_scores[i]:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            final_detections.append({
                "box": [x, y, bw, bh],
                "confidence": all_scores[i],
                "class_id": current_class_id,
                "label": classes[current_class_id]
            })
'''
# Add this return to your draw_boxes_with_nms function
def get_detection_center(image, raw_results, conf_threshold=0.98):
    h, w, _ = image.shape
    # ... (your existing NMS logic) ...
    
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_threshold, 0.4)
    
    if len(indices) > 0:
        # Take the first (highest confidence) object
        i = indices.flatten()[0]
        x, y, bw, bh = all_boxes[i]
        center_x = x + (bw // 2)
        return center_x, image
    
    return None, image


def draw_boxes_with_nms2(image, raw_results, conf_threshold=0.06, iou_threshold=0.4):
    h, w, _ = image.shape
    
    output = raw_results['best/format_conversion13'][0][0]  # (5, 8400)
    output = output.astype(np.float32)              # normalize to 0-1
    
    boxes = []
    scores = []
    
    for i in range(output.shape[1]):   # 8400 detections
        x1, y1, x2, y2, conf = output[:, i]
        
        if conf < conf_threshold:
            continue
        
        # Convert normalized coords to pixels
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)
        bw  = px2 - px1
        bh  = py2 - py1
        
        if bw <= 0 or bh <= 0:
            continue
        
        boxes.append([px1, py1, bw, bh])
        scores.append(float(conf))
    
    print(f"Boxes before NMS: {len(boxes)}")
    
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = f"Drone: {scores[i]:.2f}"
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image
