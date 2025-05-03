from ultralytics import YOLO
import cv2
import os

# Load the YOLO model
model = YOLO('yolo11n.pt')

# Load the image
image_path = 'image-2.png'
image = cv2.imread(image_path)

# Perform detection
results = model(image)

# Process and display results
for result in results:
    # Draw bounding boxes on the image
    annotated_image = result.plot()
    
    # Save the annotated image
    output_path = 'detected_image.png'
    cv2.imwrite(output_path, annotated_image)
    print(f"Results saved to {output_path}")
    
    # Print detection information
    print("\nDetection Results:")
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]
        print(f"Detected: {class_name} with confidence: {confidence:.2f}") 