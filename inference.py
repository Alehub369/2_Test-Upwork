import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

def load_model(model_path='yolo11n.pt'):
    """
    Load the YOLO model using PyTorch
    """
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model
    model = YOLO(model_path)
    model.to(device)
    
    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(model).__name__}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model

def perform_inference(model, image_path):
    """
    Perform inference on a single image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Start timing
    start_time = time.time()
    
    # Perform inference
    results = model(image)
    
    # Calculate inference time
    inference_time = time.time() - start_time
    print(f"\nInference completed in {inference_time:.4f} seconds")
    
    # Process results
    for result in results:
        # Get bounding boxes
        boxes = result.boxes
        
        # Print detection information
        print("\nDetection Results:")
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"Detected: {class_name} with confidence: {confidence:.2f}")
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            print(f"Bounding box: ({x1:.1f}, {y1:.1f}), ({x2:.1f}, {y2:.1f})")
    
    # Draw results on image
    annotated_image = results[0].plot()
    
    # Save the annotated image
    output_path = 'inference_result.png'
    cv2.imwrite(output_path, annotated_image)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    # Load the model
    model = load_model()
    
    # Perform inference on the test image
    perform_inference(model, 'image-2.png') 