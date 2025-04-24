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
    
    # Get image dimensions
    height, width = image.shape[:2]
    print(f"\nImage dimensions: {width}x{height}")
    
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
        for i, box in enumerate(boxes, 1):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            print(f"\nDetection {i}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Bounding Box:")
            print(f"    Top-left: ({x1:.1f}, {y1:.1f})")
            print(f"    Bottom-right: ({x2:.1f}, {y2:.1f})")
            print(f"    Width: {box_width:.1f}px")
            print(f"    Height: {box_height:.1f}px")
            print(f"    Area: {box_width * box_height:.1f}px²")
    
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