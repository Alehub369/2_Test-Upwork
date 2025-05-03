from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path

def load_custom_yolo():
    # Get the absolute path to the model file
    model_path = os.path.abspath('yolo11n.pt')
    print(f"Attempting to load YOLOv11 model from: {model_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"Error: File not found at {model_path}")
            return None
            
        # Load the model using ultralytics
        print("Loading YOLOv11 model...")
        model = YOLO(model_path)
        print("YOLOv11 model loaded successfully!")
        
        # Print model information
        print("\nModel Information:")
        print(f"Model type: {model.type}")
        print(f"Model task: {model.task}")
        print(f"Model names: {model.names}")
        
        # Print YOLOv11 specific architecture details
        print("\nYOLOv11 Architecture Details:")
        backbone = model.model.model
        print(f"Backbone layers: {len(backbone)}")
        
        # Get input size from first conv layer
        input_size = backbone[0].conv.weight.shape[2:]
        print(f"Input size: {input_size}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Get model size
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # Size in MB
        print(f"Model size: {model_size:.2f} MB")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None

def run_inference(model, image_path):
    """Run inference on an image using the YOLOv11 model"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
            
        print(f"\nRunning YOLOv11 inference on: {image_path}")
        results = model(image_path, conf=0.25)  # Set confidence threshold
        
        # Process results
        for result in results:
            # Get the image with detections
            img = result.plot()
            
            # Save the result
            output_path = "detection_result.jpg"
            cv2.imwrite(output_path, img)
            print(f"Detection result saved to: {output_path}")
            
            # Print detection information
            print("\nDetection Results:")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"Detected {model.names[cls]} with confidence {conf:.2f}")
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"Bounding box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        
        return results
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

if __name__ == "__main__":
    # Load the model
    model = load_custom_yolo()
    
    if model is not None:
        # Run inference on the test image
        image_path = "image-2.png"
        results = run_inference(model, image_path) 