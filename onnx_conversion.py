import torch
import torchvision
import numpy as np
from PIL import Image
import os

def convert_to_onnx(model_path, output_path, input_shape=(1, 3, 640, 640)):
    """
    Convert a YOLO model to ONNX format using PyTorch's built-in ONNX export.
    
    Args:
        model_path (str): Path to the YOLO model file
        output_path (str): Path to save the ONNX model
        input_shape (tuple): Input shape for the model (batch_size, channels, height, width)
    """
    try:
        # Load the YOLO model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.eval()
        
        # Create a dummy input tensor
        dummy_input = torch.randn(input_shape)
        
        # Export the model to ONNX format
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"Model successfully converted to ONNX format and saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error during ONNX conversion: {str(e)}")
        return False

def verify_onnx_model(model_path, image_path):
    """
    Verify the ONNX model by running inference on a test image.
    
    Args:
        model_path (str): Path to the ONNX model file
        image_path (str): Path to the test image
    """
    try:
        # Load the ONNX model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.eval()
        
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((640, 640))
        img = np.array(img)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            results = model(img)
        
        # Print detection results
        print("\nDetection Results:")
        print(results.pandas().xyxy[0])
        return True
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")
        return False

if __name__ == "__main__":
    # Paths
    model_path = "best.pt"  # Path to your YOLO model
    output_path = "model.onnx"  # Path to save the ONNX model
    test_image = "test.jpg"  # Path to a test image
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit(1)
    
    # Convert model to ONNX format
    if convert_to_onnx(model_path, output_path):
        print("\nModel conversion successful!")
        
        # Verify the model if test image exists
        if os.path.exists(test_image):
            print("\nVerifying model with test image...")
            verify_onnx_model(output_path, test_image)
        else:
            print(f"\nTest image not found at {test_image}. Skipping verification.")
    else:
        print("\nModel conversion failed.") 