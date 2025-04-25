import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path

def preprocess_image(image_path, target_size=(640, 640)):
    """
    Preprocess image for ONNX model inference
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Resize image
    resized = cv2.resize(image, target_size)
    
    # Convert to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    resized = resized.astype(np.float32) / 255.0
    
    # Transpose to NCHW format
    input_tensor = np.transpose(resized, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, 0)
    
    return input_tensor, (original_width, original_height)

def postprocess_output(output, original_size, conf_threshold=0.25):
    """
    Postprocess ONNX model output
    """
    # Get the first output (assuming it's the detection output)
    detections = output[0]
    
    # Filter detections by confidence
    mask = detections[:, 4] > conf_threshold
    detections = detections[mask]
    
    # Convert coordinates to original image size
    scale_x = original_size[0] / 640
    scale_y = original_size[1] / 640
    
    # Scale bounding boxes
    detections[:, [0, 2]] *= scale_x
    detections[:, [1, 3]] *= scale_y
    
    return detections

def draw_detections(image, detections, class_names=None):
    """
    Draw detections on image
    """
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Prepare label
        if class_names is not None:
            class_name = class_names[int(class_id)]
            label = f"{class_name} {conf:.2f}"
        else:
            label = f"{conf:.2f}"
        
        # Draw label
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def run_inference(onnx_path, image_path, conf_threshold=0.25):
    """
    Run inference using ONNX model
    """
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Preprocess image
    input_tensor, original_size = preprocess_image(image_path)
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = time.time() - start_time
    
    # Postprocess output
    detections = postprocess_output(outputs, original_size, conf_threshold)
    
    # Load original image for visualization
    image = cv2.imread(image_path)
    
    # Draw detections
    result_image = draw_detections(image, detections)
    
    # Save result
    output_path = Path(image_path).stem + "_onnx_result.jpg"
    cv2.imwrite(output_path, result_image)
    
    print(f"Inference completed in {inference_time:.4f} seconds")
    print(f"Number of detections: {len(detections)}")
    print(f"Result saved as: {output_path}")
    
    return result_image, detections

if __name__ == "__main__":
    # Paths
    onnx_model_path = "yolo11n.onnx"
    image_path = "image.png"
    
    # Run inference
    result_image, detections = run_inference(onnx_model_path, image_path) 