import torch
from ultralytics import YOLO
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import cv2

def convert_to_onnx():
    # Load the YOLOv11 model
    print("Loading YOLOv11 model...")
    model = YOLO('yolo11n.pt')
    
    # Define input shape (batch_size, channels, height, width)
    input_shape = (1, 3, 640, 640)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    print("Converting to ONNX format...")
    torch.onnx.export(
        model.model,  # model being run
        dummy_input,  # model input
        "yolo11n.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                     'output': {0: 'batch_size'}}
    )
    
    print("ONNX model saved as yolo11n.onnx")
    
    # Validate ONNX model
    print("\nValidating ONNX model...")
    onnx_model = onnx.load("yolo11n.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation successful!")

def test_onnx_inference():
    print("\nTesting ONNX model inference...")
    
    # Load and preprocess image
    image_path = "image-2.png"
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize image to match model input size
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    input_array = input_tensor.numpy()
    
    # Create ONNX Runtime session
    session = ort.InferenceSession("yolo11n.onnx")
    
    # Run inference
    outputs = session.run(
        None,
        {'input': input_array}
    )
    
    print("ONNX inference completed successfully!")
    print("Output shape:", outputs[0].shape)
    
    # Post-process the output
    predictions = outputs[0]
    
    # Reshape predictions: (1, 84, 8400) -> (1, 8400, 84)
    predictions = predictions.transpose((0, 2, 1))
    
    # Extract boxes, scores, and class predictions
    boxes = []
    scores = []
    class_ids = []
    
    conf_threshold = 0.25
    iou_threshold = 0.45
    
    # Process each detection
    for i in range(predictions.shape[1]):
        # Get scores for all classes
        class_scores = predictions[0, i, 4:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]
        
        if score > conf_threshold:
            # Get box coordinates
            x, y, w, h = predictions[0, i, :4]
            
            # Convert to corner coordinates
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            # Add to lists
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(class_id)
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    
    # Load COCO class names
    with open("coco.names", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Draw detections
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Scale boxes to original image size
    scale_x = original_size[0] / 640
    scale_y = original_size[1] / 640
    
    if len(indices) > 0:
        for i in indices:
            box = boxes[i]
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_names[class_ids[i]]} {scores[i]:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite("detection_result_onnx.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"\nDetection results saved to detection_result_onnx.jpg")
    print(f"Found {len(indices)} objects!")

if __name__ == "__main__":
    convert_to_onnx()
    test_onnx_inference() 