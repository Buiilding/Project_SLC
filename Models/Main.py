import cv2
import torch
import os
from Models.Classification_Model import Model
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
import torch
import torchvision.transforms as transforms
import cv2

def inference_video(weight_path, source_path, save_path, FPS):
    # Load the pre-trained model
    num_classes = 27  # Replace with the actual number of classes in your model
    model = Model(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # Set up video capture
    cap = cv2.VideoCapture(source_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = FPS  # Output frames per second

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, output_fps, (frame_width, frame_height))

    # Set up transforms for input frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Process each frame in the video
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_tensor = transform(frame)
        frame_tensor = frame_tensor.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            predictions = model(frame_tensor)

        # Process the predictions (e.g., draw bounding boxes on the frame)
        # Modify this part according to your detection model's output format and visualization preferences
        # Example: draw bounding boxes on the frame
        for prediction in predictions:
            class_id = prediction['class_id']
            box = prediction['box']
            confidence = prediction['confidence']
            label = f'Class: {class_id}, Confidence: {confidence:.2f}'
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the processed frame to the output video
        writer.write(frame)

        # Print progress
        frame_count += 1
        print(f'Processed frame {frame_count}/{num_frames}')

    # Release resources
    cap.release()
    writer.release()

    print('Inference completed and video saved.')

# Example usage
weight_path = '/path/to/weight.pth'
source_path = '/path/to/input/video.mp4'
save_path = '/path/to/output/video_output.mp4'
FPS = 30  # Output frames per second
inference_video(weight_path, source_path, save_path, FPS)
