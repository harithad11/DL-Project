# Define the path to your dataset
dataset_path = "/content/drive/MyDrive/DL/Project"  

!pip install torch torchvision torchaudio
!pip install ultralytics

# Use kagglehub to download the dataset
import kagglehub
dataset_path = kagglehub.dataset_download("sshikamaru/udacity-self-driving-car-dataset")

# Confirm path
print("Path to dataset files:", dataset_path)

# Required imports
import cv2
import os
import torch
from ultralytics import YOLO

# Generator function to load images in batches (modified for 100 images)
def load_images_in_batches(folder_path, batch_size=10, img_size=(640, 640), max_images=100):
    batch = []
    image_count = 0  # To track the number of images processed
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    batch.append(img)
                    image_count += 1
                    if len(batch) == batch_size:
                        yield batch  # Yield a batch of images
                        batch = []  # Reset for the next batch
                    if image_count >= max_images:
                        return  # Stop after processing 100 images
    if batch:
        yield batch  # Yield remaining images if they don't fill up the last batch

# Load YOLOv5 model (you can choose 'yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x' for different sizes)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Example usage: Load and process 100 images batch by batch
for batch in load_images_in_batches(dataset_path, batch_size=10, max_images=100):
    print(f"Processing batch of {len(batch)} images...")

    # Example: Running YOLOv5 inference on the batch
    results = model(batch)  # Perform inference on the batch
    results.show()  # Display results for visualization


