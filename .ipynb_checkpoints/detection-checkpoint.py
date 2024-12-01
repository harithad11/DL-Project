import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from IPython.display import HTML
from base64 import b64encode

def visualize_sample_images(images_path):
    """
    Visualize a few sample images from the dataset
    
    Args:
        images_path (str): Path to the image folder
    """
    sample_images = os.listdir(images_path)[:3]
    
    for img_file in sample_images:
        img_path = os.path.join(images_path, img_file)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(image_rgb)
        plt.title(img_file)
        plt.axis('off')
        plt.show()

def display_image_with_boxes(image_path, label_path, class_names):
    """
    Display an image with bounding boxes from corresponding label file
    
    Args:
        image_path (str): Path to the image
        label_path (str): Path to the label file
        class_names (list): List of class names
    """
    # Load the image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]  # Image dimensions

    # Read label file
    with open(label_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        # Parse label information
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, label.strip().split())
        x_center, y_center, bbox_width, bbox_height = int(x_center * w), int(y_center * h), int(bbox_width * w), int(bbox_height * h)

        # Calculate box coordinates
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)

        # Draw rectangle and label text
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_names[int(class_id)], (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def train_yolo_model(data_yaml, model_path='yolov8m.pt', epochs=10, batch_size=16):
    """
    Train a YOLOv8 model on the specified dataset
    
    Args:
        data_yaml (str): Path to the data configuration YAML file
        model_path (str): Path to the pre-trained model
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
    
    Returns:
        YOLO model object
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Train the model
    results = model.train(
        data=data_yaml, 
        epochs=epochs, 
        batch=batch_size, 
        imgsz=640
    )

    return model

def evaluate_model(model, data_yaml):
    """
    Evaluate the trained model
    
    Args:
        model (YOLO): Trained YOLO model
        data_yaml (str): Path to the data configuration YAML file
    """
    # Evaluate the model on validation data
    metrics = model.val(data=data_yaml)
    return metrics

def visualize_training_results(folder_path):
    """
    Visualize training result images
    
    Args:
        folder_path (str): Path to the training results folder
    """
    # Visualization types
    viz_types = {
        'Validation Batches': 'val_batch',
        'Performance Curves': ['PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png']
    }

    for title, pattern in viz_types.items():
        if isinstance(pattern, list):
            files = [file for file in os.listdir(folder_path) if file in pattern]
        else:
            files = [file for file in os.listdir(folder_path) if pattern in file]

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            img = plt.imread(file_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f"{title}: {file_name}")
            plt.axis('off')
            plt.show()

def predict_on_source(model, source, save=True, conf=0.5):
    """
    Run predictions on an image or video source
    
    Args:
        model (YOLO): Trained YOLO model
        source (str): Path to image or video
        save (bool): Whether to save the results
        conf (float): Confidence threshold
    """
    results = model.predict(
        source=source, 
        save=save, 
        conf=conf, 
        imgsz=640
    )
    return results

def display_video(video_path):
    """
    Display video using base64 encoding
    
    Args:
        video_path (str): Path to the video file
    """
    video_file = open(video_path, "rb").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width=640 controls><source src="{video_url}" type="video/mp4"></video>""")

def main():
    # Configuration
    dataset_path = 'udacity-car-dataset-1'
    train_images_path = os.path.join(dataset_path, 'train', 'images')
    test_images_path = os.path.join(dataset_path, 'test', 'images')
    test_labels_path = os.path.join(dataset_path, 'test', 'labels')
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    
    # Class names (update according to your dataset)
    class_names = [
        'biker', 'car', 'pedestrian', 'trafficLight', 
        'trafficLight-Green', 'trafficLight-GreenLeft', 
        'trafficLight-Red', 'trafficLight-RedLeft', 
        'trafficLight-Yellow', 'trafficLight-YellowLeft', 'truck'
    ]

    # Visualize sample images
    visualize_sample_images(train_images_path)

    # Display image with bounding boxes (example)
    sample_image = '1478019957687018435_jpg.rf.2bf9c3de7094f0bf93ada694a92d0afc.jpg'
    image_path = os.path.join(test_images_path, sample_image)
    label_path = os.path.join(test_labels_path, sample_image.replace('.jpg', '.txt'))
    display_image_with_boxes(image_path, label_path, class_names)

    # Train the model
    model = train_yolo_model(data_yaml)

    # Save the model
    model.save("YOLOv8m5k.pt")

    # Evaluate the model
    metrics = evaluate_model(model, data_yaml)

    # Visualize training results
    visualize_training_results('runs/detect/train22')

    # Predict on an image
    image_results = predict_on_source(model, image_path)

    # Predict on a video
    video_path = 'videos/traffic2.mp4'
    video_results = predict_on_source(model, video_path)

if __name__ == "__main__":
    main()