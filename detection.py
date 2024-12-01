import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import HTML
from base64 import b64encode

def load_model(model_path):
    """
    Load a pre-trained YOLO model
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        YOLO model object
    """
    return YOLO(model_path)

def predict_on_source(model, source, save=True, conf=0.5):
    """
    Run predictions on an image or video source
    
    Args:
        model (YOLO): Loaded YOLO model
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
    # Path to your trained model
    model_path = "YOLOv8m5k.pt"
    
    # Load the pre-trained model
    model = load_model(model_path)
    
    # Example image prediction
    image_path = 'udacity-car-dataset-1/test/images/1478019957687018435_jpg.rf.2bf9c3de7094f0bf93ada694a92d0afc.jpg'
    image_results = predict_on_source(model, image_path)
    
    # Example video prediction
    video_path = 'videos/traffic1.mp4'
    video_results = predict_on_source(model, video_path)

    video_path2 = 'videos/traffic2.mp4'
    video_results = predict_on_source(model, video_path2)
    video_path3 = 'videos/traffic3.mp4'
    video_results = predict_on_source(model, video_path3)
    video_path4 = 'videos/traffic4.mp4'
    video_results = predict_on_source(model, video_path4)

if __name__ == "__main__":
    main()