# Vehicle Detection and Tracking for Autonomous Systems

**Team Members:**
- Haritha Dhanlalji Parmar
- Srinivasa Varma Penmetsa
- Ajay Kumar Medikonda
- Balaji Koushik Varma Sayyaparaju  

## I. Problem Statement and Significance

This project addresses the challenge of real-time object detection and tracking in autonomous driving systems, particularly in urban environments like Palo Alto. Factors such as changing lighting conditions, object occlusion, and sudden appearances of objects complicate this task. Accurate detection of vehicles, pedestrians, and cyclists is crucial for safe driving decisions; failures in these systems can lead to accidents. By evaluating and refining deep learning models such as YOLO and Faster R-CNN, we aim to enhance the accuracy and reliability of autonomous systems in managing complex driving scenarios, ultimately improving overall safety.

## II. Relevant Readings

1. **"You Only Look Once: Unified, Real-Time Object Detection" (YOLO)**  
   This paper introduces YOLO, an efficient real-time object detection model that has set benchmarks for fast object detection in various applications, including self-driving cars.

2. **"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"**  
   Faster R-CNN enhances earlier RCNN models by employing a region proposal network, significantly improving speed and accuracy in object detection.

3. **"Deep Learning for Object Detection: A Comprehensive Review"**  
   This review offers insights into various deep learning models, including YOLO and RCNN, their application in object detection, and the challenges faced in autonomous driving environments.

4. **"An Overview of Object Detection: One-stage versus Two-stage Detectors"**  
   This paper compares one-stage detectors like YOLO with two-stage detectors like RCNN, elucidating the trade-offs between speed and accuracy in object detection.

## III. Data Sources

- Udacity Self-Driving Car Dataset
- KITTI Vision Benchmark Suite
- Berkeley DeepDrive (BDD100K)

## IV. Methodology

- **YOLOv5 for Object Detection**
- **Transfer Learning**
- **Faster R-CNN for Object Detection**
- **Object Tracking with SORT/DeepSORT**
- **Comparison with Other Models** (Fast R-CNN, Mask R-CNN)

## V. Evaluation Strategy

To evaluate the performance of our object detection and tracking models, we will utilize various metrics:
- **Object Detection Accuracy:** Measured using mean Average Precision (mAP) at different IoU thresholds, along with precision and recall to assess true and false detections.
- **Bounding Box Localization:** Evaluated through IoU scores and bounding box quality.
- **Tracking Performance:** Assessed using MOTA (Multiple Object Tracking Accuracy) and IDF1 to gauge tracking consistency and accuracy across frames.
- **Real-time Performance:** Measured in FPS (Frames Per Second) and latency, which are critical for rapid responses in autonomous driving.
- **Qualitative Evaluation:** Conducted through visual inspection and error analysis to identify failure modes and edge cases, such as occlusions and lighting variations.
