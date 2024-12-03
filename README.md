# Vehicle Detection and Tracking for Autonomous Systems

**Team Members:**
- Haritha Dhanlalji Parmar
- Srinivasa Varma Penmetsa
- Ajay Kumar Medikonda
- Balaji Koushik Varma Sayyaparaju  

## I. Problem Statement and Significance

Autonomous vehicle systems require robust, accurate, and real-time object detection to ensure safe navigation and decision-making in dynamic environments. The selection of an appropriate object detection model significantly impacts the system's ability to balance accuracy, speed, and scalability. However, a comprehensive evaluation of contemporary models like YOLO v8 and Faster R-CNN for such real-world applications remains underexplored. This study aims to bridge the gap by assessing the performance of these models in the context of autonomous driving, focusing on key metrics such as detection accuracy, inference speed, and resource efficiency.

## II. Relevant Readings

1. "You Only Look Once: Unified, Real-Time Object Detection" (YOLO)
This paper introduces YOLO, an efficient, real-time object detection model that has become a benchmark for fast object detection in many applications, including self-driving cars.
[YOLO Paper](https://arxiv.org/abs/1506.02640)
2. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Faster R-CNN improves over earlier RCNN models by using a region proposal network, making it faster and more accurate in detecting objects.
[Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
3. "Deep Learning for Object Detection: A Comprehensive Review"
This review provides insights into various deep learning models, including YOLO and RCNN, and their use in object detection. It also explores key challenges in autonomous driving environments.
[Deep Learning for Object Detection](https://arxiv.org/abs/1907.09408)
4. "An Overview of Object Detection: One-stage versus Two-stage Detectors"
This paper compares one-stage detectors like YOLO with two-stage detectors like RCNN, which is relevant for understanding the trade-offs between speed and accuracy in object detection.
[Object Detection Overview](https://arxiv.org/abs/2004.02190)

## III. Data Sources

1. Udacity Self-Driving Car Dataset
https://github.com/udacity/self-driving-car/tree/master/annotations
2. KITTI Vision Benchmark Suite
http://www.cvlibs.net/datasets/kitti/
3. Berkeley DeepDrive (BDD100K)
https://www.bdd100k.com/

## IV. Methodology
YOLO v8: Unified detection architecture optimized for speed with a single neural network for localization and classification. Known for speed and efficiency, making it suitable for real-time application.
Faster R-CNN: Region Proposal Network (RPN) generates proposals; separate layers handle classification and bounding boxes. Renowned for its accuracy, particularly in complex scenes, but requires more computational resources.

Data Pipeline:
Dataset: Udacity Self Driving Car Dataset, BDD100K
Data Preprocessing 
Data Augmentation
Data Loading
Model Training
Evaluation and Testing
Post-Training Processing
Data Storage


## V. Evaluation Strategy

To assess and compare the performance of YOLO v8 and Faster R-CNN for object detection in autonomous driving, we employ the following evaluation metrics:
PR Curve
F1 Curve
Confusion Matrix
These evaluations will follow the data pipeline stages: preprocessing, augmentation, and loading from datasets like Udacity and BDD100K. Results from model testing on unseen data will ensure robustness. Post-training processing will refine model outputs, while insights derived will guide optimization strategies for autonomous vehicle applications.
