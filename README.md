# BESTIA
BESTIA - Biostatistics Evaluation Systematics Traceability Integration Analysis


### **Video Analysis for Animal Counting with AI** 


#### **1. Introduction**  
AI-powered video analysis is transforming how industries track animal populations. From fisheries monitoring salmon migration to ranchers managing livestock, automated tracking enhances efficiency, accuracy, and data reliability.

#### **2. Fundamentals of Animal Tracking**  
Tracking in video involves detecting, identifying, and following animals frame by frame. Traditional tracking methods rely on manual observation, while AI-based approaches use deep learning to recognize patterns and movement automatically.

#### **3. Video Capture Setup**  
The placement and angle of the camera are crucial to ensure optimal data collection. Factors like lighting conditions, resolution, and background contrast affect AI performance, requiring careful adjustments for accurate tracking.

#### **4. Animal Detection in Video**  
Deep learning models such as YOLO and Faster R-CNN can detect animals in video feeds. These models help distinguish species, filter out irrelevant objects, and improve detection accuracy even in challenging environments.

#### **5. Challenges in Automatic Counting**  
Obstacles such as occlusion, motion blur, and environmental changes can hinder AI tracking. Addressing these challenges requires refining models, incorporating temporal analysis, and optimizing data preprocessing techniques.

#### **6. Practical Applications and Case Studies**  
AI-powered animal tracking has been successfully applied in conservation, livestock management, and fisheries. Case studies include automated salmon counting in rivers, monitoring cattle movement for health assessments, and tracking birds in large-scale ecological studies.

#### **7. Conclusions and Future of AI-Based Video Analysis**  
Future advancements in AI tracking may involve multimodal sensor integration, improved deep learning models, and enhanced real-time processing capabilities. As technology evolves, these tools will redefine how animal populations are studied and managed.


### **Tracking and Counting Methods**  
Tracking animals in video is a key challenge in automated counting. For salmon, the goal is to detect when they cross a specific line in the frame and register the event. This is achieved using computer vision algorithms that combine object detection with motion tracking. This chapter will explore techniques such as YOLO for detection and SORT for tracking, ensuring accurate counting even in variable lighting conditions and fast movement.

---

#### SALMONS
Here’s a prototype Python code snippet for counting salmon as they cross a defined line in a video, using **YOLOv8** for detection and **ByteTrack** for tracking:

```python
import cv2
import numpy as np
from ultralytics import YOLO
from supervision import LineCounter, Point, VideoSink

# Load the trained YOLOv8 model for salmon detection
model = YOLO("yolov8n.pt")  # Replace with your trained model

# Define the counting line
line_start = Point(100, 300)  # Adjust based on your video
line_end = Point(500, 300)
line_counter = LineCounter(start=line_start, end=line_end)

# Open video file
video_path = "salmon_video.mp4"  # Replace with your video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect salmon in the frame
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Extract detections

    # Filter salmon detections
    salmon_detections = [d for d in detections if d[-1] == "salmon"]  # Adjust based on class labels

    # Update line counter
    for detection in salmon_detections:
        x1, y1, x2, y2, conf, cls = detection
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        line_counter.update(Point(center_x, center_y))

    # Display count on screen
    frame = line_counter.annotate(frame)
    cv2.imshow("Salmon Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

This code should detect salmons in each frame, checks whether they cross the line, and updates the count. For better accuracy, you can train YOLO with salmon-specific images and refine the counting line placement. 






