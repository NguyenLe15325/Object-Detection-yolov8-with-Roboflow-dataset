# Import necessary libraries
from ultralytics import YOLO
import cv2

# Load trained model
# You have to provide the correct path to your model file
model = YOLO("Football_player_Detection/best.pt")

# Inference on image or video
# You have to provide the correct path to your image or video file
results = model.predict(source="Football_player_Detection/test.jpg", show=True, conf=0.5) # Set confidence threshold

# Optional: save annotated image
for r in results:
    annotated = r.plot()
    cv2.imwrite("Football_player_Detection/result.jpg", annotated)
