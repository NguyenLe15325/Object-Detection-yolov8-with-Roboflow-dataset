from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("Detection_yolov8_roboflow_dataset/best.pt")

# Inference on image or video
results = model.predict(source="Detection_yolov8_roboflow_dataset/test.jpg", show=True, conf=0.5)

# Optional: save annotated image
for r in results:
    annotated = r.plot()
    cv2.imwrite("Detection_yolov8_roboflow_dataset/result.jpg", annotated)
