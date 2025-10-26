# Import necessary libraries
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("Detection_yolov8_roboflow_dataset/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.5) # Set confidence threshold
    annotated = results[0].plot()

    # Show frame
    cv2.imshow("YOLOv8 Webcam", annotated) # Display the annotated frame

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
