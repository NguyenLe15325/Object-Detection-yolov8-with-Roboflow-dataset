# Import necessary libraries
from ultralytics import YOLO
import cv2

# Load the trained YOLOv8 model
# You have to provide the correct path to your model file
model = YOLO("Football_player_Detection/best.pt")

# Path to input video
video_path = "Football_player_Detection/input.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details for saving output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get width of the frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get height of the frames
out = cv2.VideoWriter("Football_player_Detection/output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on frame
    results = model(frame, conf=0.5) # Set confidence threshold

    # Draw detections on frame
    annotated = results[0].plot()
    out.write(annotated)

    # Optional: display live during processing
    cv2.imshow("YOLOv8 Video", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
