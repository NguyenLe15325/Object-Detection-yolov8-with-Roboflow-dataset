# YOLOv8 Object Detection with Roboflow Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenLe15325/Object-Detection-yolov8-with-Roboflow-dataset/blob/main/train_yolov8.ipynb)

Train and deploy a YOLOv8 object detection model using **Roboflow** and **Ultralytics**.

---

## 🧠 Overview

This repository provides a clean and structured workflow to:

1. Train a **YOLOv8** model on **Google Colab** using a Roboflow dataset.
2. Deploy the trained model locally for real-time inference on **images**, **videos**, or **webcam** streams.

---

## 🚀 Phase 1: Training on Google Colab

1. Open the Jupyter notebook in Colab using the button above.
2. Follow the step-by-step instructions to train the YOLOv8 model.
3. Once training completes, download the trained weights (`best.pt`) to your local machine.


---

## 💻 Phase 2: Local Deployment

### 1. Environment Setup

```bash
# Clone repo
git clone https://github.com/NguyenLe15325/Object-Detection-yolov8-with-Roboflow-dataset.git
cd Object-Detection-yolov8-with-Roboflow-dataset

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # On Linux/Mac
# OR
.venv\Scripts\activate         # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Detection Scripts

Run inference using your trained model (`best.pt`):

**For image detection:**

```bash
python detect_image.py --model best.pt --source path/to/image.jpg
```

**For video detection:**

```bash
python detect_video.py --model best.pt --source path/to/video.mp4
```

**For webcam detection:**

```bash
python detect_webcam.py --model best.pt --cam-id 0
```

---


## ⚙️ Notes

* Use `--imgsz` to adjust image size for inference.
* Use `--conf` to set detection confidence threshold (e.g., `--conf 0.5`).
* Refer to the [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com) for advanced usage.

---

## 📦 Dependencies

* Python ≥ 3.8
* ultralytics ≥ 8.0
* roboflow
* opencv-python
* torch
* torchvision

---

## 🧩 Credits

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [Roboflow](https://roboflow.com)

---
