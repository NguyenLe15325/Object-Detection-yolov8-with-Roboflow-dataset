# 🦾 YOLOv8 Object Detection with Roboflow Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NguyenLe15325/Object-Detection-yolov8-with-Roboflow-dataset/blob/main/train_yolov8.ipynb)

Train, evaluate, and deploy a **YOLOv8** object detection model using **Roboflow** for dataset management and **Ultralytics** for training.

---

## 🧠 Overview

This repository provides a complete workflow to:

1. **Train** a YOLOv8 model on **Google Colab** using a Roboflow dataset.
2. **Deploy** the trained model locally for real-time inference on **images**, **videos**, or **webcam** streams.

---

## 🚀 Phase 1: Train on Google Colab

1. Click the badge above to open the training notebook directly in Colab.
2. Follow the guided steps to connect your Roboflow account, download the dataset, and train the model.
3. After training completes, download the `best.pt` weights file for deployment.

---

## 💻 Phase 2: Deploy Locally

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/NguyenLe15325/Object-Detection-yolov8-with-Roboflow-dataset.git
cd Object-Detection-yolov8-with-Roboflow-dataset

# Create a virtual environment
python -m venv .venv
# Activate environment
source .venv/bin/activate      # macOS / Linux
# OR
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run Detection Scripts

Update your model path and input source in the script, then run it to perform detection:


You can specify options like:

* `--imgsz`: image size (e.g., `640`)
* `--conf`: confidence threshold (e.g., `0.5`)

---

## ⚽ Example Project: Football Player Detection

* **Dataset (YOLOv8 format)**:
  [Download from Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/20/download/yolov8)

* **Test Video**:
  [Soccer Game Clip](https://www.pexels.com/video/a-soccer-game-being-played-in-the-soccer-field-3507660/)

* **Test Image**:
  [Soccer Match Image](https://unsplash.com/photos/soccer-game-iiwYpGkbDgM)

---

## ⚙️ Notes

* Adjust inference parameters for accuracy and performance:

  * `--imgsz` to change image resolution
  * `--conf` to set confidence threshold
* For detailed configuration and advanced options, see the [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com).

---

## 📦 Requirements

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
