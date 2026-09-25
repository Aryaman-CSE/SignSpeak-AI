@'
# SignSpeak AI

## Real-Time Sign Language Detection

SignSpeak AI is a computer-vision and deep-learning project designed to detect American Sign Language (ASL) hand signs.

The system uses **SSD MobileNet V2 FPNLite** with the **TensorFlow Object Detection API** and transfer learning from COCO-pretrained weights. The current model is configured to recognize the **26 letters of the ASL alphabet (A–Z)**.

---

## Project Overview

```text
ASL Dataset
     ↓
Data Preprocessing
     ↓
Pascal VOC Annotations
     ↓
TFRecord Conversion
     ↓
SSD MobileNet V2 FPNLite
     ↓
Transfer Learning
     ↓
Model Fine-Tuning
     ↓
ASL Sign Detection
     ↓
A–Z Prediction