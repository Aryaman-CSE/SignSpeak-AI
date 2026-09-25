SignSpeak AI

Real-Time Sign Language Detection Using Deep Learning

SignSpeak AI is a computer-vision and deep-learning project developed to recognize American Sign Language (ASL) alphabet signs using an object-detection model.

The project uses the TensorFlow Object Detection API with SSD MobileNet V2 FPNLite and transfer learning from COCO-pretrained weights. An annotated ASL alphabet dataset is processed into TensorFlow-compatible TFRecords and used to fine-tune a 26-class object-detection model.

The current implementation establishes the core machine-learning pipeline required for ASL sign detection. Future development will extend the trained detector toward real-time webcam inference, prediction stabilization, word formation, and continuous sign-language interpretation.

1. Project Overview

Communication through sign language relies heavily on visual information such as hand shape, orientation, and position. A computer-vision system can process this visual information and identify recognizable sign patterns.

SignSpeak AI approaches this problem as an object-detection task.

Instead of only classifying an image, the model is designed to:

Locate the hand sign within an image.

Identify which ASL alphabet class it represents.

Assign a confidence score to the prediction.

Return a bounding box around the detected sign.

The current model contains 26 classes corresponding to the letters A–Z.

Core Concept

Camera / Image
      ↓
Image Preprocessing
      ↓
SSD MobileNet Object Detection
      ↓
Bounding Box Detection
      ↓
ASL Letter Classification
      ↓
A–Z Prediction
      ↓
Future: Words → Sentences

2. Problem Statement

Sign language provides an important means of communication, but conventional computer systems are primarily designed to process typed or spoken language.

The challenge addressed by this project is to develop a computer-vision pipeline capable of recognizing visual ASL hand signs and converting them into machine-understandable alphabet predictions.

The project focuses on:

Detecting hand signs from visual input.

Localizing signs using bounding boxes.

Distinguishing between 26 ASL alphabet classes.

Preparing annotated image datasets for deep-learning training.

Applying transfer learning to an object-detection model.

Building a foundation for real-time sign recognition.

3. Objectives

Develop an ASL alphabet detection system.

Recognize all 26 ASL alphabet classes from A–Z.

Use object detection rather than simple image classification.

Detect and localize hand signs using bounding boxes.

Use transfer learning to reduce training requirements.

Build the complete dataset-to-model training pipeline.

Prepare the system for future real-time webcam inference.

Establish a scalable foundation for letter-to-word and continuous sign recognition.

4. System Architecture

                    ┌──────────────────────┐
                    │    ASL Dataset       │
                    │  1,728 Images A–Z    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preprocessing    │
                    │ Verification + Split │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Pascal VOC XML       │
                    │ Bounding Annotations │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ TFRecord Conversion  │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │ SSD MobileNet V2 FPNLite       │
              │ COCO Pretrained Model          │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │ Transfer Learning     │
                    │ / Fine-Tuning         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Trained 26-Class     │
                    │ ASL Detector         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Future Real-Time     │
                    │ Webcam Detection     │
                    └──────────────────────┘

5. Dataset

The project uses an American Sign Language alphabet object-detection dataset containing images representing the complete ASL alphabet.

Dataset Statistics

Property

Value

Total Images

1,728

Classes

26

Class Range

A–Z

Annotation Format

Pascal VOC XML

Training Images

1,382

Testing Images

346

Dataset Split

80:20

Annotation Type

Bounding Boxes

Each image is paired with a Pascal VOC XML annotation containing the object class and bounding-box coordinates.

Classes

A  B  C  D  E  F  G  H  I  J  K  L  M
N  O  P  Q  R  S  T  U  V  W  X  Y  Z

6. Dataset Preparation

The dataset preparation workflow is:

Raw Dataset
     ↓
Image / XML Verification
     ↓
Class Verification
     ↓
A–Z Label Mapping
     ↓
80:20 Train/Test Split
     ↓
Pascal VOC → TFRecord
     ↓
TensorFlow Object Detection Pipeline

The final dataset split contains:

1,382 training images

346 testing images

1,728 total images

7. Annotation Format

The dataset uses the Pascal VOC XML annotation format.

Each annotation contains information such as:

Image filename

Image dimensions

Object class

Bounding-box coordinates

Minimum and maximum X coordinates

Minimum and maximum Y coordinates

These annotations provide the spatial information required by the object-detection model.

The annotations were converted into TensorFlow TFRecord format before training.

8. Label Mapping

The model uses a 26-class label map.

ID

Class

1

A

2

B

3

C

4

D

5

E

6

F

7

G

8

H

9

I

10

J

11

K

12

L

13

M

14

N

15

O

16

P

17

Q

18

R

19

S

20

T

21

U

22

V

23

W

24

X

25

Y

26

Z

The label map is stored at:

Tensorflow/workspace/annotations/label_map.pbtxt

9. TFRecord Generation

TensorFlow Object Detection models use TFRecord files for efficient training-data ingestion.

The project converts Pascal VOC annotations into:

train.record
test.record

The generated records contain:

Encoded image data

Image dimensions

Bounding-box coordinates

Class labels

Class IDs

TFRecords are generated training artifacts and are intentionally excluded from Git version control.

10. Model

SSD MobileNet V2 FPNLite

The selected architecture is:

SSD MobileNet V2 FPNLite 320×320

SSD stands for Single Shot Detector. The architecture performs object detection in a single forward pass and is designed to provide a practical balance between detection capability and computational requirements.

Model Configuration

Parameter

Value

Architecture

SSD MobileNet V2 FPNLite

Input Size

320 × 320

Detection Classes

26

Framework

TensorFlow 2.10.1

Detection API

TensorFlow Object Detection API

Initial Weights

COCO

Training Method

Transfer Learning

11. Transfer Learning

The detector is initialized using COCO-pretrained weights rather than training the object detector entirely from random initialization.

COCO Pretrained SSD MobileNet
             ↓
       Load Checkpoint
             ↓
      Configure 26 Classes
             ↓
          Fine-Tuning
             ↓
      ASL Detection Model

Transfer learning provides a starting point with learned visual representations that can be adapted to the ASL detection task.

12. Training Configuration

The model configuration is stored at:

Tensorflow/workspace/models/pipeline.config

The configuration was modified for:

num_classes: 26

The training pipeline uses:

SSD MobileNet V2 FPNLite

320×320 input resolution

COCO pretrained checkpoint

26 ASL classes

TensorFlow Object Detection API

13. Training

The model was fine-tuned for:

10,000 training steps

Training was performed on the available CPU environment.

The final recorded training step produced approximately:

Metric

Value

Classification Loss

0.0531

Localization Loss

0.0117

Regularization Loss

0.1109

Total Loss

0.1757

Training Interpretation

The reported training loss indicates that the model successfully optimized against the prepared training dataset.

However, training loss is not equivalent to detection accuracy or mAP.

A separate evaluation stage is required to obtain quantitative detection metrics such as:

Precision

Recall

mAP

Per-class performance

Detection performance on unseen samples

14. Training Pipeline

ASL Dataset
     ↓
Pascal VOC XML
     ↓
Dataset Verification
     ↓
Train/Test Split
     ↓
Label Map Creation
     ↓
TFRecord Generation
     ↓
Pipeline Configuration
     ↓
COCO Checkpoint
     ↓
SSD MobileNet Fine-Tuning
     ↓
Training Checkpoints

15. Software Environment

Technology

Version / Role

Python

3.10.x

TensorFlow

2.10.1

NumPy

1.23.5

OpenCV

4.8.1.78

TensorFlow IO

0.27.0

TF Models Official

2.10.1

LVIS

Object Detection API dependency

TensorFlow Object Detection API

Model training

OpenCV

Future camera processing

Visual Studio Code

Development environment

PowerShell

Development environment

Git

Version control

GitHub

Repository hosting

16. Project Structure

SignSpeak AI/
│
├── Tensorflow/
│   ├── scripts/
│   │   └── generate_tfrecord.py
│   │
│   └── workspace/
│       ├── annotations/
│       │   ├── label_map.pbtxt
│       │   └── .gitkeep
│       │
│       └── models/
│           ├── pipeline.config
│           └── .gitkeep
│
├── scripts/
│   ├── collect_images.py
│   ├── create_custom_tf_record.py
│   ├── create_pascal_tf_record.py
│   └── model_main_tf2.py
│
├── Tutorial.ipynb
├── .gitignore
├── README.md
└── labelimg_error.txt

Excluded Generated Files

Large or generated files are intentionally excluded from GitHub, including:

Raw image datasets

TFRecord files

Pretrained model weights

Training checkpoints

Exported models

TensorFlow Models repository

Python cache files

Virtual environments

This keeps the repository lightweight and focused on source code, configuration, and documentation.

17. Real-Time Detection Design

The planned real-time component will use OpenCV to capture frames from a camera.

Webcam
   ↓
Capture Frame
   ↓
Resize / Preprocess
   ↓
TensorFlow Inference
   ↓
Detection Boxes
   ↓
Class Predictions
   ↓
Confidence Filtering
   ↓
Display Bounding Box
   ↓
ASL Letter

A confidence threshold can be applied to reduce low-confidence predictions.

Current status: the real-time webcam component is planned and has not yet been completed.

18. Future Development

Stage 1 — Alphabet Detection

Hand Sign → A–Z

The current trained model establishes the alphabet-detection foundation.

Stage 2 — Real-Time Detection

Webcam → Detection → A–Z

The trained model will be integrated with OpenCV for live camera inference.

Stage 3 — Prediction Stabilization

Temporal smoothing and confidence filtering can reduce unstable frame-to-frame predictions.

Frame 1 → A
Frame 2 → A
Frame 3 → A
Frame 4 → A

Stable Prediction → A

Stage 4 — Word Formation

Detected letters can be accumulated to form words.

H → E → L → L → O
             ↓
           HELLO

Stage 5 — Continuous Sign Recognition

The system can eventually be extended from isolated alphabet signs toward continuous sequences of signs.

Stage 6 — Natural Language Output

Hand Signs
    ↓
Letters
    ↓
Words
    ↓
Sentences
    ↓
Natural Language

19. Current Project Status

Completed

Development environment setup

TensorFlow Object Detection API setup

ASL alphabet dataset integration

Dataset verification

Pascal VOC annotation processing

A–Z label mapping

80:20 dataset split

TFRecord generation

SSD MobileNet V2 FPNLite configuration

26-class model configuration

COCO pretrained checkpoint integration

Pipeline validation

Model fine-tuning

10,000 training steps completed

Remaining

Quantitative model evaluation

mAP calculation

Precision and recall analysis

Unseen-image testing

Webcam integration

Real-time inference

Temporal prediction stabilization

Letter-to-word conversion

Continuous sign recognition

Natural-language output

Robustness testing

Model optimization

20. Limitations

Isolated Sign Recognition

The current dataset focuses on individual ASL alphabet signs rather than continuous signing.

Dataset Dependence

Model performance depends on the visual characteristics represented in the training dataset.

Real-World Conditions

Different lighting conditions, backgrounds, camera angles, hand orientations, and distances may affect detection performance.

Evaluation

Training loss alone cannot establish real-world accuracy. Quantitative evaluation using appropriate detection metrics is required.

Continuous Language

The current model recognizes alphabet classes and does not yet perform complete sentence-level sign-language translation.

21. Team Contributions

Member 1 — Dataset & Preprocessing

ASL dataset integration and verification

Train/test preparation

Pascal VOC annotation processing

A–Z label mapping

TFRecord generation

Member 2 — Model & Configuration

SSD MobileNet V2 FPNLite setup

26-class model configuration

Pipeline configuration

Pipeline validation

COCO checkpoint integration

Member 3 — Training & System Development

TensorFlow training environment setup

Dependency and configuration resolution

Model fine-tuning

Training-loss monitoring

Detection/inference pipeline preparation

22. Key Learning Outcomes

The project provides practical experience in:

Computer vision

Object detection

Deep learning

Transfer learning

Dataset engineering

Image annotation formats

Bounding-box processing

TensorFlow Object Detection API

TFRecord generation

Model configuration

Model fine-tuning

Training debugging

Real-time inference architecture

A major engineering lesson from the project is that model performance depends not only on the neural network, but also on the consistency of the complete pipeline:

Dataset
   ↓
Annotations
   ↓
Label Mapping
   ↓
TFRecords
   ↓
Configuration
   ↓
Training
   ↓
Evaluation
   ↓
Inference

23. Reproducibility

The repository contains the source code, configuration files, label map, and scripts required to understand and reproduce the project pipeline.

Large files are intentionally excluded from GitHub to keep the repository manageable.

The complete training environment additionally requires the corresponding dataset, TensorFlow Models/Object Detection API components, and pretrained checkpoint.

24. Repository

GitHub:
https://github.com/Aryaman-CSE/SignSpeak-AI

25. Project Vision

The long-term vision of SignSpeak AI is to create a computer-vision system capable of transforming visual sign-language communication into understandable digital language.

          SIGN LANGUAGE
                │
                ▼
        HAND SIGN DETECTION
                │
                ▼
          LETTER RECOGNITION
                │
                ▼
          WORD FORMATION
                │
                ▼
       CONTINUOUS RECOGNITION
                │
                ▼
        SENTENCE GENERATION
                │
                ▼
     ACCESSIBLE COMMUNICATION

SignSpeak AI — From visual signs to machine-understandable language.
