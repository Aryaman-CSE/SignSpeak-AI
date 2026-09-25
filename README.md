SignSpeak AI

Real-Time Sign Language Detection Using Deep Learning

SignSpeak AI is a computer-vision and deep-learning project for recognizing American Sign Language (ASL) alphabet signs A–Z using an object-detection pipeline built with TensorFlow and SSD MobileNet V2 FPNLite.

<p align="center">
  <b>Dataset → Preprocessing → TFRecords → Transfer Learning → Detection</b>
</p>

Overview

SignSpeak AI treats ASL alphabet recognition as an object-detection problem rather than simple image classification.

For each input image, the system is designed to:

detect the location of the hand sign,

identify the corresponding ASL alphabet class,

assign a confidence score,

and return a bounding box around the detected sign.

The current implementation focuses on the 26 ASL alphabet classes (A–Z) and establishes the machine-learning pipeline required for future real-time webcam detection.

Pipeline

                 ┌─────────────────────┐
                 │    ASL Dataset      │
                 │  1,728 Images A–Z   │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 │   + Verification    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Pascal VOC XML      │
                 │   Annotations       │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ TFRecord Generation │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ SSD MobileNet V2    │
                 │      FPNLite        │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │   Transfer Learning │
                 │   + Fine-Tuning     │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ 26-Class ASL Model  │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Future: Live Camera │
                 │      Detection      │
                 └─────────────────────┘

Project Status

Component

Status

Development environment

✅ Complete

ASL dataset integration

✅ Complete

Dataset verification

✅ Complete

Pascal VOC processing

✅ Complete

A–Z label mapping

✅ Complete

Train/test split

✅ Complete

TFRecord generation

✅ Complete

SSD MobileNet configuration

✅ Complete

Transfer-learning setup

✅ Complete

Pipeline validation

✅ Complete

10,000-step fine-tuning

✅ Complete

Quantitative evaluation

🔄 Pending

Webcam inference

🔄 Planned

Word formation

🔄 Planned

Continuous recognition

🔄 Planned

Dataset

The project uses an ASL alphabet object-detection dataset containing annotated images for all 26 alphabet classes.

Dataset at a glance

Metric

Value

Total Images

1,728

Classes

26

Training Images

1,382

Testing Images

346

Split

80 : 20

Annotation Format

Pascal VOC XML

Annotation Type

Bounding Boxes

Classes

A  B  C  D  E  F  G  H  I  J  K  L  M
N  O  P  Q  R  S  T  U  V  W  X  Y  Z

Dataset Workflow

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

Model

SSD MobileNet V2 FPNLite

The project uses SSD MobileNet V2 FPNLite 320×320 through the TensorFlow Object Detection API.

Configuration

Value

Architecture

SSD MobileNet V2 FPNLite

Input Resolution

320 × 320

Number of Classes

26

Framework

TensorFlow 2.10.1

Detection API

TensorFlow Object Detection API

Initial Weights

COCO

Training Strategy

Transfer Learning

SSD provides single-shot object detection, while MobileNet provides a relatively lightweight feature-extraction backbone suitable for practical detection workflows.

Transfer Learning

The model was initialized from COCO-pretrained weights and adapted to the 26-class ASL detection task.

COCO Pretrained Model
        ↓
Load Checkpoint
        ↓
Configure 26 ASL Classes
        ↓
Fine-Tune
        ↓
ASL Detection Model

This approach avoids training the complete detector entirely from random initialization.

Annotation & TFRecords

The dataset uses Pascal VOC XML annotations.

Each annotation provides:

image filename,

image dimensions,

object class,

bounding-box coordinates,

object location information.

The annotations are converted into TensorFlow TFRecords for model training.

Generated records

train.record
test.record

These are generated artifacts and are intentionally excluded from version control.

Label Map

The project maps each alphabet letter to a unique class ID.

ID

Class

ID

Class

1

A

14

N

2

B

15

O

3

C

16

P

4

D

17

Q

5

E

18

R

6

F

19

S

7

G

20

T

8

H

21

U

9

I

22

V

10

J

23

W

11

K

24

X

12

L

25

Y

13

M

26

Z

Label map:

Tensorflow/workspace/annotations/label_map.pbtxt

Training

The model was fine-tuned for:

10,000 Training Steps

Training was performed on the available CPU environment.

Final recorded training values

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

Note: Training loss should not be interpreted as detection accuracy. Quantitative evaluation using metrics such as mAP, precision, and recall is still required.

Training Flow

ASL Dataset
    ↓
Pascal VOC XML
    ↓
Dataset Verification
    ↓
Train/Test Split
    ↓
Label Map
    ↓
TFRecords
    ↓
Pipeline Configuration
    ↓
COCO Checkpoint
    ↓
Fine-Tuning
    ↓
Training Checkpoints

Technology Stack

Technology

Role

Python

Core development

TensorFlow 2.10.1

Deep-learning framework

TensorFlow Object Detection API

Object-detection training

SSD MobileNet V2 FPNLite

Detection architecture

OpenCV

Image/video processing

NumPy 1.23.5

Numerical operations

Pascal VOC

Annotation format

TFRecord

TensorFlow training data

COCO Weights

Transfer-learning initialization

VS Code

Development

PowerShell

Environment and execution

Git / GitHub

Version control

Project Structure

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

Files intentionally excluded

To keep the repository manageable, the following are excluded through .gitignore:

Raw datasets

TFRecord files

Pretrained model weights

Training checkpoints

Exported models

TensorFlow Models source repository

Virtual environments

Python cache files

Generated training artifacts

Real-Time Detection

The planned real-time system will connect the trained model to an OpenCV camera feed.

┌──────────────┐
│    Webcam    │
└──────┬───────┘
       ↓
┌──────────────┐
│ Capture Frame│
└──────┬───────┘
       ↓
┌──────────────┐
│ Preprocessing│
└──────┬───────┘
       ↓
┌──────────────┐
│   Inference  │
└──────┬───────┘
       ↓
┌──────────────┐
│  Detection   │
└──────┬───────┘
       ↓
┌──────────────┐
│ A–Z + Score  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Bounding Box │
└──────────────┘

A confidence threshold can be used to filter weak predictions.

Current status: webcam inference is a planned next stage and is not yet marked as completed.

Future Roadmap

01 — Alphabet Detection

Hand Sign → A–Z

The current model establishes the alphabet-detection foundation.

02 — Live Webcam Detection

Webcam → Detection → A–Z

Integrate the trained detector with OpenCV.

03 — Prediction Stabilization

Reduce frame-to-frame prediction noise using confidence filtering and temporal smoothing.

Frame 1 → A
Frame 2 → A
Frame 3 → A
Frame 4 → A
       ↓
 Stable A

04 — Word Formation

H → E → L → L → O
             ↓
           HELLO

05 — Continuous Sign Recognition

Move from isolated alphabet signs toward continuous sign sequences.

06 — Natural Language Output

Hand Signs
    ↓
Letters
    ↓
Words
    ↓
Sentences
    ↓
Natural Language

Current Limitations

Isolated Signs

The current dataset focuses on individual alphabet signs rather than continuous signing.

Dataset Dependence

Detection performance depends on how well real-world visual conditions are represented by the training dataset.

Environmental Variation

Lighting, background, camera angle, hand orientation, distance, and occlusion may affect predictions.

Evaluation

The current training result provides loss values but does not yet provide a complete quantitative accuracy evaluation.

Continuous Translation

The current system recognizes alphabet classes and does not yet perform complete sentence-level sign-language translation.

Team Contributions

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

Learning Outcomes

This project provides practical experience with:

Computer vision

Object detection

Deep learning

Transfer learning

Dataset engineering

Pascal VOC annotations

Bounding-box processing

TFRecord generation

TensorFlow Object Detection API

Model configuration

Fine-tuning

Training and debugging

Real-time inference architecture

Git and GitHub project management

Core Engineering Principle

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

A reliable AI system depends on the consistency of the entire pipeline, not only the neural network.

Reproducibility

The repository contains the project source code, configuration files, label map, and supporting scripts.

Large generated files are excluded from GitHub.

To reproduce the complete training pipeline, the corresponding dataset, TensorFlow Models/Object Detection API components, and pretrained checkpoint are also required.

Installation

Create a Python virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\Activate.ps1

Install the primary dependencies:

pip install tensorflow==2.10.1
pip install numpy==1.23.5
pip install opencv-python==4.8.1.78
pip install tensorflow-io==0.27.0
pip install lvis
pip install tf-models-official==2.10.1

The TensorFlow Object Detection API must also be available through the TensorFlow Models repository.

Training Command

After preparing the dataset and configuring the TensorFlow Object Detection API:

$env:PYTHONPATH="$PWD\Tensorflow\models\research;$PWD\Tensorflow\models\research\slim"

Run training with:

python scripts\model_main_tf2.py `
--model_dir=Tensorflow\workspace\models\my_ssd_mobilenet `
--pipeline_config_path=Tensorflow\workspace\models\pipeline.config `
--num_train_steps=10000 `
--alsologtostderr

Development Status

Completed

Environment setup

TensorFlow Object Detection API setup

ASL dataset integration

Dataset verification

Pascal VOC processing

A–Z label mapping

80:20 dataset split

TFRecord generation

SSD MobileNet V2 FPNLite configuration

26-class configuration

COCO checkpoint integration

Pipeline validation

10,000-step fine-tuning

Next

Quantitative evaluation

mAP calculation

Precision / recall analysis

Unseen-sample testing

Webcam integration

Real-time inference

Prediction stabilization

Letter-to-word conversion

Continuous recognition

Natural-language output

Robustness testing

Model optimization

Project Vision

The long-term vision is to transform visual sign-language communication into understandable digital language.

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

SignSpeak AI

From visual signs to machine-understandable language.

Repository

GitHub — Aryaman-CSE/SignSpeak-AI
