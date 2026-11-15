# SignSpeak AI

**Real-time, hand-focused American Sign Language (ASL) alphabet recognition using computer vision and deep learning.**

SignSpeak AI uses a webcam to detect visible hands and recognize ASL alphabet signs. The current real-time application combines a **MediaPipe hand tracker**, a **MobileNetV2 image classifier** for static letters, and an **LSTM sequence model** for the movement-based letters **J** and **Z**.

> The current application recognizes isolated alphabet signs. It does not yet translate continuous ASL signing into words or sentences.

---

## Table of Contents

- [What the Application Does](#what-the-application-does)
- [Recognition Pipeline](#recognition-pipeline)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Run the Webcam Application](#run-the-webcam-application)
- [Controls](#controls)
- [Models and Required Files](#models-and-required-files)
- [How Recognition Works](#how-recognition-works)
- [Training and Development Scripts](#training-and-development-scripts)
- [Dataset and Reproducibility](#dataset-and-reproducibility)
- [Current Scope and Limitations](#current-scope-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)

---

## What the Application Does

The webcam application:

1. Captures frames from the computer's webcam.
2. Uses MediaPipe Hands to locate hand landmarks.
3. Builds a hand-centered image crop rather than classifying the whole camera frame.
4. Classifies static hand shapes into letters A–Z with MobileNetV2.
5. Uses a sequence of hand landmarks and an LSTM model to recognize movement-based J and Z.
6. Applies confidence checks and temporal voting to reduce unstable frame-by-frame predictions.
7. Displays the recognized letter, confidence, hand landmarks, bounding box, and inference FPS.

The tracker is configured for up to two hands. The display can be mirrored independently of the image sent to the classifier.

## Recognition Pipeline

```text
                 WEBCAM FRAME
                       |
                       v
              MediaPipe Hands
             Hand landmark tracking
                       |
                       v
               Hand extraction
          Hand-centered square crop
                       |
             +---------+---------+
             |                   |
             v                   v
       STATIC SIGN           HAND MOTION
       MobileNetV2           30-frame landmark
       image classifier      sequence
             |                   |
             v                   v
          A–Z scores          LSTM model
                             (J and Z)
             |                   |
             +---------+---------+
                       |
                       v
            Confidence and voting
                       |
                       v
             Stable letter output
                       |
                       v
              Webcam display
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Application runtime |
| TensorFlow 2.10.1 / Keras | Loads and runs trained neural networks |
| MobileNetV2 | Static ASL alphabet image classification |
| LSTM | Recognition of movement-based J and Z |
| MediaPipe 0.10.9 | Hand landmark detection and tracking |
| OpenCV contrib 4.8.1.78 | Webcam capture, image processing, and display |
| NumPy 1.23.5 | Numerical and sequence processing |
| Git / GitHub | Source control and project hosting |

The webcam application does not need the TensorFlow Object Detection API to run. The repository retains TensorFlow workspace configuration and label-map files from the project's earlier object-detection development stage.

## Project Structure

The repository contains the real-time application, model files, training and evaluation utilities, and the retained TensorFlow workspace configuration.

```text
SignSpeak-AI/
|
|-- realtime_signspeak.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- Tutorial.ipynb
|
|-- models/
|   |-- asl_mobilenetv2/
|   |   |-- webcam_finetuned_v2.h5
|   |   |-- class_indices.json
|   |   |-- additional model candidates, weights and reports
|   |
|   |-- dynamic_signs/
|       |-- dynamic_lstm_v2.h5
|       |-- classes_v2.npy
|       |-- other saved model versions
|
|-- Tensorflow/
|   |-- workspace/
|       |-- annotations/
|       |   |-- label_map.pbtxt
|       |
|       |-- models/
|           |-- pipeline.config
|
|-- training, data-collection and evaluation scripts
```

The tree above highlights the important files; it is not an exhaustive listing of every development utility.

## Requirements

For the documented Windows setup:

- Windows 10 or Windows 11
- Python **3.10.x**
- A working webcam
- Internet access during initial package installation
- Sufficient disk space for the Python environment and model files

The dependency versions are pinned in `requirements.txt`:

```text
tensorflow==2.10.1
numpy==1.23.5
opencv-contrib-python==4.8.1.78
mediapipe==0.10.9
protobuf==3.19.6
```

**Use Python 3.10.** TensorFlow 2.10.1 does not provide the same Windows wheel support for newer Python releases.

The application can run on CPU; a CUDA-enabled GPU is not required.

## Installation

### 1. Download the Project

Download the repository ZIP from GitHub and extract it, or clone the repository:

```powershell
git clone https://github.com/Aryaman-CSE/SignSpeak-AI.git
cd SignSpeak-AI
```

If you downloaded the ZIP, open PowerShell in the extracted project root — the folder containing `realtime_signspeak.py` and `requirements.txt`.

### 2. Create a Virtual Environment

```powershell
py -3.10 -m venv .venv
```

### 3. Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current terminal session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Package installation may take several minutes. Keep the virtual environment active when running the project.

## Run the Webcam Application

From the project root, with the virtual environment activated:

```powershell
python .\realtime_signspeak.py
```

The application loads both trained models, initializes MediaPipe hand tracking, and opens the webcam window.

**Important:** Run the script from the extracted repository structure. The application checks for `Tensorflow/workspace/models/pipeline.config` when locating the project root, even though the live recognition path loads the Keras models from the `models/` directory.

## Controls

| Key | Action |
|---|---|
| `R` | Reset prediction history |
| `F` | Toggle horizontal mirroring of the displayed camera frame |
| `Q` or `Esc` | Quit the application |

## Models and Required Files

The current webcam application loads these files:

| File | Purpose |
|---|---|
| `models/asl_mobilenetv2/webcam_finetuned_v2.h5` | Static A–Z MobileNetV2 classifier |
| `models/asl_mobilenetv2/class_indices.json` | Letter-to-class-index mapping |
| `models/dynamic_signs/dynamic_lstm_v2.h5` | LSTM model for dynamic J/Z recognition |
| `models/dynamic_signs/classes_v2.npy` | Class labels for the dynamic model |
| `Tensorflow/workspace/models/pipeline.config` | Required by the current project-root detection check |
| `Tensorflow/workspace/annotations/label_map.pbtxt` | Retained A–Z label map from the object-detection workflow |

Do not rename or move these files unless you also update the corresponding paths in the application.

The repository contains other model candidates, checkpoints, weights, and evaluation reports from development. **The live application currently loads the exact model paths listed in the table above.** The presence of another candidate model in the repository does not mean the application automatically uses it.

## How Recognition Works

### 1. Hand Detection and Cropping

MediaPipe Hands detects hand landmarks in the camera frame. The application uses the landmark coordinates to calculate a hand region, expands it to a square crop, pads regions that extend beyond the frame, and resizes the crop for classification.

This hand-focused pipeline is intended to keep the classifier focused on the hand rather than treating a face or unrelated background object as a sign.

### 2. Static Alphabet Recognition

The hand crop is resized to the MobileNetV2 input dimensions, converted from BGR to RGB, and passed through the MobileNetV2 preprocessing function.

The classifier produces class scores for the alphabet.

A prediction is accepted only when it passes the configured confidence threshold and the margin between the highest and second-highest class scores.

### 3. Dynamic J and Z Recognition

J and Z involve movement. The application maintains a sequence of 30 hand-landmark frames and passes the sequence to the LSTM model when sufficient movement is detected.

Static MobileNetV2 predictions for J and Z are suppressed so that these letters are recognized through the movement-based path.

### 4. Prediction Stabilization

The application keeps a short history of predictions for each tracked hand. A displayed letter must receive the required number of votes in the recent history.

This helps reduce rapid changes between letters when confidence fluctuates.

### 5. Display

The OpenCV window shows:

- Hand bounding box
- Hand landmark points
- Prediction state
- Confidence for a stable letter
- Estimated inference FPS

## Training and Development Scripts

The repository includes scripts used during model development, data preparation, fine-tuning, evaluation, and dynamic-sign collection.

Examples include:

- `collect_webcam_samples.py` and `collect_live_static_crops.py` — webcam sample and crop collection utilities.
- `collect_dynamic_signs.py` — collection of landmark sequences for dynamic signs.
- `fine_tune_webcam.py` and other fine-tuning scripts — model fine-tuning experiments.
- `train_dynamic_lstm.py` and `train_dynamic_lstm_v2.py` — dynamic sequence-model training utilities.
- `evaluate_static_model.py`, `evaluate_webcam_v2.py`, and round-specific evaluation scripts — model evaluation utilities.
- `build_records.py` and related TensorFlow utilities — dataset and record preparation from the earlier object-detection workflow.

These are development utilities, not a single end-to-end training command. Their input folders and output paths depend on the corresponding datasets and local training setup.

The webcam inference setup described above does not require running the training scripts.

## Dataset and Reproducibility

The project has used ASL alphabet image data, locally collected webcam samples and crops, and hand-landmark sequences for dynamic signs during development.

Large raw datasets, local webcam collections, generated training artifacts, virtual environments, and some TensorFlow development components are excluded from version control through `.gitignore`.

As a result:

- A GitHub ZIP is intended to contain the application source and committed inference model assets.
- The raw training datasets are not included in the ZIP.
- Re-training or reproducing every experiment requires obtaining the relevant datasets and preparing the expected local folder structure.
- The saved inference models are provided so the webcam application can be run without repeating model training.

Training metrics and candidate-model evaluation reports should be interpreted in the context of their specific test sets. They should not be treated as a guarantee of real-world recognition accuracy.

## Current Scope and Limitations

- The current task is isolated ASL alphabet recognition, not full sign-language translation.
- J and Z are handled as movement-based signs; other alphabet signs use static hand images.
- Recognition quality can vary with lighting, camera quality, hand orientation, distance, occlusion, background, and signing style.
- The application depends on MediaPipe successfully detecting and tracking the hand.
- The current interface displays recognized letters; it does not assemble them into words or sentences.
- The project does not currently provide a complete continuous-sign-language translation system.
- Recognition results can be uncertain. The confidence value is a model score, not a calibrated probability of correctness.

## Troubleshooting

### Python 3.10 Is Not Found

Check installed Python versions:

```powershell
py --list
```

Install Python 3.10.x and recreate the virtual environment with:

```powershell
py -3.10 -m venv .venv
```

### TensorFlow Cannot Be Installed

Confirm that the active interpreter is Python 3.10:

```powershell
python --version
python -m pip --version
```

Then retry:

```powershell
python -m pip install -r requirements.txt
```

### PowerShell Does Not Allow Environment Activation

For the current PowerShell window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### The Webcam Cannot Be Opened

- Close applications that may already be using the webcam.
- Check Windows camera privacy permissions.
- Reconnect an external webcam if applicable.
- Run the application again.

### A Required Model or Configuration File Is Missing

Check that the ZIP was fully extracted and that you are running the script from the project root.

Verify that the required paths in [Models and Required Files](#models-and-required-files) exist and have not been renamed.

### TensorFlow Prints CUDA or GPU Warnings

The application can run on CPU.

CUDA-related warnings may appear on systems without a compatible NVIDIA CUDA setup; they do not automatically mean that CPU inference has failed. Check whether the model loads and the webcam window opens.

## Future Work

Potential extensions include:

1. Improve robustness across different users, backgrounds, lighting, and camera positions.
2. Expand and systematically evaluate the training data.
3. Improve temporal handling of dynamic signs.
4. Add letter-to-word composition with an explicit user-controlled word boundary.
5. Explore continuous sign recognition and sentence-level language processing.
6. Measure performance on held-out data and real-world webcam sessions.

These are future development directions and should not be interpreted as features already implemented.

---

## Repository

**GitHub:** [Aryaman-CSE/SignSpeak-AI](https://github.com/Aryaman-CSE/SignSpeak-AI)

**Main application:** `realtime_signspeak.py`

**Dependency file:** `requirements.txt`

**Project:** SignSpeak AI — real-time, hand-focused ASL alphabet recognition.
