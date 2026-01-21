# Mastitis Detection Using Deep Learning 🐄

A deep learning solution designed to automate the detection of Mastitis in dairy cattle using image analysis. This project leverages Transfer Learning with the MobileNetV2 architecture to classify udder images as either "Healthy" or showing signs of "Mastitis".

The project includes a trained model and a user-friendly Streamlit Dashboard for real-time predictions.

### What is Mastitis?

Mastitis is a persistent inflammation of the udder tissue in dairy cows. Early detection is crucial for animal health and milk quality.

## 📂 Repository Structure

Based on the current repository layout:

```bash
MastitisDetection_Using_DeepLearning/
├── Mastitis Dataset/          # Dataset containing approx 1300 images
│   ├── Test DataSet/          # Testing images (30%)
│   │   ├── mastitis/
│   │   └── normal/
│   └── Train DataSet/         # Training images (70%)
│       ├── mastitis/
│       └── normal/
├── local_dashboard.py         # The Streamlit application script
└── mastitis_mobilenetv2.h5    # The trained Deep Learning model weights
```

## 🚀 Project Overview

Mastitis is a persistent inflammation of the udder tissue in dairy cows. Early detection is crucial for animal health and milk quality. This project uses Computer Vision to analyze thermal or visual images of udders to identify potential infection.

### Key Features:

Architecture: MobileNetV2 (Pre-trained on ImageNet) with a custom classification head.

Input Resolution: 224x224 pixels.

Classes: Binary Classification (Mastitis vs. Normal).

Interface: Interactive web dashboard allowing file uploads and instant confidence scoring.

## 🛠️ Tech Stack

Python 3.x

TensorFlow/Keras: For model building and inference.

Streamlit: For the web-based user interface.

NumPy & Pillow: For image processing.

## 💻 Installation & Usage

Follow these steps to run the project locally:

1. Clone the Repository
```bash
git clone https://github.com/Uzair12456/MastitisDetection_Using_DeepLearning.git
```
```bash
cd MastitisDetection_Using_DeepLearning
```

2. Install Dependencies

Ensure you have Python installed. Install the required libraries:
```bash
pip install tensorflow streamlit pillow numpy
```

3. Run the Dashboard
```bash
Execute the Streamlit script to launch the interface:

streamlit run local_dashboard.py
```
## Project Demo
View the working demo of the project:
[Watch Demo Video](https://github.com/Umaralp/MastitisDetection_Using_DeepLearning/blob/main/Working/Demo.mp4)

(click on 'view raw' or download the file to view)
