# Handwritten Digit Recognition App

A deep learning-based application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. The project includes both model training and an interactive GUI for real-time digit recognition.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Advanced CNN Model**: Dual-pathway architecture with multiple convolutional layers
- **Interactive GUI**: Draw digits and get instant predictions
- **High Accuracy**: Trained on MNIST dataset with robust preprocessing
- **Real-time Recognition**: Instant digit classification with confidence scores
- **User-friendly Interface**: Simple canvas for drawing with clear/undo functionality

## Model Architecture

The CNN model features:
- Dual convolutional pathways (3x3 and 5x5 filters)
- Batch normalization and dropout for regularization
- Global Average Pooling instead of Flatten
- Multiple dense layers with dropout


## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Required Libraries

Install the required packages:

```bash
pip install tensorflow numpy pandas matplotlib pillow

### Running the Application
Clone the repository:

bash
git clone https://github.com/RakeshSingh35/Handwritten-digit-model-and-app.git
cd handwritten-digit-recognition

### Run the application:
python Digit_recognition_app.py

### How to Use
Launch the application

Draw a digit (0-9) on the white canvas using your mouse

Click the "Predict" button to see the recognition result

Use "Clear" to reset the canvas and draw a new digit

The app displays both the predicted digit and confidence level

### Code Overview
Key Components
Model Definition (create_advanced_mnist_model()):

Dual convolutional pathways

Batch normalization and dropout

Global Average Pooling

GUI Application (DigitRecognizerApp class):

Tkinter-based canvas for drawing

Image preprocessing pipeline

Real-time prediction interface

Image Preprocessing:

Automatic digit cropping and centering

Size normalization to 28x28 pixels

Intensity normalization

File Descriptions
hand_written_app.py: Main application script containing both model training and GUI

mnist_cnn_model.keras: Saved trained model (created after first run)