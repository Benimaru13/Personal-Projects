# Readme.txt

Handwriting Distinguisher (Handwriting Classifier)
// Overview

Handwriting Distinguisher is a machine learning project that attempts to recognize and distinguish handwriting styles from different individuals.
The model uses a Convolutional Neural Network (CNN) to analyze handwriting images and learn the visual features that make each person’s writing unique.

This is my first version of the project and will evolve as I experiment, make mistakes, and improve the pipeline.

// Objectives

Train a CNN to classify handwriting based on the writer.

Explore image preprocessing and its effect on accuracy.

Learn how dataset size, architecture choices, and overfitting interact.

Gradually extend beyond a small personal dataset.

// Dataset

The initial dataset is intentionally small and personal.

Writers: 3

Samples per writer: 117

Total samples: 351

Source: Handwriting from myself and my siblings

Images: Currently high-resolution (≈1 MB each), to be downscaled before training

The dataset is meant for experimentation and learning, not general-purpose handwriting recognition.

Preprocessing (planned)

Convert images to grayscale

Resize to a consistent shape (124 x 124)

Normalize pixel values to [0, 1]

Optionally apply data augmentation (rotation, shear, zoom, slight noise)

// Model

A CNN built with TensorFlow/Keras.

Initial structure (subject to change):

Convolutional layers + ReLU

Max pooling

Dropout to reduce overfitting

Dense layers

Softmax output layer

Training

Split into training/validation sets

Track accuracy and loss during training

Experiment with:

learning rate

batch size

number of epochs

regularization methods (dropout, augmentation)

// Evaluation

Success criterion (for now):

Achieve ≥ 90% validation accuracy distinguishing the three writers.

Future analysis will include confusion matrices and per-class accuracy.

Tools & Libraries

Python

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Project Structure (planned)
handwriting-distinguisher/
│
├── data/                # handwriting images
├── notebooks/           # experiments
├── src/
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── results/             # metrics, saved models
└── README.md

// Roadmap

 Decide final input image resolution

 Implement preprocessing pipeline

 Train baseline CNN

 Add data augmentation

 Compare multiple architectures

 Publish trained model and results

// Limitations

Small dataset (risk of overfitting)

Limited handwriting diversity

Model may not generalize beyond the current participants

This project is primarily a learning exercise.

// Credits

Created by Chibueze.
