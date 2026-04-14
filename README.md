# AI-Powered Medical Image Analysis System

This project demonstrates how deep learning can be used to analyze medical images (e.g. X-rays, MRIs) for disease diagnosis. It includes a convolutional neural network (CNN) built with TensorFlow/Keras that classifies images (e.g. detecting pneumonia in chest X-rays). The goal is to emulate an industry workflow using public datasets and to provide a showcase for practical skills.

## Problem Statement

In many healthcare settings, doctors rely on imaging (radiology) to detect conditions like pneumonia, tumors, or fractures. Manual review of scans can be time-consuming and error-prone【7†L64-L72】. This project aims to build an **AI-powered image analysis system** that assists clinicians by automatically identifying disease in images, improving speed and accuracy【7†L64-L72】【20†L430-L438】.

## Industry Relevance

- **Hospitals/Radiology:** Automating scan analysis accelerates triage of critical cases and reduces diagnostic backlog.  
- **Diagnostic Labs/Health-Tech:** Machine learning tools supplement lab capabilities (e.g. analyzing imaging data) and enable new remote-reading services.  
- **Clinical Impact:** AI-supported diagnosis helps catch subtle findings (e.g. early tumors) that might be missed, thereby improving patient outcomes【18†L287-L294】【20†L430-L438】.

## Tech Stack

- **Language:** Python 3.x  
- **Libraries:** TensorFlow (Keras), NumPy, OpenCV (cv2), Matplotlib, scikit-learn  
- **Model:** Transfer learning with a pre-trained MobileNetV2 CNN (TensorFlow/Keras) for classification.  
- **Data:** COVID-19 radiographic dataset

## Dataset

We use [publicly available images] such as chest X-rays labeled for disease. Example datasets:

- **Lung opacity Dataset:** ~112,000 frontal X-ray images with labels【29†L254-L262】.  
- **Kaggle Pneumonia Dataset:** Thousands of pediatric X-rays labeled Normal or Pneumonia.  
- **COVID:**COVID-19 radiographic dataset.  

Images are organized into `data/train/` and `data/test/` directories by class. A *single run* uses these to train and evaluate the CNN.

## Architecture

The system pipeline is: **Input → Preprocessing → CNN Feature Extraction → Classifier → Evaluation → Visualization**. Key components:

- **Preprocessing:** Resize images to 224×224, normalize pixel values (0–1), and augment training data (flip, rotate).  
- **Feature Extractor:** MobileNetV2 base (frozen weights from ImageNet) automatically learns relevant image features.  
- **Classifier:** Dense layers on top of MobileNetV2 output to predict disease presence (sigmoid output).  
- **Training:** We train the model with labeled data (e.g. Pneumonia vs. Normal) using binary crossentropy loss.  
- **Evaluation:** Model is tested on a hold-out set; metrics (accuracy, precision, recall) and a confusion matrix are computed.  

The `src/` folder contains all code modules (preprocessing, model definition). The block diagram below illustrates this flow:


