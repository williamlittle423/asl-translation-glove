# ASL Translation Glove

This project was developed for the ENPH 454 Final Capstone Project at Queen's University by William Little and Eric deKemp.

## Overview

The **ASL Translation Glove** is a motion-tracking glove designed to recognize American Sign Language (ASL) letters in real-time. Utilizing a custom-built hardware setup and a Multi-Layer Perceptron (MLP) model, the glove translates hand movements into corresponding ASL letters with high accuracy and efficiency.

## Features

- **Real-Time Gesture Recognition:** Instantly translates hand movements into ASL letters.
- **Custom Hardware Integration:** Interfaces with Raspberry Pi 5 for data processing.
- **Efficient Data Processing:** Reduces input dimensionality from 960 to 60 features.
- **Automated Data Collection:** Simplifies gathering and storing gesture data.
- **Comprehensive Evaluation:** Includes scripts for testing and evaluating model performance.

## Software Stack

- **Python 3.10**
- **PyTorch**
- **NumPy**
- **Git & GitHub** for version control

## Data Collection

- **Duration:** 1.5-second recordings per movement, based on average ASL letter performance time.
- **Sampling Rate:** 50 Hz (20 ms intervals) across 32 time steps.
- **Data Points:** 960 data points per sample
- **Script:** `collect_input_data.py` automates sample collection and saves data to CSV.

## Data Pre-processing

To enhance model efficiency and prevent overfitting:
- **Feature Extraction:** Calculated mean and standard deviation for each axis and sensor.
- **Dimensionality Reduction:** Reduced input size from 960 to 60 features (16× reduction).

## Model Architecture

- **Type:** Multi-Layer Perceptron (MLP)
- **Mapping Function:**
  \[
  f_{\theta}: \mathbb{R}^{60} \rightarrow \mathcal{C}
  \]
  where \( \mathcal{C} \) is the set of 26 ASL letters.
- **Loss Function:** Cross-Entropy Loss for multi-class classification.

## Hyperparameter Optimization

Utilized **Grid Search** to select optimal hyperparameters:
- **Epochs:** 50
- **Hidden Layers:** 2
- **Neurons per Layer:** 512 & 256
- **Learning Rate:** 0.001

## Scripts

- **Data Collection:** `collect_input_data.py`
- **Model Training:** `train_asl_mlp.py`
- **Inference:** `infer_asl_mlp.py`
- **Evaluation:** `letter_evaluation.py`
- **Model Definition:** `asl_mlp_model.py`

## Usage

### Prerequisites

Install required Python libraries:

```bash
pip install torch numpy pandas scikit-learn
