# EEG Motor Imagery Classification 🧠🖐️

This repository contains Python code for processing Electroencephalography (EEG) data, extracting features, and classifying motor imagery (left vs. right hand) using various machine learning models. It explores two primary pipelines: a **subject-specific approach** (`specific.py`) that optimizes models for individual users, and a **generalized approach** (`general.py`) that trains a single model on combined data from all subjects.

## Table of Contents

* [About the AURA Project](#about-the-aura-project-)
* [How it Works: Two Approaches](#how-it-works-two-approaches-%EF%B8%8F)
* [Features](#features-)
* [Installation](#installation-)
* [Setup Requirements](#setup-requirements-)
* [Data](#data-)
* [Usage](#usage-%EF%B8%8F)
* [Output Logs](#output-logs-)
* [Code Structure](#code-structure-%EF%B8%8F)
* [Benchmarks](#benchmarks-)
* [License](#license-)
* [Credits](#credits-)

## About the AURA Project 🤖

This project is part of **AURA (Agentic Unified Robotics Architecture)**, an experimental system exploring intention-driven collaboration between humans and robots via Brain-Computer Interfaces (BCI).

AURA focuses on interpreting high-level user intent rather than low-level control. It enables users to select objects and destinations using discrete EEG signals like SSVEP (Steady-State Visually Evoked Potentials) or motor imagery. Once the intent is identified, the robotic system autonomously executes tasks such such as grasping and placement, optimizing motion through computer vision and intelligent planning. This reframes human-robot interaction from direct control to cognitive partnership.

**User Flow (TL;DR):**
AURA uses SSVEP for visual stimulus-driven object selection (up to four on-screen options), followed by a confirmation. Motor Imagery (MI) then provides a two-choice system for primary manipulation: grasping and releasing the selected object.

## How it Works: Two Approaches ⚙️

This repository provides two distinct EEG classification pipelines:

### 1. Subject-Specific Model Training (`specific.py`)

This approach focuses on building a personalized model for each individual user. The `specific.py` script handles this.

* **Per-Subject Data Processing:** For each subject, EEG data is loaded, preprocessed (notch filtering, bandpass filtering, and enhanced artifact removal based on amplitude and variance thresholds), and features are extracted.
* **Per-Subject Feature Selection:** `SelectKBest` with `mutual_info_classif` is applied to each subject's dataset to identify the most informative features *specific to that individual's neural patterns*.
* **Per-Subject Model Training & Selection:** Multiple machine learning models are independently trained and hyperparameter-tuned using cross-validation on the selected features for that specific subject. The model (or ensemble) that achieves the highest accuracy for that particular subject is identified as the best performer and cached.

**Benefit:** This approach aims to maximize individual performance by tailoring the model to each user's unique brain signals.

### 2. Generalized Model Training (`general.py`)

This approach trains a single model on data combined from all subjects, aiming for a model that can generalize across different users. This is the pipeline implemented in `general.py`.

* **Data Loading & Preprocessing:** EEG data for each subject is loaded, preprocessed (notch filtering, bandpass filtering, and artifact removal).
* **Feature Extraction:** For each preprocessed trial, a comprehensive set of features (including band power, time-frequency, and statistical measures) is extracted.
* **Data Consolidation:** Features and labels from *all subjects* are combined into a single, large dataset.
* **Single Feature Selection:** `SelectKBest` with `mutual_info_classif` is applied to this consolidated dataset to identify the most informative features that have the highest statistical dependency with the motor imagery classes *across all subjects*.
* **Single Model Training & Selection:** Multiple machine learning models are independently trained and hyperparameter-tuned using cross-validation on the *selected features from the combined dataset*. The model (or ensemble) that achieves the highest accuracy on a dedicated test set is identified as the best overall performer. This best model, along with its scaler and metrics, is then cached.

**Consideration (Potential Inefficiency for Personalization):** While this generalized approach simplifies deployment by providing a single model, it may be less efficient or achieve lower performance for *individual users* compared to a subject-specific model. This is because individual neural patterns and optimal features can vary significantly between people, and a generalized model might not capture these unique characteristics as effectively as a model trained specifically for one person. For personalized BCI applications, a subject-specific approach is often preferred for optimal performance.

## Features ✨

* **EEG Preprocessing:**
    * Notch filtering (50 Hz or 60 Hz configurable).
    * Bandpass filtering (8-30 Hz).
    * Enhanced artifact removal based on amplitude and variance thresholds.
* **Feature Extraction:**
    * **Traditional Features:** Mean, Standard Deviation, Peak-to-Peak, Skewness, Kurtosis, Entropy, RMS, Median.
    * **Hjorth Parameters:** Mobility and Complexity.
    * **Band Power:** Alpha (8-12 Hz) and Beta (13-30 Hz) power.
    * **Time-Frequency Features:** Mean power in Alpha and Beta bands derived from Short-Time Fourier Transform (STFT).
* **Feature Selection:** Utilizes `SelectKBest` with `mutual_info_classif` to select the most relevant features.
* **Machine Learning Models:**
    * Linear Discriminant Analysis (LDA)
    * Support Vector Machine (SVM)
    * Random Forest Classifier
    * K-Nearest Neighbors (KNN)
    * Logistic Regression
    * AdaBoost Classifier
    * Gradient Boosting Classifier
    * XGBoost Classifier
    * Ensemble Voting Classifiers (Hard and Soft)
* **Model Training & Evaluation:**
    * `train_test_split` with stratification for robust evaluation.
    * `StandardScaler` for feature scaling.
    * `GridSearchCV` for hyperparameter tuning of selected classifiers.
    * Evaluation metrics: Accuracy, Precision, Recall, F1-score.
* **Data and Model Caching:** Processed features and labels, as well as trained models and their scalers, are cached using `joblib` to speed up subsequent runs.
* **Logging:** Detailed and summary logs of the pipeline performance are generated.

## Installation 📦

To set up the environment, clone the repository and install the required Python packages listed in `requirements.txt`:

```bash
# Clone the repository (if applicable)
# git clone <repository_url>
# cd <repository_directory>

pip install -r requirements.txt
```

## Setup Requirements 📋

This project requires Python 3.8 or higher. Ensure you have `pip` installed for package management.

## Data 📊

The scripts expect EEG data in `.mat` files. Each `.mat` file should contain a structure named `eeg` with the following key fields:

* `srate`: Sampling rate (e.g., 250 Hz).
* `imagery_left`: A cell array (or similar structure) containing individual trial data for left-hand motor imagery. Each trial should be a 2D array of shape `(samples, channels)`.
* `imagery_right`: Similar to `imagery_left`, but for right-hand motor imagery.

Place your `.mat` files in a directory named `data/EEG Raw Data/` relative to the script's location. For example:

```
.
├── general.py
├── specific.py
└── data/
    └── EEG Raw Data/
        ├── s01.mat
        ├── s02.mat
        └── ...
```

## Usage ▶️

To run the **generalized classification pipeline**, execute:

```bash
python general.py
```

This script will load data from all subjects, preprocess, extract features, combine them, train and evaluate multiple models, and select the best overall model. It will attempt to load cached features and models first to speed up processing.

To run the **subject-specific classification pipeline**, execute:

```bash
python specific.py
```

This script will iterate through subjects, load their data, preprocess, extract features, train and evaluate models *individually* for each subject, and log the results. It will also attempt to load cached features and models first to speed up processing for individual subjects.

### Output Logs 📝

Upon completion, two log files will be generated in the `logs/` directory:

* `logs/details.txt`: Contains detailed results for all trained models, including metrics and selected features.
* `logs/summary.txt`: Provides a concise summary of the best model's performance.

## Code Structure 🏗️

The repository contains two main Python scripts:

* **`general.py`:** Implements the generalized model training pipeline. It loads and combines data from all subjects, performs a single feature selection step, and trains a single, generalized model.
* **`specific.py`:** Implements the subject-specific training pipeline. This involves iterating through subjects, applying preprocessing steps (notch filtering, bandpass filtering, and enhanced artifact removal), extracting features (traditional and time-frequency), performing subject-specific feature selection, and training/caching models individually for each subject.

Common functions shared between both approaches (like filtering, basic feature extraction helpers, artifact removal, caching, feature selection, and metric calculation) are designed to be modular.

## Benchmarks 📈

Detailed benchmark results and performance comparisons for different models and approaches can be found in `benchmarks.md`.

## License 📄

This project is open-source and available under the MIT License.

## Credits 🙏

The EEG dataset used for training, comprising data from 52 subjects, was obtained from:
* **GigaDB:** [https://gigadb.org/dataset/view/id/100295/](https://gigadb.org/dataset/view/id/100295/)
