# EEG Motor Imagery Classification 🧠🖐️

This repository contains Python code for processing Electroencephalography (EEG) data, extracting features, and classifying motor imagery (left vs. right hand) using various machine learning models. The pipeline includes advanced preprocessing steps, traditional EEG features, time-frequency analysis, and ensemble learning.

## Table of Contents

* [About the AURA Project](#about-the-aura-project-)
* [How it Works](#how-it-works-)
* [Features](#features-)
* [Installation](#installation-)
* [Setup Requirements](#setup-requirements-)
* [Data](#data-)
* [Usage](#usage-)
* [Output Logs](#output-logs-)
* [Code Structure](#code-structure-)
* [Benchmarks](#benchmarks-)
* [Contributing](#contributing-)
* [License](#license-)
* [Credits](#credits-)

## About the AURA Project 🤖

This project is part of **AURA (Agentic Unified Robotics Architecture)**, an experimental system exploring intention-driven collaboration between humans and robots via Brain-Computer Interfaces (BCI).

AURA focuses on interpreting high-level user intent rather than low-level control. It enables users to select objects and destinations using discrete EEG signals like SSVEP (Steady-State Visually Evoked Potentials) or motor imagery. Once the intent is identified, the robotic system autonomously executes tasks such such as grasping and placement, optimizing motion through computer vision and intelligent planning. This reframes human-robot interaction from direct control to cognitive partnership.

**User Flow (TL;DR):**
AURA uses SSVEP for visual stimulus-driven object selection (up to four on-screen options), followed by a confirmation. Motor Imagery (MI) then provides a two-choice system for primary manipulation: grasping and releasing the selected object.

## How it Works ⚙️

The core processing pipeline operates on a per-subject basis to ensure personalized model performance:

1.  **Feature Extraction & Selection:** For each subject's EEG data, a comprehensive set of features (including band power and time-frequency) is extracted. Then, `SelectKBest` with `mutual_info_classif` is used to identify and select the most informative features that have the highest statistical dependency with the motor imagery classes.
2.  **Model Training:** Multiple machine learning models (e.g., SVM, Random Forest, XGBoost, etc.) are independently trained and hyperparameter-tuned using cross-validation on the selected features for that specific subject. Ensemble methods (Voting Classifiers) are also included.
3.  **Best Model Selection:** After training and evaluating all models on a test set, the model (or ensemble) that achieves the highest accuracy for that particular subject is identified as the best performer. This best model, along with its scaler and metrics, is then cached for faster loading in subsequent runs.

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
* **Logging:** Detailed and summary logs of subject-wise performance are generated.

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

The script expects EEG data in `.mat` files. Each `.mat` file should contain a structure named `eeg` with the following key fields:

* `srate`: Sampling rate (e.g., 250 Hz).
* `imagery_left`: A cell array (or similar structure) containing individual trial data for left-hand motor imagery. Each trial should be a 2D array of shape `(samples, channels)`.
* `imagery_right`: Similar to `imagery_right`, but for right-hand motor imagery.

Place your `.mat` files in a directory named `data/EEG Raw Data/` relative to the script's location. For example:

```
.
├── app.py
└── data/
    └── EEG Raw Data/
        ├── s01.mat
        ├── s02.mat
        └── ...
```

## Usage ▶️

To run the full classification pipeline, simply execute the script:

```bash
python app.py
```

The script will iterate through subjects (from `s01.mat` to `s52.mat`), load their data, preprocess, extract features, train and evaluate models, and log the results. It will attempt to load cached features and models first to speed up processing.

### Output Logs 📝

Upon completion, two log files will be generated in the `logs/` directory:

* `logs/details.txt`: Contains detailed information for each subject, including selected features, best model, and full evaluation metrics.
* `logs/summary.txt`: Provides a concise summary of the best model and accuracy for each subject.

## Code Structure 🏗️

The script is organized into several functions, each responsible for a specific part of the EEG processing and classification pipeline:

* **Filtering:**
    * `notch_filter(data, fs, freq, quality)`: Applies a notch filter.
    * `bandpass_filter(data, lowcut, highcut, fs, order)`: Applies a bandpass filter.
* **Feature Extraction Helpers:**
    * `calculate_band_power(data, fs, band)`: Computes power in a specified frequency band.
    * `calculate_entropy(data)`: Calculates Shannon entropy.
    * `hjorth_parameters(data)`: Computes Hjorth mobility and complexity.
    * `time_frequency_features(data, fs)`: Extracts alpha and beta power using STFT.
* **Preprocessing:**
    * `remove_artifacts(trials, amp_percentile, var_percentile)`: Removes trials based on amplitude and variance thresholds.
* **Caching:**
    * `cache_trained_model(model, metrics, scaler, model_name, selected_names, cache_dir)`: Caches a trained model, its metrics, and the scaler.
    * `load_cached_model(model_name, cache_dir)`: Loads a cached model, its metrics, features, and scaler.
* **Main Pipeline Functions:**
    * `extract_features(trial_data, fs)`: Extracts a comprehensive set of traditional and time-frequency features for a single channel.
    * `load_subject_data_cached(file_path, cache_dir, overwrite)`: Loads, preprocesses, and extracts features for a subject, with caching. This is the core data pipeline.
    * `select_features(X, y, top_k)`: Performs feature selection.
    * `evaluate_metrics(y_true, y_pred)`: Calculates classification metrics.
* **Main Execution Block (`if __name__ == "__main__":`)**: Orchestrates the entire process, including model training, hyperparameter tuning, and logging.

## Benchmarks 📈

Detailed benchmark results and performance comparisons for different models and subjects can be found in `benchmarks.md`.

## Contributing 🤝

Feel free to fork the repository, open issues, or submit pull requests.

## License 📄

This project is open-source and available under the MIT License.

## Credits 🙏

The EEG dataset used for training, comprising data from 52 subjects, was obtained from:
* **GigaDB:** [https://gigadb.org/dataset/view/id/100295/](https://gigadb.org/dataset/view/id/100295/)
