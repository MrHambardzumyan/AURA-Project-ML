# Imports
import numpy as np
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch, iirnotch, stft
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import xgboost as xgb
import warnings
import os
import joblib

# Warnings
warnings.filterwarnings("ignore")

# === Notch filter at 50 Hz (or 60 Hz) ===
def notch_filter(data, fs, freq=50.0, quality=30.0):
    b, a = iirnotch(freq / (0.5 * fs), quality)
    return filtfilt(b, a, data, axis=-1)

# === Bandpass filter 8-30 Hz ===
def bandpass_filter(data, lowcut=8, highcut=30, fs=250, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, data, axis=-1)

# === Power in frequency band ===
def calculate_band_power(data, fs, band):
    f, Pxx = welch(data, fs, nperseg=512)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])

# === Entropy calculation ===
def calculate_entropy(data):
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    hist, _ = np.histogram(data, bins=50, range=(0, 1), density=True)
    return scipy_entropy(hist + 1e-8)

# === Hjorth parameters ===
def hjorth_parameters(data):
    eps = 1e-10
    first_deriv = np.diff(data)
    second_deriv = np.diff(first_deriv)
    var_zero = np.var(data) + eps
    var_d1 = np.var(first_deriv) + eps
    var_d2 = np.var(second_deriv) + eps
    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / (mobility + eps)
    return mobility, complexity

# === Traditional features per channel (updated with notch + bandpass 8-30 Hz) ===
def extract_features(trial_data, fs):
    # Notch filter (50 Hz)
    filtered_notch = notch_filter(trial_data, fs, freq=50)
    # Bandpass 8-30 Hz
    filtered = bandpass_filter(filtered_notch, 8, 30, fs)
    mean_val = np.mean(filtered)
    std_val = np.std(filtered)
    ptp_val = np.ptp(filtered)
    skew_val = skew(filtered)
    kurt_val = kurtosis(filtered)
    entropy_val = calculate_entropy(filtered)
    rms_val = np.sqrt(np.mean(filtered ** 2))
    median_val = np.median(filtered)
    mobility, complexity = hjorth_parameters(filtered)
    # Band power features in alpha, beta bands only for consistency
    band_features = [calculate_band_power(filtered, fs, b) for b in [(8,12),(13,30)]]
    # Add time-frequency features (mean power over time in alpha and beta bands)
    tf_feats = time_frequency_features(filtered, fs)
    return band_features + [mean_val, std_val, ptp_val, skew_val, kurt_val,
                            entropy_val, rms_val, median_val, mobility, complexity] + tf_feats

# === Time-frequency features via STFT ===
def time_frequency_features(data, fs):
    f, t, Zxx = stft(data, fs, nperseg=128)
    power = np.abs(Zxx) ** 2
    alpha_idx = np.where((f >= 8) & (f <= 12))[0]
    beta_idx = np.where((f >= 13) & (f <= 30))[0]
    alpha_power = np.mean(power[alpha_idx, :])
    beta_power = np.mean(power[beta_idx, :])
    return [alpha_power, beta_power]

# === Enhanced artifact removal (amplitude + variance) ===
def remove_artifacts(trials, amp_percentile=95, var_percentile=95):
    max_vals = [np.max(np.abs(trial)) for trial in trials]
    var_vals = [np.var(trial) for trial in trials]
    amp_thresh = np.percentile(max_vals, amp_percentile)
    var_thresh = np.percentile(var_vals, var_percentile)
    cleaned = [trial for trial in trials if np.max(np.abs(trial)) <= amp_thresh and np.var(trial) <= var_thresh]
    return cleaned

# === Load subject data with all preprocessing and features ===
def load_subject_data_cached(file_path, cache_dir="cache", overwrite=False):
    os.makedirs(cache_dir, exist_ok=True)
    subject_id = os.path.basename(file_path).replace(".mat", "")
    cache_path = os.path.join(cache_dir, f"{subject_id}_features.joblib")

    if os.path.exists(cache_path) and not overwrite:
        print(f"[Cache HIT] Loaded cached features for {subject_id}")
        return joblib.load(cache_path)

    print(f"[Cache MISS] Computing features for {subject_id} ...")
    data = loadmat(file_path)
    eeg = data['eeg'][0][0]
    fs = eeg['srate'][0][0]
    left_trials = remove_artifacts(eeg['imagery_left'])
    right_trials = remove_artifacts(eeg['imagery_right'])
    if not left_trials or not right_trials:
        raise ValueError("No trials remain after artifact removal.")

    features, labels = [], []
    for trial in left_trials:
        features.append(extract_features(trial, fs))
        labels.append(0)
    for trial in right_trials:
        features.append(extract_features(trial, fs))
        labels.append(1)

    X, y = np.array(features), np.array(labels)
    joblib.dump((X, y), cache_path)
    print(f"[Cache SAVE] Features cached for {subject_id}")
    return X, y

# === Cache trained models and scaler ===
def cache_trained_model(model, metrics, scaler, model_name, selected_names, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, f"{model_name}.joblib")
    metrics_path = os.path.join(cache_dir, f"{model_name}_metrics.joblib")
    scaler_path = os.path.join(cache_dir, f"{model_name}_scaler.joblib") # New scaler path

    data = {
        "metrics": metrics,
        "features": selected_names,
    }

    if not os.path.exists(model_path):
        joblib.dump(model, model_path)
        print(f"[Cache SAVE] Model {model_name} saved.")
    if not os.path.exists(metrics_path):
        joblib.dump(data, metrics_path)
        print(f"[Cache SAVE] Metrics for {model_name} saved.")
    if not os.path.exists(scaler_path): # Save the scaler
        joblib.dump(scaler, scaler_path)
        print(f"[Cache SAVE] Scaler for {model_name} saved.")

    return model_path, metrics_path, scaler_path

# === Feature selection ===
def select_features(X, y, top_k=12):
    selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.get_support(indices=True)

# === Metrics calculation ===
def evaluate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

# === Main execution ===
if __name__ == "__main__":
    feature_names = [
        'Alpha', 'Beta', 'Mean', 'STD', 'PeakToPeak',
        'Skewness', 'Kurtosis', 'Entropy', 'RMS', 'Median',
        'HjorthMobility', 'HjorthComplexity', 'TF_AlphaPower', 'TF_BetaPower'
    ]

    classifiers = {
        "LDA": LDA(),
        "SVM": SVC(probability=True, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=50, verbosity=0, eval_metric='logloss', random_state=42)
    }

    param_grids = {
        "SVM": {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        "RandomForest": {'n_estimators': [50, 100], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
        "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        "LogisticRegression": {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    }

    all_X, all_y = [], []
    failed_subjects = []

    # Load and combine data
    for subj in range(1, 53):
        subj_id = f"s{subj:02d}"
        path = f"data/EEG Raw Data/{subj_id}.mat"
        print(f">> Loading {subj_id}")
        try:
            X, y = load_subject_data_cached(path)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"[ERROR] Failed to process {subj_id}: {e}")
            failed_subjects.append(subj_id)

    if not all_X:
        raise RuntimeError("No subject data could be loaded.")

    # Combine all subject data
    X = np.vstack(all_X)
    y = np.hstack(all_y)

    # Feature selection
    X_selected, selected_idx = select_features(X, y)
    selected_names = [feature_names[i] for i in selected_idx]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    best_acc = 0
    best_model_name = ""
    best_model_obj = None
    best_metrics = {}
    trained_models = []

    # Train classifiers
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        if name in param_grids:
            clf = GridSearchCV(clf, param_grids[name], cv=3, n_jobs=-1)
            clf.fit(X_train, y_train)
            model = clf.best_estimator_
        else:
            model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = evaluate_metrics(y_test, y_pred)
        acc = metrics["accuracy"] * 100
        trained_models.append((name, model, metrics))

        if acc > best_acc:
            best_acc, best_model_name, best_model_obj, best_metrics = acc, name, model, metrics

    # Hard voting
    voting_models = [(name, model) for name, model, _ in trained_models]
    hard_vote = VotingClassifier(voting_models, voting="hard").fit(X_train, y_train)
    y_pred = hard_vote.predict(X_test)
    hard_acc = accuracy_score(y_test, y_pred) * 100
    if hard_acc > best_acc:
        best_acc, best_model_name, best_model_obj = hard_acc, "VotingHard", hard_vote
        best_metrics = evaluate_metrics(y_test, y_pred)

    # Soft voting
    soft_models = [(n, m) for n, m, _ in trained_models if hasattr(m, "predict_proba")]
    if soft_models:
        soft_vote = VotingClassifier(soft_models, voting="soft").fit(X_train, y_train)
        y_pred = soft_vote.predict(X_test)
        soft_acc = accuracy_score(y_test, y_pred) * 100
        if soft_acc > best_acc:
            best_acc, best_model_name, best_model_obj = soft_acc, "VotingSoft", soft_vote
            best_metrics = evaluate_metrics(y_test, y_pred)

    print(f"\n✅ Best Model: {best_model_name} with accuracy {best_acc:.2f}%")
    print(f"Metrics: {best_metrics}")

    # Save final model
    cache_trained_model(
        best_model_obj,
        best_metrics,
        scaler,
        f"best_model_all_subjects",
        selected_names
    )

    # --- Logging ---
    os.makedirs("logs", exist_ok=True)

    # Write summary.txt (brief summary)
    with open("logs/summary.txt", "w") as f_summary:
        f_summary.write("=== EEG Classification Summary ===\n")
        f_summary.write(f"Best Model: {best_model_name}\n")
        f_summary.write(f"Accuracy: {best_acc:.2f}%\n")
        f_summary.write("Metrics:\n")
        for metric, value in best_metrics.items():
            f_summary.write(f"  {metric.capitalize()}: {value:.4f}\n")
        f_summary.write("\nSelected Features:\n")
        f_summary.write(", ".join(selected_names) + "\n")

    print(f"[LOG] Summary saved to logs/summary.txt")

    # Write details.txt (detailed metrics for all models)
    with open("logs/details.txt", "w") as f_details:
        f_details.write("=== EEG Classification Detailed Results ===\n")
        f_details.write(f"Selected Features ({len(selected_names)}):\n")
        f_details.write(", ".join(selected_names) + "\n\n")

        f_details.write("Trained Models:\n")
        for name, model, metrics in trained_models:
            f_details.write(f"Model: {name}\n")
            for metric, value in metrics.items():
                f_details.write(f"  {metric.capitalize()}: {value:.4f}\n")
            f_details.write("\n")

        f_details.write("Voting Classifiers:\n")
        f_details.write(f"Hard Voting Accuracy: {hard_acc:.2f}%\n")
        if soft_models:
            f_details.write(f"Soft Voting Accuracy: {soft_acc:.2f}%\n")

    print(f"[LOG] Detailed results saved to logs/details.txt")
