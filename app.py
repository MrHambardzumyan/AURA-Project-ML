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

# === Load cached model and scaler ===
def load_cached_model(model_name, cache_dir="cache"):
    model_path = os.path.join(cache_dir, f"{model_name}.joblib")
    metrics_path = os.path.join(cache_dir, f"{model_name}_metrics.joblib")
    scaler_path = os.path.join(cache_dir, f"{model_name}_scaler.joblib") # New scaler path

    if os.path.exists(model_path) and os.path.exists(metrics_path) and os.path.exists(scaler_path):
        print(f"[Cache HIT] Model {model_name} loaded from cache.")
        model = joblib.load(model_path)
        metrics_data = joblib.load(metrics_path)
        metrics = metrics_data["metrics"]
        features = metrics_data["features"]
        scaler = joblib.load(scaler_path) # Load the scaler
        return model, metrics, features, scaler
    else:
        print(f"[Cache MISS] No cached model/scaler for {model_name}.")
        return None, None, None, None

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
    # Feature names updated to include time-frequency
    csp_feature_names = [f"CSPComp{i+1}_{b}" for i in range(4) for b in ['Alpha', 'Beta']]
    traditional_feature_names = [
        'Alpha', 'Beta', 'Mean', 'STD', 'PeakToPeak',
        'Skewness', 'Kurtosis', 'Entropy', 'RMS', 'Median',
        'HjorthMobility', 'HjorthComplexity', 'TF_AlphaPower', 'TF_BetaPower'
    ]
    feature_names = csp_feature_names + traditional_feature_names

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

    os.makedirs("logs", exist_ok=True)
    summary = []

    with open("logs/details.txt", "w") as f_log:
        for subj in range(1, 53):  # Change number of subjects here
            subj_id = f"s{subj:02d}"
            path = f"data/EEG Raw Data/{subj_id}.mat"
            print(f"\n>> Processing {subj_id}")
            try:
                X, y = load_subject_data_cached(path)
                f_log.write(f"\n=== {subj_id} ===\n")
            except Exception as e:
                f_log.write(f"{subj_id} load error: {e}\n")
                continue

            # Attempt to load the cached best model, its metrics, features, and scaler for this subject
            cached_model_obj, cached_best_metrics, cached_best_features, loaded_scaler = load_cached_model(f"best_model_{subj_id}")

            if cached_model_obj and loaded_scaler: # Check if both model object and scaler were successfully loaded
                # If the model exists in cache, we skip training and use the cached model
                f_log.write(f"Selected features: {cached_best_features}\n")
                f_log.write(f"Best model: {type(cached_model_obj).__name__} ({cached_best_metrics['accuracy']*100:.2f}%)\n")
                f_log.write(f"Metrics: {cached_best_metrics}\n")

                summary.append([subj_id, type(cached_model_obj).__name__, round(cached_best_metrics['accuracy']*100, 2), cached_best_metrics])
                continue  # Skip the rest of the processing for this subject


            # Proceed with model evaluation if no cached model exists
            try:
                X_selected, selected_idx = select_features(X, y)
                selected_names = [feature_names[i] for i in selected_idx]
                f_log.write(f"Selected features: {selected_names}\n")
            except Exception as e:
                f_log.write(f"{subj_id} feature selection error: {e}\n")
                continue

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.3, stratify=y, random_state=42
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            best_acc = 0
            best_model_name = ""
            best_model_obj = None # Stores the actual best model object
            best_metrics = {}
            trained = []

            # Evaluate each classifier
            for name, clf in classifiers.items():
                if name in param_grids:
                    clf_grid = GridSearchCV(clf, param_grids[name], cv=3, n_jobs=-1)
                    clf_grid.fit(X_train, y_train)
                    model = clf_grid.best_estimator_
                else:
                    model = clf.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                metrics = evaluate_metrics(y_test, y_pred)
                acc = metrics['accuracy'] * 100
                trained.append((name, model))
                if acc > best_acc:
                    best_acc, best_model_name, best_model_obj, best_metrics = acc, name, model, metrics # Correctly assign model object

            # Voting hard
            hard_vote = VotingClassifier(trained, voting="hard").fit(X_train, y_train)
            y_pred = hard_vote.predict(X_test)
            hard_acc = accuracy_score(y_test, y_pred) * 100
            if hard_acc > best_acc:
                best_acc, best_model_name, best_model_obj, best_metrics = hard_acc, "VotingHard", hard_vote, evaluate_metrics(y_test, y_pred) # Correctly assign hard_vote object

            # Voting soft
            soft_models = [(n, m) for n, m in trained if hasattr(m, 'predict_proba')]
            if soft_models:
                soft_vote = VotingClassifier(soft_models, voting="soft").fit(X_train, y_train)
                y_pred = soft_vote.predict(X_test)
                soft_acc = accuracy_score(y_test, y_pred) * 100
                if soft_acc > best_acc:
                    best_acc, best_model_name, best_model_obj, best_metrics = soft_acc, "VotingSoft", soft_vote, evaluate_metrics(y_test, y_pred) # Correctly assign soft_vote object

            f_log.write(f"Best model: {best_model_name} ({best_acc:.2f}%)\n")
            f_log.write(f"Metrics: {best_metrics}\n")
            summary.append((subj_id, best_model_name, best_acc, best_metrics))

            # Cache the best model for this subject
            if best_model_obj and best_metrics:
                cache_trained_model(best_model_obj, best_metrics, scaler, f"best_model_{subj_id}", selected_names)


    # Summary of results
    with open("logs/summary.txt", "w") as f:
        for sid, model_name, acc, metrics in summary:
            f.write(f"{sid}: Best Model - {model_name} | ({acc:.2f}%) Metrics: {metrics}\n")

    print("\n🎉 All done! Check logs/details.txt and logs/summary.txt")