import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import defaultdict
import glob
import re
from keras import layers, models, callbacks
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import importlib

# Function to extract patient ID from the filename
def extract_patient_id(filename):
    match = re.match(r"(\d{6})", os.path.basename(filename))
    return match.group(1) if match else None

# Function to load ECG data and group by patient ID
def load_ecg_data(root_dir):
    class_map = {
        "ARVC": [
            "arvc_gene_negative_definite_csv", "arvc_gene_negative_possible_csv",
            "arvc_gene_negative_probable_csv", "arvc_gene_positive_definite_csv",
            "arvc_gene_positive_possible_csv", "arvc_gene_positive_probable_csv"
        ],
        "CONTROL": ["control_csv"],
        "LQTS-negative": ["lqts_gene_negative_csv"],
        "LQTS-type1": ["lqts_type_1_csv"],
        "LQTS-type2": ["lqts_type_2_csv"]
    }

    patient_map = defaultdict(list)

    for label, dirs in class_map.items():
        for sub_dir in dirs:
            full_dir = os.path.join(root_dir, sub_dir)
            for file_path in glob.glob(os.path.join(full_dir, "*.csv")):
                patient_id = extract_patient_id(file_path)
                if patient_id:
                    patient_map[patient_id].append((file_path, label))

    return patient_map

# Function to split data without splitting patient IDs
def split_data(patient_map, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    patient_ids = list(patient_map.keys())
    train_ids, temp_ids = train_test_split(patient_ids, test_size=(val_ratio + test_ratio), random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    return train_ids, val_ids, test_ids

# Function to preprocess the ECG data
def preprocess_ecg(file_path):
    df = pd.read_csv(file_path).drop(columns=["Unnamed: 0"])
    ecg_data = df.values

    if ecg_data.shape != (2500, 8):
        print(f"Warning: Skipping file {file_path} with unexpected shape {ecg_data.shape}")
        return None

    return ecg_data

# Function to prepare the dataset
def prepare_dataset(patient_map, patient_ids):
    X = []
    y = []
    label_encoding = {"ARVC": 0, "CONTROL": 1, "LQTS-negative": 2, "LQTS-type1": 3, "LQTS-type2": 4}

    for patient_id in patient_ids:
        for file_path, label in patient_map[patient_id]:
            ecg_data = preprocess_ecg(file_path)
            if ecg_data is not None:
                X.append(ecg_data)
                y.append(label_encoding[label])

    X = np.array(X)
    y = np.array(y)
    return X, y

# Function to load a specific model
def load_model(model_name, input_shape=(2500, 8)):
    try:
        # Import the model module from the 'models' directory
        model_module = importlib.import_module(f"models.{model_name}")
        model = model_module.build_model(input_shape)
        print(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

# Main function for training and evaluation
def train_and_evaluate(root_dir, model):
    print("Loading data...")
    patient_map = load_ecg_data(root_dir)

    print("Splitting data...")
    train_ids, val_ids, test_ids = split_data(patient_map)

    print("Preparing datasets...")
    X_train, y_train = prepare_dataset(patient_map, train_ids)
    X_val, y_val = prepare_dataset(patient_map, val_ids)
    X_test, y_test = prepare_dataset(patient_map, test_ids)

    print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}, Test data shape: {X_test.shape}")

    print("Training model...")
    batch_size = min(32, len(X_train)) 
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=batch_size)

    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Predicting on the test set
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Binarize the labels for ROC-AUC calculation
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    auc_score = roc_auc_score(y_test_binarized, y_pred_probs, average='macro', multi_class='ovr')
    print(f"Macro-Averaged ROC-AUC Score: {auc_score:.2f}")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ARVC", "CONTROL", "LQTS-negative", "LQTS-type1", "LQTS-type2"]))

    return model, history