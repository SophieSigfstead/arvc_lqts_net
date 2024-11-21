import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import defaultdict
import glob
import re
from keras import layers, models
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import importlib

# Function to extract patient ID from the filename
def extract_patient_id(filename):
    match = re.match(r"(\d{6})", os.path.basename(filename))
    return match.group(1) if match else None

def load_ecg_data(root_dir):

    class_map = {
        "ARVC": [
            "arvc_gene_negative_definite_csv", "arvc_gene_positive_probable_csv",
            "arvc_gene_negative_probable_csv", "arvc_gene_positive_definite_csv"
        ],
        "CONTROL_LQTS": ["control_csv", "lqts_gene_negative_csv", "lqts_type_1_csv", "lqts_type_2_csv"],
    }

    # Load metadata files
    arvc_control_qt = pd.read_csv('METADATA_Gene_Positive_vs_(Gene_Negative+Unaffected_Relatives+Control).csv')
    arvc_control_qt['filename'] = arvc_control_qt['filename'].str.replace('.xml', '.csv', regex=False)

    stollery_qt = pd.read_csv('stollery_qc_2023.01.16-3.csv')

    regular_lqts = pd.read_csv('ecg_qc2022.12.21-21.csv')
    
    # Collect bad ECG filenames
    bad_ecgs_arvc = arvc_control_qt[arvc_control_qt['qc'] != "good"]['filename'].tolist()
    bad_ecgs_stollery = stollery_qt[stollery_qt['qc'] != "Good"]['file'].tolist()
    bad_ecgs_lqts = regular_lqts[regular_lqts['qc'] != "Good"]['file'].tolist()
    print(len(bad_ecgs_arvc))
    print(len(bad_ecgs_stollery))
    print(len(bad_ecgs_lqts))

    bad_count = 0
    bad_ecgs = set(bad_ecgs_arvc + bad_ecgs_stollery + bad_ecgs_lqts)  # Combine all bad ECGs into a set
    print("Number of bad ecgs" , len(list(set(bad_ecgs_arvc + bad_ecgs_stollery + bad_ecgs_lqts))))
    
    patient_map = defaultdict(list) 
    for label, dirs in class_map.items():
        for sub_dir in dirs:
            full_dir = os.path.join(root_dir, sub_dir)
            for file_path in glob.glob(os.path.join(full_dir, "*.csv")):
                # Extract filename from file path
                file_name = os.path.basename(file_path)

                # Exclude bad ECGs
                if file_name in bad_ecgs:
                    bad_count +=1
                    continue

                # Extract patient ID
                patient_id = extract_patient_id(file_path)
                if patient_id:
                    patient_map[patient_id].append((file_path, label))
    print("Bad count", bad_count)
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

# Function to apply augmentations
def augment_ecg_data(ecg_data):
    noise = np.random.normal(0, 0.05, ecg_data.shape)
    ecg_data_noisy = ecg_data + noise

    scale_factor = np.random.uniform(0.9, 1.1)
    ecg_data_scaled = ecg_data_noisy * scale_factor

    jitter_factor = np.random.normal(0, 0.02, ecg_data.shape)
    ecg_data_jittered = ecg_data_scaled + jitter_factor

    return ecg_data_jittered

# Function to prepare the dataset
def prepare_dataset(patient_map, patient_ids, augment=False):
    X = []
    y = []
    label_encoding = {"ARVC": 0, "CONTROL_LQTS": 1}

    for patient_id in patient_ids:
        for file_path, label in patient_map[patient_id]:
            ecg_data = preprocess_ecg(file_path)
            if ecg_data is not None:
                X.append(ecg_data)
                y.append(label_encoding[label])
                if augment:
                    augmented_data = augment_ecg_data(ecg_data)
                    X.append(augmented_data)
                    y.append(label_encoding[label])

    return np.array(X), np.array(y, dtype=np.int32)

# Function to load a specific model
def load_model(model_name, input_shape=(2500, 8)):
    try:
        model_module = importlib.import_module(f"models_2_class.{model_name}")
        model = model_module.build_model(input_shape)
        print(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

# Main function for training and evaluation
def train_and_evaluate(root_dir, model, epochs, batch_size, experiment):
    patient_map = load_ecg_data(root_dir)
    train_ids, val_ids, test_ids = split_data(patient_map)

    X_train, y_train = prepare_dataset(patient_map, train_ids, augment=False)
    X_val, y_val = prepare_dataset(patient_map, val_ids, augment=False)
    X_test, y_test = prepare_dataset(patient_map, test_ids, augment=False)

    # Print data types for debugging
    print(f"X_train shape: {X_train.shape}, y_train dtype: {y_train.shape}")
    print(f"X_tval dtype: {X_val.shape}, y_train dtype: {y_val.shape}")
    print(f"X_train dtype: {X_test.shape}, y_train dtype: {y_test.shape}")

    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    experiment.log_metric("Test_Loss", test_loss)
    experiment.log_metric("Test_Accuracy", test_accuracy)

    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion Matrix:")
    print(cm)
    experiment.log_confusion_matrix(y_true=y_test, y_predicted=y_pred, labels=[0, 1])

    # Calculate Sensitivity and Specificity

    tp = cm[1, 1]  # True Positives
    fp = cm[0, 1]  # False Positives
    fn = cm[1, 0]  # False Negatives
    tn = cm[0, 0]  # True Negatives

    # Calculate Sensitivity and Specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    experiment.log_metric("Test_Sensitivity", sensitivity.mean())
    experiment.log_metric("Test_Specificity", specificity.mean())

    # ROC-AUC Score
    y_test_binarized = label_binarize(y_test, classes=[0, 1])
    auc_score = roc_auc_score(y_test_binarized, y_pred, average='macro', multi_class='ovr')
    print(f"Macro-Averaged ROC-AUC Score: {auc_score:.2f}")
    experiment.log_metric("Test_ROC_AUC", auc_score)

    return model, history