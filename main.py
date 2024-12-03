import numpy as np
import os
import pickle
import pandas as pd
import logging  # Import logging
import argparse  # Import argparse

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Uncomment the following lines if you need to attach a debugger
# import ptvsd
# ptvsd.enable_attach(log_dir=os.path.dirname(__file__))
# ptvsd.wait_for_attach(timeout=15)

from util import options, data_processing
from ml_models import autoencoder


def setup_logging(log_file):
    """
    Sets up logging to file and console.

    Parameters:
    - log_file (str): Path to the log file.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Define the format for logs
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def train(args):
    logging.info(f"Starting training for model: {args.model_name}")

    # Step 1: Prepare the data for training
    process_data = data_processing.processData(args.data_path)
    X_train, Y_train, X_val, Y_val = process_data.prepareTrainingData()

    # Display shapes and label distribution
    logging.info(f"Training data shape: X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    logging.info(f"Validation data shape: X_val: {X_val.shape}, Y_val: {Y_val.shape}")
    logging.info("Label distribution in training data:")
    logging.info(Y_train.value_counts())

    # Check if there are normal samples (label=0) and anomaly samples (label=1)
    num_normal_samples = (Y_train == 0).sum()
    num_anomaly_samples = (Y_train == 1).sum()
    logging.info(f"Number of normal samples (label=0): {num_normal_samples}")
    logging.info(f"Number of anomaly samples (label=1): {num_anomaly_samples}")

    if num_normal_samples == 0 or num_anomaly_samples == 0:
        logging.error("Insufficient class distribution. Ensure that both normal and anomaly samples are present.")
        raise ValueError("Insufficient class distribution.")

    # Step 2: Handle Class Imbalance with SMOTE
    logging.info("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    logging.info(
        f"After SMOTE, training data shape: X_train_resampled: {X_train_resampled.shape}, Y_train_resampled: {Y_train_resampled.shape}")
    logging.info("Label distribution after SMOTE:")
    logging.info(pd.Series(Y_train_resampled).value_counts())

    # Step 3: Initialize and compile the AutoEncoder model
    model = autoencoder.AutoEncoder(X_train_resampled.shape[1])
    compiled_model = model.compile_model()
    logging.info("AutoEncoder model compiled successfully.")

    # Step 4: Rescale the training data
    x_scale = process_data.dataScaling(X_train_resampled)
    logging.info(f"Rescaled training data shape: {x_scale.shape}")

    # Step 5: Feature Selection using RFE
    logging.info("Performing Recursive Feature Elimination (RFE) for feature selection...")
    # Initialize Logistic Regression for RFE
    lr_estimator = LogisticRegression(solver="lbfgs", max_iter=1000)
    rfe = RFE(estimator=lr_estimator, n_features_to_select=50, step=10)
    rfe.fit(x_scale, Y_train_resampled)
    X_train_rfe = rfe.transform(x_scale)
    X_val_rfe = rfe.transform(process_data.dataScaling(X_val))
    logging.info(f"After RFE, training data shape: {X_train_rfe.shape}, validation data shape: {X_val_rfe.shape}")

    # Save the RFE object
    rfe_save_path = os.path.join(args.ckpt_path, "rfe_selector.pkl")
    with open(rfe_save_path, 'wb') as file:
        pickle.dump(rfe, file)
    logging.info(f"RFE selector saved at: {rfe_save_path}")

    # Step 6: Extract hidden representations from AutoEncoder
    # Separate normal and anomaly data based on labels
    x_norm = X_train_resampled[Y_train_resampled == 0]
    x_fraud = X_train_resampled[Y_train_resampled == 1]

    logging.info(f"Normal data shape (x_norm): {x_norm.shape}")
    logging.info(f"Anomaly data shape (x_fraud): {x_fraud.shape}")

    # Define indices for extracting hidden representations
    norm_start, norm_end = 9000, 15000
    fraud_start, fraud_end = 9000, 15000

    norm_end = min(norm_end, x_norm.shape[0])
    fraud_end = min(fraud_end, x_fraud.shape[0])

    logging.info(f"Extracting hidden representations for normal samples from index {norm_start} to {norm_end}.")
    norm_hid_rep = model.getHiddenRepresentation(compiled_model).predict(x_norm[norm_start:norm_end])

    logging.info(f"Extracting hidden representations for anomaly samples from index {fraud_start} to {fraud_end}.")
    fraud_hid_rep = model.getHiddenRepresentation(compiled_model).predict(x_fraud[fraud_start:fraud_end])

    # Combine hidden representations and create corresponding labels
    rep_x = np.append(norm_hid_rep, fraud_hid_rep, axis=0)
    y_n = np.zeros(norm_hid_rep.shape[0])
    y_f = np.ones(fraud_hid_rep.shape[0])
    rep_y = np.append(y_n, y_f)

    logging.info(f"Combined hidden representation shape: {rep_x.shape}")
    logging.info(f"Combined labels shape: {rep_y.shape}")

    # Step 7: Split the representations into training and validation sets for the classifier
    train_x, val_x, train_y, val_y = train_test_split(
        rep_x, rep_y, test_size=0.25, random_state=42
    )
    logging.info(f"Classifier training data shape: {train_x.shape}, labels shape: {train_y.shape}")
    logging.info(f"Classifier validation data shape: {val_x.shape}, labels shape: {val_y.shape}")

    # Step 8: Hyperparameter Tuning with GridSearchCV for Logistic Regression
    logging.info("Performing hyperparameter tuning for Logistic Regression using GridSearchCV...")
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'saga']
    }
    lr = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(train_x, train_y)
    best_lr = grid_search.best_estimator_
    logging.info(f"Best Logistic Regression parameters: {grid_search.best_params_}")

    # Evaluate the best Logistic Regression model
    pred_y_lr = best_lr.predict(val_x)
    logging.info("Classification Report for Logistic Regression after GridSearchCV:")
    logging.info("\n%s", classification_report(val_y, pred_y_lr))
    logging.info("Accuracy Score: %s", accuracy_score(val_y, pred_y_lr))

    # Step 9: Train Random Forest Classifier
    logging.info("Training Random Forest Classifier as an alternative to Logistic Regression...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(train_x, train_y)
    pred_y_rf = rf.predict(val_x)
    logging.info("Classification Report for Random Forest Classifier:")
    logging.info("\n%s", classification_report(val_y, pred_y_rf))
    logging.info("Accuracy Score: %s", accuracy_score(val_y, pred_y_rf))

    # Step 10: Choose the Best Model Based on F1-Score
    f1_lr = classification_report(val_y, pred_y_lr, output_dict=True)['1.0']['f1-score']
    f1_rf = classification_report(val_y, pred_y_rf, output_dict=True)['1.0']['f1-score']

    if f1_lr > f1_rf:
        best_model = best_lr
        model_type = "LogisticRegression"
        logging.info("Selected Logistic Regression as the best model based on F1-Score.")
    else:
        best_model = rf
        model_type = "RandomForestClassifier"
        logging.info("Selected Random Forest Classifier as the best model based on F1-Score.")

    # Step 11: Save the Best Model to Disk
    classifier_save_path = os.path.join(args.ckpt_path, f"best_classifier_{model_type}.pkl")
    with open(classifier_save_path, 'wb') as file:
        pickle.dump(best_model, file)
    logging.info(f"Best classifier ({model_type}) saved at: {classifier_save_path}")

    # Step 12: Save the AutoEncoder model to Disk
    # Sanitize model_name by removing trailing dots
    sanitized_model_name = args.model_name.rstrip('.')
    autoencoder_save_path = os.path.join(args.ckpt_path,
                                         sanitized_model_name + '.keras')  # e.g., './checkpoints/autoencoder.keras'
    model.save_load_models(path=autoencoder_save_path, model=compiled_model)
    logging.info(f"AutoEncoder model saved at: {autoencoder_save_path}")

    # Step 13: Load the saved AutoEncoder model for further processing
    loaded_autoencoder = model.save_load_models(path=autoencoder_save_path, mode="load")
    logging.info("AutoEncoder model loaded for hidden representation extraction.")

    # Step 14: Generate and Display Final Classification Metrics
    logging.info("\nFinal Classification Report for Best Model:")
    if model_type == "LogisticRegression":
        final_pred_y = pred_y_lr
    else:
        final_pred_y = pred_y_rf
    final_report = classification_report(val_y, final_pred_y)
    logging.info("\n%s", final_report)

    final_accuracy = accuracy_score(val_y, final_pred_y)
    logging.info("Final Accuracy Score: %s", final_accuracy)

    logging.info("Training process completed successfully.")


def test(args):
    logging.info(f"Starting testing for model: {args.model_name}")

    # Step 1: Read and preprocess the test data
    process_data = data_processing.processData(args.data_path)
    X_test = process_data.prepareTestData()
    logging.info(f"Test data shape: {X_test.shape}")

    # Step 2: Rescale the test data using the same scaler used during training
    x_scale = process_data.dataScaling(X_test)
    logging.info(f"Rescaled test data shape: {x_scale.shape}")

    # Step 3: Load the trained AutoEncoder model
    model = autoencoder.AutoEncoder(X_test.shape[1])
    # Sanitize model_name by removing trailing dots
    sanitized_model_name = args.model_name.rstrip('.')
    autoencoder_save_path = os.path.join(args.ckpt_path,
                                         sanitized_model_name + '.keras')  # e.g., './checkpoints/autoencoder.keras'
    compiled_model = model.compile_model()
    compiled_model = model.save_load_models(path=autoencoder_save_path, mode="load")
    logging.info("AutoEncoder model loaded for hidden representation extraction.")

    # Step 4: Extract hidden representations from the AutoEncoder
    hidden_representation = model.getHiddenRepresentation(compiled_model)

    # Step 5: Load the trained Best Classifier
    # Assuming you saved the best classifier with the model type in the filename
    best_classifier_lr_path = os.path.join(args.ckpt_path, "best_classifier_LogisticRegression.pkl")
    best_classifier_rf_path = os.path.join(args.ckpt_path, "best_classifier_RandomForestClassifier.pkl")

    if os.path.exists(best_classifier_lr_path):
        loaded_classifier = pickle.load(open(best_classifier_lr_path, 'rb'))
        model_type = "LogisticRegression"
        logging.info("Best Logistic Regression classifier loaded from disk.")
    elif os.path.exists(best_classifier_rf_path):
        loaded_classifier = pickle.load(open(best_classifier_rf_path, 'rb'))
        model_type = "RandomForestClassifier"
        logging.info("Best Random Forest classifier loaded from disk.")
    else:
        logging.error("No trained classifier found in the checkpoint directory.")
        raise FileNotFoundError("Trained classifier not found.")

    # Step 6: Load the RFE selector
    rfe_save_path = os.path.join(args.ckpt_path, "rfe_selector.pkl")
    if os.path.exists(rfe_save_path):
        with open(rfe_save_path, 'rb') as file:
            rfe = pickle.load(file)
        logging.info("RFE selector loaded successfully.")

        # Apply RFE to test data
        x_test_rfe = rfe.transform(x_scale)
        logging.info(f"Test data after RFE: {x_test_rfe.shape}")
    else:
        logging.error("RFE selector not found. Ensure that it was saved during training.")
        raise FileNotFoundError("RFE selector not found.")

    # Step 7: Generate Predictions in Batches to Manage Memory Usage
    batch_size = 5000  # Adjust based on your system's memory capacity
    num_samples = len(x_test_rfe)
    logging.info(f"Number of test samples: {num_samples}")
    predictions = []

    for idx in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
        end_idx = min(idx + batch_size, num_samples)
        batch_x = x_test_rfe[idx:end_idx]
        hidden_batch = hidden_representation.predict(batch_x)
        pred_y = loaded_classifier.predict(hidden_batch)
        predictions.extend(pred_y)
        logging.info(f"Processed batch {idx} to {end_idx}.")

    # Step 8: Save Predictions to a CSV File
    output_filename = "test_output.csv"
    process_data.save_df(predictions, num_samples, output_filename)
    logging.info(f"Predictions saved to {output_filename}.")


def main():
    parser = argparse.ArgumentParser(description="Network Anomaly Detection using AutoEncoder and Classifier.")
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the input data CSV file.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'test', 'colab'],
        help="Mode to run the script: 'train', 'test', or 'colab'."
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='./checkpoints/',
        help='Path to save or load model checkpoints.'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='autoencoder',
        help='Name of the AutoEncoder model.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default='main.log',
        help='Path for the log file.'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of worker processes for parallel reading. Defaults to (CPU cores - 1).'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)
    logging.info("=== Starting Main Script ===")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Data Path: {args.data_path}")
    logging.info(f"Checkpoint Path: {args.ckpt_path}")
    logging.info(f"Model Name: {args.model_name}")

    # Ensure checkpoint directory exists
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
        logging.info(f"Created checkpoint directory at: {args.ckpt_path}")

    # Execute based on the selected mode
    if args.mode == "train":
        logging.info("Mode: Training")
        train(args)
    elif args.mode == "test":
        logging.info("Mode: Testing")
        test(args)
    elif args.mode == "colab":
        logging.info("Mode: Colab")
        # Add any colab-specific functionality if needed
    else:
        logging.error("Invalid mode selected. Choose from 'train', 'test', or 'colab'.")
        raise Exception("Invalid mode selected. Choose from 'train', 'test', or 'colab'.")

    logging.info("=== Main Script Completed ===")


if __name__ == "__main__":
    main()
