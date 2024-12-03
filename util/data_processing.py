import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
import seaborn as sns

import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split


class processData():
    def __init__(self, file_path):
        self.raw_data = pd.read_csv(file_path)
        # Standardize column names
        self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
        # Optional: Verify column names
        print("Columns in raw data:", self.raw_data.columns.tolist())

    def save_df(self, predictions, num_samples, file_path):
        self.raw_data["label"] = predictions[:num_samples]
        self.raw_data.to_csv(file_path, index=False)

    def visualizeData(self):
        """
        Visualize the data,
        """
        pass

    def dataScaling(self, x):
        # Perform Min-Max Scaling
        scaler = MinMaxScaler()
        # Before scaling, ensure no Inf or -Inf
        x = x.replace([np.inf, -np.inf], np.nan)
        # Fill any remaining NaN after replacement with median
        x = x.fillna(x.median())
        x_scale = scaler.fit_transform(x)
        return x_scale

    def prepareTrainingData(self):
        # Convert 'label' column to string, lowercase, and strip whitespace
        self.raw_data['label'] = self.raw_data['label'].astype(str).str.lower().str.strip()
        print("Unique labels after standardization:", self.raw_data['label'].unique())

        # Identify unique labels
        LABELS = self.raw_data["label"].unique()  # Get unique values of label column
        print("Unique labels before mapping:", LABELS)

        # Define normal labels
        normal_labels = ['benign']  # Add other normal labels if present

        # Replace normal labels with 0 and others with 1
        self.raw_data["label"] = self.raw_data["label"].replace(normal_labels, 0)
        self.raw_data = self.raw_data.infer_objects(copy=False)  # Handle downcasting
        anomaly_labels = [label for label in LABELS if label not in normal_labels]
        self.raw_data["label"] = self.raw_data["label"].replace(anomaly_labels, 1)
        self.raw_data = self.raw_data.infer_objects(copy=False)  # Handle downcasting

        # Handle duplicate columns
        duplicate_columns = self.raw_data.columns[self.raw_data.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"Duplicated columns found: {duplicate_columns}")
            self.raw_data = self.raw_data.loc[:, ~self.raw_data.columns.duplicated()]
            print("Duplicated columns removed.")

        data = self.raw_data

        # Identify numeric and categorical columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns

        # Impute missing values
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        data[categorical_cols] = data[categorical_cols].fillna("None")

        # Replace Inf and -Inf with NaN, then impute again
        data = data.replace([np.inf, -np.inf], np.nan)
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        # Optionally, cap extremely large values
        threshold = 1e6  # Adjust based on domain knowledge
        data[numeric_cols] = data[numeric_cols].clip(upper=threshold)

        # Encode categorical columns
        columnsToEncode = list(categorical_cols)

        for feature in columnsToEncode:
            try:
                le = LabelEncoder()  # Instantiate a new LabelEncoder for each feature
                data[feature] = le.fit_transform(data[feature])
            except Exception as e:
                print(f'Error encoding {feature}: {e}')

        # Explicitly infer objects again if any changes occurred
        data = data.infer_objects(copy=False)

        # Verify no non-finite values remain
        if not np.isfinite(data).all().all():
            print("Data contains non-finite values. Replacing them with median.")
            data = data.replace([np.inf, -np.inf], np.nan)
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

        # Print label distribution after mapping
        label_counts = data['label'].value_counts()
        print("Label distribution after mapping:")
        print(label_counts)

        # Split the data into train and validation
        Train, Val = train_test_split(
            data, test_size=0.2, random_state=1, shuffle=True)
        print(f"Train set shape: {Train.shape}")
        print(f"Validation set shape: {Val.shape}")

        # Split features and target
        features = [feature for feature in Train.columns.tolist() if feature != "label"]
        target = "label"

        X_train = Train[features]
        Y_train = Train[target]

        X_val = Val[features]
        Y_val = Val[target]

        print("Data processed ...")
        return (X_train, Y_train, X_val, Y_val)

    def prepareTestData(self):
        data = self.raw_data
        # Drop the first column if it's an unnamed index column
        if 'unnamed: 0' in data.columns:
            data = data.drop('unnamed: 0', axis=1)
            print("Dropped 'Unnamed: 0' column from test data.")
        data = data.fillna("None") 

        # Handle duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"Duplicated columns found in test data: {duplicate_columns}")
            data = data.loc[:, ~data.columns.duplicated()]
            print("Duplicated columns removed from test data.")

        # Convert 'label' column to string, lowercase, and strip whitespace (if present)
        if 'label' in data.columns:
            data['label'] = data['label'].astype(str).str.lower().str.strip()
            # Map labels if necessary (e.g., 'benign' to 0, others to 1)
            # You can uncomment the following lines if test data includes labels
            # normal_labels = ['benign']
            # data["label"] = data["label"].replace(normal_labels, 0)
            # anomaly_labels = [label for label in data["label"].unique() if label not in normal_labels]
            # data["label"] = data["label"].replace(anomaly_labels, 1)
            data = data.infer_objects(copy=False)

        # Identify categorical columns
        categorical_cols = data.select_dtypes(include=['category', 'object']).columns

        # Encode categorical columns
        for feature in categorical_cols:
            try:
                le = LabelEncoder()  # Instantiate a new LabelEncoder for each feature
                data[feature] = le.fit_transform(data[feature])
            except Exception as e:
                print(f'Error encoding {feature}: {e}')

        # Explicitly infer objects again if any changes occurred
        data = data.infer_objects(copy=False)

        print("Test data processed.")
        return data
