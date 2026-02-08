import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    features = np.load('data/ECG_Features.npy')
    labels = np.load('data/EEG_labels.npy')

    print(f"\nDataset: \n\t{features.shape[0]} samples\n\t{features.shape[1]} features")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    binary_labels = ((labels == 'N') | (labels == '·')).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features_scaled, binary_labels, test_size=0.3, random_state=42, stratify=binary_labels)

    return x_train, x_test, y_train, y_test
