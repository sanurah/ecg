import numpy as np
from sklearn.svm import OneClassSVM
from metrics_calculator import calculate_metrics


def run_classical(x_train, x_test, y_train, y_test, subset=None):
    print("\n" + "=" * 50)
    print("Classical: RBF kernel")
    print("=" * 50 + "\n")

    x_train_normal = x_train[y_train == 1]
    if subset:
        x_train_normal = x_train_normal[:subset]
        x_test = x_test[:subset]
        y_test = y_test[:subset]

    print(f"Training samples (normal): {len(x_train_normal)}")
    print(f"Test samples: {len(x_test)} (normal: {np.sum(y_test == 1)}, anomaly: {np.sum(y_test == 0)})")

    print("\nTraining Model...")
    model = OneClassSVM(gamma='scale', nu=0.3)
    model.fit(x_train_normal)

    print("\nTesting...")
    predictions = model.predict(x_test)

    return calculate_metrics(predictions, y_test)
