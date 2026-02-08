import numpy as np


def calculate_metrics(predictions, y_test):
    y_pred = (predictions == 1).astype(int)

    tp = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 1) & (y_test == 1))
    fn = np.sum((y_pred == 1) & (y_test == 0))

    accuracy = np.mean(y_pred == y_test)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"\nResults:")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }
