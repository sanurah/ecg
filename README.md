# ECG Anomaly Detection

Quantum vs Classical anomaly detection for ECG data using One-Class SVM.

## Structure

```
ecg/
├── main.py                 # Main entry point
├── classical_detector.py   # Classical One-Class SVM
├── quantum_detector.py     # Quantum Kernel SVM
├── preprocessor.py         # Data preprocessing and loading
├── metrics_calculator.py   # Performance metrics calculation
├── visualizer.py           # Results visualization
├── requirements.txt        # Dependencies
├── results.png             # Visualization output after execution (root level)
├── data/
│   ├── ECG_Features.npy    # ECG features (112,570 samples × 8 features)
│   └── EEG_labels.npy      # Labels (N=normal, others=anomaly)
└── output_samples/
    ├── results.png         # Visualization output
    ├── ibm_results.png     # Visualization output (IBM backend)
    ├── output.log          # Console output log
    └── ibm_output.log      # Console output log (IBM backend)
```

## Requirements

- Python 3.10
- Qiskit
- Qiskit Machine Learning
- Qiskit IBM Runtime
- NumPy
- Scikit-learn
- Matplotlib

## Installation

1. Create and activate a virtual environment (Optional):

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

**Notes:**

- The quantum detector uses a subset of 100 samples by default due to computational constraints. You can adjust
  this by modifying the fourth parameter in the `run_quantum()` call in `main.py`.
- If you want to run the IBM backend, pelase set your IBM token to the `IBM_TOKEN` in  `quantum_detector.py` file.
  Default execution will be on a local simulator.

## Dataset

The two files are stored in the data folder.
Provided by MIT-BIH Arrhythmia Database:

- 112,570 ECG samples
- 8 features per sample
- Binary classification: Normal ('N') vs. Anomaly

## Methods

### Classical

- One-Class SVM with RBF kernel
- Trained only on normal samples
- Full dataset evaluation

### Quantum

- Angle encoding (RY gates, 8 qubits)
- Quantum kernel via fidelity
- Subset evaluation (computational constraints)

## Output

- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Visualization saved as PNG


