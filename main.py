#!/usr/bin/env python3
from classical_detector import run_classical
from preprocessor import load_and_prepare_data
from quantum_detector import run_quantum
from visualizer import plot_results


def main():
    try:
        x_train, x_test, y_train, y_test = load_and_prepare_data()

        classical_results = run_classical(x_train, x_test, y_train, y_test)
        quantum_results = run_quantum(x_train, x_test, y_train, y_test, 100)

        plot_results(classical_results, quantum_results)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
