import matplotlib.pyplot as plt
import numpy as np


def plot_results(classical_results, quantum_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cm_c = classical_results['confusion_matrix']
    axes[0, 0].imshow(cm_c, cmap='Blues')
    axes[0, 0].set_title('Classical Confusion Matrix')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['Normal', 'Anomaly'])
    axes[0, 0].set_yticklabels(['Normal', 'Anomaly'])
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i, cm_c[i, j], ha="center", va="center")

    cm_q = quantum_results['confusion_matrix']
    axes[0, 1].imshow(cm_q, cmap='Greens')
    axes[0, 1].set_title('Quantum Confusion Matrix')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_xticklabels(['Normal', 'Anomaly'])
    axes[0, 1].set_yticklabels(['Normal', 'Anomaly'])
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, cm_q[i, j], ha="center", va="center")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    classical_values = [classical_results[m.lower()] for m in metrics]
    quantum_values = [quantum_results[m.lower()] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 0].bar(x - width / 2, classical_values, width, label='Classical', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width / 2, quantum_values, width, label='Quantum', color='green', alpha=0.7)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Classical vs Quantum Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)

    axes[1, 1].axis('off')
    table_data = []
    for i, metric in enumerate(metrics):
        table_data.append([metric, f"{classical_values[i]:.4f}", f"{quantum_values[i]:.4f}"])

    table_data.append(['', '', ''])
    table_data.append(['Qubits', '-', '8'])
    table_data.append(['Circuit Depth', '-', '1'])
    table_data.append(['Shots', '-', '1024'])
    table = axes[1, 1].table(cellText=table_data, colLabels=['Metric', 'Classical', 'Quantum'],
                             cellLoc='center', loc='center', colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (i, j), cell in table.get_celld().items():
        if i == 0 or j == 0:
            cell.set_facecolor('lightgray')
    axes[1, 1].set_title('Results Summary')

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)

    print("\n" + "=" * 50 + "\n")
    print("Results saved to results.png")
