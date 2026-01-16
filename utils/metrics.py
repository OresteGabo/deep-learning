import torch
import time
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def print_detailed_report(y_true, y_pred):
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Earthquake']))

def measure_complexity(model, input_size=(1, 512)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    # Measure inference time
    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    avg_inference = (time.time() - start) / 100

    # Count parameters
    params = sum(p.numel() for p in model.parameters())

    return params, avg_inference

def plot_training_results(train_losses, test_accuracies, model_name):
    # Ensure the assets directory exists
    if not os.path.exists('assets'):
        os.makedirs('assets')

    # --- IMAGE 1: LOSS CONVERGENCE ---
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    plt.title(f'{model_name}: Training Loss Convergence', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    loss_path = f'assets/{model_name}_loss.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to free up memory

    # --- IMAGE 2: ACCURACY ---
    plt.figure(figsize=(8, 6))
    plt.plot(test_accuracies, label='Test Accuracy', color='#2ca02c', linewidth=2)
    plt.title(f'{model_name}: Test Accuracy Evolution', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    acc_path = f'assets/{model_name}_accuracy.png'
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {loss_path} and {acc_path}")