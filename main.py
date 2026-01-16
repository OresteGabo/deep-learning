import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import get_dataloaders
from utils.metrics import measure_complexity, plot_training_results, print_detailed_report

# Model imports
from models.mlp import EarthquakeMLP
from models.cnn import EarthquakeCNN
from models.rnn import EarthquakeRNN
from models.hybrid import SeismicNet  # Your new Hybrid model

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_epochs = 100


def run_experiment(model_class, name, train_loader, test_loader):
    model = model_class().to(device)

    # 1. WEIGHTED LOSS: Address class imbalance (approx 3:1 ratio)
    # This forces the model to care more about the minority 'Earthquake' class
    weights = torch.tensor([1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 2. ONECYCLE LR: Helps escape local minima (the 74% trap)
    # It starts low, ramps up, then cools down.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    params, inf_time = measure_complexity(model)
    print(f"\n--- Starting Experiment: {name} ---")
    print(f"Parameters: {params:,} | Inference: {inf_time:.6f}s")

    epoch_losses = []
    epoch_accuracies = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # Step OneCycleLR after every batch
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = outputs.argmax(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Calculate Accuracy
        correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        accuracy = 100 * correct / len(all_labels)
        epoch_accuracies.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f'assets/best_{name}.pth')

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}% - LR: {current_lr:.6f}")

    # 3. DETAILED EVALUATION: Final Report & Confusion Matrix
    print(f"\nFinal Results for {name}:")
    print_detailed_report(all_labels, all_preds)

    plot_training_results(epoch_losses, epoch_accuracies, name)
    print(f"Final Accuracy: {epoch_accuracies[-1]:.2f}% | Best: {best_acc:.2f}%")


if __name__ == "__main__":
    # Ensure your data_loader implements the SeismicTransform (Z-score + Noise)
    train_ld, test_ld = get_dataloaders('data/Earthquakes_TRAIN.tsv', 'data/Earthquakes_TEST.tsv')

    models_to_test = [
        (EarthquakeMLP, "MLP"),
        (EarthquakeCNN, "CNN"),
        (EarthquakeRNN, "RNN"),
        (SeismicNet, "Hybrid_SeismicNet")  # Added the professional refactor
    ]

    for m_class, name in models_to_test:
        run_experiment(m_class, name, train_ld, test_ld)