import torch
import torch.nn as nn
from utils.data_loader import get_dataloaders
from utils.metrics import measure_complexity, plot_training_results

# Direct imports from the individual model files
from models.mlp import EarthquakeMLP
from models.cnn import EarthquakeCNN
from models.rnn import EarthquakeRNN

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_epochs = 100

def run_experiment(model_class, name, train_loader, test_loader):
    model = model_class().to(device)

    # We use a slightly lower learning rate for CNN/RNN to avoid the "straight line" trap
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # Scheduler: Reduces learning rate if the loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    criterion = nn.CrossEntropyLoss()

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

            # OPTIONAL: Standardize X on the fly if not done in data_loader
            # X = (X - X.mean()) / (X.std() + 1e-6)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        # Step the scheduler
        scheduler.step(avg_loss)

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = outputs.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        accuracy = 100 * correct / total
        epoch_accuracies.append(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), f'assets/best_{name}.pth')

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}% - LR: {current_lr}")

    plot_training_results(epoch_losses, epoch_accuracies, name)
    print(f"Final Accuracy: {epoch_accuracies[-1]:.2f}% | Best: {best_acc:.2f}%")

if __name__ == "__main__":
    train_ld, test_ld = get_dataloaders('data/Earthquakes_TRAIN.tsv', 'data/Earthquakes_TEST.tsv')

    models_to_test = [
        (EarthquakeMLP, "MLP"),
        (EarthquakeCNN, "CNN"),
        (EarthquakeRNN, "RNN")
    ]

    for m_class, name in models_to_test:
        run_experiment(m_class, name, train_ld, test_ld)