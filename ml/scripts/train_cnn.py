#!/usr/bin/env python3
"""
NAVIN ML — 1D-CNN Activity Classifier (PyTorch)
Input: (batch, 9 channels, 20 timesteps)
Output: 3 classes — STAIRS(0), STATIONARY(1), WALKING(2)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ============================================================================
# PATHS
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 10       # early stopping: stop if val loss doesn't improve for 10 epochs
DROPOUT = 0.2
NUM_CLASSES = 3

# ============================================================================
# MODEL
# ============================================================================
class ActivityCNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=3):
        super().__init__()

        # Conv Block 1: 9 channels → 32 filters, kernel=3
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        # Conv Block 2: 32 → 64 filters, kernel=3
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)

        # GlobalAveragePooling → collapses time axis → shape becomes (batch, 64)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Final classifier: 64 → 3 classes
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch, 9, 20)

        x = self.conv1(x)         # (batch, 32, 20)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)         # (batch, 64, 20)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gap(x)           # (batch, 64, 1)
        x = x.squeeze(-1)         # (batch, 64)

        x = self.fc(x)            # (batch, 3)
        return x


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    # Reshape: (N, 20, 9) → (N, 9, 20) — PyTorch Conv1d expects channels first
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    # Convert to PyTorch tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader, y_test


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (outputs.argmax(1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * len(y_batch)
            correct += (outputs.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("NAVIN ML — 1D-CNN Training (PyTorch)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, test_loader, y_test = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Model
    model = ActivityCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>10.4f} | {val_acc:>8.1%}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    test_loss, test_acc, y_pred = evaluate(model, test_loader, criterion, device)

    with open(os.path.join(DATA_DIR, 'class_names.json')) as f:
        label_map = json.load(f)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    test_f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*55}")
    print(f"TEST RESULTS")
    print(f"{'='*55}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.1%}")
    print(f"Test F1:   {test_f1:.4f}")

    print(f"\n{classification_report(y_test, y_pred, target_names=class_names)}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"{'':>12} {'STAIRS':>10} {'STATION':>10} {'WALKING':>10}  <- predicted")
    for i, name in enumerate(class_names):
        row = '  '.join(f'{cm[i,j]:>8}' for j in range(len(class_names)))
        print(f"  {name:>10} {row}")

    # Save model
    model_path = os.path.join(MODEL_DIR, 'cnn_activity.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    # Save metrics
    metrics = {
        'model': '1D-CNN (2 conv layers)',
        'framework': 'PyTorch',
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'dropout': DROPOUT,
            'epochs_trained': len(history['train_loss']),
            'patience': PATIENCE,
        },
        'total_params': total_params,
        'test_f1_weighted': round(float(test_f1), 4),
        'test_accuracy': round(float(test_acc), 4),
        'test_classification_report': classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        ),
        'test_confusion_matrix': cm.tolist(),
        'training_history': history,
    }
    metrics_path = os.path.join(RESULTS_DIR, 'cnn_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")


if __name__ == '__main__':
    main()
