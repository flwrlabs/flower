import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x) # The Skip Connection

class ResMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_blocks=3, dropout=0.3):
        super().__init__()
        
        # --- SHARED BODY ---
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden_dim, dropout))
        self.body = nn.Sequential(*layers)
        
        # --- LOCAL HEAD ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # Binary Classification
        )

    def forward(self, x):
        features = self.body(x)
        return self.head(features)

def train(net, trainloader, epochs, lr, weight_decay, device):
    net.to(device)
    
    # 1. BCE is standard for binary clinical tasks
    criterion = nn.BCEWithLogitsLoss(reduction="sum").to(device)
    optimizer = torch.optim.AdamW(net.parameters(), 
                                  lr=lr,
                                  weight_decay=weight_decay)
    
    net.train()
    total_loss = 0.0

    ds_size = len(trainloader.dataset)
    for _ in range(epochs):
        for batch in trainloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device).float().unsqueeze(1) # Reshape for BCE (N, 1)
            
            optimizer.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    metrics = {
        "train_loss": float(total_loss / (ds_size * epochs)),
        "num-examples": ds_size
    }
    return metrics

def test(net, testloader, device):
    """Validate the model on the test set using clinical metrics."""
    net.to(device)
    # Using 'sum' reduction for loss calculation across batches
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    
    total_loss = 0.0
    all_y_true = []
    all_y_probs = []

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device).float().unsqueeze(1)
            
            logits = net(x)
            total_loss += criterion(logits, y).item()
            
            # Convert logits to probabilities for AUROC/AUPRC
            probs = torch.sigmoid(logits)
            
            all_y_true.append(y.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_y_true)
    y_probs = np.concatenate(all_y_probs)

    # Calculate metrics
    loss = total_loss / len(testloader.dataset)
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    
    # Optional: Simple threshold at 0.5 for accuracy
    preds = (y_probs > 0.5).astype(int)
    accuracy = (preds == y_true).mean()

    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "num-examples": len(testloader.dataset),
    }
    return metrics


def create_resmlp(f: int, t: int):
    return ResMLP(input_dim= f, output_dim = t, hidden_dim=128, n_blocks=3, dropout=0.5,)