"""fedhyb: A Flower Baseline."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import logging
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

# ---------------------------------------
# Set Device (GPU if available, else CPU)
# ---------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ---------------------------------------
# Multiclass Classification Model
# ---------------------------------------
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.hid1 = nn.Linear(num_feature, 34)   # Input layer -> Hidden layer 1
        self.hid2 = nn.Linear(34, 24)            # Hidden layer 1 -> Hidden layer 2
        # self.drop2 = nn.Dropout(0.25)              # Optional dropout (commented out)
        self.oupt = nn.Linear(24, num_class)         # Final output layer

        # Xavier initialization
        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x))
        z = torch.tanh(self.hid2(z))
        # z = self.drop2(z)
        z = self.oupt(z)
        return z

# ---------------------------------------
# Accuracy Metric
# ---------------------------------------
def multi_acc(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_true).float()
    acc = correct_pred.sum() / len(correct_pred)
    return torch.round(acc * 100)

# ---------------------------------------
# Load data and initialize model
# ---------------------------------------


# ---------------------------------------
# Training Function
# ---------------------------------------
def train(model, train_loader, val_loader, EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    print("Begin training...")

    for e in tqdm(range(1, EPOCHS + 1)):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()

        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # Validation
        val_epoch_loss = 0
        val_epoch_acc = 0
        model.eval()

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        # Log training and validation stats
      #  print(f'Epoch {e:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} '
       #       f'| Val Loss: {val_epoch_loss/len(val_loader):.5f} '
       #       f'| Train Acc: {train_epoch_acc/len(train_loader):.3f}% '
       #       f'| Val Acc: {val_epoch_acc/len(val_loader):.3f}% '
       #      )

        # Track training stats globally
        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
      

# ---------------------------------------
# Validation Function (Inference + Report)
# ---------------------------------------
def validation(model, val_loader, current_round, server_round, logger):
    criterion = nn.CrossEntropyLoss()
    y_pred_list = []
    val_round_loss = 0
    val_round_acc = 0

    model.eval()
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            _, y_pred_tags = torch.max(y_val_pred, dim=1)
            y_pred_list.extend(y_pred_tags.cpu().numpy())

            val_round_loss += val_loss.item()
            val_round_acc += val_acc.item()

    # Final reporting
    avg_loss = val_round_loss / len(val_loader)
    avg_acc = val_round_acc / len(val_loader)
    true_labels = []
    for _, y_val_batch in val_loader:
      true_labels.extend(y_val_batch.cpu().numpy())


   # print(f'| Val Loss: {avg_loss:.5f} | Val Acc: {avg_acc:.3f}')
    if current_round==server_round:     
      logger.info(classification_report(true_labels, y_pred_list))

    return avg_loss, avg_acc
    
def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
