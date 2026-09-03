import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flwr_datasets import FederatedDataset

from fedSSL.model import SimClr, SimClrPredictor, get_parameters
from fedSSL.utils import transform_fn


def evaluate_gb_model(num_classes, tune_encoder, model_path, batch_size, train_split, epochs):
    simclr_predictor = SimClrPredictor(num_classes, tune_encoder)

    load_model(simclr_predictor, model_path)

    trainset, testset = load_centralized_data(train_split)
    trainloader = DataLoader(trainset, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    
    fine_tune_predictor(simclr_predictor, trainloader, criterion, epochs, device)
    
    loss, accuracy = evaluate(simclr_predictor, testloader, criterion, device)

    return loss, accuracy


def load_centralized_data(train_split):
    fds = FederatedDataset(dataset="uoft-cs/cifar10", partitioners={'train': 1, 'test': 1})

    cen_train_data = fds.load_split("train")
    cen_train_data = cen_train_data.with_transform(transform_fn(False))

    if train_split < 1.0:
        cen_train_data = cen_train_data.train_test_split(test_size=(1 - train_split), shuffle=True, seed=42)['train']

    cen_test_data = fds.load_split("test")
    cen_test_data = cen_test_data.with_transform(transform_fn(False))

    return cen_train_data, cen_test_data


def load_model(simclr_predictor, model_path):
    simclr = SimClr()
    print("Loading pre-trained model...")
    state_dict = torch.load(model_path)
    simclr.load_state_dict(state_dict)
    weights = get_parameters(simclr)
    simclr_predictor.set_encoder_parameters(weights)


def fine_tune_predictor(simclr_predictor, trainloader, criterion, epochs, device):
    optimizer = optim.Adam(simclr_predictor.parameters(), lr=3e-4)
    simclr_predictor.to(device)
    simclr_predictor.train()

    with tqdm(total=epochs * len(trainloader), desc=f'Downstream Finetune', position=0, leave=True) as pbar:
        for epoch in range(epochs):
            batch = 0
            for item in trainloader:
                x, labels = item['img'], item['label']
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs = simclr_predictor(x)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                batch += 1


def evaluate(simclr_predictor, testloader, criterion, device):
    simclr_predictor.eval()
    total = 0
    correct = 0
    loss = 0
    batch = 0
    num_batches = len(testloader)
    
    with tqdm(total=num_batches, desc=f'Global Model Test', position=0, leave=True) as pbar:

        with torch.no_grad():
            for item in testloader:
                x, labels = item['img'], item['label']
                x, labels = x.to(device), labels.to(device)
                
                logits = simclr_predictor(x)
                values, predicted = torch.max(logits, 1)  
                
                total += labels.size(0)
                loss += criterion(logits, labels).item()
                correct += (predicted == labels).sum().item()

                pbar.update(1)
                batch += 1
  
    return loss / batch, correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tune-encoder", type=bool, default=False)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-split", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    loss, accuracy = evaluate_gb_model(args.num_classes, args.tune_encoder, args.model_path, args.batch_size,
                                       args.train_split, args.epochs)

    print(f"Loss: {loss}, Accuracy: {accuracy}")
