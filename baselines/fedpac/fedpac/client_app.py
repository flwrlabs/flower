"""fedpac: A Flower Baseline."""

import torch
import copy

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fedpac.dataset import load_data
from fedpac.model import Net, fedavg_train, test, train, get_weights, set_weights, test, train
from fedpac.utils import get_centroid

class FlowerClient(NumPyClient):
    """A class defining the client."""

    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.num_classes = self.net.num_classes
        self.feature_extractor = self.get_feature_extractor()
        self.feature_centroid = get_centroid(self.feature_extractor)
        self.class_sizes = self.get_class_sizes()
        self.class_fractions = self.get_class_fractions()
        self.avg_head = []  # Add new attribute to store average heads

    def get_class_sizes(self):
        dataloader = self.trainloader
        sizes = torch.zeros(self.num_classes)
        for _images, labels in dataloader:
            for i in range(self.num_classes):
                sizes[i] = sizes[i] + (i == labels).sum()
        return sizes

    def get_class_fractions(self):
        total = len(self.trainloader.dataset)
        return self.class_sizes / total

    def get_statistics(self):
        dim = self.net.state_dict()[self.net.classifier_layers[0]][0].shape[0]
        feat_dict = self.get_feature_extractor()
        for k in feat_dict.keys():
            feat_dict[k] = torch.stack(feat_dict[k])

        py = self.class_fractions
        py2 = torch.square(py)
        v = 0
        h_ref = torch.zeros((self.num_classes, dim), device=self.device)
        datasize = torch.tensor(len(self.trainloader)).to(self.device)
        for k in feat_dict.keys():
            feat_k = feat_dict[k]
            num_k = feat_k.shape[0]
            feat_k_mu = feat_k.mean(dim=0)
            h_ref[k] = py[k] * feat_k_mu
            v += (
                py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))
            ).item()
            v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / datasize.item()
        return (v, h_ref)

    def get_feature_extractor(self):
        """Extract feature extractor layers."""
        feature_extractors = {}
        model = self.net
        train_data = self.trainloader
        with torch.no_grad():
            for inputs, labels in train_data:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                features, outputs = model(inputs)
                feature_extractor = features.clone().detach().cpu()
                for i in range(len(labels)):
                    if labels[i].item() not in feature_extractors.keys():
                        feature_extractors[labels[i].item()] = []
                        feature_extractors[labels[i].item()].append(
                            feature_extractor[i, :]
                        )
                    else:
                        feature_extractors[labels[i].item()] = [feature_extractor[i, :]]

        return feature_extractors

    def get_classifier_head(self):
        w = copy.deepcopy(self.net.state_dict())
        keys = self.net.classifier_layers
        for k in keys:
            w[k] = torch.zeros_like(w[k])

        w0 = 0
        for i in range(len(self.avg_head)):
            w0 += self.avg_head[i]
            for k in keys:
                w[k] += self.avg_head[i] * self.net.state_dict()[k]

        for k in keys:
            w[k] = torch.div(w[k], w0)

        return w

    def update_classifier(self, classifier):
        local_weight = self.net.state_dict()
        classifier_keys = self.net.classifier_layers
        for k in local_weight.keys():
            if k in classifier_keys:
                local_weight[k] = classifier[k]
        self.net.load_state_dict(local_weight)

    def fit(self, parameters, config):
        """Train model using PAC strategy."""
        set_weights(self.net, parameters)
        
        # Get statistics before training
        v, h_ref = self.get_statistics()
        
        # Regular training
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        
        # Get statistics after training
        v_new, h_ref_new = self.get_statistics()
        
        # Update classifier head based on statistics
        if v_new > v:
            classifier = self.get_classifier_head()
            self.update_classifier(classifier)
        
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "v": v_new,
                "h_ref": h_ref_new,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate model using this client's data."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    net = Net()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
