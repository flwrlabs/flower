from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50


class NtxentLoss(nn.Module):
    def __init__(self, device, temperature=0.5):
        super(NtxentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.batch_size = None
        self.criterion = nn.CrossEntropyLoss(reduction="mean").to(device)

    def forward(self, z_i, z_j):
        self.batch_size = z_i.size(0)
        z_i = F.normalize(z_i)
        z_j = F.normalize(z_j)
        
        z = torch.cat((z_i, z_j), dim=0)

        sim_matrix = torch.matmul(z, z.T) / self.temperature
        sim_matrix.fill_diagonal_(-float('inf'))

        labels = torch.arange(self.batch_size, 2 * self.batch_size).to(self.device)
        labels = torch.cat((labels, labels - self.batch_size))  
        
        loss = self.criterion(sim_matrix, labels)
        return loss


class Mlp(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.in_features = dim
        self.out_features = projection_size
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


class SimClr(nn.Module):
    def __init__(self, projection_size=2048, projection_hidden_size=4096):
        super(SimClr, self).__init__()

        self.encoder = resnet50(weights=None)
        self.encoded_size = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        
        self.projected_size = projection_size                
        self.proj_head = Mlp(self.encoded_size, projection_size, projection_hidden_size)
        self.isInference = False
        
    def setInference(self, isInference):
        self.isInference = isInference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder(x)
        if self.isInference:
            return e1
        return self.proj_head(e1)


class SimClrTransform:
    def __init__(self, size=32):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.size = size

        self.base_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()]
        )

        self.test_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

    def __call__(self, x, augment_data):
        if not augment_data:
            return self.test_transform(x)

        return self.base_transform(x), self.base_transform(x)


class SimClrPredictor(nn.Module):
    def __init__(self, num_classes, tune_encoder=False):
        super(SimClrPredictor, self).__init__()

        self.simclr = SimClr()
        self.linear_predictor = nn.Linear(self.simclr.encoded_size, num_classes)
        self.simclr.setInference(True)

        if not tune_encoder:
            for param in self.simclr.parameters():
                param.requires_grad = False

    def set_encoder_parameters(self, weights):
        set_parameters(self.simclr, weights)

    def forward(self, x):
        self.simclr.setInference(True)
        features = self.simclr(x)
        output = self.linear_predictor(features)
        return output


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
