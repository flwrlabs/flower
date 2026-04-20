# client.py
import flwr as fl
from GLFC import GLFC_model
from ResNet import resnet18_cbam
from myNetwork import network, LeNet
import torch
from collections import OrderedDict
import argparse
import torch.nn as nn
import numpy as np
from ProxyServer import *
from mini_imagenet import *
from tiny_imagenet import *
from torchvision import transforms


def args_parser():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument('--numclass', type=int, default=100, help='总类别数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--task_size', type=int, default=10, help='每个任务的类别数')
    parser.add_argument('--memory_size', type=int, default=2000, help='记忆库大小')
    parser.add_argument('--epochs_local', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备')

    # 客户端参数
    parser.add_argument('--client_id', type=int, default=0, help='客户端ID')

    args = parser.parse_args()
    return args


def weights_init(m):
    """初始化模型权重"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


def local_train(model, client_id, model_g, task_id, model_old, ep_g, old_client_0):
    """本地训练函数"""
    # 设置训练模式
    model.train()

    # 加载全局模型参数
    if model_g is not None:
        model.load_state_dict(model_g.state_dict())

    # 获取训练数据
    train_loader = model.get_train_loader(task_id)

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=model.learning_rate,
                                momentum=0.9,
                                weight_decay=5e-4)

    # 初始化原型梯度
    proto_grad = None
    if task_id > 0 and client_id not in old_client_0:
        proto_grad = model.compute_prototype_grad()

    # 本地训练
    for epoch in range(model.epochs_local):
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(model.device)
            labels = labels.to(model.device)

            optimizer.zero_grad()

            # 根据任务类型选择不同的训练策略
            if task_id == 0:
                loss = model.train_first_task(images, labels)
            else:
                if client_id in old_client_0:
                    loss = model.train_old_task(images, labels, task_id)
                else:
                    loss = model.train_new_task(images, labels, task_id, model_old)

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Client {client_id}, Task {task_id}, '
                      f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    return model.state_dict(), proto_grad


def model_local_eval(model, test_dataset, task_id, task_size, device):
    """本地评估函数"""
    model.eval()
    correct = 0
    total = 0

    # 计算当前任务的类别范围
    start_class = task_id * task_size
    end_class = (task_id + 1) * task_size

    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 只评估当前任务的类别
            mask = (labels >= start_class) & (labels < end_class)
            if not mask.any():
                continue

            images = images[mask]
            labels = labels[mask]

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


class GLFCClient(fl.client.NumPyClient):
    def __init__(self, model, args, client_id):
        self.model = model
        self.args = args
        self.client_id = client_id
        self.device = args.device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # 获取当前任务信息
        task_id = config["task_id"]
        ep_g = config["ep_g"]
        model_old = config.get("model_old", None)

        # 本地训练
        local_model, proto_grad = local_train(
            self.model,
            self.client_id,
            self.model,
            task_id,
            model_old,
            ep_g,
            config.get("old_client_0", [])
        )

        return self.get_parameters(config={}), len(self.model.train_dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        task_id = config["task_id"]
        acc = model_local_eval(
            self.model,
            self.model.test_dataset,
            task_id,
            self.args.task_size,
            self.args.device
        )

        return 0.0, len(self.model.test_dataset), {"accuracy": acc}


def main():
    # 解析命令行参数
    args = args_parser()

    # 初始化模型
    feature_extractor = resnet18_cbam()
    encode_model = LeNet(num_classes=100)
    encode_model.apply(weights_init)

    # 数据集准备
    train_transform = transforms.Compose([
        transforms.RandomCrop((args.img_size, args.img_size), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.24705882352941178),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    # 加载数据集
    if args.dataset == 'cifar100':
        train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
        test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)
    elif args.dataset == 'tiny_imagenet':
        train_dataset = Tiny_Imagenet('./tiny-imagenet-200', train_transform=train_transform,
                                      test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = train_dataset
    else:
        train_dataset = Mini_Imagenet('./train', train_transform=train_transform, test_transform=test_transform)
        train_dataset.get_data()
        test_dataset = train_dataset

    model = GLFC_model(
        args.numclass,
        feature_extractor,
        args.batch_size,
        args.task_size,
        args.memory_size,
        args.epochs_local,
        args.learning_rate,
        train_dataset,
        args.device,
        encode_model
    )

    # 启动客户端
    client = GLFCClient(model, args, args.client_id)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


if __name__ == "__main__":
    main()
