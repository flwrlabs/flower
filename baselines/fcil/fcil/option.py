import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")
    parser.add_argument('--method', type=str, default='glfc', help="name of method")
    parser.add_argument('--iid_level', type=int, default=6, help='number of data classes for local clients')
    parser.add_argument('--numclass', type=int, default=10, help="number of data classes in the first task")
    parser.add_argument('--img_size', type=int, default=32, help="size of images")
    parser.add_argument('--device', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--batch_size', type=int, default=128, help='size of mini-batch')
    parser.add_argument('--task_size', type=int, default=10, help='number of data classes each task')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--memory_size', type=int, default=2000, help='size of exemplar memory')
    parser.add_argument('--epochs_local', type=int, default=20, help='local epochs of each global round')
    parser.add_argument('--learning_rate', type=float, default=2.0, help='learning rate')
    parser.add_argument('--num_clients', type=int, default=30, help='initial number of clients')
    parser.add_argument('--local_clients', type=int, default=10, help='number of selected clients each round')
    parser.add_argument('--epochs_global', type=int, default=100, help='total number of global rounds')
    parser.add_argument('--tasks_global', type=int, default=10, help='total number of tasks')
    args = parser.parse_args()
    return args