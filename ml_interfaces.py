import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm

from nni.compression.pruning import L1NormPruner, L2NormPruner
from nni.compression.speedup import ModelSpeedup
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torch.optim import SGD

from models.vgg import *


def build_model(args):
    if args.model_name == 'vgg11_bn':
        model = vgg11_bn()
    else:
        raise NotImplementedError
    
    if os.path.exists(args.weight):
        model.load_state_dict(torch.load(args.weight))
    else:
        print("No weight file found, use default weight.")
    
    model = model.cuda()
    
    return model

def build_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), 1e-2)
    else:
        raise NotImplementedError
    
    return optimizer

def build_dataloader(args):
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])    
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        raise NotImplementedError
    
    return train_loader, test_loader
    
def build_criterion(args):
    criterion = nn.CrossEntropyLoss()
    return criterion

def train_model(model, opt, criterion, train_loader, mode='train'):
    model.train()
    if mode == 'train':
        epochs = args.train_epoch
    elif mode == 'tune':
        epochs = args.tune_epoch
    for e in tqdm(range(epochs)):
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            opt.step()
            
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_dataset_length = 64
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            test_loss += loss(output, target)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            break
    test_loss /= test_dataset_length
    accuracy = 100. * correct / test_dataset_length
    print('Average test loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, test_dataset_length, accuracy))

def prune_model(args, config, model):
    if args.pruner == 'l1_norm':
        pruner = L1NormPruner(model, config)
    elif args.pruner == 'l2_norm':
        pruner = L2NormPruner(model, config)
    else:
        raise NotImplementedError
    
    _, masks = pruner.compress()
    pruner.unwrap_model()
    ModelSpeedup(model, torch.rand(64, 3, 32, 32).cuda(), masks).speedup_model()
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vgg11_bn")
    parser.add_argument("--weight", type=str, default="weights/vgg11_bn.pt")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--pruner", type=str, default="l2_norm")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--train_epoch", type=int, default=30)
    parser.add_argument("--tune_epoch", type=int, default=10)
    
    args = parser.parse_args()
    
    
    config_list = [{
        'op_types': ['Linear', 'Conv2d'],
        'exclude_op_names': ['classifier.6', 'features.25'],
        # 'op_names': ['features.0', 'features.4', 'features.8', 'features.11', 'features.15'],
        'sparse_ratio': 0.5
    }]
    
    model = build_model(args)
    train_loader, test_loader = build_dataloader(args)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)
    
    print("Before pruning:")
    print(model)
    # train_model(model, optimizer, criterion, train_loader, mode='train')
    evaluate_model(model, test_loader)
    # torch.save(model.state_dict(), 'weights/vgg11_bn.pt')
    print("...")
    
    model = prune_model(args, config_list, model)
    print("After pruning:")
    print(model)
    train_model(model, optimizer, criterion, train_loader, mode='tune')
    evaluate_model(model, test_loader)
    