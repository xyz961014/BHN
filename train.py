from pathlib import Path
import argparse
import numpy as np
import json
import ipdb
import sys
import datetime
import os
from resnet import get_resnet

import os
import pytorch_lightning as pl
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")

    parser.add_argument("--model", type=str,
                        default="resnet34",
                        choices= ["resnet18", "resnet34"],
                        help="Network to train")
    parser.add_argument("--dataset", type=str,
                        default="cifar-10",
                        choices= ["cifar-10", "cifar-100"],
                        help="Dataset to train")

    parser.add_argument("--data_root", type=str, default="data",
                        help="directory of the data")
    parser.add_argument("--validation_ratio",type=float, default=0.1,
                        help="ratio to split the training set to get the validation set")
    parser.add_argument("--batch_size",type=int, default=128,
                        help="batch size")
    parser.add_argument("--eval_batch_size",type=int, default=32,
                        help="test batch size")
    parser.add_argument("--num_epochs",type=int, default=10,
                        help="training epochs")
    parser.add_argument("--lr",type=float, default=2e-3,
                        help="learning rate")
    parser.add_argument("--momentum",type=float, default=0.9,
                        help="momentum in SGD optimizer")
    parser.add_argument("--weight_decay",type=float, default=1e-3,
                        help="L2 regularization")
    return parser.parse_args()


class ModelLightning(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=self.args.lr, 
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        return optimizer


def main(args):
    
    if args.dataset == "cifar-10":
        # Load CIFAR-10 dataset

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=transform)

    elif args.dataset == "cifar-100":
        train_dataset = CIFAR100(root=args.data_root, train=True, download=True, transform=transform)
        test_dataset = CIFAR100(root=args.data_root, train=False, download=True, transform=transform)

    val_size = int(args.validation_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True,
                              num_workers=11)
    val_loader = DataLoader(test_dataset, 
                            batch_size=args.eval_batch_size,
                            num_workers=11)
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.eval_batch_size,
                             num_workers=11)


    # Instantiate the Lightning module
    model = get_resnet(model_name=args.model, dataset=args.dataset)
    model = ModelLightning(model, args)
    
    # Train the model
    trainer = pl.Trainer(max_epochs=args.num_epochs)
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    trainer.test(model, test_loader)

if __name__ == "__main__":
    args = parse_args()
    main(args)
