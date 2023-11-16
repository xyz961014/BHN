from pathlib import Path
import argparse
import numpy as np
import json
import ipdb
import sys
import os
from datetime import datetime
from tqdm import tqdm

from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch

from resnet import get_resnet
from data_loader import split_cifar, parse_ckpt_name, merge_clean_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed")

    parser.add_argument("--model", type=str,
                        default="resnet34",
                        choices= ["resnet18", "resnet34", "resnet50"],
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
    parser.add_argument("--num_epochs",type=int, default=200,
                        help="training epochs")
    parser.add_argument("--lr",type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--momentum",type=float, default=0.9,
                        help="momentum in SGD optimizer")
    parser.add_argument("--weight_decay",type=float, default=5e-4,
                        help="L2 regularization")

    # data split
    parser.add_argument("--clean_ratio",type=float, default=0.4, # 0.4 * 50000 = 20000
                        help="keep clean_ratio of the training data clean and add noise to the rest")
    parser.add_argument("--train_ratio",type=float, default=0.5, # 0.5 * 20000 = 10000
                        help="the ratio of training set in clean data, the rest is calibration set")

    # lr_scheduler
    parser.add_argument("--lr_decay_factor",type=float, default=0.1,
                        help="factor of learning rate decay, gamma in StepLR")
    parser.add_argument("--lr_decay_epochs",type=int, default=160,
                        help="the period to decay learning rate")

    # run name to save
    parser.add_argument("--name", type=str, default="default",
                        help="run name")

    # load checkpoint
    parser.add_argument("--skip_first_training", action="store_true",
                        help="skip training on the clean set")
    parser.add_argument("--load_path", type=str, default=None,
                        help="run path to load checkpoint from")


    return parser.parse_args()


class ModelLightning(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = argparse.Namespace(**args)
        self.save_hyperparameters()

        self.model = get_resnet(model_name=self.args.model, dataset=self.args.dataset)
        self.calibration_scores = torch.zeros(0)
        self.calibration_accuracy = 0.
    
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
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.args.lr_decay_epochs,
                                                    gamma=self.args.lr_decay_factor)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_epoch_start(self):
        # Get current learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=False, on_epoch=True, prog_bar=True)


    def compute_calibration_scores(self, calibration_loader):
        self.eval()
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            scores = []
            accuracies = []
            for batch in tqdm(calibration_loader, desc="Computing calibration scores"):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self(x)
                batch_score = -F.cross_entropy(y_hat, y, reduction="none")
                batch_accuracy = (y_hat.argmax(dim=1) == y).float()
                scores.append(batch_score)
                accuracies.append(batch_accuracy)
            self.calibration_scores = torch.cat(scores, dim=0)
            accuracies = torch.cat(accuracies, dim=0)
            self.calibration_accuracy = accuracies.sum() / accuracies.numel()
            print("Calibration set accuracy: {:.4f}".format(self.calibration_accuracy))

        return self.calibration_scores


def main(args):

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    args.name = args.name + f"_{formatted_time}"


    # Define data preprocessing
    if args.dataset in ["cifar-10", "cifar-100"]:
        train_transform = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    # Load CIFAR-10 dataset
    if args.dataset == "cifar-10":
        train_dataset = CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)

    elif args.dataset == "cifar-100":
        train_dataset = CIFAR100(root=args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=args.data_root, train=False, download=True, transform=test_transform)

    if args.dataset in ["cifar-10", "cifar-100"]:
        clean_train_dataset, calibration_dataset, noise_eval_dataset = split_cifar(dataset=train_dataset, 
                                                                                   clean_ratio=args.clean_ratio,
                                                                                   train_ratio=args.train_ratio)
    ###############################################
    # TODO: Add noise to noise_eval_dataset
    noise_eval_dataset = noise_eval_dataset
    ###############################################
    
    # Create data loaders
    clean_train_loader = DataLoader(clean_train_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=True,
                                    num_workers=11)
    calibration_loader = DataLoader(calibration_dataset, 
                                    batch_size=args.eval_batch_size,
                                    num_workers=11)
    noise_eval_loader = DataLoader(noise_eval_dataset,
                                   batch_size=args.eval_batch_size,
                                   num_workers=11)
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.eval_batch_size,
                             num_workers=11)


    # Instantiate the Lightning module
    model = None

    if args.load_path is not None and Path(args.load_path).exists():
        load_path = Path(args.load_path)
        hparams_file = load_path / "hparams.yaml"
        ckpt_files = []
        for ckpt_file in load_path.rglob("*.ckpt"):
            ckpt_files.append(ckpt_file)
        if len(ckpt_files) > 0 and hparams_file.exists():
            ckpt_files.sort(key=lambda f: parse_ckpt_name(f.name), reverse=True)
            model = ModelLightning.load_from_checkpoint(ckpt_files[0], hparams_file=hparams_file)

    if model is None:
        model = ModelLightning(vars(args)) # use vars in order to save hparams safely in a yaml file
    
    if not args.skip_first_training:
        # Train the model on clean training set
        logger = TensorBoardLogger("lightning_logs", 
                                   name=args.name, 
                                   version="clean")
        trainer = Trainer(logger=logger, max_epochs=args.num_epochs)
        trainer.fit(model, clean_train_loader, calibration_loader)
    else:
        trainer = Trainer(logger=False)

    # Test the model 
    trainer.test(model, test_loader)

    # Compute scores on calibration set
    calibration_scores = model.compute_calibration_scores(calibration_loader)

    ###############################################
    # TODO: Inference on noise_eval_dataset
    ###############################################

    ###############################################
    # TODO: Compute noise detection metrics
    ###############################################

    # Train the model on all clean data
    clean_prediction = torch.empty(len(noise_eval_dataset)).uniform_(0.5, 1).bernoulli() # a fake one
    all_train_dataset = merge_clean_dataset(clean_train_dataset, calibration_dataset, noise_eval_dataset,
                                            clean_prediction=clean_prediction)
    all_train_loader = DataLoader(all_train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True,
                                  num_workers=11)

    # reinitialize model
    model = ModelLightning(vars(args)) 

    logger = TensorBoardLogger("lightning_logs", 
                               name=args.name, 
                               version="all")
    trainer = Trainer(logger=logger, max_epochs=args.num_epochs)
    trainer.fit(model, all_train_loader)

    # Test the model 
    trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
