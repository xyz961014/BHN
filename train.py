from pathlib import Path
import argparse
import numpy as np
import json
import sys
import os
from datetime import datetime
from tqdm import tqdm

from lightning import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn.functional as F
import torch

from omegaconf import OmegaConf

import add_noise
from resnet import get_resnet
from data_loader import split_cifar, parse_ckpt_name, NoisyLabelDataset, CleanClothing1M, NoisyClothing1M
from performance_metrics import fdr, recall, precision, f1

import ipdb

# Parsing CLI and config files
def parse_args():

    parser = argparse.ArgumentParser(description='Lightning')
    parser.add_argument('--config', type=str,
                        default='config.yaml', help='config file')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    params = OmegaConf.load(
        Path(Path(__file__).parent.resolve() / 'configs' / args.config))
    return params

# One can add extra parameters here, that are not specified in the yaml file.
# For instance : 
# params.optimizer.learning_rate_fc = 0.01
def additional_config_parameters(params):
    pass 


class ModelLightning(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
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
                y_hat = self.forward(x)
                batch_score = -F.cross_entropy(y_hat, y, reduction="none")
                batch_accuracy = (y_hat.argmax(dim=1) == y).float()
                scores.append(batch_score)
                accuracies.append(batch_accuracy)
            self.calibration_scores = torch.cat(scores, dim=0)
            accuracies = torch.cat(accuracies, dim=0)
            self.calibration_accuracy = accuracies.sum() / accuracies.numel()
            print("Calibration set accuracy: {:.4f}".format(self.calibration_accuracy))

        return self.calibration_scores

    
    def eval_pvalues(self, noise_eval_loader):
        m = self.calibration_scores.shape[0]
        self.eval()
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            pvalues_list = []
            for batch in tqdm(noise_eval_loader, desc="Computing p-values"):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device) 
                y_hat = self.forward(x)
                noisy_scores = -F.cross_entropy(y_hat, y, reduction="none")
                # Trick to compute the pvalue. L = eval length
                noisy_scores = noisy_scores.unsqueeze(0).T # [L] -> [1, L]
                noisy_scores = noisy_scores.repeat(1, m) # [1, L] -> [m, L]
                card = torch.sum(self.calibration_scores <= noisy_scores, dim=1)
                pvalues = (card + 1) / (m + 1)
                pvalues_list.append(pvalues)
            pvalues_tensor = torch.cat(pvalues_list, dim=0)
        return pvalues_tensor


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
    elif args.dataset == "clothing1M":
        train_transform = transforms.Compose([
            transforms.Resize(256),  # Resize the image to 256 x 256
            transforms.CenterCrop(224),  # Crop the middle 224 x 224
            transforms.RandomCrop(size=224, padding=32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),  # Resize the image to 256 x 256
            transforms.CenterCrop(224),  # Crop the middle 224 x 224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
            
    # Load CIFAR-10 dataset
    if args.dataset == "cifar-10":
        rain_dataset = CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=args.data_root, train=False, download=True, transform=test_transform)

    elif args.dataset == "cifar-100":
        train_dataset = CIFAR100(root=args.data_root, train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=args.data_root, train=False, download=True, transform=test_transform)

    elif args.dataset == "clothing1M":
        clean_train_dataset = CleanClothing1M(
            subset_list_file='clean_train_key_list.txt',
            annotation_file='clean_label_kv.txt',
            dataset_dir=args.data_root,
            transform=train_transform)
        calibration_dataset = CleanClothing1M(
            subset_list_file='clean_val_key_list.txt',
            annotation_file='clean_label_kv.txt',
            dataset_dir=args.data_root,
            transform=test_transform)
        test_dataset = CleanClothing1M(
            subset_list_file='clean_test_key_list.txt',
            annotation_file='clean_label_kv.txt',
            dataset_dir=args.data_root,
            transform=test_transform)
        noise_eval_dataset = NoisyClothing1M(
            annotation_file='noisy_label_kv.txt',
            dataset_dir=args.data_root,
            transform=test_transform
        )
        noise_dataset = NoisyClothing1M(
            annotation_file='noisy_label_kv.txt',
            dataset_dir=args.data_root,
            transform=train_transform
        )

    if args.dataset in ["cifar-10", "cifar-100"]:
        clean_train_dataset, calibration_dataset, noise_eval_dataset = split_cifar(dataset=train_dataset, 
                                                                                   clean_ratio=args.clean_ratio,
                                                                                   train_ratio=args.train_ratio)
        # Adding noise
        true_labels = torch.tensor(noise_eval_dataset.dataset.targets)[noise_eval_dataset.indices]
        if args.noise_type == "symmetric":
            noisy_labels = add_noise.symmetric(eta=args.noise_eta, 
                                            labels=true_labels.unsqueeze(1), 
                                            K=len(noise_eval_dataset.dataset.classes)).squeeze(1)
        elif args.noise_type == "asymmetric":
            noisy_labels = add_noise.asymmetric(eta=args.noise_eta, 
                                                labels=true_labels.unsqueeze(1), 
                                                K=len(noise_eval_dataset.dataset.classes)).squeeze(1)
        elif args.noise_type == "instance":
            features = torch.tensor(noise_eval_dataset.dataset.data)[noise_eval_dataset.indices].float()
            features = features.reshape(features.size(0), -1)
            noisy_labels = add_noise.instanceDependent(eta=args.noise_eta, 
                                                    features=features,
                                                    labels=true_labels.unsqueeze(1), 
                                                    K=len(noise_eval_dataset.dataset.classes)).squeeze(1)

        noise_eval_dataset = NoisyLabelDataset(noise_eval_dataset, noisy_labels)

    
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
        elif len(ckpt_files) > 0:
            try:
                model = ModelLightning.load_from_checkpoint(ckpt_files[0])
            except Exception as e:
                print("Failed to load the model, {}.".format(e))
        else:
            print("No ckpt found in {}.".format(args.load_path))

    if model is None:
        model = ModelLightning(args)
    
    if not args.skip_first_training:
        if args.logger == 'wandb':
            logger = WandbLogger(
                project='lightning_logs',
                save_dir=".",
                name=args.name,
                prefix="clean",
                version=f"{args.name}_clean",
                config=OmegaConf.to_container(args),
            )
        elif args.logger == 'tensorboard':
            logger = TensorBoardLogger("lightning_logs", 
                                       name=args.name, 
                                       version="clean"
                                       )
        else:
            logger = False    
        trainer = Trainer(logger=logger, max_epochs=args.num_epochs)
        # Train the model on clean training set
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

    # calibration_scores as model attribute, shape [M]
    # noise_eval_dataset the dataset containing (X, Y~)

    noisy_pvalues = model.eval_pvalues(noise_eval_loader).cpu()

    sorted_noisy_pvalue, indices = torch.sort(noisy_pvalues)

    N = sorted_noisy_pvalue.shape[0]
    bh_threshold = (torch.arange(N) + 1) * args.alpha / N
    bh_indice = torch.nonzero((sorted_noisy_pvalue <= bh_threshold))
    if bh_indice.numel() > 0:
        k = bh_indice.max()
    else:
        k = 0

    #while k <= N and sorted_noisy_pvalue[k-1] <= k * args.alpha / N:
    #    k+=1

    preds = torch.ones_like(noisy_pvalues)
    preds[indices[:k]] = 0
    preds = preds.bool()

    # Compute scores if we know true labels (not the case for clothing1M)
    if args.dataset in ["cifar-10", "cifar-100"]:
        fdr_score = fdr(true_labels, noisy_labels, preds)
        recall_score = recall(true_labels, noisy_labels, preds)
        f1_score = f1(true_labels, noisy_labels, preds)
        print("=" * 20 + "Noise Detection Evaluation" + "=" * 20)
        print("FDR: {:7.4f} | Recall: {:7.4f} | F1: {:7.4f}".format(fdr_score, recall_score, f1_score))
        print("=" * 66)

    if args.train_final:
        # Train the model on detected clean data
        sampler = None
        if args.dataset == 'clothing1M':
            detected_clean_dataset = noise_dataset.get_clean_dataset(clean_prediction=preds)
            targets = torch.tensor(detected_clean_dataset.dataset.img_labels.loc[
                                                            detected_clean_dataset.indices
                                                            ][1].values)
            class_count = torch.bincount(targets)
            class_weights = 1. / class_count
            sample_weights = class_weights[targets]
            sampler = WeightedRandomSampler(sample_weights, args.mini_batch_retrain_size*args.batch_size)
        else:
            detected_clean_dataset = noise_eval_dataset.get_clean_dataset(clean_prediction=preds)

        detected_clean_loader = DataLoader(detected_clean_dataset,
                                        sampler=sampler,
                                        batch_size=args.batch_size, 
                                        shuffle=True,
                                        num_workers=11)

        # reinitialize model
        model = ModelLightning(args) 

        if args.logger == 'wandb':
            logger = WandbLogger(
                project='lightning_logs',
                save_dir=".",
                name=args.name,
                prefix="detected_clean",
                version=f"{args.name}_detected_clean",
                config=OmegaConf.to_container(args),
            )
        elif args.logger == 'tensorboard':
            logger = TensorBoardLogger("lightning_logs", 
                                       name=args.name, 
                                       version="detected_clean"
                                       )
        else:
            logger = False

        trainer = Trainer(logger=logger, max_epochs=args.num_epochs)
        trainer.fit(model, detected_clean_loader, calibration_loader)

        # Test the model 
        trainer.test(model, test_loader)


if __name__ == "__main__":
    args = parse_args()
    cli_args = OmegaConf.from_cli()
    args = OmegaConf.merge(args, cli_args)
    # An example to use command line to specify arguments is
    # python train.py --config <file> lr=0.01 name=new_test
    main(args)
