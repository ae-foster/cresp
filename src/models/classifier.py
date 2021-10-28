import math
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import encode_dataset
from pytorch_lightning import metrics
import logging
from src.architectures import MLP
from hydra.utils import instantiate
from collections import defaultdict
from utils import split_context_target

log = logging.getLogger(__name__)


class Evaluator(nn.Module):
    def __init__(self, criterion, representation_dim, dataset, encoder, cfg, device):
        super(Evaluator, self).__init__()
        self.representation_dim = representation_dim
        self.dataset = dataset
        self.clf = None
        self.device = device
        self.cfg = cfg
        self.encoder = encoder
        self.criterion = criterion
        self.step = 0

    def get_clf(self):
        dim_in = (
            self.representation_dim
            + (self.dataset.targeted and not self.encoder.module._targeted) * self.dataset.num_covariates
        )
        return MLP(dim_in, self.dataset.num_classes).to(self.device)

    def fit(self, train_loader, hard_reset=True):
        if hard_reset or self.clf is None:
            self.clf = self.get_clf()
        optimizer = instantiate(self.cfg.optim.object, params=self.clf.parameters())

        self.encoder.eval()
        if self.cfg.optim.name == "lbfgs":
            with torch.no_grad():
                X, y = encode_dataset(self.encoder, train_loader, self.device, self.dataset.targeted, progress_bar=True)
            metrics = self.train_lbfgs(X, y, optimizer, self.cfg.optim.reg, self.cfg.optim.num_lbfgs_steps)
        else:
            self.clf.train()
            torch.set_grad_enabled(True)  # TODO: Does not work without
            for epoch in range(self.cfg.optim.num_epochs):
                metrics = self.train_evaluator_sgd(train_loader, optimizer, epoch)
        return metrics

    def train_lbfgs(self, X, y, optimizer, reg, num_lbfgs_steps):
        self.clf.train()
        accuracy = metrics.Accuracy().to(X.device)
        # t = tqdm(range(num_lbfgs_steps), bar_format="{desc}{bar}{r_bar}", file=sys.stdout)
        t = range(num_lbfgs_steps)

        for _ in t:

            def closure():
                optimizer.zero_grad()
                accuracy.reset()
                preds = self.clf(X)
                loss = self.criterion(preds, y)
                l2_norm = sum([param.pow(2).sum() for name, param in self.clf.named_parameters() if "weight" in name])
                acc = accuracy(preds, y) if isinstance(self.criterion, nn.CrossEntropyLoss) else math.nan
                loss += reg * l2_norm

                loss.backward()

                # t.set_description("Train clf => Loss: %.2f | Acc: %.2f%% " % (loss.item(), 100.0 * acc))
                return loss

            train_loss = optimizer.step(closure)
            self.step += 1
        train_loss = self.criterion(self.clf(X), y)
        return dict(loss=train_loss.item(), acc=accuracy.compute())

    def evaluate(self, test_loader):
        self.eval()
        test_loss = 0
        accuracy = metrics.Accuracy().to(self.device)
        targeted = self.dataset.targeted
        encoder = self.encoder
        with torch.no_grad():
            t = tqdm(enumerate(test_loader), total=len(test_loader), bar_format="{desc}{bar}{r_bar}")
            for batch_idx, ((x, xi), y) in t:
                x, xi, y = x.to(self.device), xi.to(self.device), y.to(self.device)
                if targeted:
                    (y, y_context), (_, _) = split_context_target(y, y, index=0)
                    y = y.squeeze(-1)
                representation = encoder(xi, x, evaluation=True, targeted=targeted)
                preds = self.clf(representation)
                loss = self.criterion(preds, y)

                test_loss += loss.item() * x.shape[0]  # since drop_last=False
                acc = accuracy(preds, y) if isinstance(self.criterion, nn.CrossEntropyLoss) else math.nan

                t.set_description("Test clf => Loss: %.2f | Acc: %.2f%% " % (loss.item(), 100.0 * acc))

        return dict(loss=test_loss / len(test_loader.dataset), acc=accuracy.compute())

    def evaluate_tensor(self, X_test, y_test):
        self.eval()
        accuracy = metrics.Accuracy().to(self.device)
        with torch.no_grad():
            preds = self.clf(X_test)
            loss = self.criterion(preds, y_test)

            test_loss = loss.item()
            acc = accuracy(preds, y_test) if isinstance(self.criterion, nn.CrossEntropyLoss) else math.nan

        return dict(loss=test_loss, acc=accuracy.compute().item())

    def scan(self, train_loader, val_loader, test_loader, hard_reset=True):
        prop_array = self.cfg.prop_scan_array
        reg_array = self.cfg.reg_weight_array

        self.encoder.eval()
        targeted = self.dataset.targeted
        if self.cfg.optim.name == "lbfgs":
            with torch.no_grad():
                print("Encoding train")
                X, y = encode_dataset(self.encoder, train_loader, self.device, targeted, progress_bar=False)
                print("Encoding validation")
                X_val, y_val = encode_dataset(self.encoder, val_loader, self.device, targeted, progress_bar=False)
                print("Encoding test")
                X_test, y_test = encode_dataset(self.encoder, test_loader, self.device, targeted, progress_bar=False)
            val_results = defaultdict(dict)
            test_results = defaultdict(dict)
            results = {}
            for prop in prop_array:
                split_size = round(X.shape[0] * prop)
                X_this, y_this = X[:split_size, ...], y[:split_size, ...]
                for reg in reg_array:
                    print(f"Training on {100*prop}% of train data with reg weight {reg}")
                    if hard_reset or self.clf is None:
                        self.clf = self.get_clf()
                    optimizer = instantiate(self.cfg.optim.object, params=self.clf.parameters())
                    self.train_lbfgs(X_this, y_this, optimizer, reg, self.cfg.optim.num_lbfgs_steps)
                    val_metrics = self.evaluate_tensor(X_val, y_val)
                    test_metrics = self.evaluate_tensor(X_test, y_test)
                    val_results[prop][reg] = val_metrics
                    test_results[prop][reg] = test_metrics
                best_reg = max(val_results[prop].keys(), key=lambda k: val_results[prop][k]["acc"])
                results[prop] = test_results[prop][best_reg]
        else:
            raise AttributeError("Scan only works for L-BFGS right now.")

        return val_results, test_results, results
