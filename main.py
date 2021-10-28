"""Train an net using Contrastive Learning."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import random_split, DataLoader, Subset
from src import models
from src.dataset import rebracket_collate

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import pytorch_lightning as pl
from utils import OmegaConf
from utils import ModelCheckpoint, DebugCallback, CSVLogger, TensorBoardLogger, WandbLogger, LightningModule
import logging

log = logging.getLogger(__name__)


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
        self.collator = instantiate(self.cfg.dataset.collator)

    def setup(self, stage=None, seed=None):
        log.info(f"Preparing data for {stage} stage")
        if stage == "fit":
            data = instantiate(self.cfg.dataset.object)
            if self.cfg.dataset.split[0] < 1.0:
                split_size = [round(len(data) * prop) for prop in self.cfg.dataset.split]
                split_size[-1] = len(data) - sum(split_size[:-1])
            else:
                split_size = self.cfg.dataset.split
            seed = seed if seed is not None else self.cfg.seed
            generator = torch.Generator().manual_seed(seed)
            self._train_set, self._val_set, self._test_set = random_split(data, split_size, generator=generator)

            # Sample dataset for training classifier
            N = len(self._train_set)
            indices = torch.randperm(N, generator=generator).tolist()
            indices = indices[: round(N * self.cfg.clf.prop)]
            self._clf_train_set = Subset(self._train_set, indices)
            if self.cfg.fix_clf_train:
                self.fix_clf_train_set()

    def fix_clf_train_set(self):
        """Convert clf train set to a fixed tensor dataset. Suitable for semi-supervised evaluation."""
        loader = DataLoader(
            self._clf_train_set,
            batch_size=self.cfg.clf.optim.batch_size,
            shuffle=True,
            **self.loader_kwargs,
            collate_fn=self.collator(self.cfg.n_views_test),
        )
        X, Xi, Y = [], [], []
        for (x, xi), y in tqdm(loader):
            X.append(x.cpu())
            Xi.append(xi.cpu())
            Y.append(y.cpu())
        X = torch.cat(X, dim=0)
        Xi = torch.cat(Xi, dim=0)
        Y = torch.cat(Y, dim=0)
        self._clf_train_set = torch.utils.data.TensorDataset(X, Xi, Y)

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            **self.loader_kwargs,
            drop_last=True,
            collate_fn=self.collator(self.cfg.n_views_train),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            **self.loader_kwargs,
            collate_fn=self.collator(self.cfg.n_views_train),
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_set,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            **self.loader_kwargs,
            collate_fn=self.collator(self.cfg.n_views_train),
        )


class ClfDataModule(pl.LightningDataModule):
    def __init__(self, data_module, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_module = data_module
        self.loader_kwargs = data_module.loader_kwargs
        self.collator = data_module.collator

    @property
    def _train_set(self):
        return self.data_module._clf_train_set

    @property
    def _val_set(self):
        return self.data_module._val_set

    @property
    def _test_set(self):
        return self.data_module._test_set

    def train_dataloader(self):
        if self.cfg.fix_clf_train:
            collator = rebracket_collate
        else:
            collator = self.collator(self.cfg.n_views_test)
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.clf.optim.batch_size,
            shuffle=True,
            **self.loader_kwargs,
            collate_fn=collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            **self.loader_kwargs,
            collate_fn=self.collator(self.cfg.n_views_test),
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_set,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            **self.loader_kwargs,
            collate_fn=self.collator(self.cfg.n_views_test),
        )


class EvaluatorCallback(pl.callbacks.Callback):
    def __init__(self, evaluator, data_module):
        super().__init__()
        self.evaluator = evaluator
        self.data_module = data_module

    def on_epoch_start(self, *args, **kwargs):
        print()

    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.cfg.clf_freq > 0) and (
            pl_module.current_epoch % pl_module.cfg.clf_freq == (pl_module.cfg.clf_freq - 1)
        ):
            metrics = self.evaluator.fit(self.data_module.train_dataloader())
            [pl_module.log(f"train/clf/{k}", v, on_step=False, on_epoch=True) for k, v in metrics.items()]
            metrics = self.evaluator.evaluate(self.data_module.val_dataloader())
            [pl_module.log(f"val/clf/{k}", v, on_step=False, on_epoch=True) for k, v in metrics.items()]

    def on_test_epoch_end(self, trainer, pl_module):
        self.evaluator.fit(self.data_module.train_dataloader())
        metrics = self.evaluator.evaluate(self.data_module.test_dataloader())
        [pl_module.log(f"test/clf/{k}", v, on_step=False, on_epoch=True) for k, v in metrics.items()]


@hydra.main(config_path="config", config_name="main")
def main(cfg):
    run_path = os.getcwd()
    dirname = HydraConfig.get().job.override_dirname
    success_path, ckpt_path = os.path.join(run_path, "success.txt"), None

    cfg.num_gpus = torch.cuda.device_count()
    cfg.num_workers = 4 * cfg.num_gpus if cfg.num_workers == -1 else cfg.num_workers
    gpus = 1 if torch.cuda.is_available() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For reproducibility purposes
    cfg.seed = cfg.run if cfg.seed == 0 else int(torch.randint(0, 2 ** 32 - 1, (1,)).item())
    pl.seed_everything(cfg.seed)

    # Init PyTorch Lightning model ⚡
    model = instantiate(cfg.model, cfg)
    criterion = instantiate(cfg.dataset.criterion)
    evaluator = models.Evaluator(criterion, cfg.representation_dim, cfg.dataset, model.encoder, cfg.clf, device)

    # Init PyTorch Lightning datamodule ⚡
    data_module = DataModule(cfg)
    clf_data_module = ClfDataModule(data_module, cfg)

    # Init PyTorch Lightning loggers ⚡
    Path(cfg.paths.logs).mkdir(parents=True, exist_ok=True)  # Create logs dir
    loggers = [instantiate(logger_conf) for logger_conf in cfg.logger.values()] if "logger" in cfg else []

    # Init PyTorch Lightning callbacks ⚡
    callbacks = [instantiate(cfg.checkpoint)]
    if model.name in ["ssl", "cnp"]:
        callbacks.append(EvaluatorCallback(evaluator, clf_data_module))
        callbacks.append(DebugCallback(cfg, data_module, evaluator))

    if not cfg.clf.only:
        # Init PyTorch Lightning trainer ⚡
        trainer = instantiate(
            cfg.trainer, gpus=gpus, resume_from_checkpoint=ckpt_path, callbacks=callbacks, logger=loggers
        )
        log.info("Stage : Training")
        if model.name == "sup":
            data_module.setup("fit")
            trainer.fit(model, clf_data_module)
        else:
            trainer.fit(model, data_module)

        log.info("Stage : Testing")
        trainer.test()

        if not trainer.interrupted:
            trainer.logger.save()
            log.info("Success!")
            open(success_path, "w+").close()
            return model.hp_metric  # value used for hyperparameter minimization with ax or nevergrad

    else:
        log.info("Load encoder and train classifier...")
        from pytorch_lightning.utilities.cloud_io import load as pl_load

        checkpoint = pl_load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        evaluator.encoder = evaluator.encoder.to(device)

        clf_data_module.data_module.setup("fit", seed=model.cfg.seed)
        if cfg.clf.scan:
            val_results, test_results, results = evaluator.scan(
                clf_data_module.train_dataloader(), clf_data_module.val_dataloader(), clf_data_module.test_dataloader()
            )
            print("Val", val_results, "Test", test_results, "Best val->test", results)

        else:
            metrics = {}
            metrics["train"] = evaluator.fit(clf_data_module.train_dataloader())
            metrics["val"] = evaluator.evaluate(clf_data_module.val_dataloader())
            metrics["test"] = evaluator.evaluate(clf_data_module.test_dataloader())
            for split, metrics in metrics.items():
                for logger in loggers:
                    results = {f"{split}/clf/{k}": v for k, v in metrics.items()}
                    logger.log_metrics(results)
                    logger.save()


if __name__ == "__main__":
    cudnn.benchmark = True

    main()
