import torch.nn as nn
from pytorch_lightning.metrics import Accuracy
from hydra.utils import instantiate
from src import architectures
from .cnp import EncoderCNP as Encoder
from utils import LightningModule


class ModelSupervised(LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.cov_net = instantiate(cfg.cov_net)
        self.obs_net = instantiate(cfg.obs_net, num_channels=cfg.dataset.shape[0])
        enc_net = instantiate(cfg.enc_net, num_channels=cfg.dataset.shape[0] + cfg.enc.num_channels)
        enc = instantiate(cfg.enc, net=enc_net, dataset=cfg.dataset)
        agg = instantiate(cfg.agg)
        self_attention = instantiate(cfg.self_attn)
        self.encoder = nn.DataParallel(Encoder(enc, agg, self.cov_net, self.obs_net, self_attention))

        self.clf = architectures.MLP(cfg.representation_dim, cfg.dataset.num_classes)
        self.criterion = instantiate(cfg.dataset.criterion)
        self.train_acc, self.val_acc, self.test_acc = Accuracy(), Accuracy(), Accuracy()
        self.name = "sup"

    def compute_batch_metrics(self, batch, batch_idx, stage, *args, **kwargs):
        (x, xi), targets = batch
        representation = self.encoder(xi, x, evaluation=True)
        preds = self.clf(representation)
        loss = self.criterion(preds, targets)
        metrics = dict(loss=loss)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            accuracy = getattr(self, f"{stage}_acc")
            metrics["acc"] = accuracy(preds, targets)
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.compute_batch_metrics(batch, batch_idx, "train")
        [
            (
                self.log(f"train/{k}", v, prog_bar=k in ["acc"], logger=True, on_epoch=False),
                self.log(f"train/clf/{k}", v, prog_bar=k in ["acc"], logger=True, on_epoch=False),
            )
            for k, v in metrics.items()
        ]
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.compute_batch_metrics(batch, batch_idx, "val")
        [self.log(f"val/{k}", v, on_step=False, on_epoch=True, logger=True) for k, v in metrics.items()]
        [self.log(f"val/clf/{k}", v, on_step=False, on_epoch=True, logger=True) for k, v in metrics.items()]

    def test_step(self, batch, batch_idx):
        metrics = self.compute_batch_metrics(batch, batch_idx, "test")
        [self.log(f"test/{k}", v, on_step=False, on_epoch=True, logger=True) for k, v in metrics.items()]
        [self.log(f"test/clf/{k}", v, on_step=False, on_epoch=True, logger=True) for k, v in metrics.items()]
