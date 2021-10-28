import os
from pathlib import Path
import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger as BaseCSVLogger
from pytorch_lightning.loggers import TensorBoardLogger as BaseTensorBoardLogger
from pytorch_lightning.loggers import WandbLogger as BaseWandbLogger
from hydra.utils import instantiate
import wandb
import matplotlib.pyplot as plt
from utils.helper import split_context_target, encode_dataset
from .viz import scatter_plot
from omegaconf import OmegaConf


class NoneHydra:
    def __init__(self, *args, **kwargs):
        pass

    def __bool__(self):
        return False


class LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hp_metric = 0.0
        if "no_hyp_save" in cfg:
            pass
        else:
            cfg = OmegaConf.to_container(cfg, resolve=True)  # Otherwise OmegaList and co may not play with yaml
            self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim.object, params=self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        scheduler = [scheduler] if scheduler else []
        return [optimizer], scheduler

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict

    def on_epoch_start(self, *args, **kwargs):
        print()

    # Keep track of best validation accuracy for hyperparameter optimization and tensorboard plotting
    def validation_epoch_end(self, validation_step_outputs):
        hp_metric = self.trainer.logged_metrics["val/loss"].cpu()  # -self.val_acc.compute().cpu()
        self.hp_metric = self.hp_metric if self.hp_metric < hp_metric else hp_metric

    # def test_epoch_end(self, outputs):
    #     self.logger[-1].log_hyperparams(self.hparams, metrics={"hp_metric": self.hp_metric})

    def training_step(self, batch, batch_idx):
        metrics = self.compute_batch_metrics(batch, batch_idx, "train")
        [self.log(f"train/{k}", v, prog_bar=k in ["acc"], logger=True, on_epoch=False) for k, v in metrics.items()]
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.compute_batch_metrics(batch, batch_idx, "val")
        [self.log(f"val/{k}", v, on_step=False, on_epoch=True, logger=True) for k, v in metrics.items()]

    def test_step(self, batch, batch_idx):
        metrics = self.compute_batch_metrics(batch, batch_idx, "test")
        [self.log(f"test/{k}", v, on_step=False, on_epoch=True, logger=True) for k, v in metrics.items()]


class ModelCheckpoint(BaseModelCheckpoint):
    """ Force checkpoint to override .ckpt file if existing """

    def _get_metric_interpolated_filepath_name(self, ckpt_name_metrics, epoch, step, del_filepath):
        return self.format_checkpoint_name(epoch, step, ckpt_name_metrics)


class CSVLogger(BaseCSVLogger):
    """ """

    @rank_zero_only
    def log_hyperparams(self, params, metrics=None):
        params = self._convert_params(params)
        self.experiment.log_hparams(params)

    @property
    def log_dir(self):
        return self.root_dir

    def create_and_get_image_dir(self, name, step):
        root_dir = "logs/images"
        Path(root_dir).mkdir(parents=True, exist_ok=True)
        path = os.path.join(os.getcwd(), root_dir, "{}_{:03d}.png".format(name, step))
        return path

    def log_image(self, name, image, step):
        path = self.create_and_get_image_dir(name, step)
        save_image(image, path)

    def log_plot(self, name, plt, step):
        path = self.create_and_get_image_dir(name, step)
        plt.savefig(path, dpi=300, bbox_inches="tight")


class TensorBoardLogger(BaseTensorBoardLogger):
    def log_image(self, name, image, step):
        self.experiment.add_image(name, image, step)

    def log_plot(self, name, plt, step):
        pass


class WandbLogger(BaseWandbLogger):
    def log_image(self, name, image, step):
        self.experiment.log({name: [wandb.Image(image.cpu(), caption=step)]}, commit=False)

    def log_plot(self, name, plt, step):
        self.experiment.log({name: [wandb.Image(plt, caption=step)]}, commit=False)


def call_debug(func):
    def wrapper_call_debug(self, pl_module, *args, **kwargs):
        if (pl_module.cfg.debug_freq > 0) and (
            pl_module.current_epoch % pl_module.cfg.debug_freq == (pl_module.cfg.debug_freq - 1)
        ):
            func(self, pl_module, *args, **kwargs)

    return wrapper_call_debug


class DebugCallback(Callback):
    def __init__(self, cfg, data_module, evaluator):
        super().__init__()
        self.cfg = cfg
        self.data_module = data_module
        self.evaluator = evaluator

    def log_image(self, pl_module, name, image):
        for logger in pl_module.logger:
            logger.log_image(name, image, pl_module.current_epoch)

    def log_plot(self, pl_module, name, plt):
        for logger in pl_module.logger:
            logger.log_plot(name, plt, pl_module.current_epoch)

    def viz_input(self, pl_module, K=5):
        """ vizualize some raw input images of same object """

        images = torch.stack([self.data_module.train_dataloader().dataset[k].make_views([0])[0][0] for k in range(K)])
        V = images.shape[1]
        images = images.reshape(-1, *images.shape[2:])
        image = make_grid(images, nrow=V, padding=0)

        self.log_image(pl_module, "input_image", image)

    @call_debug
    def representation_umap(self, pl_module):
        import umap

        # from sklearn.decomposition import PCA

        dataloader = self.data_module.val_dataloader()
        with torch.no_grad():
            X, y = encode_dataset(pl_module.encoder, dataloader, pl_module.device)
        Z = umap.UMAP().fit_transform(X.cpu())
        plt = scatter_plot(Z, labels=y.cpu())
        self.log_plot(pl_module, "repr", plt)

    @call_debug
    def reconstruction_1d(self, pl_module):
        """ Reconstruction for CNP model """
        from .viz import gp_plot

        dataloader = self.data_module.test_dataloader()
        batch_size = 1
        fig, axes = plt.subplots(1, batch_size, sharex=False, sharey=True, figsize=(10, 4))
        axes = axes if isinstance(axes, list) else [axes]

        bound = self.cfg.dataset.collator.bound
        n_views_test = self.cfg.n_views_test
        covariates = torch.distributions.Uniform(low=-bound, high=bound).sample(
            torch.Size([batch_size, n_views_test, 1])
        )
        for i in range(batch_size):
            func_object = dataloader.dataset[i]
            function = func_object.function
            x_grid = torch.linspace(-bound, bound, 100).to(pl_module.device)
            y_grid = function(x_grid)
            (x, xi), labels = func_object.make_views(covariates[i])

            xi_context, x_context = (
                xi.to(pl_module.device).unsqueeze(0),
                x.to(pl_module.device).unsqueeze(0),
            )
            xi_target, x_target = x_grid.view(1, -1, 1), y_grid.view(1, -1, 1)

            representation = self.evaluator.encoder(xi_context, x_context, evaluation=True)
            if self.evaluator.clf is not None:
                preds = self.evaluator.clf(representation)
                preds = [round(pred.item(), 2) for pred in preds.cpu().view(-1).numpy()]
            else:
                preds = math.nan
            labels = [round(pred.item(), 2) for pred in labels.cpu().view(-1).numpy()]
            title = f"l/p={labels}/{preds}"

            if pl_module.name == "cnp":
                x_pred_mean, x_pred_scale = pl_module.forward(xi_context, x_context, xi_target)
            else:
                x_pred_mean, x_pred_scale = x_target, torch.zeros_like(x_target)

            xi_context, x_context, xi_target, x_target, x_pred_mean, x_pred_scale = (
                tensor.detach().cpu().view(-1).numpy()
                for tensor in [
                    xi_context,
                    x_context,
                    xi_target,
                    x_target,
                    x_pred_mean,
                    x_pred_scale,
                ]
            )
            gp_plot(axes[i], xi_target, x_target, xi_context, x_context, x_pred_mean, x_pred_scale, title)
        self.log_plot(pl_module, "rec", plt)
        plt.clf()
        plt.close(fig)

    @call_debug
    def viz_reconstruction(self, pl_module, K=40):
        dataloader = self.data_module.val_dataloader()
        batch = next(iter(dataloader))
        (x, xi), y = batch
        x, xi, y = x[:K, ...].to(pl_module.device), xi[:K, ...].to(pl_module.device), y[:K, ...].to(pl_module.device)

        encoder = self.evaluator.encoder
        if self.evaluator.clf is not None:
            representation = encoder(xi, x, evaluation=True, targeted=self.evaluator.dataset.targeted)
            preds = self.evaluator.clf(representation)
            prob = F.softmax(preds, dim=-1)[:, [1]]
            prob = (1 - prob.view(-1, 1, 1, 1)) * torch.ones_like(x[:, 0, ...])
        else:
            prob = torch.zeros_like(x[:, 0, ...])

        (xi_target, xi_context), (x_target, x_context), (y_target, y_context) = split_context_target(xi, x, y, index=0)

        if pl_module.name == "cnp":
            x_prediction_mean, _ = pl_module.forward(xi_context, x_context, xi_target)
            x_prediction_mean = x_prediction_mean.reshape(x_target.shape)
        else:
            x_prediction_mean = x_target

        x_target, x_prediction_mean = x_target.squeeze(1), x_prediction_mean.squeeze(1)
        label = (1 - y_target.view(-1, 1, 1, 1)) * torch.ones_like(x_target)

        cat_tensors = torch.cat(
            [x_context.transpose(0, 1).flatten(end_dim=1), x_prediction_mean, x_target, prob, label], dim=0
        )
        img_all = make_grid(cat_tensors, nrow=x_prediction_mean.shape[0], padding=2)

        self.log_image(pl_module, "all_image", img_all)

    def on_train_start(self, trainer, pl_module):
        if "input" in self.cfg.debug:
            self.viz_input(pl_module)
        if "rec" in self.cfg.debug:
            self.viz_reconstruction(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if "repr" in self.cfg.debug:
            self.representation_umap(pl_module)
        if "rec_1d" in self.cfg.debug:
            self.reconstruction_1d(pl_module)
        if "rec" in self.cfg.debug:
            self.viz_reconstruction(pl_module)