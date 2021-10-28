import torch
import torch.nn as nn
from utils import split_context_target, half_context_target, collapse_batch_dim, LightningModule
from .critic import CriticCresp, CriticSimCLR
from hydra.utils import instantiate
from .encoder import Encoder


class EncoderDecoderSSL(Encoder):
    """ Inspired by https://github.com/3springs/attentive-neural-processes/blob/master/neural_processes/models/neural_process/model.py#L83 """

    def __init__(self, enc, agg, dec, cov_net, obs_net, self_attention, targeted=True):
        super(EncoderDecoderSSL, self).__init__(enc, agg, cov_net, obs_net, self_attention)
        self._dec = dec
        self._targeted = targeted

    def reshape_targets(self, context, xi_target):
        # xi_target is of shape [B, NT, R]
        # context is of shape [B, R]
        batch_dim, target_dim = xi_target.shape[0:2]
        context = context.unsqueeze(1).expand(batch_dim, target_dim, *context.shape[1:])
        context, xi_target = collapse_batch_dim(context), collapse_batch_dim(xi_target)
        return context, xi_target

    def decode(self, context, xi_target, uncollapse=False):
        batch_dim, target_dim = xi_target.shape[0:2]
        context, xi_target = self.reshape_targets(context, xi_target)
        predictive = self._dec(xi_target, context)
        if uncollapse:
            predictive = predictive.reshape(batch_dim, target_dim, *predictive.shape[1:])
        return predictive

    def forward(self, b_xi, b_x, evaluation=False, targeted=False):
        """
        This method encoders the entire tensor `b_x` to produce a large bank of negative samples.
        It then splits these encodings into context and target.
        The context is aggregated and projected to form a predictive representation for the target.
        :param b_xi: Covariates
        :param b_x: Observations
        :param evaluation: Whether this is evaluation mode (in which case, encode everything)
        :return:
        """
        # Shapes: [B, n_views, xi_dim], [B, n_views, C, W, H]
        e_pair, e_xi, e_x = self.permute_encode(b_x, b_xi)
        if evaluation:
            if not targeted:  # untargeted task
                return self.postprocess_context(e_xi, e_pair)
            else:  # targeted task
                (e_target_xi, e_context_xi), (e_target_x, _), (_, e_context_pair) = split_context_target(
                    e_xi, e_x, e_pair, index=0
                )
                encoded_context = self.postprocess_context(e_context_xi, e_context_pair, e_target_xi)
                if self._targeted is True:  # targeted model
                    if self._agg.name not in ["cross_attn", "kernel"]:  # already include a "target net"
                        encoded_context = self.decode(encoded_context, e_target_xi)
                else:
                    encoded_context = torch.cat([encoded_context, e_target_xi.squeeze(-1)], dim=-1)
                return encoded_context
        else:
            if self._targeted is True:
                (e_target_xi, e_context_xi), (e_target_x, _), (_, e_context_pair) = split_context_target(
                    e_xi, e_x, e_pair, index=0
                )
                encoded_context = self.postprocess_context(e_context_xi, e_context_pair, e_target_xi)
                if self._agg.name not in ["cross_attn", "kernel"]:
                    encoded_context = self.decode(encoded_context, e_target_xi)
                target = e_target_x.reshape(*encoded_context.shape)
                return encoded_context, target
            elif self._targeted is False:
                (encoded_xi1, encoded_xi2), (encoded_x1, encoded_x2) = half_context_target(e_xi, e_pair)
                encoded1 = self.postprocess_context(encoded_xi1, encoded_x1)
                encoded2 = self.postprocess_context(encoded_xi2, encoded_x2)
                return encoded1, encoded2
            else:
                raise ValueError(
                    f"Unexpected setting for parameter `targeted`: got {self._targeted}. " "Choose from True, False."
                )


class LightningSSL(LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.cov_net = instantiate(cfg.cov_net)
        self.obs_net = instantiate(cfg.obs_net, num_channels=cfg.dataset.shape[0])
        self.enc_net = instantiate(cfg.enc_net, num_channels=cfg.dataset.shape[0] + cfg.enc.num_channels)
        self.targeted = cfg.targeted
        enc = instantiate(cfg.enc, net=self.enc_net)

        agg = instantiate(cfg.agg)
        self_attention = instantiate(cfg.self_attn)
        if self.targeted is True:
            self.target_net = instantiate(cfg.target_net)
            dec = instantiate(cfg.dec, net=self.target_net, dataset=cfg.dataset)
        else:
            dec = None
        self.encoder = nn.DataParallel(
            EncoderDecoderSSL(enc, agg, dec, self.cov_net, self.obs_net, self_attention, targeted=self.targeted)
        )
        if self.targeted:  # applies to True
            self.critic = CriticCresp(**cfg.critic)
        else:  # applies to False
            self.critic = CriticSimCLR(**cfg.critic)
        self.name = "ssl"

    def encode(self, xi, x):
        return self.encoder(xi, x)

    def forward(self, x, xi):
        predictive_representation, target_representation = self.encode(xi, x)
        scores, pseudotargets = self.critic(target_representation, predictive_representation)
        return scores, pseudotargets

    def compute_batch_metrics(self, batch, *args, **kwargs):
        (x, xi), _ = batch
        scores, pseudotargets = self.forward(x, xi)
        loss = self.criterion(scores, pseudotargets)
        return dict(loss=loss)
