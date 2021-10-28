import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from utils import split_context_target, collapse_batch_dim, LightningModule
from hydra.utils import instantiate
from .encoder import Encoder


class EncoderCNP(Encoder):
    def __init__(self, enc, agg, cov_net, obs_net, self_attention):
        super(EncoderCNP, self).__init__(enc, agg, cov_net, obs_net, self_attention)

    def forward(self, b_xi_context, b_x_context, evaluation=False, targeted=False, **kwargs):
        e_pair, e_xi, e_x = self.permute_encode(b_x_context, b_xi_context)
        if evaluation and targeted:
            (e_target_xi, e_context_xi), (e_target_x, _), (_, e_context_pair) = split_context_target(
                e_xi, e_x, e_pair, index=0
            )
            encoded_context = self.postprocess_context(e_context_xi, e_context_pair)
            encoded_context = torch.cat([encoded_context, e_target_xi.squeeze(-1)], dim=-1)
            return encoded_context
        else:
            return self.postprocess_context(e_xi, e_pair)


def diagonal_quadratic_covariance_activation(pre_activation, min_cov):
    return min_cov + pre_activation.pow(2)


def diagonal_softplus_covariance_activation(pre_activation, min_cov):
    return min_cov + F.softplus(pre_activation)


def diagonal_quadratic_softplus_covariance_activation(pre_activation, min_cov):
    return min_cov + F.softplus(pre_activation).pow(2)


covariance_activation_functions = {
    "diagonal_quadratic": diagonal_quadratic_covariance_activation,
    "diagonal_softplus": diagonal_softplus_covariance_activation,
    "diagonal_softplus_quadratic": diagonal_quadratic_softplus_covariance_activation,
}


class LightningCNP(LightningModule):
    def __init__(self, cfg, min_cov, covariance_activation_function, **kwargs):
        super().__init__(cfg)
        self.save_hyperparameters("min_cov", "covariance_activation_function")

        self.covariance_activation_function = partial(
            covariance_activation_functions[covariance_activation_function], min_cov=min_cov
        )
        self.cov_net = instantiate(cfg.cov_net)
        self.obs_net = instantiate(cfg.obs_net, num_channels=cfg.dataset.shape[0])
        enc_net = instantiate(cfg.enc_net, num_channels=cfg.dataset.shape[0] + cfg.enc.num_channels)
        enc = instantiate(cfg.enc, net=enc_net, dataset=cfg.dataset)
        agg = instantiate(cfg.agg)
        self_attention = instantiate(cfg.self_attn)
        self.encoder = nn.DataParallel(EncoderCNP(enc, agg, self.cov_net, self.obs_net, self_attention))
        self.decoder = nn.DataParallel(instantiate(cfg.dec_net, num_channels=cfg.dataset.shape[0]))
        self.name = "cnp"

    def encode(self, xi_context, x_context):
        return self.encoder(xi_context, x_context)

    def decode(self, representation, xi_target):
        # Expand the dim-1 of the representation to match the number of targets
        batch_shape = max(representation.shape[0], xi_target.shape[0])
        num_target = xi_target.shape[1]
        representation = representation.unsqueeze(1).expand(batch_shape, num_target, -1)
        xi_target = torch.flatten(xi_target.float(), end_dim=1)
        xi_target = self.cov_net(xi_target).reshape(batch_shape, num_target, -1)
        means, scales = self.decoder(torch.cat([representation, xi_target], dim=-1))
        scales = self.covariance_activation_function(scales)
        # scales = 0.01 * torch.ones_like(means)
        return means, scales

    def forward(self, xi_context, x_context, xi_target):
        return self.decode(self.encode(xi_context, x_context), xi_target)

    def compute_batch_metrics(self, batch, *args, **kwargs):
        (x, xi), labels = batch
        (xi_target, xi_context), (x_target, x_context) = split_context_target(xi, x, index=0)
        x_prediction_mean, x_prediction_scale = self.forward(xi_context, x_context, xi_target)
        dist = torch.distributions.Normal(x_prediction_mean, x_prediction_scale)
        x_target = x_target.reshape(*x_prediction_mean.shape)
        log_ll = dist.log_prob(x_target).mean()
        return dict(loss=-log_ll)
