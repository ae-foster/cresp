import torch
from torch import nn
from src.architectures import MLP
from utils import collapse_batch_dim


class MeanAgg(nn.Module):
    """ Compute the mean of context representations """

    def __init__(self, **kwargs):
        super(MeanAgg, self).__init__()
        self.name = "mean"

    def forward(self, _, encoded, *args):
        return encoded.mean(1)


class FirstAgg(nn.Module):
    """ Only keep the first context view and drop the other ones """

    def __init__(self, **kwargs):
        super(FirstAgg, self).__init__()
        self.name = "first"

    def forward(self, _, encoded, *args):
        return encoded[:, 0, ...]


class KernelAgg(nn.Module):
    """ Apply kernel weighting """

    def __init__(self, kernel, lambda_reg, learn, **kwargs):
        super(KernelAgg, self).__init__()
        self.name = "kernel"
        self.kernel = kernel
        for param in self.kernel.parameters():
            param.requires_grad = learn
        self.lambda_reg = lambda_reg

    def forward(self, context_xi, encoded, target_xi):
        K = self.kernel(context_xi).evaluate()
        reg = self.lambda_reg * torch.eye(K.shape[1]).repeat(K.shape[0], 1, 1).to(K)
        # if target_xi is None or not self.target_cov:
        # k_star = torch.ones(*encoded.shape[:-1], 1).to(encoded)
        # else:
        k_star = self.kernel(context_xi, target_xi).evaluate()
        weights = torch.bmm((K + reg).inverse(), k_star)
        weights = nn.functional.softmax(weights, dim=1)

        return (weights * encoded).sum(1)


class CrossAttentionAgg(nn.Module):
    """ """

    def __init__(self, attender, **kwargs):
        super(CrossAttentionAgg, self).__init__()
        self.name = "cross_attn"
        self.attender = attender

    def forward(self, encoded_xi, encoded_x, target_xi):
        # keys, queries, values
        representation = self.attender(encoded_xi, target_xi, encoded_x).squeeze(1)
        return representation


class SimpleEnc(nn.Module):
    """ Do not use the covariate, i.e. simply encode the view """

    def __init__(self, **kwargs):
        super(SimpleEnc, self).__init__()

    def forward(self, encoded_x, encoded_xi):
        return encoded_x


class CatEnc(nn.Module):
    """ Concatenate covariate with view """

    def __init__(self, net, **kwargs):
        super(CatEnc, self).__init__()
        self._net = net

    def forward(self, encoded_x, encoded_xi):
        return self._net(torch.cat([encoded_xi, encoded_x], dim=-1))


class ChannelEnc(nn.Module):
    """Transform with linear layer covariate before being concatenated to the observation a new channel dim,
    then feed it to the encoder to get representations.
    """

    def __init__(self, net, dataset, **kwargs):
        super(ChannelEnc, self).__init__()
        self._net = net
        self.dataset = dataset

    def forward(self, encoded_x, encoded_xi):
        context_xi_channel = encoded_xi.view(-1, 1, *self.dataset.shape[1:])  # B*V, C, H, W
        return self._net(torch.cat([encoded_x, context_xi_channel], dim=-3))  # B*V, H'


class Gated(nn.Module):
    def __init__(self, in_dim, representation_dim, gating, active, pointwise, extra_layer, **kwargs):
        super(Gated, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, representation_dim)
        if extra_layer:
            self.inner = nn.Sequential(
                nn.ReLU(), nn.Linear(representation_dim, representation_dim), nn.BatchNorm1d(representation_dim)
            )
        else:
            self.inner = nn.Identity()

        self.gating = gating
        if self.gating:
            if pointwise:
                self.fc2 = nn.Linear(in_dim, representation_dim)
            else:
                self.fc2 = nn.Linear(in_dim, 1)
            if active == "tanh":
                self.activation = nn.Tanh()
            elif active == "sigmoid":
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unexpected activation {active}")

    def forward(self, x):
        # This extra 1 dimension breaks batch norm among other things
        if len(x.shape) == 3:
            x = x.squeeze(1)
        normed = self.bn(x)
        representation = self.fc1(normed)
        representation = self.inner(representation)
        if self.gating:
            multiplicative = self.activation(self.fc2(normed))
            representation = multiplicative * representation
        return representation  # [B, V, H']


class Encoder(nn.Module):
    """https://github.com/MJHutchinson/SteerableCNP/blob/dc77e712c7eb05a3c4931d9fbb46412d40927c34/steer_cnp/cnp.py"""

    def __init__(self, enc, agg, cov_net, obs_net, self_attention):
        super(Encoder, self).__init__()
        self._enc = enc
        self._agg = agg
        self._cov_net = cov_net
        self._obs_net = obs_net
        self._self_attention = self_attention
        self._targeted = False

    def permute_encode(self, x, xi):
        """Encodes xi using `cov_net`, then feeds x and encoded xi into the `enc`, which further processes them into
        a set of encodings. Permutation is applied for BatchNorm optimization."""
        # Permuting these dimensions should help with BatchNorm issue
        batch_dim, context_dim = x.shape[0:2]
        x, xi = x.transpose(0, 1), xi.transpose(0, 1)
        x, xi = collapse_batch_dim(x), collapse_batch_dim(xi)
        # Encode xi and x
        xi_encoded = self._cov_net(xi)
        x_encoded = self._obs_net(x)
        # This first encodes x, then combines xi_encoded and x_encoded and encodes the result
        encodings = self._enc(x_encoded, xi_encoded)
        xi_encoded = xi_encoded.reshape(context_dim, batch_dim, *xi_encoded.shape[1:]).transpose(0, 1)
        x_encoded = x_encoded.reshape(context_dim, batch_dim, *x_encoded.shape[1:]).transpose(0, 1)
        encodings = encodings.reshape(context_dim, batch_dim, *encodings.shape[1:]).transpose(0, 1)
        # We may be interested in using xi_encoded later
        return encodings, xi_encoded, x_encoded

    def postprocess_context(self, encoded_xi, encoded_pair, encoded_target_xi=None):
        """Applies self-attention and aggregation to the context."""
        encoded_pair = self._self_attention(encoded_pair)
        aggregated = self._agg(encoded_xi, encoded_pair, encoded_target_xi)
        return aggregated