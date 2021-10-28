import torch
from torch import nn


class BaseCritic(nn.Module):
    def __init__(self, latent_dim, projection_dim, linear, temperature=1.0):
        super(BaseCritic, self).__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        layers = []
        if not linear:
            layers.extend([
                nn.Linear(latent_dim, latent_dim, bias=False),  # w1
                nn.BatchNorm1d(latent_dim),                     # bn1
                nn.ReLU(),
            ])
        layers.extend([
            nn.Linear(latent_dim, projection_dim, bias=False), # w2
            nn.BatchNorm1d(projection_dim, affine=False),      # bn2
        ])
        self.project = nn.Sequential(*layers)


class CriticSimCLR(BaseCritic):
    def forward(self, h1, h2):
        z1, z2 = self.project(h1), self.project(h2)
        sim11 = self.cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        return raw_scores, targets


class CriticCresp(BaseCritic):
    def forward(self, h_target, h_predictive):
        z_target, z_predictive = self.project(h_target), self.project(h_predictive)  # [B, Z]
        # # So, target indices vary over dimension 1 and predictive indices over dimension 0
        # # Cross entropy loss will normalize over dimension 1: normalized self-supervised task over targets.. ok
        sim_scores = self.cossim(z_target.unsqueeze(0), z_predictive.unsqueeze(1)) / self.temperature  # [B, B]
        d = sim_scores.shape[-1]
        targets = torch.arange(d, dtype=torch.long, device=sim_scores.device)
        return sim_scores, targets


class MLPCriticCresp(CriticCresp):
    def __init__(self, net, temperature=1.0):
        super(MLPCriticCresp, self).__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.project = net