import torch
from torch.utils.data._utils.collate import default_collate
from omegaconf.listconfig import ListConfig


def random_views_collate_fn(num_views):
    """
    Returns a collate function suitable for use with multi-view object datasets.
    This collate function selects `num_views` from the total number of views at random without replacement.
    The choice is re-randomized at each call.
    """

    def collate_views(batch):
        rand_perms = [torch.randperm(o.total_views) for o in batch]
        realised_batches = [o.make_views(r[:num_views]) for o, r in zip(batch, rand_perms)]
        return default_collate(realised_batches)

    return collate_views


def deterministic_views_collate_fn(num_views):
    """
    Returns a collate function suitable for use with multi-view object datasets.
    This collate function selects `num_views` from the total number of views deterministically.
    The same subset of views is always returned.
    """

    def collate_views(batch):
        realised_batches = [o.make_views(list(range(min(num_views, o.total_views)))) for o in batch]
        return default_collate(realised_batches)

    return collate_views


def one_dim_collate_fn(num_views, bound):
    """"""

    def collate_views(batch):
        bounds = bound if isinstance(bound, ListConfig) else [-bound, bound]
        covariates = torch.distributions.Uniform(low=bounds[0], high=bounds[1]).sample(
            torch.Size([len(batch), num_views, 1])
        )
        realised_batches = [o.make_views(covariates[i]) for i, o in enumerate(batch)]
        return default_collate(realised_batches)

    return collate_views


def rebracket_collate(batch):
    batch = [((x, xi), y) for (x, xi, y) in batch]
    return default_collate(batch)
