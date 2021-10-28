import sys
import torch
from tqdm import tqdm


def collapse_batch_dim(x):
    return x.reshape(-1, *x.shape[2:]) if x is not None else None


def split_context_target(*tensors, index=0, dim=1):
    assert index == 0
    assert dim == 1
    output = [(tensor[:, [0], ...], tensor[:, 1:, ...]) for tensor in tensors]
    return output


def half_context_target(*tensors, dim=1):
    assert dim == 1
    initial_shape = tensors[0].shape[dim]
    split_point = initial_shape // 2
    output = [(tensor[:, :split_point, ...], tensor[:, split_point:, ...]) for tensor in tensors]
    return output


def encode_dataset(encoder, dataloader, device, targeted, progress_bar=False):
    encoder.eval()

    store = []
    with torch.no_grad():
        iterator = enumerate(dataloader)
        if progress_bar:
            iterator = tqdm(iterator, total=len(dataloader), bar_format="{desc}{bar}{r_bar}", file=sys.stdout)
        for batch_idx, ((x, xi), y) in iterator:
            x, xi, y = x.to(device), xi.to(device), y.to(device)
            if targeted:
                (y, y_context), (_, _) = split_context_target(y, y, index=0)
                y = y.squeeze(-1)
            representation = encoder(xi, x, evaluation=True, targeted=targeted)
            store.append((representation, y))
            if progress_bar:
                iterator.set_description("Encoded %d/%d" % (batch_idx + 1, len(dataloader)))

    X, y = zip(*store)
    X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)
    return X, y
