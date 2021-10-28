import math
import torch
from torch.distributions import Uniform, Bernoulli


def gaussian(func, xi, x, epsilon):
    return x + epsilon * torch.randn(x.shape)


def bimodal(func, xi, x, epsilon):
    return x + epsilon * Bernoulli(probs=0.5).sample(x.shape)


class OneDimFunc:
    """"""

    def __init__(self, N, epsilon, noise, transform=None, **kwargs):
        super().__init__()
        self.N = N
        self.noise = noise
        self.epsilon = epsilon
        self.transform = eval(transform) if transform else None
        self.params, self.labels = self.sample_functions(N)

    def sample_functions(self, N):
        raise NotImplementedError()

    def make_function(self, coeffs):
        raise NotImplementedError()

    def __len__(self):
        """
        Return number of total models in the loaded dataset.
        """
        return self.N

    def __getitem__(self, index: int):
        """"""
        params = self.params[index]
        labels = self.labels[index]
        function = self.make_function(*params)
        return OneDimFuncObject(function, params, labels, self.noise, self.epsilon, self.transform)


class OneDimFuncObject:
    def __init__(self, function, params, labels, noise, epsilon, transform):
        self.noise = noise
        self.epsilon = epsilon
        self.transform = transform
        self.function = function
        self.labels = labels
        self.params = params

    def make_views(self, covariates):
        views = self.function(covariates)
        views = self.noise(self.function, covariates, views, self.epsilon)
        return (views, covariates), self.labels


class SineFunc(OneDimFunc):
    """"""

    def __init__(self, period, amplitude, phase, **kwargs):
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        super().__init__(**kwargs)

    def sample_functions(self, N):
        periods = Uniform(low=self.period[0], high=self.period[1]).sample(torch.Size([N]))
        amplitudes = Uniform(low=self.amplitude[0], high=self.amplitude[1]).sample(torch.Size([N]))
        phases = Uniform(low=0, high=self.phase).sample(torch.Size([N]))
        params = torch.stack([periods, amplitudes, phases]).t()
        target = torch.stack([phases, amplitudes, periods]).t()
        return params, target.view(-1, 3)

    def make_function(self, period, amplitude, phase):
        return lambda x: amplitude * torch.sin(2 * math.pi / period * x + phase)