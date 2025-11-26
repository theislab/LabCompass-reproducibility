import abc
import math

import torch


class LambdaScheduler(abc.ABC):
    @abc.abstractmethod
    def compute_lambda_t(
        self,
        t,
        *args
    ):
        raise NotImplementedError

class ConstantLambdaScheduler(LambdaScheduler):
    def __init__(
        self,
        lambda_val:float,
    ):
        self.lambda_val = lambda_val
    
    def compute_lambda_t(self, t, *args, **kwargs):
        return self.lambda_val * torch.ones((*t.shape[:-1], 1), device=t.device)


class ExponentialDecayScheduler(LambdaScheduler):
    def __init__(
        self,
        lmin,
        lmax,
        gamma,
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.gamma = gamma

    def compute_lambda_t(self, t, *args, **kwargs):
        b = (self.lmin - self.lmax)/(math.exp(-self.gamma)-1)
        a = self.lmax - b
        return a + b*torch.exp(-self.gamma*t)
