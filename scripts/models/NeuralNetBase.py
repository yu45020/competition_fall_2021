import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetBase(pl.LightningModule):

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optim

    def count_parameters(self):
        return f"{sum(p.numel() for p in self.parameters()):,}"

    @staticmethod
    def criteria(input, target):
        return mse_loss(input, target)

    @staticmethod
    def criteria_weighted(input, target, weight):
        return (weight * (input - target) ** 2).mean()

    @staticmethod
    def prepare_dataset(x_cont, x_cat, y, weight=None, return_dataloader=False, batch_size=1024):
        if y is not None:
            if weight is None:
                weight = np.ones_like(y)

            out = TensorDataset(torch.FloatTensor(x_cont),
                                torch.LongTensor(x_cat),
                                torch.FloatTensor(y),
                                torch.FloatTensor(weight)
                                )
        else:
            print("No Y")
            out = TensorDataset(torch.FloatTensor(x_cont),
                                torch.LongTensor(x_cat))
        if return_dataloader:
            return DataLoader(out, batch_size=batch_size, shuffle=False)
        else:
            return out

    def prepare_module(self, x_train_category, x_train_continuous, y_train, weight_train,
                       x_val_category, x_val_continuous, y_val,
                       x_test_test_category, x_test_continuous, y_test,
                       batch_size,
                       num_workers=0,
                       ):
        assert all(i.shape[1] == 1 for i in (y_train, y_test, y_val))

        data_train = self.prepare_dataset(x_train_continuous, x_train_category, y_train, weight_train)

        data_val = self.prepare_dataset(x_val_continuous, x_val_category, y_val)

        data_test = self.prepare_dataset(x_test_continuous, x_test_test_category, y_test)

        return pl.LightningDataModule.from_datasets(train_dataset=data_train,
                                                    val_dataset=data_val,
                                                    test_dataset=data_test,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers)

        # return {"train": data_train, "val": data_val, "test": data_test}

    @staticmethod
    def check_point_callback(*args, **kwargs):
        return ModelCheckpoint(*args, **kwargs, )


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def __init__(self, dim=-1):
        super(GEGLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_, gate = x.chunk(2, dim=self.dim)
        return x_ * F.gelu(gate)


class RMSNorm(nn.Module):
    # copy from
    # https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class GradualWarmupScheduler(_LRScheduler, ReduceLROnPlateau):
    # copy from
    # https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier=1.0, total_epoch=10, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=10, min_lr=1e-4, verbose=True)
        # LambdaLR(optimizer, lr_lambda=lambda epoch: 1e-4)  # after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, metrics=None, epoch=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                self.after_scheduler.step(metrics, epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
