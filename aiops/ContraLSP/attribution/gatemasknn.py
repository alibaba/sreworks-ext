import torch as th
import torch.nn as nn
import numpy as np
from captum._utils.common import _run_forward
from sklearn.cluster import KMeans
import random
from typing import Callable, Union
from collections import Counter

from tint.models import Net, MLP
import warnings
warnings.filterwarnings("ignore")


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size=20, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = th.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class GateMaskNN(nn.Module):
    """
    Extremal Mask NN model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
        factor_dilation (float): Ratio between the final and the
            initial size regulation factor. Default to 100
    References
        #. `Learning Perturbations to Explain Time Series Predictions <https://arxiv.org/abs/2305.18840>`_
        #. `Understanding Deep Networks via Extremal Perturbations and Smooth Masks <https://arxiv.org/abs/1910.08485>`_
    """

    def __init__(
        self,
        forward_func: Callable,
        model: nn.Module = None,
        batch_size: int = 32,
        factor_dilation: float = 10.0,
        based: float = 0.5,
        pooling_method: str = 'sigmoid',
        use_win: bool = False,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "forward_func", forward_func)
        self.model = model
        self.batch_size = batch_size

        self.input_size = None
        self.win_size = None
        self.sigma = None
        self.channels = None
        self.T = None
        self.reg_multiplier = None
        self.mask = None
        self.based = based
        self.factor_dilation = factor_dilation
        self.pooling_method = pooling_method
        self.use_win = use_win

    def init(self, input_size: tuple, batch_size: int = 32,
             win_size: int = 5, sigma: float = 0.5, n_epochs: float = 100) -> None:
        self.input_size = input_size
        self.batch_size = batch_size
        self.win_size = win_size
        self.sigma = sigma
        self.channels = input_size[2]
        self.T = input_size[1]
        self.reg_multiplier = np.exp(
            np.log(self.factor_dilation) / n_epochs
        )

        self.moving_avg = MovingAvg(self.win_size)

        self.mask = nn.Parameter(th.Tensor(*input_size))

        self.trendnet = nn.ModuleList()
        for i in range(self.channels):
            self.trendnet.append(MLP([self.T, 32, self.T], activations='relu'))

        self.reset_parameters()

    def hard_sigmoid(self, x):
        return th.clamp(x, 0.0, 1.0)

    def reset_parameters(self) -> None:
        self.mask.data.fill_(0.5)
        # In the first training step, µd is 0.0
        # self.mask.data.fill_(0.0)

    def forward(
        self,
        x: th.Tensor,
        batch_idx,
        baselines,
        target,
        *additional_forward_args,
    ) -> (th.Tensor, th.Tensor):

        mu = self.mask[
            self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)
        ]
        noise = th.randn(x.shape)
        mask = mu + self.sigma * noise.normal_() * self.training
        mask = self.refactor_mask(mask, x)

        # hard sigmoid
        mask = self.hard_sigmoid(mask)

        # If model is provided, we use it as the baselines
        if self.model is not None:
            baselines = self.model(x - baselines)

        # Mask data according to samples
        # We eventually cut samples up to x time dimension
        # x1 represents inputs with important features masked.
        # x2 represents inputs with unimportant features masked.
        mask = mask[:, : x.shape[1], ...]
        x1 = x * mask + baselines * (1.0 - mask)
        x2 = x * (1.0 - mask) + baselines * mask

        # Return f(perturbed x)
        return (
            _run_forward(
                forward_func=self.forward_func,
                inputs=x1,
                target=target,
                additional_forward_args=additional_forward_args,
            ),
            _run_forward(
                forward_func=self.forward_func,
                inputs=x2,
                target=target,
                additional_forward_args=additional_forward_args,
            ),
        )

    def trend_info(self, x):
        if self.use_win:
            trend = self.moving_avg(x)
        else:
            trend = x
        trend_out = th.zeros(trend.shape,
                             dtype=trend.dtype).to(trend.device)
        for i in range(self.channels):
            trend_out[:, :, i] = self.trendnet[i](trend[:, :, i])

        # front = x[:, 0:1, :]
        # x_f = th.cat([front, x], dim=1)[:, :-1, :]
        # res = th.abs(x - x_f)
        return trend_out

    def refactor_mask(self, mask, x):

        trend = self.trend_info(x)
        mask = mask + self.based

        if self.pooling_method == 'sigmoid': #https://paperswithcode.com/method/silu
            mask = mask * th.sigmoid(
                mask * trend
            )
        elif self.pooling_method == 'softmax':  #https://arxiv.org/pdf/1910.08485.pdf, no work.
            mask = mask * th.softmax(
                mask * trend, dim=1
            )
        elif self.pooling_method == 'cdf':  #https://paperswithcode.com/method/gelu
            mask = 0.5 * (
                    mask + mask * th.erf(mask * trend)
            )
        elif self.pooling_method == 'none':  # no smoothness
            mask = mask
        else:
            assert False, f"Unknown pooling method {self.pooling_method}"
        return mask

    def representation(self, x):
        mask = self.refactor_mask(self.mask, x)
        mask = self.hard_sigmoid(mask)
        return mask.detach().cpu()


class GateMaskNet(Net):
    """
    Gate mask model as a Pytorch Lightning model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        preservation_mode (bool): If ``True``, uses the method in
            preservation mode. Otherwise, uses the deletion mode.
            Default to ``True``
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
        lambda_1 (float): Weighting for the mask loss. Default to 1.
        lambda_2 (float): Weighting for the model output loss. Default to 1.
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0
    """

    def __init__(
        self,
        forward_func: Callable,
        preservation_mode: bool = True,
        model: nn.Module = None,
        batch_size: int = 32,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        factor_dilation=10.0,
        based=0.5,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        mask = GateMaskNN(
            forward_func=forward_func,
            model=model,
            batch_size=batch_size,
            factor_dilation=factor_dilation,
            based=based
        )

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

        self.preservation_mode = preservation_mode
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.based = based

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, baselines, target, *additional_forward_args = batch

        # If additional_forward_args is only one None,
        # set it to None
        if additional_forward_args == [None]:
            additional_forward_args = None

        # Get perturbed output
        # y_hat1 is computed by masking important features
        # y_hat2 is computed by masking unimportant features
        if additional_forward_args is None:
            y_hat1, y_hat2 = self(x.float(), batch_idx, baselines, target)
        else:
            y_hat1, y_hat2 = self(
                x.float(),
                batch_idx,
                baselines,
                target,
                *additional_forward_args,
            )

        # Get unperturbed output for inputs and baselines
        y_target1 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )

        # Add L1 loss
        mask_ = self.net.mask[
                self.net.batch_size
                * batch_idx: self.net.batch_size * (batch_idx + 1)
                ]
        reg = 0.5 + 0.5 * th.erf(self.net.refactor_mask(mask_, x) / (self.net.sigma * np.sqrt(2)))
        # reg = self.net.refactor_mask(mask_, x).abs()

        # trend_reg = self.net.trend_info(x).abs().mean()
        mask_loss = self.lambda_1 * reg.mean()
        # mask_loss = self.lambda_1 * th.sum(reg, dim=[1,2]).mean()

        triplet_loss = 0
        if self.net.model is not None:
            condition = self.net.model(x - baselines)
            if self.net.batch_size > 1:
                triplet_loss = self.lambda_2 * self._triplet_loss(condition)
            else:
                triplet_loss = self.lambda_2 * condition.abs().mean()

        # Add preservation and deletion losses if required
        if self.preservation_mode:
            main_loss = self.loss(y_hat1, y_target1)
        else:
            main_loss = -1. * self.loss(y_hat2, y_target1)

        loss = main_loss + mask_loss + triplet_loss

        # test log
        _test_mask = self.net.representation(x)
        test = (_test_mask[_test_mask > 0]).sum()
        lambda_1_t = self.lambda_1
        lambda_2_t = self.lambda_2
        reg_t = reg.mean()
        print(test, reg_t, triplet_loss, lambda_1_t, lambda_2_t, main_loss)

        return loss

    def _triplet_loss(self, condition):
        _, ts_dim, num_dim = condition.shape
        points = condition.reshape(-1, ts_dim * num_dim).detach().numpy()
        num_cluster = 2
        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(points)
        cluster_label = kmeans.labels_
        num_cluster_set = Counter(cluster_label)

        # loss of each cluster
        loss_cluster = condition.abs().mean()   # placeholder
        for i in range(num_cluster):
            if num_cluster_set[i] < 2:
                continue
            cluster_i = points[np.where(cluster_label == i)]
            distance_i = kmeans.transform(cluster_i)[:, i]
            dist_positive = th.DoubleTensor([1])
            dist_negative = th.DoubleTensor([1])
            if num_cluster_set[i] >= 250:
                num_positive = 50
            else:
                num_positive = int(num_cluster_set[i] / 5 + 1)

            # select anchor and positive
            anchor_positive = np.argpartition(distance_i, num_positive)[:(num_positive + 1)]
            # torch anchor
            representation_anc = th.from_numpy(points[anchor_positive[0]])
            # transfer 1D to 3D
            representation_anc = th.reshape(representation_anc, (1, 1, np.shape(points)[1]))

            # positive part
            for l in range(1, num_positive + 1):
                # torch positive
                representation_pos = th.from_numpy(points[anchor_positive[l]])
                # transfer 1D to 3D
                representation_pos = th.reshape(representation_pos, (1, 1, np.shape(points)[1]))
                anchor_minus_positive = representation_anc - representation_pos
                dist_positive += th.norm(anchor_minus_positive, p=1)/np.shape(points)[1]
            dist_positive = dist_positive / num_positive

            # negative part
            for k in range(num_cluster):
                dist_cluster_k_negative = th.DoubleTensor([1])
                if k == i:
                    continue
                else:
                    # select negative
                    if num_cluster_set[k] >= 250:
                        num_negative_cluster_k = 50
                    else:
                        num_negative_cluster_k = int(num_cluster_set[k] / 5 + 1)

                    negative_cluster_k = random.sample(range(points[kmeans.labels_ == k][:, 0].size),
                                                       num_negative_cluster_k)
                    for j in range(num_negative_cluster_k):
                        # torch negative
                        representation_neg = th.from_numpy(points[kmeans.labels_ == k][negative_cluster_k[j]])
                        # transfer 1D to 3D
                        representation_neg = th.reshape(representation_neg, (1, 1, np.shape(points)[1]))
                        anchor_minus_negative = representation_anc - representation_neg
                        dist_cluster_k_negative += th.norm(anchor_minus_negative, p=1)/np.shape(points)[1]

                dist_cluster_k_negative = dist_cluster_k_negative / num_negative_cluster_k
                dist_negative += dist_cluster_k_negative
            dist_negative = dist_negative / (num_cluster - 1)
            #  loss =  -(margin + positives - negatives)
            if self.preservation_mode:
                loss_values = th.max((-dist_positive + dist_negative - 1)[0], th.tensor(0.)) / num_cluster
            else:
                loss_values = th.max(-(-dist_positive + dist_negative - 1)[0], th.tensor(0.)) / num_cluster
            loss_cluster += loss_values

        return loss_cluster

    def on_train_epoch_end(self) -> None:
        # Increase the regulator coefficient
        # self.lambda_1 *= self.net.reg_multiplier
        pass

    def configure_optimizers(self):
        params = [{"params": self.net.mask}]
        if self.net.model is not None:
            params += [{"params": self.net.model.parameters()}]
        if self.net.trendnet is not None:
            params += [{"params": self.net.trendnet.parameters()}]
        # params = self.net.parameters()
        if self._optim == "adam":
            optim = th.optim.Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}
