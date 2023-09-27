import torch
import numpy as np
import random as rd
import argparse
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from tint.attr import (
    ExtremalMask,
    DynaMask,
)
from tint.attr.models import (
    ExtremalMaskNet,
    MaskNet,
)
from tint.models import MLP, RNN

from utils.tools import print_results, plot_example_box
from attribution.gatemasknn import *
from attribution.gate_mask import *

rd.seed(42)
np.random.seed(42)
torch.manual_seed(42)
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unit Testing')
    parser.add_argument('--feature_num', default=3, type=int)
    parser.add_argument('--ts', default=10*2, type=int)
    parser.add_argument('--bs', default=100, type=int)
    args = parser.parse_args()

    explainers = ["gatemask","extrmask", "dynamask"]

    X = torch.randint(low=0, high=5, size=(args.bs, args.ts, args.feature_num)).float()

    def black_box(input):
        num_samples, time_len, num_f = input.shape
        output = torch.zeros((num_samples, time_len, 1))    # (bs, T, 1)

        # level 1
        output[:25, 4*2:9*2, :] = input[:25, 0*2:5*2, 0:1]

        # level 2
        output[25:50,  4*2:, :] = input[25:50,  4*2:, 1:2]

        # level 3
        output[50:, 0*2:2*2, :] = (input[50:, 0*2:2*2, 0:1] + input[50:, 0*2:2*2, 2:])**2
        return output.reshape(-1, time_len, 1)

    true_saliency = torch.zeros(X.shape)
    true_saliency[:25, 0*2:5*2, 0:1] = 1
    true_saliency[25:50, 4*2:, 1:2] = 1
    true_saliency[50:, 0*2:2*2, 0:1], true_saliency[50:, 0*2:2*2, 2:] = 1, 1
    print("-===============", true_saliency.sum())
    for i in range(args.bs):
        plot_example_box(true_saliency, i, "./plot/demo2plot/true_{}.png".format(i))

    if "gatemask" in explainers:
        trainer = Trainer(
            max_epochs=50,
            accelerator="cpu",
            log_every_n_steps=2,
            callbacks=[EarlyStopping('train_loss', patience=10, mode='min')],
        )
        mask = GateMaskNet(
            forward_func=black_box,
            model=nn.Sequential(
                RNN(
                    input_size=X.shape[-1],
                    rnn="gru",
                    hidden_size=X.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * X.shape[-1], X.shape[-1]]),
            ),
            lambda_1=0.1,  # 0.1 for our lambda is suitable
            lambda_2=0.1,
            optim="adam",
            lr=0.1,
        )
        explainer = GateMask(black_box)
        _attr = explainer.attribute(
            X,
            trainer=trainer,
            mask_net=mask,
            batch_size=args.bs,
            win_size=5,
            sigma=0.5,
        )
        gatemask_saliency = _attr.clone().detach()
        print_results(gatemask_saliency, true_saliency)
        # plot_example_box(gatemask_saliency, 0)
        # plot_example_box(gatemask_saliency, 49)
        # plot_example_box(gatemask_saliency, 99)

        for i in range(args.bs):
            plot_example_box(gatemask_saliency, i, "./plot/demo2plot/gatemask_{}.png".format(i))

    if "extrmask" in explainers:
        trainer = Trainer(
            max_epochs=50,
            accelerator='cpu',
            log_every_n_steps=2,
            callbacks=[EarlyStopping('train_loss', patience=10, mode='min')],
        )
        mask = ExtremalMaskNet(
            forward_func=black_box,
            model=nn.Sequential(
                RNN(
                    input_size=X.shape[-1],
                    rnn="gru",
                    hidden_size=X.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * X.shape[-1], X.shape[-1]]),
            ),
            optim="adam",
            lr=0.1,
        )
        explainer = ExtremalMask(black_box)
        _attr = explainer.attribute(
            X,
            trainer=trainer,
            mask_net=mask,
            batch_size=args.bs,
        )
        nnmask_saliency = _attr.clone().detach().numpy()
        print_results(nnmask_saliency, true_saliency)

        # plot_example_box(nnmask_saliency, 0)
        # plot_example_box(nnmask_saliency, 49)
        # plot_example_box(nnmask_saliency, 99)

        for i in range(args.bs):
            plot_example_box(nnmask_saliency, i, "./plot/demo2plot/extrmask_{}.png".format(i))

    if "dynamask" in explainers:
        from attribution.mask_group import MaskGroup
        from attribution.perturbation import GaussianBlur
        from utils.losses import mse_multiple

        pert = GaussianBlur(device=device)  # We use a Gaussian Blur perturbation operator
        mask_group = MaskGroup(perturbation=pert, device=device, random_seed=42)
        mask_group.fit_multiple(
            f=black_box,
            X=X,
            area_list=np.arange(0.01, 0.51, 0.01),
            loss_function_multiple=mse_multiple,
            n_epoch=50,
            learning_rate=0.1,
        )
        thresh = 0.01 * torch.ones(args.bs)
        mask = mask_group.get_extremal_mask_multiple(thresh)  # The mask with the lowest error is selected
        dynamask_saliency = mask.clone().detach().numpy()
        print_results(dynamask_saliency, true_saliency)

        # plot_example_box(dynamask_saliency, 0)
        # plot_example_box(dynamask_saliency, 49)
        # plot_example_box(dynamask_saliency, 99)
        #
        for i in range(args.bs):
            plot_example_box(dynamask_saliency, i, "./plot/demo2plot/dynamask_{}.png".format(i))
