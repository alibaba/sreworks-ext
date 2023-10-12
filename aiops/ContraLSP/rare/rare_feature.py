import argparse
import os
import pickle as pkl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from statsmodels.tsa.arima_process import ArmaProcess
from attribution.gate_mask import GateMask
from attribution.gatemasknn import *
from utils.tools import print_results
import torch.nn as nn
from attribution.mask_group import MaskGroup
from attribution.perturbation import GaussianBlur
from tint.attr import ExtremalMask
from tint.attr.models import ExtremalMaskNet
from tint.models import MLP, RNN
from attribution.explainers import *
from utils.losses import mse_multiple


explainers = ["dynamask", "nnmask", "gatemask", "fo", "fp", "ig", "shap"]
seed_everything(42)


def run_experiment(
    cv: int = 0,
    N_ex: int = 100,
    T: int = 50,
    N_features: int = 50,
    N_select: int = 5,
    save_dir: str = "./results/rare_feature/",
):
    """Run experiment.

    Args:
        cv: Do the experiment with different cv to obtain error bars.
        N_ex: Number of time series to generate.
        T: Length of each time series.
        N_features: Number of features in each time series.
        N_select: Number of features that are truly salient.
        save_dir: Directory where the results should be saved.

    Returns:
        None
    """
    # Create the saving directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Initialize useful variables
    random_seed = cv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Generate the input data
    ar = np.array([2, 0.5, 0.2, 0.1])  # AR coefficients
    ma = np.array([2])  # MA coefficients
    data_arima = ArmaProcess(ar=ar, ma=ma).generate_sample(nsample=(N_ex, T, N_features), axis=1)
    X = torch.tensor(data_arima, device=device, dtype=torch.float32)

    # Initialize the saliency tensors
    true_saliency = torch.zeros(size=(N_ex, T, N_features), device=device, dtype=torch.int64)
    # The truly salient features are selected randomly via a random permutation
    perm = torch.randperm(N_features, device=device)
    true_saliency[:, int(T / 4) : int(3 * T / 4), perm[:N_select]] = 1
    with open(os.path.join(save_dir, f"true_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(true_saliency, file)
    print("==============The mean of true_saliency is", true_saliency.sum(), "==============\n" + 100 * "=")

    # The white box only depends on the truly salient features
    def f(input):
        output = torch.zeros(input.shape, device=input.device)
        output[:, int(T / 4) : int(3 * T / 4), perm[:N_select]] = input[:, int(T / 4) : int(3 * T / 4), perm[:N_select]]
        output = (output ** 2).sum(dim=-1)
        return output.unsqueeze(-1)

    if "gatemask" in explainers:
        trainer = Trainer(
            max_epochs=200,
            accelerator="cpu",
            log_every_n_steps=2,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = GateMaskNet(
            forward_func=f,
            model=nn.Sequential(
                RNN(
                    input_size=X.shape[-1],
                    rnn="gru",
                    hidden_size=X.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * X.shape[-1], X.shape[-1]]),
            ),
            lambda_1=0.1,   # 0.01 for our lambda is suitable
            lambda_2=0.1,
            optim="adam",
            lr=0.1,
        )
        explainer = GateMask(f)
        _attr = explainer.attribute(
            X,
            trainer=trainer,
            mask_net=mask,
            batch_size=N_ex,
            sigma=0.5,
        )
        gatemask_saliency = _attr.clone().detach()
        with open(os.path.join(save_dir, f"gatemask_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(gatemask_saliency, file)
        print("==============gatemask==============")
        print_results(gatemask_saliency, true_saliency)

    if "nnmask" in explainers:
        trainer = Trainer(
            max_epochs=200,
            accelerator='cpu',
            log_every_n_steps=2,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = ExtremalMaskNet(
            forward_func=f,
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
        explainer = ExtremalMask(f)
        _attr = explainer.attribute(
            X,
            trainer=trainer,
            mask_net=mask,
            batch_size=50,
        )
        nnmask_saliency = _attr.clone().detach().numpy()
        with open(os.path.join(save_dir, f"nnmask_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(nnmask_saliency, file)
        print("==============nnmask==============")
        print_results(nnmask_saliency, true_saliency)

    if "dynamask" in explainers:
        pert = GaussianBlur(device=device)  # We use a Gaussian Blur perturbation operator
        mask_group = MaskGroup(perturbation=pert, device=device, random_seed=random_seed, verbose=True)
        mask_group.fit_multiple(
            f=f,
            X=X,
            area_list=np.arange(0.011, 0.041, 0.002),
            loss_function_multiple=mse_multiple,
            n_epoch=200,
            size_reg_factor_dilation=1000,
            size_reg_factor_init=0.01,
            learning_rate=0.1,
        )
        thresh = 0.01 * torch.ones(N_ex)
        mask = mask_group.get_extremal_mask_multiple(thresh)  # The mask with the lowest error is selected
        dynamask_saliency = mask.clone().detach().numpy()
        with open(os.path.join(save_dir, f"dynamask_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(dynamask_saliency, file)
        print("==============dynamask==============")
        print_results(dynamask_saliency, true_saliency)

    def f(input):
        output = torch.zeros(input.shape, device=input.device)
        output[int(T / 4) : int(3 * T / 4), perm[:N_select]] = input[int(T / 4) : int(3 * T / 4), perm[:N_select]]
        output = (output ** 2).sum(dim=-1)
        return output.unsqueeze(-1)

    if "fo" in explainers:
        fo_saliency = torch.zeros(size=true_saliency.shape, device=device)
        for k in range(N_ex):  # We compute the attribution for each individual time series
            x = X[k, :, :]
            # Feature Occlusion attribution
            fo = FO(f=f)
            fo_attr = fo.attribute(x)
            fo_saliency[k, :, :] = fo_attr
        # Save everything in the directory
        with open(os.path.join(save_dir, f"fo_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(fo_saliency, file)
        print("==============fo==============")
        print_results(fo_saliency, true_saliency)

    if "fp" in explainers:
        fp_saliency = torch.zeros(size=true_saliency.shape, device=device)
        for k in range(N_ex):  # We compute the attribution for each individual time series
            x = X[k, :, :]
            # Feature Permutation attribution
            fp = FP(f=f)
            fp_attr = fp.attribute(x)
            fp_saliency[k, :, :] = fp_attr
        with open(os.path.join(save_dir, f"fp_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(fp_saliency, file)
        print("==============fp==============")
        print_results(fp_saliency, true_saliency)

    if "ig" in explainers:
        ig_saliency = torch.zeros(size=true_saliency.shape, device=device)
        for k in range(N_ex):  # We compute the attribution for each individual time series
            x = X[k, :, :]
            # Integrated Gradient attribution
            ig = IG(f=f)
            ig_attr = ig.attribute(x)
            ig_saliency[k, :, :] = ig_attr
        with open(os.path.join(save_dir, f"ig_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(ig_saliency, file)
        print("==============ig==============")
        print_results(ig_saliency, true_saliency)

    if "shap" in explainers:
        shap_saliency = torch.zeros(size=true_saliency.shape, device=device)
        for k in range(N_ex):  # We compute the attribution for each individual time series
            x = X[k, :, :]
            # Sampling Shapley Value attribution
            shap = SVS(f=f)
            shap_attr = shap.attribute(x)
            shap_saliency[k, :, :] = shap_attr
        with open(os.path.join(save_dir, f"shap_saliency_{cv}.pkl"), "wb") as file:
            pkl.dump(shap_saliency, file)
        print("==============shap==============")
        print_results(shap_saliency, true_saliency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default=0, type=int)
    parser.add_argument("--print_result", default=True, type=bool)
    parser.add_argument("--save_dir", default="./results/rare_feature/", type=str)
    args = parser.parse_args()

    if args.print_result:
        from utils.tools import process_results
        process_results(5, explainers, args.save_dir)
    else:
        for i in range(5):
            run_experiment(cv=i, save_dir=args.save_dir)
