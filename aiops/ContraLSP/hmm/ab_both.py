import multiprocessing as mp
import os
from pytorch_lightning.callbacks import EarlyStopping

import sys

from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List
from utils.tools import print_results
from tint.datasets import HMM
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)
from tint.models import MLP, RNN

from abstudy.gatemasknn_no_both import *
from abstudy.gate_mask_noboth import GateMask
from hmm.classifier import StateClassifierNet


def main(
    explainers: List[str],
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    is_train: bool = True,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    output_file: str = "mymask_ablation_both.csv",
):
    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    print(accelerator)
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    hmm = HMM(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = StateClassifierNet(
        feature_size=3,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    classifier.load_state_dict(th.load("./model/classifier_{}_{}".format(fold, seed)))

    # Get data for explainers
    with lock:
        x_train = hmm.preprocess(split="train")["x"].to(device)
        x_test = hmm.preprocess(split="test")["x"].to(device)
        y_test = hmm.preprocess(split="test")["y"].to(device)
        true_saliency = hmm.true_saliency(split="test").to(device)

    print("==============The sum of true_saliency is", true_saliency.sum(), "==============\n" + 70 * "=")
    # if not os.path.exists("./results/"):
    #     os.makedirs("./results/")
    # np.save('./results/true.npy', true_saliency.detach().numpy())

    # Switch to eval
    classifier.eval()
    classifier.zero_grad()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "gate_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            log_every_n_steps=2,
            deterministic=deterministic,
            logger=TensorBoardLogger(
                save_dir=".",
                version=random.getrandbits(128),
            ),
        )
        mask = GateMaskNet(
            forward_func=classifier,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            optim="adam",
            lr=0.01,
        )
        explainer = GateMask(classifier)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=x_test.shape[0],
            sigma=0.5,
        )
        attr["gate_mask"] = _attr.to(device)
        print_results(attr["gate_mask"], true_saliency)

    with open(output_file, "a") as fp, lock:
        for k, v in attr.items():
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "gate_mask",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train thr rnn classifier.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=3,  #2 is best
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=1,
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results.csv",
        help="Where to save the results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # for i in [1,2,3,4]:
    #     main(
    #         explainers=["gate_mask"],
    #         device=args.device,
    #         fold=i,
    #         seed=args.seed,
    #         deterministic=args.deterministic,
    #         is_train=args.train,
    #         lambda_1=args.lambda_1,
    #         lambda_2=args.lambda_2,
    #     )

    #
    from utils.tools import process_results_by_file
    process_results_by_file(5, args.explainers, path="mymask_ablation_both.csv")
