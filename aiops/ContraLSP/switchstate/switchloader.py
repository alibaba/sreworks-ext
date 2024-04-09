import numpy as np
import os
import pickle as pkl
import torch as th

from tint.datasets.dataset import DataModule


file_dir = os.path.dirname(__file__)


def logit(x):
    # The real logit should be '1.0 / (1 + np.exp(-x))', but we use -2 as in
    # https://github.com/JonathanCrabbe/Dynamask for fair comparison.
    return 1.0 / (1 + np.exp(-2 * x))


class Switch(DataModule):

    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "switchstate",
            "data",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        n_folds: int = None,
        fold: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            n_folds=n_folds,
            fold=fold,
            num_workers=num_workers,
            seed=seed,
        )

    def download(
        self,
        train_size: int = 800,
        test_size: int = 200,
        signal_length: int = 200,
        split: str = "train",
    ):
        file = os.path.join(self.data_dir, f"{split}_")

        if split == "train":
            with open('./simulated_data_l2x/state_dataset_x_train.pkl',"rb") as f:
                features = pkl.load(f)
            with open('./simulated_data_l2x/state_dataset_y_train.pkl',"rb") as f:
                labels = pkl.load(f)
            with open('./simulated_data_l2x/state_dataset_importance_train.pkl',"rb") as f:
                importance_score = pkl.load(f)
            with open('./simulated_data_l2x/state_dataset_states_train.pkl',"rb") as f:
                all_states = pkl.load(f)
        elif split == "test":
            with open('./simulated_data_l2x/state_dataset_x_test.pkl',"rb") as f:
                features = pkl.load(f)
            with open('./simulated_data_l2x/state_dataset_y_test.pkl',"rb") as f:
                labels = pkl.load(f)
            with open('./simulated_data_l2x/state_dataset_importance_test.pkl',"rb") as f:
                importance_score = pkl.load(f)
            with open('./simulated_data_l2x/state_dataset_states_test.pkl',"rb") as f:
                all_states = pkl.load(f)
        else:
            raise NotImplementedError

        with open(
            os.path.join(self.data_dir, file + "features.npz"), "wb"
        ) as fp:
            pkl.dump(obj=features, file=fp)
        with open(
            os.path.join(self.data_dir, file + "labels.npz"), "wb"
        ) as fp:
            pkl.dump(obj=labels, file=fp)
        with open(
            os.path.join(self.data_dir, file + "importance.npz"), "wb"
        ) as fp:
            pkl.dump(obj=importance_score, file=fp)
        with open(
            os.path.join(self.data_dir, file + "states.npz"), "wb"
        ) as fp:
            pkl.dump(obj=all_states, file=fp)

    def preprocess(self, split: str = "train") -> dict:
        file = os.path.join(self.data_dir, f"{split}_")

        with open(
            os.path.join(self.data_dir, file + "features.npz"), "rb"
        ) as fp:
            features = pkl.load(file=fp)
        with open(
            os.path.join(self.data_dir, file + "labels.npz"), "rb"
        ) as fp:
            labels = pkl.load(file=fp)

        return {
            "x": th.from_numpy(features).float(),
            "y": th.from_numpy(labels).long(),
        }

    def prepare_data(self):
        """"""
        if not os.path.exists(
            os.path.join(self.data_dir, "train_features.npz")
        ):
            self.download(split="train")
        if not os.path.exists(
            os.path.join(self.data_dir, "test_features.npz")
        ):
            self.download(split="test")

    def true_saliency(self, split: str = "train") -> th.Tensor:
        file = os.path.join(self.data_dir, f"{split}_")

        with open(
            os.path.join(self.data_dir, file + "features.npz"), "rb"
        ) as fp:
            features = pkl.load(file=fp)
        #
        # # Load the true states that define the truly salient features
        # # and define A as in Section 3.2:
        # with open(
        #     os.path.join(self.data_dir, file + "importance.npz"), "rb"
        # ) as fp:
        #     true_saliency = pkl.load(file=fp)
        #     true_saliency = th.from_numpy(true_saliency)

        # Load the true states that define the truly salient features
        # and define A as in Section 3.2:
        with open(
            os.path.join(self.data_dir, file + "states.npz"), "rb"
        ) as fp:
            true_states = pkl.load(file=fp)

            true_saliency = th.zeros(features.shape)
            for exp_id, time_slice in enumerate(true_states):
                for t_id, feature_id in enumerate(time_slice):
                    true_saliency[exp_id, t_id, feature_id] = 1
            true_saliency = true_saliency.long()
        return true_saliency

