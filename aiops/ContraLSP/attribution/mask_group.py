import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
import pandas as pd
from attribution.mask import Mask
from attribution.perturbation import Perturbation
from tqdm import tqdm


class MaskGroup:

    def __init__(
        self,
        perturbation: Perturbation,
        device,
        random_seed: int = 987,
        deletion_mode: bool = False,
        verbose: bool = True,
    ):
        self.perturbation = perturbation
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose
        self.deletion_mode = deletion_mode
        self.mask_list = None
        self.area_list = None
        self.f = None
        self.X = None
        self.n_epoch = None
        self.T = None
        self.N_features = None
        self.Y_target = None
        self.masks_tensor = None
        self.mask_tensor = None
        self.hist = None

    def fit(
        self,
        X,
        f,
        area_list,
        loss_function,
        n_epoch: int = 1000,
        initial_mask_coeff: float = 0.5,
        size_reg_factor_init: float = 0.1,
        size_reg_factor_dilation: float = 100,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        time_reg_factor: float = 0,
    ):

        # Ensure that the area list is sorted
        area_list.sort()
        self.area_list = area_list
        N_area = len(area_list)
        # Create a list of masks
        mask_list = []
        # Initialize the random seed and the attributes
        t_fit = time.time()
        torch.manual_seed(self.random_seed)
        reg_factor = size_reg_factor_init
        error_factor = 1 - 2 * self.deletion_mode  # In deletion mode, the error has to be maximized
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epoch)
        self.f = f
        self.X = X
        self.n_epoch = n_epoch
        self.T, self.N_features = X.shape
        self.Y_target = f(X)
        # The initial mask tensor has all coefficients set to initial_mask_coeff
        self.masks_tensor = initial_mask_coeff * torch.ones(size=(N_area, self.T, self.N_features), device=self.device)
        # The target is the same for each mask so we simply repeat it along the first axis
        Y_target_group = self.Y_target.clone().detach().unsqueeze(0).repeat(N_area, 1, 1)
        # Create a copy of the extremal tensor that is going to be trained, the optimizer and the history
        masks_tensor_new = self.masks_tensor.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([masks_tensor_new], lr=learning_rate, momentum=momentum)
        hist = torch.zeros(3, 0)
        # Initializing the reference vector used in the regulator
        reg_ref = torch.ones((N_area, self.T * self.N_features), dtype=torch.float32, device=self.device)
        for i, area in enumerate(self.area_list):
            reg_ref[i, : int((1 - area) * self.T * self.N_features)] = 0.0
        # Run the optimization
        for k in range(n_epoch):
            # Measure the loop starting time
            t_loop = time.time()
            # Generate perturbed input and outputs
            if self.deletion_mode:
                X_pert = self.perturbation.apply_extremal(X=X, extremal_tensor=1 - masks_tensor_new)
            else:
                X_pert = self.perturbation.apply_extremal(X=X, extremal_tensor=masks_tensor_new)
            Y_pert = torch.stack([f(x_pert) for x_pert in torch.unbind(X_pert, dim=0)], dim=0)

            # Evaluate the overall loss (error [L_e] + size regulation [L_a] + time variation regulation [L_c])
            error = loss_function(Y_pert, Y_target_group)
            masks_tensor_sorted = masks_tensor_new.reshape(N_area, self.T * self.N_features).sort(dim=1)[0]
            size_reg = ((reg_ref - masks_tensor_sorted) ** 2).mean()
            time_reg = (torch.abs(masks_tensor_new[:, 1 : self.T - 1, :] - masks_tensor_new[:, : self.T - 2, :])).mean()
            loss = error_factor * error + reg_factor * size_reg + time_reg_factor * time_reg
            # Apply the gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Ensures that the constraint is fulfilled
            masks_tensor_new.data = masks_tensor_new.data.clamp(0, 1)
            # Save the error and the regulator
            metrics = torch.tensor([error, size_reg, time_reg]).cpu().unsqueeze(1)
            hist = torch.cat((hist, metrics), dim=1)
            # Increase the regulator coefficient
            reg_factor *= reg_multiplicator
            # Measure the loop ending time
            t_loop = time.time() - t_loop
            if self.verbose and k%200==0:
                print(
                    f"Epoch {k + 1}/{n_epoch}: error = {error.data:.3g} ; "
                    f"size regulator = {size_reg.data:.3g} ; time regulator = {time_reg.data:.3g} ;"
                    f" time elapsed = {t_loop:.3g} s"
                )
                print((masks_tensor_new.sum()).data/N_area)

        # Update the mask and history tensor, print the final message
        self.masks_tensor = masks_tensor_new.clone().detach().requires_grad_(False)
        self.hist = hist
        t_fit = time.time() - t_fit
        print(
            f"The optimization finished: error = {error.data:.3g} ; size regulator = {size_reg.data:.3g} ;"
            f" time regulator = {time_reg.data:.3g} ; time elapsed = {t_fit:.3g} s"
        )

        # Store the individual mask coefficients in distinct mask objects
        for index, mask_tensor in enumerate(self.masks_tensor.unbind(dim=0)):
            mask = Mask(
                perturbation=self.perturbation, device=self.device, verbose=False, deletion_mode=self.deletion_mode
            )
            mask.mask_tensor = mask_tensor
            mask.hist = self.hist
            mask.f = self.f
            mask.X = self.X
            mask.n_epoch = self.n_epoch
            mask.T, mask.N_features = self.T, self.N_features
            mask.Y_target = self.Y_target
            mask.loss_function = loss_function
            mask_list.append(mask)
        self.mask_list = mask_list

    def fit_multiple(
        self,
        X,
        f,
        area_list,
        loss_function_multiple,
        use_last_timestep_only: bool = False,
        n_epoch: int = 200,
        initial_mask_coeff: float = 0.5,
        size_reg_factor_init: float = 0.1,
        size_reg_factor_dilation: float = 100,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        time_reg_factor: float = 0.01,
    ):
        # Ensure that the area list is sorted
        area_list.sort()
        self.area_list = area_list
        N_area = len(area_list)
        # Create a list of masks
        mask_list = []
        # Initialize the random seed and the attributes
        t_fit = time.time()
        torch.manual_seed(self.random_seed)
        reg_factor = size_reg_factor_init
        error_factor = 1 - 2 * self.deletion_mode  # In deletion mode, the error has to be maximized
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epoch)
        self.f = f
        self.X = X
        self.n_epoch = n_epoch
        num_samples, self.T, self.N_features = X.shape
        self.Y_target = f(X) # num_samples, num_time, num_state=2
        if use_last_timestep_only:
            self.Y_target = self.Y_target[:, -1:, :]
        # The initial mask tensor has all coefficients set to initial_mask_coeff
        self.masks_tensor = initial_mask_coeff * torch.ones(size=(N_area, num_samples, self.T, self.N_features), device=self.device)
        # The target is the same for each mask so we simply repeat it along the first axis
        Y_target_group = self.Y_target.clone().detach().unsqueeze(0).repeat(N_area, 1, 1, 1)
        # Create a copy of the extremal tensor that is going to be trained, the optimizer and the history
        masks_tensor_new = self.masks_tensor.clone().detach().requires_grad_(True)
        # optimizer = optim.SGD([masks_tensor_new], lr=learning_rate, momentum=momentum)
        optimizer = optim.Adam([masks_tensor_new], lr=learning_rate)

        metrics = []
        # Initializing the reference vector used in the regulator
        reg_ref = torch.ones((N_area, num_samples, self.T * self.N_features), dtype=torch.float32, device=self.device)
        for i, area in enumerate(self.area_list):
            reg_ref[i, :, :int((1 - area) * self.T * self.N_features)] = 0.0

        # Run the optimization
        for k in tqdm(range(n_epoch)):
            # Measure the loop starting time
            t_loop = time.time()
            # Generate perturbed input and outputs
            if self.deletion_mode:
                X_pert = self.perturbation.apply_extremal_multiple(X=X, extremal_tensor=1 - masks_tensor_new)
            else:
                X_pert = self.perturbation.apply_extremal_multiple(X=X, extremal_tensor=masks_tensor_new)

            # x_pert (num_samples, T, num_feature)
            # f(x_pert) = (num_sample, T, num_state)
            # y_pert = (num_area, num_sample, T, num_state)

            # x_pert (T, num_feature)
            # f(x_pert) = (T, num_state)
            # Y_pert = (n_area, T, num_state)
            X_pert_flatten = X_pert.reshape(N_area * num_samples, self.T, self.N_features)
            Y_pert_flatten = f(X_pert_flatten) # (N_area * num_samples, T, num_state)
            if use_last_timestep_only:
                Y_pert = Y_pert_flatten.reshape(N_area, num_samples, 1, -1)
            else:
                Y_pert = Y_pert_flatten.reshape(N_area, num_samples, self.T, -1)

            # Evaluate the overall loss (error [L_e] + size regulation [L_a] + time variation regulation [L_c])
            error = loss_function_multiple(Y_pert, Y_target_group) # (num_sample)
            masks_tensor_sorted = masks_tensor_new.reshape(N_area, num_samples, self.T * self.N_features).sort(dim=2)[0]
            size_reg = ((reg_ref - masks_tensor_sorted) ** 2).mean(dim=[0, 2])
            masks_tensor_diff = masks_tensor_new[:, :, 1: self.T - 1, :] - masks_tensor_new[:, :, :self.T - 2, :]
            time_reg = (torch.abs(masks_tensor_diff)).mean(dim=[0, 2, 3])
            loss = error_factor * error + reg_factor * size_reg + time_reg_factor * time_reg
            # Apply the gradient step
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            # Ensures that the constraint is fulfilled
            masks_tensor_new.data = masks_tensor_new.data.clamp(0, 1)
            # Save the error and the regulator
            metric = torch.stack([error, size_reg, time_reg], dim=1).detach().cpu().numpy()
            metrics.append(metric)
            # Increase the regulator coefficient
            reg_factor *= reg_multiplicator
            # Measure the loop ending time
            t_loop = time.time() - t_loop
            if self.verbose and k%20==0:
                print(
                    f"Epoch {k + 1}/{n_epoch}: error = {error.mean().data:.3g} ; "
                    f"size regulator = {size_reg.mean().data:.3g} ; time regulator = {time_reg.mean().data:.3g} ;"
                    f" time elapsed = {t_loop:.3g} s"
                )
                print((masks_tensor_new.sum()/(N_area*num_samples)).data, X.shape[2]*X.shape[1])

        # Update the mask and history tensor, print the final message
        self.masks_tensor = masks_tensor_new.clone().detach().requires_grad_(False) # (N_area, num_samples, T, nfeat)
        self.hist = torch.from_numpy(np.stack(metrics, axis=2))
        t_fit = time.time() - t_fit

        print(
            f"The optimization finished: error = {error.mean().data:.3g} ; size regulator = {size_reg.mean().data:.3g} ;"
            f" time regulator = {time_reg.mean().data:.3g} ; time elapsed = {t_fit:.3g} s"
        )

        # Store the individual mask coefficients in distinct mask objects
        for index, mask_tensor in enumerate(self.masks_tensor.unbind(dim=0)):
            mask = Mask(
                perturbation=self.perturbation, device=self.device, verbose=False, deletion_mode=self.deletion_mode
            )
            mask.mask_tensor = mask_tensor
            mask.hist = self.hist
            mask.f = self.f
            mask.X = self.X
            mask.n_epoch = self.n_epoch
            mask.T, mask.N_features = self.T, self.N_features
            mask.Y_target = self.Y_target
            mask.loss_function = loss_function_multiple
            mask_list.append(mask)
        self.mask_list = mask_list

    def get_best_mask(self):
        """This method returns the mask with lowest error."""
        error_list = [mask.get_error() for mask in self.mask_list]
        best_index = error_list.index(min(error_list))
        print(
            f"The mask of area {self.area_list[best_index]:.2g} is"
            f" the best with error = {error_list[best_index]:.3g}."
        )
        return self.mask_list[best_index]

    def get_extremal_mask(self, threshold):
        """This method returns the extremal mask for the acceptable error threshold (called epsilon in the paper)."""
        error_list = [mask.get_error() for mask in self.mask_list]
        # If the minimal error is above the threshold, the best we can do is select the mask with lowest error
        if min(error_list) > threshold:
            return self.get_best_mask()
        else:
            for id_mask, error in enumerate(error_list):
                if error < threshold:
                    print(
                        f"The mask of area {self.area_list[id_mask]:.2g} is"
                        f" extremal with error = {error_list[id_mask]:.3g}."
                    )
                    return self.mask_list[id_mask]

    def get_extremal_mask_multiple(self, thresholds):
        """This method returns the extremal mask for the acceptable error threshold (called epsilon in the paper)."""
        error_list = torch.stack([mask.get_error_multiple() for mask in self.mask_list], dim=1)
        mask_stacked = torch.stack([mask.mask_tensor for mask in self.mask_list])
        num_area, num_samples, num_times, num_features = mask_stacked.shape
        # If the minimal error is above the threshold, the best we can do is select the mask with lowest error
        thres_mask = torch.min(error_list, dim=1)[0] > thresholds
        best_mask = torch.argmin(error_list, dim=1) #(num_sample)
        error_mask = (error_list < thresholds.view(-1, 1)) * torch.arange(-len(self.mask_list), 0).view(1, -1).to(self.device)
        first_mask = torch.argmin(error_mask, dim=1)
        indexes = torch.where(thres_mask, best_mask, first_mask) # (num_sample)
        selected_masks = torch.gather(mask_stacked, 0, indexes.view(1, num_samples, 1, 1).expand(1, num_samples, num_times, num_features))
        self.mask_tensor = selected_masks.reshape(num_samples, num_times, num_features)
        return selected_masks.reshape(num_samples, num_times, num_features) #(num_samples, num_times, num_features)

    def plot_errors(self):
        """This method plots the error as a function of the mask size."""
        sns.set()
        error_list = [mask.get_error() for mask in self.mask_list]
        plt.plot(self.area_list, error_list)
        plt.title("Errors for the various masks")
        plt.xlabel("Mask area")
        plt.ylabel("Error")
        plt.show()

    def get_smooth_mask(self, sigma=1):
        """This method smooths the mask tensor by applying a temporal Gaussian filter for each feature.

        Args:
            sigma: Width of the Gaussian filter.

        Returns:
            torch.Tensor: The smoothed mask.
        """
        # Define the Gaussian smoothing kernel
        T_axis = torch.arange(1, self.T + 1, dtype=int, device=self.device)
        T1_tensor = T_axis.unsqueeze(1).unsqueeze(2)
        T2_tensor = T_axis.unsqueeze(0).unsqueeze(2)
        kernel_tensor = torch.exp(-1.0 * (T1_tensor - T2_tensor) ** 2 / (2.0 * sigma ** 2))
        kernel_tensor = torch.divide(kernel_tensor, torch.sum(kernel_tensor, 0))
        kernel_tensor = kernel_tensor.repeat(1, 1, self.N_features)
        # Smooth the mask tensor by applying the kernel
        mask_tensor_smooth = torch.einsum("sti,si->ti", kernel_tensor, self.mask_tensor)
        return mask_tensor_smooth

    def extract_submask(self, mask_tensor, ids_time, ids_feature):
        """This method extracts a submask specified with specified indices.

        Args:
            mask_tensor: The tensor from which data should be extracted.
            ids_time: List of the times that should be extracted.
            ids_feature: List of the features that should be extracted.

        Returns:
            torch.Tensor: Submask extracted based on the indices.
        """
        # If no identifiers have been specified, we use the whole data
        if ids_time is None:
            ids_time = [k for k in range(self.T)]
        if ids_feature is None:
            ids_feature = [k for k in range(self.N_features)]
        # Extract the relevant data in the mask
        submask_tensor = mask_tensor.clone().detach().requires_grad_(False).cpu()
        submask_tensor = submask_tensor[ids_time, :]
        submask_tensor = submask_tensor[:, ids_feature]
        return submask_tensor

    def plot_mask(self, idx=0, ids_time=None, ids_feature=None, smooth: bool = False, sigma: float = 1.0):
        """This method plots (part of) the mask.

        Args:
            ids_time: List of the times that should appear on the plot.
            ids_feature: List of the features that should appear on the plot.
            smooth: True if the mask should be smoothed before plotting.
            sigma: Width of the smoothing Gaussian kernel.

        Returns:
            None
        """
        sns.set()
        # Smooth the mask if required
        if smooth:
            mask_tensor = self.get_smooth_mask(sigma)
        else:
            mask_tensor = self.mask_tensor
        # Extract submask from ids
        submask_tensor_np = self.extract_submask(mask_tensor[idx], ids_time, ids_feature).numpy()
        df = pd.DataFrame(data=np.transpose(submask_tensor_np), index=ids_feature, columns=ids_time)
        # Generate heatmap plot
        color_map = sns.diverging_palette(10, 133, as_cmap=True)
        heat_map = sns.heatmap(data=df, cmap=color_map, cbar_kws={"label": "Mask"}, vmin=0, vmax=1)
        plt.xlabel("Time")
        plt.ylabel("Feature Number")
        plt.title("Mask coefficients over time")
        plt.show()