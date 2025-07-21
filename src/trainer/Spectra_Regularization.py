import torch
from .base import Base_Trainer
from tabulate import tabulate
from utils import (
    asinh_ratio_loss,
    log_10sigma,
    r_squared,
    mean_relative_error,
    mean_squared_logarithmic_error,
    mean_squared_loss,
)
from matplotlib import pyplot as plt
import wandb
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Spectra_Regularization_Trainer(Base_Trainer):
    def __init__(self, model, model_cfg, opt_cfg, dataset_cfg, tc_rng, total_train_steps=None):
        super().__init__(model, model_cfg, opt_cfg, dataset_cfg, tc_rng, total_train_steps)
        # set binary control labels
        self.random_loss = opt_cfg.random_loss
        # accumulate flux or asinh(flux) to the model
        self.apply_asinh_accumulate = model_cfg.apply_asinh_accumulate
        # do we calculate mse loss in normalized or inv_normalized space
        self.normalize_mse_loss = model_cfg.normalize_mse_loss
        self.loss_names = ["loss", "flux_per_ky_loss", "flux_loss"]

    def get_input_target(self, data):
        # input, flux_per_ky, flux
        input = data[0]
        gt_flux_per_ky = data[1]
        gt_flux = data[2]
        return input, gt_flux_per_ky, gt_flux

    def accumulate(self, data):
        data = self.move_to_device(data)
        if self.apply_asinh_accumulate:
            # apply asinh to the fluxes
            input, target_flux_per_ky, target_flux = self.get_input_target(data)
            data_trans = (input, torch.asinh(target_flux_per_ky), torch.asinh(target_flux))
            self.model.accumulate(data_trans)
        else:
            self.model.accumulate(data)

    def _model_forward(self, data):
        input, _, _ = self.get_input_target(data)
        normalized_pred_per_ky = self.model(input)
        return normalized_pred_per_ky

    def get_pred(self, data):
        data = self.move_to_device(data)
        # (B, nky, 4), inv normalization is aways conducted in the model
        pred_flux_per_ky = self._model_forward(data)
        # if accumulated asinh, then we need to transform it back to real space using sinh
        if self.apply_asinh_accumulate:
            pred_flux_per_ky = torch.sinh(pred_flux_per_ky)
        # get pred_flux
        pred_flux = torch.sum(pred_flux_per_ky, dim=1)

        return pred_flux_per_ky, pred_flux

    def _loss_fn(self, data):
        # get pred fluxes, always in real
        pred_flux_per_ky, pred_flux = self.get_pred(data)
        # get gt fluxes, aways in real
        _, gt_flux_per_ky, gt_flux = self.get_input_target(data)

        # transform fluxes accordigly, asinh is must, then normalize if needed
        gt_flux_per_ky_trans = torch.asinh(gt_flux_per_ky)
        gt_flux_trans = torch.asinh(gt_flux)
        pred_flux_per_ky_trans = torch.asinh(pred_flux_per_ky)
        pred_flux_trans = torch.asinh(pred_flux)
        if self.normalize_mse_loss:
            gt_flux_per_ky_trans = self.model._targetNormalizerPerWavenumber(gt_flux_per_ky_trans, accumulate=False)
            gt_flux_trans = self.model._targetNormalizer(gt_flux_trans, accumulate=False)
            pred_flux_per_ky_trans = self.model._targetNormalizerPerWavenumber(pred_flux_per_ky_trans, accumulate=False)
            pred_flux_trans = self.model._targetNormalizer(pred_flux_trans, accumulate=False)

        if self.random_loss:
            target_log10_max = gt_flux.new_tensor(list(self.dataset_cfg.target_log10_max))
            target_log10_min = gt_flux.new_tensor(list(self.dataset_cfg.target_log10_min))
            flux_loss = asinh_ratio_loss(
                gt_flux, pred_flux, target_log10_max, target_log10_min, tc_rng=self.tc_rng
            )
        else:
            flux_loss = mean_squared_loss(gt_flux_trans, pred_flux_trans)
        flux_per_ky_loss = mean_squared_loss(gt_flux_per_ky_trans, pred_flux_per_ky_trans)
        w_target = self.model_cfg.w_target
        w_spectra = self.model_cfg.w_spectra
        loss = w_target * flux_loss + w_spectra * flux_per_ky_loss
        return loss, flux_per_ky_loss, flux_loss

    def iter(self, data):
        data = self.move_to_device(data)
        loss, _, _ = self._loss_fn(data)
        loss.backward()
        # Gradient clipping
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_cfg.gnorm_clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.train_step += 1

    def get_test_loss(self, dataloader):
        """
        Calculate loss

        Args:
            dataloader: The data loader to get data.
        """
        losses = []

        for data in dataloader:
            data = self.move_to_device(data)

            # For calculate losses
            loss = self.get_loss(data)

            losses.append(loss[0])
        return sum(losses) / len(losses)

    def get_metrics(self, data):
        # move to device
        data = self.move_to_device(data)
        # get pred fluxes, always in real
        _, pred_flux = self.get_pred(data)
        # get gt fluxes, aways in real
        _, _, gt_flux = self.get_input_target(data)

        sigma = log_10sigma(gt_flux, pred_flux)
        R_sq = r_squared(gt_flux, pred_flux)
        MRE = 100 * mean_relative_error(gt_flux, pred_flux)  # in percentage
        MSLE = mean_squared_logarithmic_error(gt_flux, pred_flux)

        # Convert metrics to numpy arrays
        R_sq = R_sq.cpu().detach().numpy()
        MRE = MRE.cpu().detach().numpy()
        sigma = sigma.cpu().detach().numpy()
        MSLE = MSLE.cpu().detach().numpy()

        return R_sq, MRE, sigma, MSLE

    def print_metrics(self, data, prefix):
        """
        Print evaluation metrics in a tabular format.

        Args:
            data: PyTorch Geometric Data object containing input and target data.
            prefix: String prefix for the output table.
        """
        R_sq, MRE, sigma, MSLE = self.get_metrics(data)

        channel_len = len(R_sq)
        list_elements = []
        headers = ["Channel", "RSq", "MRE", "Sigma", "MSLE"]
        channel_names = [
            "OUT_G_elec",
            "OUT_Q_elec",
            "OUT_Q_ions",
            "OUT_P_ions",
        ]

        for cid in range(channel_len):
            row = [f"{prefix}, channel:{channel_names[cid]}", R_sq[cid], MRE[cid], sigma[cid], MSLE[cid]]
            list_elements.append(row)

        print(tabulate(list_elements, headers=headers, tablefmt="grid"))

    def eval_plot(self, data, prefix, board):
        """
        Plot evaluation figures for analysis.

        Args:
            data: PyTorch Geometric Data object containing input and target data.
            prefix: String prefix for the plot filename.
            board: Boolean flag for logging to Weights & Biases (wandb).
        """
        # get pred fluxes, always in real
        _, pred_flux = self.get_pred(data)
        # get gt fluxes, aways in real
        _, _, gt_flux = self.get_input_target(data)

        pred = pred_flux.cpu().detach().numpy()
        target = gt_flux.cpu().detach().numpy()

        self.plot_data(pred, target)

        # Step 11: Save the figure with the given prefix
        plt.savefig(prefix + ".png")

        # Step 12: Optionally log the figure to Weights & Biases if board is True
        if board:
            wandb.log({"example": wandb.Image(prefix + ".png")})

        plt.close()
        return

    def plot_data(self, pred, target):
        # Step 4: Create a 2x2 grid of subplots for visualization
        fig, axs = plt.subplots(2, 2)  # 2x2 layout for 4 channels
        fig.tight_layout()
        axs = axs.flatten()  # Flatten the grid to iterate over it easily

        # Step 5: Loop over each channel (dimension) in the target tensor
        for k in range(target.shape[-1]):
            curr_axs = axs[k]  # Select the current axis
            curr_pred = pred[:, k]  # Get predictions for the current channel
            curr_targ = target[:, k]  # Get targets for the current channel

            # Step 6: Define binning and ticks for the histogram
            if k < 3:
                b = np.logspace(-2, 2, num=50, base=10)  # Logarithmic bins for the first 3 channels
                ticks = (
                    [10**-1, 10**0, 10**1],
                    [r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"],
                )
            else:
                b = np.logspace(-1, 2.5, num=50, base=10)  # Different binning for the 4th channel
                ticks = ([10**0, 10**1, 10**2], [r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])

            # Step 7: Plot a 2D histogram for the current channel
            _, _, _, img = curr_axs.hist2d(
                curr_targ,
                curr_pred,
                bins=(b, b),
                cmin=1,  # Minimum count for color binning
                cmap="magma_r",  # Colormap ('magma_r' is the reversed magma colormap)
            )

            # Step 8: Add a reference line, set log scales, and apply ticks
            curr_axs.plot([0, 1], [0, 1], transform=curr_axs.transAxes, color="blue", linewidth=0.75)
            curr_axs.set_xscale("log")
            curr_axs.set_yscale("log")
            curr_axs.set_xticks(ticks[0])

            # Step 9: Conditionally add labels to axes
            if k in [0, 2]:
                curr_axs.set_ylabel("TGLF-NN")  # Y-label for specific channels
                if k == 2:
                    curr_axs.set_xlabel("TGLF")  # X-label for the 3rd channel
            elif k == 3:
                curr_axs.set_xlabel("TGLF")  # X-label for the 4th channel

            # Step 10: Add a colorbar to the current axis
            fig.colorbar(img, ax=curr_axs)
        return


    def board_loss_over_loopers(self, dataloader, prefix, board, data_size):
        """
        Log the loss to wandb.

        Args:
            dataloader: The data loader to get data.
            prefix: Prefix for the loss output.
            board: Flag to determine if the loss should be logged to wandb.
            data_size: The size of BAL dataset.
        """
        losses = []
        flux_per_ky_losses = []
        flux_losses = []

        R_sqs = []
        MREs = []
        sigmas = []
        MSLEs = []

        for data in dataloader:
            data = self.move_to_device(data)

            # For calculate losses
            loss = self.get_loss(data)

            losses.append(loss[0])
            flux_per_ky_losses.append(loss[1])
            flux_losses.append(loss[2])

            # For calculate other metrics
            R_sq, MRE, sigma, MSLE = self.get_metrics(data)
            R_sqs.append(R_sq)
            MREs.append(MRE)
            sigmas.append(sigma)
            MSLEs.append(MSLE)

        loss_mean = sum(losses) / len(losses)
        print(f"Data size: {data_size}, {prefix}_loss: {loss_mean}")
        flux_per_ky_loss_mean = sum(flux_per_ky_losses) / len(flux_per_ky_losses)
        flux_loss_mean = sum(flux_losses) / len(flux_losses)
        print(f"{prefix}_flux_loss: {flux_loss_mean}, {prefix}_flux_per_ky_loss: {flux_per_ky_loss_mean}")

        R_sq_mean = np.mean(np.array(R_sqs), axis=0)
        MRE_mean = np.mean(np.array(MREs), axis=0)
        sigma_mean = np.mean(np.array(sigmas), axis=0)
        MSLE_mean = np.mean(np.array(MSLEs), axis=0)

        channel_len = len(R_sq_mean)
        list_elements = []
        headers = ["Channel", "RSq", "MRE", "Sigma", "MSLE"]
        channel_names = self.dataset_cfg.target_keys

        for cid in range(channel_len):
            row = [
                f"{prefix}, channel:{channel_names[cid]}",
                R_sq_mean[cid],
                MRE_mean[cid],
                sigma_mean[cid],
                MSLE_mean[cid],
            ]
            list_elements.append(row)
        print(tabulate(list_elements, headers=headers, tablefmt="grid"))

        if board:
            log_map = {"data size": data_size, f"{prefix}_loss": loss_mean}
            log_map[f"{prefix}_flux_per_ky_loss"] = flux_per_ky_loss_mean
            log_map[f"{prefix}_flux_loss"] = flux_loss_mean

            channel_names = self.dataset_cfg.target_keys
            for cid in range(channel_len):
                log_map[f"{channel_names[cid]}_RSq"] = R_sq_mean[cid]
                log_map[f"{channel_names[cid]}_MRE"] = MRE_mean[cid]
                log_map[f"{channel_names[cid]}_Sigma"] = sigma_mean[cid]
                log_map[f"{channel_names[cid]}_MSLE"] = MSLE_mean[cid]
            log_map["mean_RSq"] = np.mean(R_sq_mean)
            log_map["mean_MRE"] = np.mean(MRE_mean)
            log_map["mean_Sigma"] = np.mean(sigma_mean)
            log_map["mean_MSLE"] = np.mean(MSLE_mean)

            wandb.log(log_map)
