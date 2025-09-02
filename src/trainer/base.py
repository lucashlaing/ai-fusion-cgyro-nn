import numpy as np
import torch
from utils import WarmupCosineDecayScheduler, mean_squared_loss
import wandb
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Base_Trainer:
    def __init__(self, model, model_cfg, opt_cfg, dataset_cfg, tc_rng, total_train_steps=None):
        """
        Initialize the Base_Trainer.

        Args:
            model: The model to be trained.
            model_cfg: Model configuration.
            opt_cfg: Optimizer configuration.
            dataset_cfg: Dataset configuration.
            tc_rng: Random number generator for TensorCore operations.
            total_train_steps: total training steps.
        """
        self.model = model
        self.model_cfg = model_cfg
        self.opt_cfg = opt_cfg
        self.dataset_cfg = dataset_cfg
        self.tc_rng = tc_rng
        # if torch.cuda.device_count() > 1:
        #    print("Using", torch.cuda.device_count(), "GPUs!")
        #    self.model = torch.nn.DataParallel(self.model)
        #    print("Model wrapped by DataParallel", flush=True)

        self.device = device
        self.model.to(device)
        print("Model moved to {}".format(device), flush=True)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=opt_cfg.peak_lr,
            weight_decay=opt_cfg.weight_decay,
        )

        if total_train_steps is None:
            total_train_steps = opt_cfg.decay_steps

        self.lr_scheduler = WarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup=total_train_steps//10,
            max_iters=total_train_steps,
        )

        print(self.model, flush=True)
        self.train_step = 0
        self.loss_names = ["loss"] # NOTE override if needed

    # =====================================================================
    # Methods that need to be implemented in child classes
    # =====================================================================

    def _model_forward(self, data):
        """
        A wrapper to call model.forward.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Tuple of average mean squared error and predicted target.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("_model_forward need to be implemented in child class.")

    def get_input_target(self, data):
        """
        Extract the label data from the input data.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Tuple of label data and label mask.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("get_input_target need to be implemented in child class.")

    def accumulate(self, data):
        """
        Calculate the relative error for each channel and output both mean and std.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Tuple of mean and std of the error.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("accumulate need to be implemented in child class.")

    def _loss_fn(self, data):
        """
        Calculate the loss function.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Loss value in RMSE.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("_loss_fn need to be implemented in child class.")

    def get_metrics(self, data):
        """
        Calculate the relative error for each channel and output both mean and std.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Tuple of mean and std of the error.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("get_metrics need to be implemented in child class.")

    def print_metrics(self, data, prefix):
        """
        Print metrics.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the metrics output.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("print_metrics need to be implemented in child class.")

    def eval_plot(self, data, prefix):
        """
        Plot evaluation figures that are helpful for analysis.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the plot output.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("eval_plot need to be implemented in child class.")

    # =====================================================================
    # Methods that do not need modifications in child classes
    # =====================================================================

    def get_pred(self, data):
        """
        Get the prediction.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Predictions as torch tensor
        """
        data = self.move_to_device(data)
        predict = self._model_forward(data)
        return predict

    def move_to_device(self, data):
        """
        Move data to the specified device.

        Args:
            data: Data to move to device (list, tuple or torch.Tensor).

        Returns:
            Data moved to device.
        """
        if isinstance(data, (list, tuple)):
            return [self.move_to_device(d) for d in data]
        else:
            return data.to(self.device)

    def iter(self, data):
        """
        Train the model for one iteration.

        Args:
            data: PyTorch Geometric Data object containing input data.
        """
        data = self.move_to_device(data)
        loss = self._loss_fn(data)

        loss.backward()

        # Gradient clipping
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.opt_cfg.gnorm_clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.train_step += 1

    def save(self, save_dir):
        """
        Save the model parameters.

        Args:
            save_dir: Directory to save the model parameters.
        """
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(model.state_dict(), f"{save_dir}/{self.train_step}_params.pth")
        print(f"Saved to {save_dir}, step {self.train_step}")

    def restore(self, save_dir, step, restore_opt_state=True):
        """
        Restore the model parameters.

        Args:
            save_dir: Directory to restore the model parameters from.
            step: Training step to restore.
            restore_opt_state: Flag to restore optimizer state (default: True).
        """
        params_path = f"{save_dir}/{step}_params.pth"
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(torch.load(params_path, map_location=device))
        print(f"Restored params from {save_dir}, step {step}")

    def get_loss(self, data):
        """
        Get the loss value.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            Loss value as np.ndarray.
        """
        data = self.move_to_device(data)
        loss = self._loss_fn(data)
        return loss

    def board_loss(self, data, prefix, board):
        """
        Log the loss to wandb.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the loss output.
            board: Flag to determine if the loss should be logged to wandb.
        """
        loss = self.get_loss(data)
        print(f"train step: {self.train_step}, {prefix}_loss: {loss}")
        if board:
            log_map = {"step": self.train_step}
            if isinstance(loss, tuple):
                for i in range(len(loss)):
                    log_map[f"{prefix}_{self.loss_names[i]}"] = loss[i]
            else:
                log_map[f"{prefix}_loss"] = loss
            wandb.log(log_map)

    def board_loss_over_data_size(self, data, prefix, board, data_size):
        """
        Log the loss to wandb.

        Args:
            data: PyTorch Geometric Data object containing input data.
            prefix: Prefix for the loss output.
            board: Flag to determine if the loss should be logged to wandb.
            data_size: The size of BAL dataset.
        """
        data = self.move_to_device(data)
        loss = self.get_loss(data)
        print(f"Data size: {data_size}, {prefix}_loss: {loss}")
        if board:
            # data size and trainer losses
            log_map = {"data size": data_size}
            if isinstance(loss, tuple):
                for i in range(len(loss)):
                    log_map[f"{prefix}_{self.loss_names[i]}"] = loss[i]
            else:
                log_map[f"{prefix}_loss"] = loss

            # Additional losses
            R_sq, MRE, sigma, MSLE = self.get_metrics(data)
            channel_len = len(R_sq)
            channel_names = self.dataset_cfg.target_keys

            for cid in range(channel_len):
                log_map[f"{channel_names[cid]}_RSq"] = R_sq[cid]
                log_map[f"{channel_names[cid]}_MRE"] = MRE[cid]
                log_map[f"{channel_names[cid]}_Sigma"] = sigma[cid]
                log_map[f"{channel_names[cid]}_MSLE"] = MSLE[cid]
            log_map["mean_RSq"] = np.mean(R_sq)
            log_map["mean_MRE"] = np.mean(MRE)
            log_map["mean_Sigma"] = np.mean(sigma)
            log_map["mean_MSLE"] = np.mean(MSLE)

            wandb.log(log_map)

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
            losses.append(loss)
        return sum(losses) / len(losses)

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
        R_sqs = []
        MREs = []
        sigmas = []
        MSLEs = []

        for data in dataloader:
            data = self.move_to_device(data)

            # For calculate losses
            loss = self.get_loss(data)
            losses.append(loss)

            # For calculate other metrics
            R_sq, MRE, sigma, MSLE = self.get_metrics(data)
            R_sqs.append(R_sq)
            MREs.append(MRE)
            sigmas.append(sigma)
            MSLEs.append(MSLE)

        loss_mean = sum(losses) / len(losses)
        print(f"Data size: {data_size}, {prefix}_loss: {loss_mean}")

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
