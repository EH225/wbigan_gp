"""
This module contains training routines for a Wasserstein Bi-GAN.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

import torch
import psutil
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import wraps
from tqdm.auto import tqdm
from typing import Tuple, Callable, Dict, List
import logging, gc
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import get_device, get_amp_dtype, generate_loss_plots, save_images
from torch_models.generator import Generator
from torch_models.encoder import Encoder
from torch_models.discriminator import Discriminator
from torch_models.shared_components import ClassEmbedding
from dataset_utils import get_class_labels


def infinite_loader(dataloader: DataLoader):
    """
    Infinitely yields batches of data from the input dataloader (dl) without caching batches.
    """
    while True:
        for batch in dataloader:
            yield batch


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def compute_with_amp(func):
    """
    Decorator that wraps func in automatic mixed precision (AMP) evaluation context managers from
    pytorch if self.amp_dtype is not None. This does not alter the inputs or outputs of func.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.amp_dtype is not None:
            with torch.autocast(device_type=self.device, dtype=self.amp_dtype):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """
    Sets the requires_grad_ property for all parameters in a given module.

    :param module: The input module whose parameters will be affected.
    :param requires_grad: True or False indicating if gradient tracking should be enabled or disabled.
    :returns: None, the internal state of module is edited inplace.
    """
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def compute_mmd(x, y, sigmas=(1, 2, 4, 8, 16)):
    """
    x: (B, D) encoder outputs
    y: (B, D) prior samples
    """

    xx = torch.cdist(x, x).pow(2)
    yy = torch.cdist(y, y).pow(2)
    xy = torch.cdist(x, y).pow(2)

    mmd = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2)

        Kxx = torch.exp(-gamma * xx)
        Kyy = torch.exp(-gamma * yy)
        Kxy = torch.exp(-gamma * xy)

        mmd += Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

    return mmd


class Trainer:

    def __init__(self, config: Dict, dataloaders: Dict, **kwargs):
        """
        Creates a trainer object for training a Wasserstein Bi-GAN model. This class wrapper has methods
        for loading models and the training state from a recent checkpoint, saving a model and training
        state, and running a training loop to make gradient updates.

        :param config: An input config dictionary file detailing the configuration parameters of the models
            and for training.
        :param dataloaders: A dictionary of torch dataloaders with keys "train" and "val".
        """
        super().__init__()
        self.config = config  # Record config parameters passed
        setattr(self, "z_dim", config["models"]["z_dim"])
        setattr(self, "num_classes", config["models"]["num_classes"])
        setattr(self, "image_dim", config["models"]["image_dim"])

        ### Set up folders for the output (samples images), losses, and model checkpoints
        results_folder = os.path.join(CURRENT_DIR, "results", str(config["name"]))
        self.results_folder = results_folder  # A directory where the checkpoints will be saved
        self.checkpoints_folder = os.path.join(self.results_folder, "checkpoints")
        self.losses_folder = os.path.join(self.results_folder, "losses")
        self.pretrain_losses_folder = os.path.join(self.results_folder, "pretrain_losses")
        self.pretrain_samples_folder = os.path.join(self.results_folder, "pretrain_samples")
        self.samples_folder = os.path.join(self.results_folder, "samples")
        for directory in [self.results_folder, self.checkpoints_folder, self.losses_folder,
                          self.pretrain_losses_folder, self.pretrain_samples_folder, self.samples_folder]:
            os.makedirs(directory, exist_ok=True)  # Create the directory if not already there

        self.class_labels = get_class_labels(config["dataset"])  # A dict mapping int:str for each class label

        #### Set up logging during training
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Remove any existing handlers so re-running a notebook cell doesn't duplicate logs
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        # Log to file
        file_handler = logging.FileHandler(
            os.path.join(self.results_folder, "train.log"),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Log through tqdm
        tqdm_handler = TqdmLoggingHandler()
        tqdm_handler.setFormatter(formatter)
        self.logger.addHandler(tqdm_handler)

        self.logger.setLevel(logging.INFO)

        ### Configure the 3 models used in Bi-GAN training
        self.generator = Generator(self.z_dim, self.image_dim)
        self.encoder = Encoder(self.z_dim, self.image_dim)
        self.discriminator = Discriminator(self.z_dim, self.image_dim)
        self.class_embedding = ClassEmbedding(self.num_classes, self.z_dim)
        self.models = [self.generator, self.encoder, self.discriminator, self.class_embedding]
        for model in self.models:  # Report the number of trainable parameters in each model
            self.logger.info(f"{model.name}: {sum(p.numel() for p in model.parameters())} parameters")

        ### Set up other training variables required
        self.device = get_device()  # Auto-detect what device to use for training
        # Save a pointers to the train and validation dataloaders
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.step = 0  # Training step counter, will train until this reaches num_steps
        self.train_losses, self.val_losses = [], []  # Aggregate loss values during training

    def extract_config_params(self, config_dict: dict) -> None:
        """
        This method extracts relevant parameters from config_dict and sets them as attributes of self.
        e.g. self.critic_updates = config_dict["critic_updates"].
        """
        ### Extract training parameters from the config dict
        defaults = [("batch_size", 64), ("lr", 1.0e-4), ("weight_decay", 0.0e-0), ("num_steps", 100000),
                    ("adam_betas", (0.9, 0.999)), ("grad_clip", 1.0), ("use_amp", True),
                    ("use_latest_checkpoint", True), ("eval_every", 10000), ("save_every", 5000),
                    ("critic_updates", 5), ("lambda_val", 10.0)  # These last 2 are specific to training only
                    ]
        for param_name, default_val in defaults:  # Extract from config dict if possible, otherwise use
            # the default value for each parameter defined immediately above
            default_val = tuple(default_val) if param_name == "adam_betas" else default_val
            setattr(self, param_name, config_dict.get(param_name, default_val))

    def create_optimizers(self, config_dict: dict) -> None:
        """
        Creates optimizers for each of the models using parameters recorded in config_dict. Optimizers are
        created and saved as attributes to self.
        """
        ### Configure optimizers for training each model, exclude bias and norm layers from weight decay
        self.amp_dtype = get_amp_dtype(self.device) if config_dict["use_amp"] else None
        norm_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)
        for model in self.models:  # Create a separate Adam optimizer for each model
            decay_params, no_decay_params = [], []

            for module in model.modules():
                for name, param in module.named_parameters(recurse=False):
                    if not param.requires_grad:  # Skip over if no gradient tracking
                        continue

                    if isinstance(module, norm_layers):
                        # Exclude any kind of batch / group / layer norm from weight decay
                        no_decay_params.append(param)
                    elif name == "bias":  # Also exclude any bias terms from weight decay as well
                        no_decay_params.append(param)
                    else:  # All others will have weight decay applied to them
                        decay_params.append(param)

            # Check that all params are fully partitioned across decay_params and no_decay_params, check that
            # there is no overlap and also that the total number across both subsets sums to the exp. total
            assert len(set(decay_params).intersection(set(no_decay_params))) == 0
            assert (len(decay_params) + len(no_decay_params) == sum(p.requires_grad
                                                                    for p in model.parameters()))
            opt = torch.optim.Adam([
                {'params': decay_params, 'weight_decay': config_dict["weight_decay"]},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=config_dict["lr"], betas=config_dict["adam_betas"])
            setattr(self, f"opt_{model.name}", opt)  # e.g. self.opt_generator

        # Create 1 grad scaler for all models to use
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None

    def load_latest_checkpoint(self, pretrain: bool = False) -> None:
        """
        Loads in weights and optimizer states cached to disk of the latest checkpoint if called.
        """
        all_checkpoints = os.listdir(self.checkpoints_folder)  # Get all files listed in the directory
        # Split into pretrain and non-pretrain checkpoints
        pretrain_checkpoints = [x for x in all_checkpoints if x.startswith("pretrain-model")
                                and x.endswith(".pt")]
        model_checkpoints = [x for x in all_checkpoints if x.startswith("model") and x.endswith(".pt")]

        # If there are any prior checkpoints, attempt to load the latest one
        if pretrain and len(pretrain_checkpoints) > 0:
            last_checkpoint = max([int(x.replace("pretrain-model-", "").replace(".pt", ""))
                                   for x in pretrain_checkpoints if x.endswith(".pt")])
            self.load(last_checkpoint, True, False)

        elif len(all_checkpoints) > 0:  # Load model checkpoints first, if none, then load pretrained instead
            if len(model_checkpoints) > 0:
                last_checkpoint = max([int(x.replace("model-", "").replace(".pt", ""))
                                       for x in model_checkpoints if x.endswith(".pt")])
                self.load(last_checkpoint, False, False)

            elif len(pretrain_checkpoints) > 0:
                last_checkpoint = max([int(x.replace("pretrain-model-", "").replace(".pt", ""))
                                       for x in pretrain_checkpoints if x.endswith(".pt")])
                self.load(last_checkpoint, True, True)

    def save(self, milestone: int, pretrain: bool = False) -> None:
        """
        Saves the weights and training state of the models for the current milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :param pretrain: A bool flag indicating if this milestone is a pretraining milestone.
        :returns: None. Writes the weights and losses to disk.
        """
        file_name = f"pretrain-model-{milestone}.pt" if pretrain else f"model-{milestone}.pt"
        checkpoint_path = os.path.join(self.checkpoints_folder, file_name)
        self.logger.info(f"Saving model to {checkpoint_path}.")
        data = {"step": self.step}
        for model in self.models:
            data[model.name] = getattr(self, model.name).state_dict()  # Model weights
            data[f"opt_{model.name}"] = getattr(self, f"opt_{model.name}").state_dict()  # Optimizer
        if self.scaler is not None:
            data["scaler"] = self.scaler.state_dict()
        torch.save(data, checkpoint_path)

        # Save down all the loss values produced by models training since the last caching
        if pretrain:
            train_loss_cols = ["step", "prior_loss", "recon_loss", "latent_cycle_loss"]
            val_loss_cols = []
        else:
            train_loss_cols = ["step", "G_loss", "E_loss", "D_loss", "D_loss_real", "D_loss_fake",
                               "grad_penalty"]
            val_loss_cols = ["step", "E_avg", "E_std", "E_NLL", "D_real", "D_fake"]

        # Convert the train losses to a pd.DataFrame and save down the results
        df = pd.DataFrame(self.train_losses, columns=train_loss_cols)
        losses_folder = self.pretrain_losses_folder if pretrain else self.losses_folder
        df.to_csv(os.path.join(losses_folder, f"train-losses-{milestone}.csv"))

        # Convert the validation losses to a pd.DataFrame and save down the results
        if len(self.val_losses) > 0:
            df = pd.DataFrame(self.val_losses, columns=val_loss_cols)
            df.to_csv(os.path.join(self.losses_folder, f"val-losses-{milestone}.csv"))

    def load(self, milestone: int, pretrain: bool = False, weights_only: bool = False) -> None:
        """
        Loads in the cached weights and training state from disk for a particular milestone.

        :param milestone: An integer denoting the training timestep at which the model weights were saved.
        :param pretrain: A bool flag indicating if this milestone is a pretraining milestone.
        :param weights_only: If True, then only model weights are loaded, nothing else.
        :returns: None. Weights are loaded into the model.
        """
        file_name = f"pretrain-model-{milestone}.pt" if pretrain else f"model-{milestone}.pt"
        checkpoint_path = os.path.join(self.checkpoints_folder, file_name)
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        self.logger.info(f"Loading model from {checkpoint_path}.")

        if weights_only:
            self.logger.info("Loading only model weights, leaving all else as default")
            for model in self.models:
                getattr(self, model.name).load_state_dict(checkpoint_data[model.name])  # Model weights

        else:  # Load everything from the checkpoint, step counter, model weights, opt state, scaler
            self.step = checkpoint_data["step"]
            for model in self.models:
                getattr(self, model.name).load_state_dict(checkpoint_data[model.name])  # Model weights
                getattr(self, f"opt_{model.name}").load_state_dict(checkpoint_data[f"opt_{model.name}"])

            if self.scaler is not None and "scaler" in checkpoint_data:
                self.scaler.load_state_dict(checkpoint_data["scaler"])

        # Losses are not loaded in, they are saved to disk periodically with the model weights and are not
        # needed to continue training. The losses obtained by training will be cached again at the next save

        # Move the models and optimizers to the same device to continue training or for inference
        for model in self.models:
            getattr(self, model.name).to(self.device)  # Move the model to the correct device
            # Move the optimizer parameters to the correct device
            for state in getattr(self, f"opt_{model.name}").state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

    def update_lr(self, new_lr: float) -> None:
        """
        Sets all optimizers to have a new learning rate specified by new_lr.
        """
        for model in self.models:
            for param_group in getattr(self, f"opt_{model.name}").param_groups:
                param_group["lr"] = new_lr
        self.logger.info(f"Learning rates updated to: {new_lr}")

    def report_memory_usage(self) -> None:
        """
        Reports the current memory usage on the CPU and GPU if available via logging.
        """
        process = psutil.Process(os.getpid())

        cpu_ram_gb = process.memory_info().rss / (1024 ** 3)

        if torch.cuda.is_available():
            gpu_alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)

            self.logger.info(
                f"RAM={cpu_ram_gb:.2f} GB | "
                f"GPU alloc={gpu_alloc_gb:.2f} GB | "
                f"GPU reserved={gpu_reserved_gb:.2f} GB"
            )
        else:
            self.logger.info(f"RAM={cpu_ram_gb:.2f} GB")

    @compute_with_amp
    def compute_G_loss(self, batch: Dict) -> torch.Tensor:
        """
        Computes the generator loss (G_loss).
            1. Zeros the generator and class embedding optimizer gradients
            2. Computes and returns the generator loss (-1) * D(G(z), z, class_embed)

        :param batch: An input batch of data from the dataloader.
        :returns: G_loss, the generator loss averaged over the batch.
        """
        for model_name in ["generator", "class_embedding"]:
            getattr(self, f"opt_{model_name}").zero_grad(set_to_none=True)

        class_id = batch["class_id"].to(self.device, non_blocking=True)  # (B, )
        class_embed = self.class_embedding(class_id)  # (B, z_zim)
        batch_size = len(class_id)  # Will be <= self.batch_size
        z = torch.randn(batch_size, self.z_dim, device=self.device)  # (B, z_dim)
        x_fake = self.generator(z, class_embed)  # Generate fake images (B, 3, 128, 128)
        set_requires_grad(self.discriminator, False)  # Freeze the critic to save memory

        # Train the generator to produce images that the critic assigns high scores to
        G_loss = (-1) * self.discriminator(x_fake, z, class_embed).mean()

        set_requires_grad(self.discriminator, True)  # Unfreeze the critic model parameters
        return G_loss

    @compute_with_amp
    def compute_E_loss(self, batch: Dict) -> torch.Tensor:
        """
        Computes the encoder loss (E_loss).
            1. Zeros the encoder and class embedding optimizer gradients
            2. Computes and returns the encoder loss (-1) * D(x_real, E(x_real), class_embed)

        :param batch: An input batch of data from the dataloader.
        :returns: E_loss, the encoder loss averaged over the batch.
        """
        for model_name in ["encoder", "class_embedding"]:
            getattr(self, f"opt_{model_name}").zero_grad(set_to_none=True)

        x_real = batch["image"].to(self.device, non_blocking=True)  # (B, 3, 128, 128)
        class_id = batch["class_id"].to(self.device, non_blocking=True)  # (B, )
        class_embed = self.class_embedding(class_id)  # (B, z_zim)
        z_pred = self.encoder(x_real, class_embed)  # Encoder z prediction (B, z_dim)
        set_requires_grad(self.discriminator, False)  # Freeze the critic to save memory

        # Train the encoder to produce z vectors that the critic assigns high scores to
        E_loss = (-1) * self.discriminator(x_real, z_pred, class_embed).mean()
        set_requires_grad(self.discriminator, True)  # Unfreeze the critic model parameters

        # Add regularization to encourage the z_pred distribution to directly match that of the prior (N, I)
        latent_reg = (z_pred.mean(dim=0) - 0.0).pow(2).mean()  # Regularize towards each z_dim to be mean 0
        latent_reg += (z_pred.std(dim=0) - 1.0).pow(2).mean()  # Regularize towards each z_dim to be stddev 1
        latent_reg += (z_pred.pow(2).sum(dim=1).mean() - self.z_dim).pow(2)  # Apply L2 regularization
        E_loss += latent_reg  # Add the regularization penalty to encourage N(0, 1) behavior
        return E_loss

    @compute_with_amp
    def compute_D_loss(self, batch: Dict) -> torch.Tensor:
        """
        Computes the discriminator loss (D_loss).
            1. Zeros the discriminator and class embedding optimizer gradients
            2. Computes and returns the discriminator loss:
                D(G(z), z, class_embed) - D(x_real, E(x_real), class_embed)

        :param batch: An input batch of data from the dataloader.
        :returns: D_loss, the discriminator loss averaged over the batch.
        """
        for model_name in ["discriminator", "class_embedding"]:
            getattr(self, f"opt_{model_name}").zero_grad(set_to_none=True)

        x_real = batch["image"].to(self.device, non_blocking=True)  # (B, 3, 128, 128)
        class_id = batch["class_id"].to(self.device, non_blocking=True)  # (B, 1)
        class_embed = self.class_embedding(class_id)  # (B, z_zim)
        batch_size = len(x_real)  # Will be <= self.batch_size
        z = torch.randn(batch_size, self.z_dim, device=self.device)  # (B, z_dim)

        with torch.no_grad():  # For the discriminator loss, no gradients to other models needed
            x_fake = self.generator(z, class_embed)  # Generate fake images (B, 3, 128, 128)
            z_pred = self.encoder(x_real, class_embed)  # Encoder z prediction (B, z_dim)

        set_requires_grad(self.discriminator, True)  # Make sure gradients are being tracked

        # The discriminator wants to give the highest scores to the true data pairs
        # and lower scores to the fake data pairs, detach to prevent gradients into the
        # other models i.e. the encoder and generator models
        D_loss_real = self.discriminator(x_real, z_pred, class_embed).mean()
        D_loss_fake = self.discriminator(x_fake, z, class_embed).mean()
        # Maximize: E[D(x_real, E(x_real))] - E[D(G(z), z)] subject to the Lipschitz F1 penalty
        D_loss = D_loss_fake - D_loss_real

        # Add a 1-Lipschitz condition proxy to smooth the discriminator scoring function between the 2
        # distributions, create linearly interpolated x data between the fake and real dist as an eval point
        alpha = torch.rand(batch_size, 1, 1, 1, device=x_real.device)  # Required for 4d image broadcasting
        # Linearly interpolate between (G(z), z) and (x_real, E(x_real)) with random weights
        x_interp = alpha * x_fake + (1 - alpha) * x_real  # (B, 3, 128, 128)
        alpha = alpha.reshape(batch_size, 1)  # Reshape into (B, 1) for broadcasting with (B, z_dim)
        z_interp = alpha * z + (1 - alpha) * z_pred  # (B, z_dim)

        x_interp.requires_grad_(True)  # x_fake and x_real don't have gradients, turn them on for this calc
        z_interp.requires_grad_(True)  # z_pred and z don't have gradients, turn them on for this calc
        # Here we will penalize gradients both with respect to the interpolated image and z-vector, to enforce
        # the condition on the joint feature space which is required in the bi-GAN setting
        scores = self.discriminator(x_interp, z_interp, class_embed).squeeze()  # Compute critic scores (B, )
        # Compute the gradient of the critic scores wrt the input interpolated x_interp
        # (B, 3, 128, 128), (B, z_dim)
        grad_x, grad_z = torch.autograd.grad(outputs=scores, inputs=[x_interp, z_interp],
                                             grad_outputs=torch.ones_like(scores), create_graph=True)
        # Flatten (B, 3, 128, 128) -> (B, 3*128*128), partial derivatives wrt to each image as row vectors
        # then compute the Euclidean (L2) norm of each row ||grad_x D(x,z)||_2
        grad_x = grad_x.reshape(batch_size, -1)  # (B, 3, 128, 128) -> (B, 49152)
        grad = torch.cat([grad_x, grad_z], dim=1)  # (B, 49152 + z_dim)
        grad_norm = grad.norm(2, dim=1)  # (B, )
        # Penalize deviations from 1, as shown in the WGAN-GP paper i.e. a 1-Lipschitz function satisfies
        # ||grad_x D(x,z)||_2 == 1 along the sampled interpolated points between the 2 distributions,
        # real and fake. Compute the MSE of the grad_norm vs 1.0 everywhere
        # If it's too steep, the gradient norm is greater than 1, so the penalty pushes it to flatten
        # If it's too flat, the gradient norm is less than 1, so the penalty encourages steeper slopes
        # During training, this nudges the critic toward having a gradient magnitude close to 1 on the
        # interpolated points, which is the smoothness condition WGAN-GP is enforcing
        grad_penalty = ((grad_norm - 1) ** 2).mean()  # Compute the L2 norm of the gradient
        D_loss += self.lambda_val * grad_penalty

        if self.step % 500 == 0:
            print(f"\nStep: {self.step}")
            print("   ",
                  f"D_real={D_loss_real.item():.1f}",
                  f"D_fake={D_loss_fake.item():.1f}",
                  )

            print("   ",
                  f"|z|={z.norm(dim=1).mean():.2f}",
                  f"|z_pred|={z_pred.norm(dim=1).mean():.2f}",
                  )

            print("   ",
                  f"GP={grad_penalty.item():.2f}",
                  f"grad_norm={grad_norm.mean().item():.2f}",
                  )

        return D_loss, D_loss_real, D_loss_fake, grad_penalty

    def compute_gradients(self, loss: torch.Tensor) -> None:
        """
        This function essentially performs loss.backward(), but handles using a scaler for certain AMP.

        :param loss: A torch.Tensor with gradient tracking.
        :returns: None, gradients are computed and that information is stored in the optimizer of each model.
        """
        # Compute gradients with a backwards pass using auto-diff
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimizer_step(self, model: nn.Module) -> float:
        """
        Performs a gradient update step on the input model, this assumes compute_gradients has already been
        called prior to this method.

        :param model: The model whose parameters are to be updated using the gradients wrt the loss.
        :returns: The grad_norm computed with gradient clipping or np.NaN if no grad clipping is done.
        """
        grad_norm = np.nan  # Set a default value in case self.grad_clip is None
        opt = getattr(self, f"opt_{model.name}")

        if self.scaler is not None:
            if self.grad_clip is not None:  # Apply grad clipping if applicable
                self.scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.scaler.step(opt)  # Update the model parameters by taking a gradient step
        else:
            if self.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            opt.step()  # Update the model parameters by taking a gradient step
        return float(grad_norm)

    def pretrain(self, new_lr: float = None) -> None:
        """
        Runs pre-training on the Generator, Encoder, and ClassEmbeddings models using an auto-encoder
        reconstruction loss approach to pre-train these models to a baseline level of performance before
        introducing the full adversarial training paradigm by adding the discriminator.

        Here we will use real images (x_real) from the training set and compute:
            z_pred = E(x_real) with regularization towards the N(0, I) prior
            x_hat = G(E(x_real)) with an L1 loss vs the original real images
            z_pred = E(G(z)) with an L2 loss vs the original z vector

        This will help the generator learn to create realistic looking images and the encoder to learn to
        create z vectors near the expected N(0, I) prior latent distribution.

        Pass in new_lr to update the learning rate to new_lr after loading the last checkpoint (if applicable)
        and training continues.
        """
        config_dict = self.config["pretraining"]  # Use the pretraining config settings
        self.extract_config_params(config_dict)  # Set param values as attributes of self
        self.create_optimizers(config_dict)  # Init optimizers with config params
        if config_dict.get("use_latest_checkpoint", True):
            self.load_latest_checkpoint(pretrain=True)

        if new_lr is not None:  # If provided, update the learning rates of all models before training
            self.update_lr(new_lr)

        self.logger.info(f"Starting Pre-Training, device={self.device}, amp_dtype={self.amp_dtype}")
        for model in self.models:  # Report the learning rate and weight decay of all the models
            self.logger.info(model.name)
            for i, param_group in enumerate(getattr(self, f"opt_{model.name}").param_groups):
                self.logger.info(f"lr={param_group['lr']}, wd={param_group['weight_decay']}")
                break  # Show for only the first parameter group, assume all are the same

            model.to(self.device)  # Move the model to the correct device if not already there
            model.train()  # Make sure to set the model to train mode for training

        inf_dataloader = infinite_loader(self.train_dataloader)  # This does not cache batches

        with tqdm(initial=self.step, total=self.num_steps) as pbar:
            while self.step < self.num_steps:  # Run all pre-training iterations

                for model_name in ["generator", "encoder", "class_embedding"]:
                    getattr(self, f"opt_{model_name}").zero_grad(set_to_none=True)

                batch = next(inf_dataloader)
                x_real = batch["image"].to(self.device, non_blocking=True)  # (B, 3, 128, 128)
                class_id = batch["class_id"].to(self.device, non_blocking=True)  # (B, 1)
                class_embed = self.class_embedding(class_id)  # (B, z_zim)

                ### Compute z_pred = E(x_real) with regularization towards the N(0, I) prior
                z_pred = self.encoder(x_real, class_embed)  # Encoder z prediction (B, z_dim)
                # Add the regularization penalty to encourage N(0, 1) behavior
                mean_loss = z_pred.mean(dim=0).pow(2).mean()  # Encourage each dim of z_pred to be mean zero
                std_loss = (z_pred.std(dim=0) - 1).pow(2).mean()  # and stddev 1
                latent_reg = (z_pred.pow(2).sum(dim=1).mean() - self.z_dim).pow(2)  # Apply L2 regularization
                prior_loss = mean_loss + std_loss + latent_reg

                ### Compute x_hat = G(E(x_real) + noise) with an L1 loss vs the original real images
                # Add a little noise to the encoder outputs so that the generator learns to handle the region
                # around E(x_real) and not just the exact outputs directly, this is a regularizing effect
                alpha = torch.rand(len(z_pred), 1, device=self.device)  # (B, 1) random numbers [0. 1]
                alpha *= min(0.5, (self.step / self.num_steps))  # Limit to at most [0.0, 0.5]
                z_noisy = (1 - alpha) * z_pred + alpha * torch.randn_like(z_pred, device=self.device)
                x_hat = self.generator(z_noisy, class_embed)  # Generate reconstructions (B, 3, 128, 128)
                recon_loss = F.l1_loss(x_hat, x_real)

                # Compute z_cycle = E(G(z)) with an L2 loss vs the original z vector
                batch_size = len(x_real)  # Will be <= self.batch_size
                z = torch.randn(batch_size, self.z_dim, device=self.device)  # (B, z_dim)
                z_cycle = self.encoder(self.generator(z, class_embed), class_embed)  # (B, z_dim)
                latent_cycle_loss = F.smooth_l1_loss(z_cycle, z)  # L1 loss wrt the latent vector

                mmd_loss = compute_mmd(z_pred, z)  # Further regularization towards the prior

                ### Compute a gradient update now that the loss has been computed
                loss = 0.1 * prior_loss + 10.0 * recon_loss + 1.0 * latent_cycle_loss + 10.0 * mmd_loss
                self.compute_gradients(loss)  # Call backwards() on the loss to compute gradients
                G_grad = self.optimizer_step(self.generator)  # Update model params of G
                E_grad = self.optimizer_step(self.encoder)  # Update model params of E
                CE_grad = self.optimizer_step(self.class_embedding)  # Update the class embedding model params
                if self.scaler is not None:  # Only call update() iff using this approach
                    self.scaler.update()

                pbar.set_postfix_str(
                    f"prior_loss: {prior_loss.item():.2f}, mmd_loss: {mmd_loss:.2f} "
                    f"recon_loss: {recon_loss.item():.2f}, "
                    f"latent_cycle_loss: {latent_cycle_loss.item():.2f}, G_grad: {G_grad:.2f}, "
                    f"E_grad: {E_grad:.2f}, CE_grad: {CE_grad:.2f}"
                )

                ### Aggregate all the loss values for each timestep, record separately for each
                self.train_losses.append((self.step, prior_loss.item(), recon_loss.item(),
                                          latent_cycle_loss.item()))
                self.step += 1

                ### Periodically run evaluation metrics on the validation data set, always on the last iter
                if self.step % self.eval_every == 0 or self.step == self.num_steps:
                    with torch.no_grad():  # Compute without gradient tracking
                        self.generate_samples(pretrain=True)  # Generate some samples using random z-values
                        # Also save samples of reconstructed images i.e. G(E(x_real))
                        file_name = f"reconstructions-{self.step}.png"
                        titles = class_id[:40].detach().cpu().tolist()
                        titles = [f"{i} {self.class_labels[i]}" for i in titles]
                        save_images(x_hat[:40].detach().cpu(), titles, 5,
                                    os.path.join(self.pretrain_samples_folder, file_name))
                        # Print some diagnostic stats on how the encoder outputs look
                        print(f"Avg L2 Norm (z - z_cycle): {(z - z_cycle).norm(dim=1).mean():.2f}")
                        print(f"mean_loss: {mean_loss:.3f}, std_loss: {std_loss:.3f}")
                        print(f"Avg |E(x)|, {z_pred.mean(dim=1).abs().mean(dim=0):.2f}",
                              f"Avg std(x), {z_pred.std(dim=1).mean(dim=0):.2f}")

                ### Periodically save the model weights to disk, always on the last iter too
                if self.step % self.save_every == 0 or self.step == self.num_steps:
                    self.save(self.step, True)
                    # Clear the list of losses after each save, store only the ones from the last save to
                    # the next save
                    self.train_losses, self.val_losses = [], []
                    # Generate new loss plots after saving additional loss data to disk
                    generate_loss_plots(self.pretrain_losses_folder, self.results_folder)
                    torch.cuda.empty_cache()
                    gc.collect()  # This will slow down training if called too often

                del batch, x_real, class_id, class_embed, z_pred, z, prior_loss, recon_loss
                del latent_cycle_loss, loss, G_grad, E_grad, CE_grad
                pbar.update(1)

    def train(self, new_lr: float = None, freeze_encoder: bool = False) -> None:
        """
        Runs the training loop for the Wasserstein Bi-GAN until completion for self.num_steps total
        training iterations.

        Pass in new_lr to update the learning rate to new_lr after loading the last checkpoint (if applicable)
        and training continues.
        """
        config_dict = self.config["training"]  # Use the Bi-GAN training config settings
        self.extract_config_params(config_dict)  # Set param values as attributes of self
        self.create_optimizers(config_dict)  # Init optimizers with config params
        if config_dict.get("use_latest_checkpoint", True):
            self.load_latest_checkpoint(pretrain=False)

        if new_lr is not None:  # If provided, update the learning rates of all models before training
            self.update_lr(new_lr)

        if freeze_encoder is True:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.logger.info(f"Starting Training, device={self.device}, amp_dtype={self.amp_dtype}")
        for model in self.models:  # Report the learning rate and weight decay of all the models
            self.logger.info(model.name)
            for i, param_group in enumerate(getattr(self, f"opt_{model.name}").param_groups):
                self.logger.info(f"lr={param_group['lr']}, wd={param_group['weight_decay']}")
                break  # Show for only the first parameter group, assume all are the same

            model.to(self.device)  # Move the model to the correct device if not already there
            model.train()  # Make sure to set the model to train mode for training

        inf_dataloader = infinite_loader(self.train_dataloader)  # This does not cache batches

        with tqdm(initial=self.step, total=self.num_steps) as pbar:
            while self.step < self.num_steps:  # Run until all training iterations are complete
                ### Perform K updates to the critic model first
                for _ in range(self.critic_updates):
                    batch = next(inf_dataloader)
                    D_loss, D_loss_real, D_loss_fake, grad_penalty = self.compute_D_loss(batch)
                    self.compute_gradients(D_loss)  # Call backwards() on the loss to compute gradients
                    D_grad = self.optimizer_step(self.discriminator)  # Update model params of D
                    self.optimizer_step(self.class_embedding)  # Update the class embedding model params
                    if self.scaler is not None:  # Only call update() iff using this approach
                        self.scaler.update()

                ### Then perform 1 updated to the generator and encoder models
                batch = next(inf_dataloader)
                # Generator update
                G_loss = self.compute_G_loss(batch)  # Compute the G loss over this batch with grads
                self.compute_gradients(G_loss)  # Call backwards() on the loss to compute gradients
                G_grad = self.optimizer_step(self.generator)  # Update model params of G
                self.optimizer_step(self.class_embedding)  # Update the class embedding model params
                if self.scaler is not None:  # Only call update() iff using this approach
                    self.scaler.update()

                # Encoder update
                if freeze_encoder is False:
                    E_loss = self.compute_E_loss(batch)  # Compute the E loss over this batch with grads
                    self.compute_gradients(E_loss)  # Call backwards() on the loss to compute gradients
                    E_grad = self.optimizer_step(self.encoder)  # Update model params of E
                    self.optimizer_step(self.class_embedding)  # Update the class embedding model params
                    if self.scaler is not None:  # Only call update() iff using this approach
                        self.scaler.update()
                else:  # If we have frozen the encoder, then no gradient updates will be made to it
                    E_loss = torch.tensor(0.0, device=self.device)
                    E_grad = np.nan

                pbar.set_postfix(
                    G_loss=f"{G_loss.item():.1f}", G_grad=f"{G_grad:.1f}",
                    E_loss=f"{E_loss.item():.1f}", E_grad=f"{E_grad:.1f}",
                    D_loss=f"{D_loss.item():.1f}", D_grad=f"{D_grad:.1f}", )

                ### Aggregate all the loss values for each timestep, record separately for each
                self.train_losses.append((self.step, G_loss.item(), E_loss.item(), D_loss.item(),
                                          D_loss_real.item(), D_loss_fake.item(), grad_penalty.item()))
                self.step += 1

                if self.step % 1000 == 0:
                    # self.report_lr_opt_state()  # Report info about the current learning rate and opt state
                    self.report_memory_usage()  # Report info about the memory usage

                ### Periodically run evaluation metrics on the validation data set, always on the last iter
                if self.step % self.eval_every == 0 or self.step == self.num_steps:
                    with torch.no_grad():  # Compute without gradient tracking
                        self.run_eval()

                ### Periodically save the model weights to disk, always on the last iter too
                if self.step % self.save_every == 0 or self.step == self.num_steps:
                    self.save(self.step, False)
                    # Clear the list of losses after each save, store only the ones from the last save to
                    # the next save
                    self.train_losses, self.val_losses = [], []
                    # Generate new loss plots after saving additional loss data to disk
                    generate_loss_plots(self.losses_folder, self.results_folder)
                    torch.cuda.empty_cache()
                    gc.collect()  # This will slow down training if called too often

                del batch, G_loss, E_loss, D_loss
                pbar.update(1)

    @compute_with_amp
    def generate_samples(self, pretrain: bool = False, seed: int = None):
        """
        Runs G(z) to create a grid of images, 1 for each class. Results are saved to disk.
        """
        for model in [self.generator, self.class_embedding]:
            model.eval()

        class_id = torch.tensor(list(range(self.class_embedding.num_classes)), device=self.device)  # (B, )
        class_embed = self.class_embedding(class_id)  # (B, z_zim)

        rng = torch.Generator(device=self.device)  # Get up a random number generator
        if seed is not None:  # Set the seed if one is provided for replicability
            rng.manual_seed(seed)
        z = torch.randn(len(class_id), self.z_dim, device=self.device, generator=rng)  # (B, z_dim)
        x_fake = self.generator(z, class_embed)  # Compute G(z) i.e. the synthetic images
        # Save down the results to a grid of images, one for each image class
        titles = [f"{i} {self.class_labels[i]}" for i in range(self.class_embedding.num_classes)]
        samples_folder = self.pretrain_samples_folder if pretrain else self.samples_folder
        save_images(x_fake, titles, 5, os.path.join(samples_folder, f"sample-{self.step}.png"))

        for model in [self.generator, self.class_embedding]:
            model.train()

    @compute_with_amp
    def run_eval(self):
        """
        Orchestrates an evaluation run of the model, which is intended to be run periodically during training
        to track the performance of the models over time.
        """
        for model in self.models:  # Switch all models to eval mode
            model.eval()

        ### 1). Generator Eval - Generate a few sample images for each class so that we can track the
        # progression of  the generator model over time. We expect to see image clarity gradually improve
        # and hope to avoid mode collapse
        self.generate_samples(pretrain=False, seed=2026)

        ### Encoder and Discriminator Eval
        z_pred_all = []
        D_loss_components = []
        for batch in self.val_dataloader:
            x_real = batch["image"].to(self.device, non_blocking=True)  # (B, 3, 128, 128)
            class_id = batch["class_id"].to(self.device, non_blocking=True)  # (B, 1)
            class_embed = self.class_embedding(class_id)  # (B, z_zim)
            batch_size = len(x_real)  # Will be <= self.batch_size
            z = torch.randn(batch_size, self.z_dim, device=self.device)  # (B, z_dim)

            z_pred = self.encoder(x_real, class_embed)  # Encoder z prediction (B, z_dim)
            z_pred_all.append(z_pred)

            x_fake = self.generator(z, class_embed)  # Generate fake images (B, 3, 128, 128)
            D_loss_real = self.discriminator(x_real, z_pred, class_embed).mean().item()
            D_loss_fake = self.discriminator(x_fake, z, class_embed).mean().item()
            D_loss_components.append((D_loss_real, D_loss_fake))

        ### 2). Evaluation Eval - Estimate the performance of the encoder model by seeing how close the
        # y_pred outputs match the prior i.e. N(0, I). We expect to see the per dim mean approach 0 and
        # the per dim stddev converge to 1. We can compute NLL and a KL divergence as well.
        z_pred = torch.concat(z_pred_all, dim=0)  # (N, z_dim) concatenate all the z_pred together
        encoder_metrics = [
            z_pred.mean(dim=0).abs().mean().item(),  # Avg(|E(x)|) for each z-dim
            z_pred.std(dim=0).mean().item(),  # Avg(Stddev(x)) for each z-dim
            (-1) * (-0.5 * (z_pred.pow(2) + math.log(2 * math.pi)).sum(dim=0)).mean().item(),  # Avg(NLL)
        ]

        ### 3). Discriminator Eval - Track the performance of the model using the discriminator, track the
        # discriminator loss on the eval set and record the components for x_real, x_fake, and grad penalty
        discriminator_metrics = [
            np.array([x[0] for x in D_loss_components]).mean(),  # Mean D_loss_real
            np.array([x[1] for x in D_loss_components]).mean(),  # Mean D_loss_fake
        ]
        self.val_losses.append([self.step] + encoder_metrics + discriminator_metrics)  # Record for caching

        for model in self.models:  # Switch all models back to train mode
            model.train()
