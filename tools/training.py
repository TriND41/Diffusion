import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as distributed
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from model.diffusion import Diffusion
from resolvers.image import ImageProcessor

from handlers.symbols import CheckpointKey
from handlers.checkpoint import CheckpointManager, load_checkpoint
from handlers.logging import Logger
from handlers.early_stopping import EarlyStopping
from handlers import gradient
from handlers.loading import UnconditionalImageGenerationDataset
from handlers.configs import DiffusionConfig, ImageProcessorConfig
from handlers.criterion import ImageGenerationCriterion
from handlers.symbols import CheckpointKey
import handlers.gradient as gradient

import torchsummary
from tqdm import tqdm
import logging
from typing import Literal, Optional, Tuple, Dict, Any

class Trainer:
    def __init__(
        self,
        rank: int,
        # Image config
        input_size: int = 28,
        input_channels: int = 1,
        # Model configs
        timesteps: int = 1000,
        bilinear: bool = False,
        # Optimization
        lr: Optional[float] = None,
        # Checkpoint
        checkpoint_path: Optional[str] = None,
        checkpoint_folder: str = "./checkpoints",
        n_saved_checkpoints: int = 3,
        save_checkpoint_after_steps: Optional[int] = None,
        save_checkpoint_after_epochs: int = 1,
        # Early stopping
        early_stopping: bool = False,
        n_patiences: int = 3,
        observe: Literal['loss', 'score'] = 'loss',
        score_type: Literal['down', 'up'] = 'down',
        # Logging
        logging: bool = False,
        logging_project: str = "Image Generation - Diffusion",
        logging_name: Optional[str] = None
    ) -> None:
        self.rank = rank

        checkpoint: Optional[Dict[str, Any]] = None
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = load_checkpoint(checkpoint_path)

            self.hyper_params = DiffusionConfig(**checkpoint[CheckpointKey.HYPER_PARAMS])
            self.image_configs = ImageProcessorConfig(**checkpoint[CheckpointKey.IMAGE_PROCESSOR])
        else:
            self.image_configs = ImageProcessorConfig(input_size, input_channels)
            self.hyper_params = DiffusionConfig(input_channels, timesteps, bilinear)

        # Processing
        self.image_processor = ImageProcessor(**self.image_configs.__dict__)

        # Model
        self.model = Diffusion(**self.hyper_params)
        self.model.to(rank)
        if distributed.is_initialized():
            self.model = DDP(self.model, device_ids=[rank])

        # Optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr if lr is not None else 3e-3)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        # Evaluation
        self.criterion = ImageGenerationCriterion()
        self.criterion.to(rank)

        # Load weights
        self.n_steps, self.n_epochs = 0, 0
        if checkpoint is not None:
            self.__load_weights(checkpoint)

        if lr is not None:
            self.optimizer.param_groups[0]['lr'] = lr

        if self.rank == 0:
            # Checkpoint
            self.checkpoint_manager = CheckpointManager(checkpoint_folder, n_savings=n_saved_checkpoints)
            self.save_checkpoint_after_steps = save_checkpoint_after_steps
            self.save_checkpoint_after_epochs = save_checkpoint_after_epochs

            # Early Stopping
            self.early_stopping: Optional[EarlyStopping] = None
            if early_stopping:
                self.early_stopping = EarlyStopping(n_patiences, score_type)
                self.observe = observe

            # Logging
            self.logger: Optional[Logger] = None
            if logging:
                self.logger = Logger(logging_project, logging_name)

            # Summary
            print("\nModel Summary:")
            torchsummary.summary(self.model)

    def __load_weights(self, checkpoint: Dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint[CheckpointKey.MODEL])
        self.optimizer.load_state_dict(checkpoint[CheckpointKey.OPTIMIZER])
        self.scheduler.load_state_dict(checkpoint[CheckpointKey.SCHEDULER])

        self.n_steps = checkpoint[CheckpointKey.ITERATION]
        self.n_epochs = checkpoint[CheckpointKey.EPOCH]

    def __save_checkpoint(self, logging: bool = False) -> None:
        checkpoint = {
            # Params
            CheckpointKey.HYPER_PARAMS: self.hyper_params.__dict__,
            CheckpointKey.IMAGE_PROCESSOR: self.image_configs.__dict__,
            # Weights
            CheckpointKey.MODEL: self.model.state_dict(),
            CheckpointKey.OPTIMIZER: self.optimizer.state_dict(),
            CheckpointKey.SCHEDULER: self.scheduler.state_dict(),
            # Info
            CheckpointKey.ITERATION: self.n_steps,
            CheckpointKey.EPOCH: self.n_epochs
        }
        self.checkpoint_manager.save_checkpoint(checkpoint, self.n_epochs, self.n_steps, logging=logging)
    
    def __configure_dataloader(self, dataset: UnconditionalImageGenerationDataset, batch_size: int = 1) -> DataLoader:
        if not distributed.is_initialized():
            sampler = RandomSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, num_replicas=distributed.get_world_size(), rank=self.rank)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=dataset.collate
        )
    
    def train(
        self, 
        dataset: UnconditionalImageGenerationDataset,
        batch_size: int,
        num_epochs: int = 1,
        # Val
        val_dataset: Optional[UnconditionalImageGenerationDataset] = None,
        val_batch_size: int = 1,
        # Extra
        fp16: bool = False,
        gradient_clipping: bool = False,
        clipping_value: Optional[float] = None
    ) -> None:
        dataloader = self.__configure_dataloader(dataset, batch_size=batch_size)

        val_dataloader: Optional[DataLoader] = None
        if val_dataset is not None:
            val_dataloader = self.__configure_dataloader(val_dataset, batch_size=val_batch_size)

        scaler = torch.GradScaler(enabled=fp16)
        for epoch in range(num_epochs):
            if distributed.is_initialized():
                dataloader.sampler.set_epoch(self.n_epochs)
            
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}\n==========================")

            self.model.train()
            train_loss = 0.0
            train_gradient_norm = 0.0
            for inputs in tqdm(dataloader):
                with torch.no_grad():
                    inputs = torch.tensor(inputs, dtype=torch.float, device=self.rank)
                    noises = torch.randn(inputs.size(), dtype=torch.float, device=self.rank)

                # Forward
                with torch.autocast(device_type='cuda', enabled=fp16):
                    outputs = self.model(inputs, noises)
                    with torch.autocast(device_type='cuda', enabled=False):
                        loss = self.criterion.mse_loss(outputs, noises)
                        assert torch.isnan(loss) ==  False
            
                # Backward
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                if gradient_clipping:
                    gradient.clip_gradient(self.model.parameters(), value=clipping_value)
                grad_norm = gradient.compute_gradient_norm(self.model.parameters())
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)

                # Update iteration
                self.n_steps += 1
                scaler.update()
                train_loss += loss
                train_gradient_norm += grad_norm

            # Update epoch
            self.n_epochs += 1

            train_loss /= len(dataloader)
            train_gradient_norm /= len(len(dataloader))
            if distributed.is_initialized():
                distributed.all_reduce(train_loss, op=distributed.ReduceOp.AVG)
                distributed.all_reduce(train_gradient_norm, op=distributed.ReduceOp.AVG)

            # Log
            if self.rank == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                print(f"Train Loss: {(train_loss):.4f}")
                print(f"Train Gradient Norm: {(train_gradient_norm):.4f}")
                print(f"Learning Rate: {current_lr}")

                if epoch % self.save_checkpoint_after_epochs == self.save_checkpoint_after_epochs - 1 or epoch == num_epochs - 1:
                    self.__save_checkpoint(logging=True)

                if self.logger is not None:
                    self.logger.log({
                        'train_loss': train_loss,
                        'gradient_norm': train_gradient_norm,
                        'learning_rate': current_lr
                    }, self.n_steps)

                if self.early_stopping is not None and val_dataloader is None:
                        self.early_stopping.step(train_loss)
            
            # Update lr
            self.scheduler.step()

            # Validation
            if val_dataloader is not None:
                pass
            
            # Early stop
            if self.rank == 0:
                if self.early_stopping is not None and self.early_stopping.early_stop():
                    break