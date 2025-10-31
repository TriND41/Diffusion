import sys
sys.path.append('.')

import os
import torch
import torch.multiprocessing as mp
from tools.training import Trainer
from handlers.loading import UnconditionalImageGenerationDataset
from handlers.distribution import setup, cleanup
from typing import Literal, Optional

def train(
    rank: int,
    world_size: int,
    # Training
    train_path: str,
    num_train_samples: Optional[int] = None,
    train_batch_size: int = 1,
    num_epochs: int = 1,
    fp16: bool = False,
    gradient_clipping: bool = False,
    clipping_value: Optional[float] = None,
    # Validation
    val_path: Optional[str] = None,
    num_val_samples: Optional[int] = None,
    val_batch_size: int = 1,
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
    try:
        if world_size > 1:
            setup(rank, world_size)

        trainer = Trainer(
            rank=rank,
            input_size=input_size,
            input_channels=input_channels,
            timesteps=timesteps,
            bilinear=bilinear,
            lr=lr,
            checkpoint_path=checkpoint_path,
            checkpoint_folder=checkpoint_folder,
            n_saved_checkpoints=n_saved_checkpoints,
            save_checkpoint_after_steps=save_checkpoint_after_steps,
            save_checkpoint_after_epochs=save_checkpoint_after_epochs,
            early_stopping=early_stopping,
            n_patiences=n_patiences,
            observe=observe,
            score_type=score_type,
            logging=logging,
            logging_project=logging_project,
            logging_name=logging_name
        )

        train_dataset = UnconditionalImageGenerationDataset(
            manifest=train_path,
            processor=trainer.image_processor,
            num_examples=num_train_samples
        )

        val_dataset: Optional[UnconditionalImageGenerationDataset] = None
        if val_path is not None and os.path.exists(val_path):
            val_dataset = UnconditionalImageGenerationDataset(
                manifest=val_path,
                processor=trainer.image_processor,
                num_examples=num_val_samples
            )
            
        trainer.train(
            dataset=train_dataset,
            batch_size=train_batch_size,
            num_epochs=num_epochs,
            val_dataset=val_dataset,
            val_batch_size=val_batch_size,
            fp16=fp16,
            gradient_clipping=gradient_clipping,
            clipping_value=clipping_value
        )

    except Exception as e:
        raise ValueError(str(e))
    
    finally:
        if world_size > 1:
            cleanup()

def main(
    # Training
    train_path: str,
    num_train_samples: Optional[int] = None,
    train_batch_size: int = 1,
    num_epochs: int = 1,
    fp16: bool = False,
    gradient_clipping: bool = False,
    clipping_value: Optional[float] = None,
    # Validation
    val_path: Optional[str] = None,
    num_val_samples: Optional[int] = None,
    val_batch_size: int = 1,
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
    assert os.path.exists(train_path)

    n_gpus = torch.cuda.device_count()
    assert n_gpus > 1

    if n_gpus == 1:
        train(
            0, n_gpus,
            train_path, num_train_samples, train_batch_size,
            num_epochs, fp16, gradient_clipping, clipping_value,
            val_path, num_val_samples, val_batch_size,
            input_size, input_channels,
            timesteps, bilinear,
            lr,
            checkpoint_path, checkpoint_folder, n_saved_checkpoints, save_checkpoint_after_steps, save_checkpoint_after_epochs,
            early_stopping, n_patiences, observe, score_type,
            logging, logging_project, logging_name
        )
    elif n_gpus > 1:
        mp.spawn(
            fn=train,
            args=(
                n_gpus,
                train_path, num_train_samples, train_batch_size,
                num_epochs, fp16, gradient_clipping, clipping_value,
                val_path, num_val_samples, val_batch_size,
                input_size, input_channels,
                timesteps, bilinear,
                lr,
                checkpoint_path, checkpoint_folder, n_saved_checkpoints, save_checkpoint_after_steps, save_checkpoint_after_epochs,
                early_stopping, n_patiences, observe, score_type,
                logging, logging_project, logging_name
            ),
            nprocs=n_gpus,
            join=True
        )

if __name__ == '__main__':
    import fire
    fire.Fire(main)