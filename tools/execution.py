import os
import torch
import numpy as np
from model.diffusion import Diffusion
from resolvers.image import ImageProcessor
from handlers.configs import DiffusionConfig, ImageProcessorConfig
from handlers.checkpoint import load_checkpoint
from handlers.symbols import CheckpointKey
import logging
from typing import Union

logger = logging.getLogger(__name__)

class Executor:
    def __init__(self, checkpoint_path: str, device: Union[str, int, torch.device] = 'cpu') -> None:
        assert os.path.exists(checkpoint_path)
        checkpoint = load_checkpoint(checkpoint_path)
        logger.info(f"Loaded Checkpoint from {checkpoint_path}")

        # Declare Model
        hyper_params = DiffusionConfig(**checkpoint[CheckpointKey.HYPER_PARAMS])
        self.model = Diffusion(**hyper_params.__dict__)
        self.model.load_state_dict(checkpoint[CheckpointKey.MODEL])
        self.model.eval()
        self.model.to(device)

        # Declare Processing
        image_configs = ImageProcessorConfig(**checkpoint[CheckpointKey.IMAGE_PROCESSOR])
        self.image_processor = ImageProcessor(**image_configs.__dict__)
        
        # Declare Info
        self.input_channels = image_configs.input_channels
        self.input_size = image_configs.input_size
        self.timesteps = hyper_params.timesteps
        self.device = device
    
    @torch.no_grad()
    @torch.inference_mode()
    def run(self) -> np.ndarray:
        samples = torch.randn(
            (1, self.input_channels, self.input_size, self.input_size),
            dtype=torch.float,
            device=self.device
        )

        for t in range(self.timesteps):
            samples = self.model.reverse(samples, t)

        return samples.squeeze(0).permute([1, 2, 0]).cpu().numpy()