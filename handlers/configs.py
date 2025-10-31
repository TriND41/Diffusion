from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    channels: int
    timesteps: int = 1000
    bilinear: bool = False

@dataclass
class ImageProcessorConfig:
    input_size: int
    input_channels: int