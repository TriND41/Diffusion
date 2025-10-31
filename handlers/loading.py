import os
import numpy as np
from torch.utils.data import Dataset
from resolvers.image import ImageProcessor
import io
from typing import Optional, List

class UnconditionalImageGenerationDataset(Dataset):
    def __init__(self, manifest: str, processor: ImageProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.data = io.open(manifest, encoding='utf8').read().strip().split("\n")
        if num_examples is not None:
            self.data = self.data[:num_examples]

        self.processor = processor

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> np.ndarray:
        image_path = self.data[index]
        image = self.processor.load_image(image_path)
        return image
    
    def collate(self, images: List[np.ndarray]) -> np.ndarray:
        images = self.processor(images)
        return images