import numpy as np
import cv2 as cv
from typing import List

class ImageProcessor:
    def __init__(self, input_size: int = 28, input_channels: int = 1) -> None:
        self.input_size = input_size
        self.input_channels = input_channels

    def load_image(self, path: str) -> np.ndarray:
        image = cv.imread(path)
        if self.input_channels == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return image / 255
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, dsize=(self.input_size, self.input_size))
    
    def __call__(self, images: List[np.ndarray]) -> np.ndarray:
        images = [self.resize(image) for image in images]
        images = np.stack(images, axis=0)
        if self.input_channels == 3:
            images = np.transpose(images, [0, 3, 1, 2])
        else:
            images = np.expand_dims(images, axis=1)
        return images