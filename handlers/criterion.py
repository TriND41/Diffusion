import torch
import torch.nn as nn

class ImageGenerationCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__mse_criterion = nn.MSELoss()

    def mse_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        outputs = outputs.float()
        targets = targets.float()
        return self.__mse_criterion(outputs, targets)