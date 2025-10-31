import torch
from typing import Optional, Iterator

def gradient_clipping(parameters: Iterator[torch.Tensor], clipping: bool = False, norm_type: int = 2, max_norm: int = 1, clipping_value: Optional[float] = None) -> torch.Tensor:
    parameters = list(filter(lambda param: param.grad is not None, parameters))
    
    norm_type = float(norm_type)
    if clipping_value is not None:
        clipping_value = float(clipping_value)
    
    total_norm = 0.0
    for param in parameters:
        param_norm = param.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
        
    total_norm = total_norm ** (1. / norm_type)

    if clipping:
        if clipping_value is None:
            torch.nn.utils.clip_grad.clip_grad_norm_(parameters, max_norm=max_norm, norm_type=norm_type)
        else:
            torch.nn.utils.clip_grad.clip_grad_value_(parameters, clip_value=clipping_value)

    return total_norm

def clip_gradient(parameters: Iterator[torch.Tensor], value: Optional[float] = None) -> None:
    parameters = list(filter(lambda param: param.grad is not None, parameters))

    if value is None:
        torch.nn.utils.clip_grad.clip_grad_norm_(parameters, norm_type=2)
    else:
        torch.nn.utils.clip_grad.clip_grad_value_(parameters, clip_value=value)

def compute_gradient_norm(parameters: Iterator[torch.Tensor]) -> torch.Tensor:
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.data.norm(2) ** 2
    return total_norm**0.5