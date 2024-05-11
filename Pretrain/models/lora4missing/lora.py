"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time

class LoRALayer():
    """
    Base lora class
    """
    def __init__(
            self,
            r,
            lora_alpha,
         ):
        self.r = r
        self.lora_alpha = lora_alpha

    def reset_parameters(self):
        raise NotImplementedError



class LoRAProj(nn.Linear, LoRALayer):
    def __init__(self, r, lora_alpha, in_features, out_features):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        LoRALayer.__init__(self, r, lora_alpha)
        nn.Linear.__init__(self, in_features, out_features)
        # Lora configuration
        self.lora_A = nn.ParameterList([self.weight.new_zeros((r, in_features)) for _ in range(4)])
        self.lora_B = nn.ParameterList([self.weight.new_zeros((out_features, r)) for _ in range(4)])
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            for idx in range(4):
                nn.init.kaiming_uniform_(self.lora_A[idx], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[idx])


    def forward(self, x, idx):
        result = F.linear(x, self.weight, bias=self.bias)
        out = (x @ self.lora_A[idx].T @ self.lora_B[idx].T*self.scaling)
        result += out
        return result

class LoRALinear(nn.Linear, LoRALayer):
    def __init__(self, r, lora_alpha, in_features, out_features):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        LoRALayer.__init__(self, r, lora_alpha)
        nn.Linear.__init__(self, in_features, out_features)
        # Lora configuration
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight, bias=self.bias)
        out = (x @ self.lora_A.T @ self.lora_B.T*self.scaling)
        result += out
        return result


class LoRAConv3d(nn.Conv3d, LoRALayer):
    def __init__(self, r, lora_alpha, in_channels: int, out_channels: int, kernel_size: int | F.Tuple[int], stride: int | F.Tuple[int] = 1, padding: str | int | F.Tuple[int] = 0, dilation: int | F.Tuple[int] = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        LoRALayer.__init__(self, r, lora_alpha)
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_channels*kernel_size*kernel_size*kernel_size)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels, r)))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Conv3d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(
            input,
            self.weight+(self.lora_B@self.lora_A).view(self.weight.shape)*self.scaling,
            self.bias
        )
