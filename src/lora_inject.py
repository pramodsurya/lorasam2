import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """
    Linear layer with additive LoRA adapters that preserves the pretrained base weight.

    y = x @ (W^T + (alpha/r) * (B @ A)^T) + b

    Where A: (r, in), B: (out, r). A is initialized with zeros, B with zeros (or small init), so
    initial delta is zero and the layer behaves like the original.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0

        # Register the frozen base weight/bias
        self.weight = nn.Parameter(base.weight.data.clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            # Kaiming init for A, zeros for B is common; we'll use small init for stability
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r and self.lora_A is not None and self.lora_B is not None:
            delta_w = torch.matmul(self.lora_B, self.lora_A) * self.scaling  # (out,in)
            w_eff = self.weight + delta_w
        else:
            w_eff = self.weight
        return torch.nn.functional.linear(x, w_eff, self.bias)


class LoRAInjector:
    """
    Recursively wrap selected nn.Linear modules in a model with LoRA adapters, preserving
    the original weights. Exposes utilities to freeze non-LoRA params and to extract
    LoRA-only state dicts.
    """

    def __init__(self, rank: int = 8, alpha: int = 16, target_modules=None):
        self.rank = rank
        self.alpha = alpha
        # If target_modules is None, default to wrapping all Linear layers.
        self.target_modules = set(m.lower() for m in (target_modules or []))

    def _should_wrap(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not self.target_modules:
            return True
        lname = name.lower()
        return any(t in lname for t in self.target_modules)

    def apply(self, root: nn.Module):
        for name, child in list(root.named_children()):
            if self._should_wrap(name, child):
                setattr(root, name, LoRALinear(child, r=self.rank, alpha=self.alpha))
            else:
                self.apply(child)

    @staticmethod
    def mark_only_lora_trainable(model: nn.Module):
        for n, p in model.named_parameters():
            if ("lora_A" in n) or ("lora_B" in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    @staticmethod
    def lora_state_dict(model: nn.Module):
        return {k: v.cpu() for k, v in model.state_dict().items() if ("lora_A" in k) or ("lora_B" in k)}

    @staticmethod
    def load_lora_state_dict(model: nn.Module, state: dict):
        missing, unexpected = model.load_state_dict(state, strict=False)
        return missing, unexpected
