import torch

assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with CUDA support."
device = torch.device("cuda:0")

from typing import *
import abc

class Module(abc.ABC):
    device: Optional[torch.device]
    inputs: Tuple[torch.Tensor, ...]

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device

    @abc.abstractmethod
    def parameters(self) -> List[torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        self.inputs = inputs
        return self.forward(*inputs)

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device] = None):
        super().__init__(device)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features, device=device, requires_grad=True) / in_features
        self.bias = torch.zeros(out_features, device=device, requires_grad=True)

    def parameters(self) -> List[torch.Tensor]:
        return [self.weight, self.bias]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t() + self.bias

    def backward(self, dLdout: torch.Tensor) -> torch.Tensor:
        grad_input = dLdout @ self.weight
        self.weight.grad = dLdout.t() @ self.inputs[0]
        self.bias.grad = dLdout.sum(dim=0)
        return grad_input

    def __repr__(self)-> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})"


class ReLU(Module):
    def parameters(self) -> Iterator[torch.Tensor]:
        return []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0)

    def backward(self, dLdout: torch.Tensor) -> torch.Tensor:
        grad_input = dLdout.clone()
        grad_input[self.inputs[0] <= 0] = 0
        return grad_input

    def __repr__(self)-> str:
        return "ReLU()"

class CrossEntropyLoss(Module):
    def parameters(self) -> Iterator[torch.Tensor]:
        return []

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b = logits.shape[0]
        return -logits.softmax(dim=-1).log()[torch.arange(b, device=self.device), targets].mean()

    def backward(self, dLdout: torch.Tensor) -> torch.Tensor:
        logits, targets = self.inputs
        b = logits.shape[0]
        grad_logits = logits.softmax(dim=-1)
        grad_logits[torch.arange(b, device=self.device), targets] -= 1
        grad_logits /= b
        return grad_logits

    def __repr__(self)-> str:
        return "CrossEntropyLoss()"


class Network(Module):
    layers: List[Module]

    def __init__(self, input_dim, device=None):
        super().__init__(device)
        self.layers = [
            Linear(input_dim, 8192, device=device),
            ReLU(device=device),
            Linear(8192, 8192, device=device),
            ReLU(device=device),
            Linear(8192, 10, device=device),
        ]

    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def backward(self, dLdout: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[::-1]:
            dLdout = layer.backward(dLdout)
        return dLdout

    def __repr__(self) -> str:
        repr_str = "Network(\n"
        for layer in self.layers:
            repr_str += "    " + repr(layer) + "\n"
        repr_str += ')'
        return repr_str


if __name__ == "__main__":
    model = Network(input_dim=64, device=device)
    input = torch.randn(2, 64, device=device)
    print(model)
    print('output=\n', model(input))
    assert model.backward(torch.randn(2, 10, device=device)).shape == input.shape
    print('backward works!')