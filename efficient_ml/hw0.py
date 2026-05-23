import random
import numpy as np
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

# To ensure reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

transforms = {
    "train": Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
    ]),
    "test": ToTensor(),
}

dataset = {}
for split in ["train", "test"]:
    dataset[split] = CIFAR10(root="./data",
                             train=(split == "train"),
                             download=True,
                             transform=transforms[split])

samples = [[] for _ in range(10)]
print(dataset)
for img, label in dataset["test"]:
    if len(samples[label]) < 4:
        samples[label].append(img)

plt.figure(figsize=(20, 9))
for index in range(40):
    label = index % 10
    image = samples[label][index // 10]

    # Convert the image from (C, H, W) to (H, W, C) for visualization
    image = image.permute(1, 2, 0)
    label = dataset["test"].classes[label]

    plt.subplot(4, 10, index + 1)
    plt.imshow(image)
    plt.title(label)
    plt.axis("off")
plt.show()

dataflow = {}
for split in ["train", "test"]:
    dataflow[split] = DataLoader(dataset[split],
                                 batch_size=512,
                                 shuffle=(split == "train"),
                                 num_workers=0,
                                 pin_memory=True)

for inputs, targets in dataflow["train"]:
    print(f"Input batch shape: {inputs.shape}, dtype: {inputs.dtype}")
    print(f"Target batch shape: {targets.shape}, dtype: {targets.dtype}")
    break

# VGG model
class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()
        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}_{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                add("conv", nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(inplace=True))
                in_channels = x
            else:
                add("pool", nn.MaxPool2d(kernel_size=2, stride=2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        # average pooling [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


model = VGG().cuda()
print(model.backbone)
print(model.classifier)

num_params = 0
for param in model.parameters():
    if param.requires_grad:
        num_params += param.numel()
print(f"Total number of trainable parameters: {num_params}")

# what does MACs mean?
# MACs stands for Multiply-Accumulate Operations,
# which is a common metric used to measure the
# computational complexity of a neural network.
# It counts the number of times a multiplication
# and an addition are performed during a forward pass through the network.
# This metric is particularly important for understanding the efficiency of a model,
# especially when deploying it on resource-constrained devices.
num_macs = profile_macs(model, torch.zeros(1, 3, 32, 32).cuda())
print(f"Total number of MACs for a single forward pass: {num_macs}")

criterion = nn.CrossEntropyLoss()

optimizer = SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=5e-4)

num_epochs = 20
steps_per_epoch = len(dataflow["train"])

lr_lambda = lambda step: np.interp(
    [step / steps_per_epoch],
    [0, num_epochs * 0.3, num_epochs],
    [0, 1, 0]
)[0]

steps = np.arange(num_epochs * steps_per_epoch)
plt.plot(steps, [lr_lambda(step) * 0.4 for step in steps])
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid("on")
plt.show()

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

def train(
        model: nn.Module,
        dataflow: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
) -> None:
    model.train()
    total_loss = 0.0
    for inputs, targets in tqdm(dataflow, desc="Training", leave=False):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(dataflow.dataset)
    print(f"Average training loss: {avg_loss:.4f}")

@torch.inference_mode()
def evaluate(model: nn.Module, dataflow: DataLoader) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in tqdm(dataflow, desc="eval", leave=False):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)

        num_samples += targets.size(0)
        num_correct += (predictions == targets).sum().item()
    return (num_correct / num_samples * 100)


for epoch in tqdm(range(1, num_epochs + 1)):
    train(model, dataflow["train"], criterion, optimizer, scheduler)
    accuracy = evaluate(model, dataflow["test"])
    print(f"Test Accuracy: {accuracy:.2f}%")

plt.figure(figsize=(20, 10))
for index in range(40):
    img, label = dataset["test"][index]
    model.eval()
    with torch.inference_mode():
        pred = model(img.unsqueeze(0).cuda()).argmax(dim=1)

    img = img.permute(1, 2, 0)
    pred = dataset["test"].classes[pred]
    label = dataset["test"].classes[label]

    plt.subplot(4, 10, index + 1)
    plt.imshow(img)
    plt.title(f"GT: {label}\nPred: {pred}")
    plt.axis("off")
plt.show()