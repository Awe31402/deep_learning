import copy
import math
import random
import time
from collections import defaultdict, OrderedDict
from typing import Union, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm

assert torch.cuda.is_available(), \
    "CUDA is not available. Please check your GPU settings."

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def download_url(url, model_dir='.', overwrite=False):
    import os, sys, ssl
    from urllib.request import urlretrieve
    ssl.create_default_context = ssl._create_unverified_context
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)

    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            print(f"Downloading {url} to {cached_file}")
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        os.remove(os.path.join(model_dir, 'download.lock'))
        sys.stderr.write(f"Error downloading {url}: {e}\n")
        return None

# VGG model
class VGG(nn.Module):
    ARCH = [64, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self) -> None:
        super().__init__()
        layers = []
        counts = defaultdict(int)

        def add(name: str, layer: nn.Module) -> None:
            layers.append((f"{name}{counts[name]}", layer))
            counts[name] += 1

        in_channels = 3
        for x in self.ARCH:
            if x != 'M':
                add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
                add("bn", nn.BatchNorm2d(x))
                add("relu", nn.ReLU(inplace=True))
                in_channels = x
            else:
                add("pool", nn.MaxPool2d(2))

        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)

        # average pooling [N, 512, 2, 2] => [N, 512]
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def train(
        model: nn.Module,
        dataflow: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LambdaLR,
        callbacks = None
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

        if callbacks is not None:
            for callback in callbacks:
                callback()

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

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

def get_sparisty(tensor: torch.Tensor) -> float:
    return 1 - float(tensor.count_nonzero() / tensor.numel())

def get_model_sparsity(model: nn.Module) -> float:
    num_nonzeros = 0
    num_elements = 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero().item()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only = False) -> int:
    num_params = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_params += param.count_nonzero().item()
        else:
            num_params += param.numel()
    return num_params

def get_model_size(model: nn.Module, data_width=32,count_nonzero_only = False) -> float:
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def fine_grained_pruning(tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    實作細粒度大小剪枝
    """
    # 步驟 1: 計算剪枝後的零數 (num_zeros)
    num_elements = tensor.numel()
    num_zeros = int(round(num_elements * target_sparsity))

    # 處理邊界條件
    if num_zeros == 0:
        mask = torch.ones_like(tensor, dtype=torch.bool)
    elif num_zeros >= num_elements:
        mask = torch.zeros_like(tensor, dtype=torch.bool)
    else:
        # 步驟 2: 計算權重張量的「重要性」
        importance = torch.abs(tensor)

        # 步驟 3: 計算修剪「閾值」
        threshold = torch.kthvalue(importance.view(-1), num_zeros).values

        # 步驟 4: 根據「閾值」計算剪枝「掩碼」 (mask)
        mask = torch.gt(importance, threshold)

    # 原處將權重乘以遮罩以應用剪枝 (masked_fill_)
    tensor.masked_fill_(~mask, 0)

    return mask

def test_fine_grained_prune(
        test_tensor=torch.tensor([[-0.46, -0.40, 0.39, 0.19, 0.37],
                              [0.00, 0.40, 0.17, -0.15, 0.16],
                              [-0.20, -0.23, 0.36, 0.25, 0.03],
                              [0.24, 0.41, 0.07, 0.13, -0.15],
                              [0.48, -0.09, -0.36, 0.12, 0.45]]),
        test_mask=torch.tensor([[True, True, False, False, False],
                            [False, True, False, False, False],
                            [False, False, False, False, False],
                            [False, True, False, False, False],
                            [True, False, False, False, True]]),
        target_sparsity=0.75,
        target_nonzeros=None):
    def plot_matrix(tensor, ax, title):
        ax.imshow(tensor.cpu().numpy()==0, vmin=0, vmax=1, cmap="tab20c")
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(i, j, f"{tensor[j, i]:.2f}",
                               ha="center", va="center", color="black")

    test_tensor = test_tensor.clone()
    fig, axes = plt.subplots(1, 2, figsize=(6, 10))
    ax_left, ax_right = axes.ravel()
    plot_matrix(test_tensor, ax_left, "Original Tensor")

    sparsity_before_pruning = get_sparisty(test_tensor)
    mask = fine_grained_pruning(test_tensor, target_sparsity)
    sparsity_after_pruning = get_sparisty(test_tensor)
    sparsity_of_mask = get_sparisty(mask)

    plot_matrix(test_tensor, ax_right, "Pruned Tensor")
    fig.tight_layout()
    plt.show()

    print('* Test fine_grained_prune()')
    print(f'    target sparsity: {target_sparsity:.2f}')
    print(f'        sparsity before pruning: {sparsity_before_pruning:.2f}')
    print(f'        sparsity after pruning: {sparsity_after_pruning:.2f}')
    print(f'        sparsity of pruning mask: {sparsity_of_mask:.2f}')

    if target_nonzeros is None:
        if test_mask.equal(mask):
            print('* Test passed.')
        else:
            print('* Test failed.')
    else:
        if mask.count_nonzero() == target_nonzeros:
            print('* Test passed.')
        else:
            print('* Test failed.')

def plot_weight_distribution(model, output_fig_name="weight_distribution.png",
                             bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(3,3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().view(-1).cpu().numpy()
                param_cpu = param_cpu[param_cpu != 0]
                ax.hist(param_cpu, bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().view(-1).cpu().numpy(), bins=bins, density=True,
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(output_fig_name)


class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                masks[name] = fine_grained_pruning(param, sparsity_dict[name])
        return masks

@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                          in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            fine_grained_pruning(param.detach(), target_sparsity=sparsity)
            acc = evaluate(model, dataloader)
            if verbose:
                print(f'\rsparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(f'\rsparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]')
        accuracies.append(accuracy)
    return sparsities, accuracies

def plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy, output_fig_name="sensitivity_curves.png"):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)),figsize=(15,8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(80, 95)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend([
                'accuracy after pruning',
                f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
            ])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(output_fig_name)

def plot_num_parameters_distribution(model, filename="num_parameters_distribution.png"):
    num_parameters = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            num_parameters[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    # get pretrained model
    checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
    checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
    model = VGG().cuda()
    # model = VGG()
    print(f"=> loading checkpoint '{checkpoint_url}'")
    model.load_state_dict(checkpoint['state_dict'])
    recover_model = lambda: model.load_state_dict(checkpoint['state_dict'])

    # cifar10 dataset
    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=512,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )

    # Observe weighting distribution of the pretrained model
    plot_weight_distribution(model)

    # test fine grained pruning
    test_fine_grained_prune()

    # Question 3
    print("\n--- Question 3 ---")
    target_sparsity = 0.60
    test_fine_grained_prune(target_sparsity=target_sparsity, target_nonzeros=10)

    # Cells 46 to 54
    print("\n--- Sensitivity Scan (Cells 51-54) ---")
    dense_model_accuracy = evaluate(model, dataloader['test'])
    print(f"dense model has accuracy={dense_model_accuracy:.2f}%")

    sparsities, accuracies = sensitivity_scan(
        model, dataloader['test'], scan_step=0.1, scan_start=0.4, scan_end=1.0)
    
    plot_sensitivity_scan(model, sparsities, accuracies, dense_model_accuracy)
    print("Sensitivity curves saved as sensitivity_curves.png")

    plot_num_parameters_distribution(model)

    # Question 5
    print("\n--- Question 5 ---")
    recover_model()

    sparsity_dict = {
        'backbone.conv0.weight': 0.15,
        'backbone.conv1.weight': 0.40,
        'backbone.conv2.weight': 0.60,
        'backbone.conv3.weight': 0.70,
        'backbone.conv4.weight': 0.75,
        'backbone.conv5.weight': 0.85,
        'backbone.conv6.weight': 0.85,
        'backbone.conv7.weight': 0.85,
        'classifier.weight': 0.40,
    }

    pruner = FineGrainedPruner(model, sparsity_dict)
    print(f'After pruning with sparsity dictionary')
    for name, sparsity in sparsity_dict.items():
        print(f'  {name}: {sparsity:.2f}')
    print(f'The sparsity of each layer becomes')
    for name, param in model.named_parameters():
        if name in sparsity_dict:
            print(f'  {name}: {get_sparisty(param):.2f}')

    dense_model_size = get_model_size(model)
    sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
    sparse_model_accuracy = evaluate(model, dataloader['test'])
    print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% before fine-tuning")

    # Fine-tune the model
    num_finetune_epochs = 5
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    criterion = nn.CrossEntropyLoss()

    best_sparse_model_checkpoint = dict()
    best_accuracy = 0
    print(f'Finetuning Fine-grained Pruned Sparse Model')
    for epoch in range(num_finetune_epochs):
        train(model, dataloader['train'], criterion, optimizer, scheduler,
              callbacks=[lambda: pruner.apply(model)])
        accuracy = evaluate(model, dataloader['test'])
        is_best = accuracy > best_accuracy
        if is_best:
            best_sparse_model_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            best_accuracy = accuracy
        print(f'    Epoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')

    # Load the best sparse model checkpoint and show final accuracy
    model.load_state_dict(best_sparse_model_checkpoint['state_dict'])
    final_sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"Sparse model final size={final_sparse_model_size / MiB:.2f} MiB = {final_sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
    final_sparse_model_accuracy = evaluate(model, dataloader['test'])
    print(f"Sparse model accuracy after fine-tuning: {final_sparse_model_accuracy:.2f}%")