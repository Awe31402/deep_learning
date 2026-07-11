# **MIT 6.5940 EfficientML.ai Lab 2: Quantization (量化)**

本 Colab 筆記本提供了 Lab 2 量化的程式碼和框架。你可以在這裡完成你的解答。

完成此實驗後，請填寫此 [回饋表單](https://forms.gle/Q17ttphz5wLQQcRn9)。我們很想聽聽你的想法或關於我們如何改進此實驗的回饋！

## 實驗目標

在本次作業中，你將練習對經典的神經網路模型進行量化，以減少模型大小和延遲。本次作業的目標如下：

- 理解**量化**的基本概念
- 實作並應用 **K-Means 量化**
- 針對 K-Means 量化實作並應用**量化感知訓練 (Quantization-Aware Training)**
- 實作並應用**線性量化 (Linear Quantization)**
- 針對線性量化實作並應用**僅整數推論 (Integer-Only Inference)**
- 基本了解量化帶來的效能提升（例如加速）
- 理解這些量化方法之間的差異與權衡

## 目錄

主要包含兩個部分：***K-Means 量化***與***線性量化***。

總共有 ***10*** 個問題：
- *K-Means 量化* 部分有 ***3*** 個問題（問題 1-3）。
- *線性量化* 部分有 ***6*** 個問題（問題 4-9）。
- 問題 10 比較了 K-Means 量化與線性量化。

# 環境設定

首先，安裝所需的套件並下載數據集和預訓練模型。這裡我們使用 CIFAR10 數據集和 VGG 網路，這與我們在 Lab 0 教學中使用的是相同的。




```python
print('Installing torchprofile...')
!pip install torchprofile 1>/dev/null
print('Installing fast-pytorch-kmeans...')
! pip install fast-pytorch-kmeans 1>/dev/null
print('All required packages have been successfully installed!')
```

    Installing torchprofile...
    Installing fast-pytorch-kmeans...
    All required packages have been successfully installed!



```python
import copy
import math
import random
from collections import OrderedDict, defaultdict

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *

from torchprofile import profile_macs

assert torch.cuda.is_available(), \
"The current runtime does not have CUDA support." \
"Please go to menu bar (Runtime - Change runtime type) and select GPU"
```


```python
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
```




    <torch._C.Generator at 0x7ef9740f31b0>




```python
def download_url(url, model_dir='.', overwrite=False):
    import os, sys
    from urllib.request import urlretrieve
    target_dir = url.split('/')[-1]
    model_dir = os.path.expanduser(model_dir)
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, target_dir)
        cached_file = model_dir
        if not os.path.exists(cached_file) or overwrite:
            sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
            urlretrieve(url, cached_file)
        return cached_file
    except Exception as e:
        # 移除 lock 檔案，以便下次執行下載。
        os.remove(os.path.join(model_dir, 'download.lock'))
        sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
        return None
```


```python
class VGG(nn.Module):
  ARCH = [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

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
        # conv-bn-relu
        add("conv", nn.Conv2d(in_channels, x, 3, padding=1, bias=False))
        add("bn", nn.BatchNorm2d(x))
        add("relu", nn.ReLU(True))
        in_channels = x
      else:
        # maxpool
        add("pool", nn.MaxPool2d(2))
    add("avgpool", nn.AvgPool2d(2))
    self.backbone = nn.Sequential(OrderedDict(layers))
    self.classifier = nn.Linear(512, 10)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # backbone: [N, 3, 32, 32] => [N, 512, 2, 2]
    x = self.backbone(x)

    # avgpool: [N, 512, 2, 2] => [N, 512]
    # x = x.mean([2, 3])
    x = x.view(x.shape[0], -1)

    # classifier: [N, 512] => [N, 10]
    x = self.classifier(x)
    return x
```


```python
def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None
) -> None:
  model.train()

  for inputs, targets in tqdm(dataloader, desc='train', leave=False):
    # 將數據從 CPU 移動到 GPU
    inputs = inputs.cuda()
    targets = targets.cuda()

    # 重設梯度（清除上一次迭代的梯度）
    optimizer.zero_grad()

    # 前向推論
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # 反向傳播
    loss.backward()

    # 更新優化器和學習率排程器
    optimizer.step()
    scheduler.step()

    if callbacks is not None:
        for callback in callbacks:
            callback()
```


```python
@torch.inference_mode()
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  extra_preprocess = None
) -> float:
  model.eval()

  num_samples = 0
  num_correct = 0

  for inputs, targets in tqdm(dataloader, desc="eval", leave=False):
    # 將數據從 CPU 移動到 GPU
    inputs = inputs.cuda()
    if extra_preprocess is not None:
        for preprocess in extra_preprocess:
            inputs = preprocess(inputs)

    targets = targets.cuda()

    # 推論
    outputs = model(inputs)

    # 將對數機率 (logits) 轉換為類別索引
    outputs = outputs.argmax(dim=1)

    # 更新指標
    num_samples += targets.size(0)
    num_correct += (outputs == targets).sum()

  return (num_correct / num_samples * 100).item()
```

輔助函數（計算 FLOPs、模型大小等）


```python
def get_model_flops(model, inputs):
    num_macs = profile_macs(model, inputs)
    return num_macs
```


```python
def get_model_size(model: nn.Module, data_width=32):
    """
    計算模型大小（以位元為單位）
    :param data_width: 每個元素佔用的位元數
    """
    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
```

定義用於驗證的雜項函數。


```python
def test_k_means_quantize(
    test_tensor=torch.tensor([
        [-0.3747,  0.0874,  0.3200, -0.4868,  0.4404],
        [-0.0402,  0.2322, -0.2024, -0.4986,  0.1814],
        [ 0.3102, -0.3942, -0.2030,  0.0883, -0.4741],
        [-0.1592, -0.0777, -0.3946, -0.2128,  0.2675],
        [ 0.0611, -0.1933, -0.4350,  0.2928, -0.1087]]),
    bitwidth=2):
    def plot_matrix(tensor, ax, title, cmap=ListedColormap(['white'])):
        ax.imshow(tensor.cpu().numpy(), vmin=-0.5, vmax=0.5, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(j, i, f'{tensor[i, j].item():.2f}',
                                ha="center", va="center", color="k")

    fig, axes = plt.subplots(1,2, figsize=(8, 12))
    ax_left, ax_right = axes.ravel()

    print(test_tensor)
    plot_matrix(test_tensor, ax_left, 'original tensor')

    num_unique_values_before_quantization = test_tensor.unique().numel()
    k_means_quantize(test_tensor, bitwidth=bitwidth)
    num_unique_values_after_quantization = test_tensor.unique().numel()
    print('* Test k_means_quantize()')
    print(f'    target bitwidth: {bitwidth} bits')
    print(f'        num unique values before k-means quantization: {num_unique_values_before_quantization}')
    print(f'        num unique values after  k-means quantization: {num_unique_values_after_quantization}')
    assert num_unique_values_after_quantization == min((1 << bitwidth), num_unique_values_before_quantization)
    print('* Test passed.')

    plot_matrix(test_tensor, ax_right, f'{bitwidth}-bit k-means quantized tensor', cmap='tab20c')
    fig.tight_layout()
    plt.show()
```


```python
def test_linear_quantize(
    test_tensor=torch.tensor([
        [ 0.0523,  0.6364, -0.0968, -0.0020,  0.1940],
        [ 0.7500,  0.5507,  0.6188, -0.1734,  0.4677],
        [-0.0669,  0.3836,  0.4297,  0.6267, -0.0695],
        [ 0.1536, -0.0038,  0.6075,  0.6817,  0.0601],
        [ 0.6446, -0.2500,  0.5376, -0.2226,  0.2333]]),
    quantized_test_tensor=torch.tensor([
        [-1,  1, -1, -1,  0],
        [ 1,  1,  1, -2,  0],
        [-1,  0,  0,  1, -1],
        [-1, -1,  1,  1, -1],
        [ 1, -2,  1, -2,  0]], dtype=torch.int8),
    real_min=-0.25, real_max=0.75, bitwidth=2, scale=1/3, zero_point=-1):
    def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=ListedColormap(['white'])):
        ax.imshow(tensor.cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                datum = tensor[i, j].item()
                if isinstance(datum, float):
                    text = ax.text(j, i, f'{datum:.2f}',
                                    ha="center", va="center", color="k")
                else:
                    text = ax.text(j, i, f'{datum}',
                                    ha="center", va="center", color="k")
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fig, axes = plt.subplots(1,3, figsize=(10, 32))
    plot_matrix(test_tensor, axes[0], 'original tensor', vmin=real_min, vmax=real_max)
    _quantized_test_tensor = linear_quantize(
        test_tensor, bitwidth=bitwidth, scale=scale, zero_point=zero_point)
    _reconstructed_test_tensor = scale * (_quantized_test_tensor.float() - zero_point)
    print('* Test linear_quantize()')
    print(f'    target bitwidth: {bitwidth} bits')
    print(f'        scale: {scale}')
    print(f'        zero point: {zero_point}')
    assert _quantized_test_tensor.equal(quantized_test_tensor)
    print('* Test passed.')
    plot_matrix(_quantized_test_tensor, axes[1], f'2-bit linear quantized tensor',
                vmin=quantized_min, vmax=quantized_max, cmap='tab20c')
    plot_matrix(_reconstructed_test_tensor, axes[2], f'reconstructed tensor',
                vmin=real_min, vmax=real_max, cmap='tab20c')
    fig.tight_layout()
    plt.show()

```

```python
def test_quantized_fc(
    input=torch.tensor([
        [0.6118, 0.7288, 0.8511, 0.2849, 0.8427, 0.7435, 0.4014, 0.2794],
        [0.3676, 0.2426, 0.1612, 0.7684, 0.6038, 0.0400, 0.2240, 0.4237],
        [0.6565, 0.6878, 0.4670, 0.3470, 0.2281, 0.8074, 0.0178, 0.3999],
        [0.1863, 0.3567, 0.6104, 0.0497, 0.0577, 0.2990, 0.6687, 0.8626]]),
    weight=torch.tensor([
        [ 1.2626e-01, -1.4752e-01,  8.1910e-02,  2.4982e-01, -1.0495e-01,
          -1.9227e-01, -1.8550e-01, -1.5700e-01],
        [ 2.7624e-01, -4.3835e-01,  5.1010e-02, -1.2020e-01, -2.0344e-01,
           1.0202e-01, -2.0799e-01,  2.4112e-01],
        [-3.8216e-01, -2.8047e-01,  8.5238e-02, -4.2504e-01, -2.0952e-01,
           3.2018e-01, -3.3619e-01,  2.0219e-01],
        [ 8.9233e-02, -1.0124e-01,  1.1467e-01,  2.0091e-01,  1.1438e-01,
          -4.2427e-01,  1.0178e-01, -3.0941e-04],
        [-1.8837e-02, -2.1256e-01, -4.5285e-01,  2.0949e-01, -3.8684e-01,
          -1.7100e-01, -4.5331e-01, -2.0433e-01],
        [-2.0038e-01, -5.3757e-02,  1.8997e-01, -3.6866e-01,  5.5484e-02,
           1.5643e-01, -2.3538e-01,  2.1103e-01],
        [-2.6875e-01,  2.4984e-01, -2.3514e-01,  2.5527e-01,  2.0322e-01,
           3.7675e-01,  6.1563e-02,  1.7201e-01],
        [ 3.3541e-01, -3.3555e-01, -4.3349e-01,  4.3043e-01, -2.0498e-01,
          -1.8366e-01, -9.1553e-02, -4.1168e-01]]),
    bias=torch.tensor([ 0.1954, -0.2756,  0.3113,  0.1149,  0.4274,  0.2429, -0.1721, -0.2502]),
    quantized_bias=torch.tensor([ 3, -2,  3,  1,  3,  2, -2, -2], dtype=torch.int32),
    shifted_quantized_bias=torch.tensor([-1,  0, -3, -1, -3,  0,  2, -4], dtype=torch.int32),
    calc_quantized_output=torch.tensor([
        [ 0, -1,  0, -1, -1,  0,  1, -2],
        [ 0,  0, -1,  0,  0,  0,  0, -1],
        [ 0,  0,  0, -1,  0,  0,  0, -1],
        [ 0,  0,  0,  0,  0,  1, -1, -2]], dtype=torch.int8),
    bitwidth=2, batch_size=4, in_channels=8, out_channels=8):
    def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=ListedColormap(['white'])):
        ax.imshow(tensor.cpu().numpy(), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                datum = tensor[i, j].item()
                if isinstance(datum, float):
                    text = ax.text(j, i, f'{datum:.2f}',
                                    ha="center", va="center", color="k")
                else:
                    text = ax.text(j, i, f'{datum}',
                                    ha="center", va="center", color="k")

    output = torch.nn.functional.linear(input, weight, bias)

    quantized_weight, weight_scale, weight_zero_point = \
        linear_quantize_weight_per_channel(weight, bitwidth)
    quantized_input, input_scale, input_zero_point = \
        linear_quantize_feature(input, bitwidth)
    _quantized_bias, bias_scale, bias_zero_point = \
        linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale)
    assert _quantized_bias.equal(_quantized_bias)
    _shifted_quantized_bias = \
        shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point)
    assert _shifted_quantized_bias.equal(shifted_quantized_bias)
    quantized_output, output_scale, output_zero_point = \
        linear_quantize_feature(output, bitwidth)

    _calc_quantized_output = quantized_linear(
        quantized_input, quantized_weight, shifted_quantized_bias,
        bitwidth, bitwidth,
        input_zero_point, output_zero_point,
        input_scale, weight_scale, output_scale)
    assert _calc_quantized_output.equal(calc_quantized_output)

    reconstructed_weight = weight_scale * (quantized_weight.float() - weight_zero_point)
    reconstructed_input = input_scale * (quantized_input.float() - input_zero_point)
    reconstructed_bias = bias_scale * (quantized_bias.float() - bias_zero_point)
    reconstructed_calc_output = output_scale * (calc_quantized_output.float() - output_zero_point)

    fig, axes = plt.subplots(3,3, figsize=(15, 12))
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    plot_matrix(weight, axes[0, 0], 'original weight', vmin=-0.5, vmax=0.5)
    plot_matrix(input.t(), axes[1, 0], 'original input', vmin=0, vmax=1)
    plot_matrix(output.t(), axes[2, 0], 'original output', vmin=-1.5, vmax=1.5)
    plot_matrix(quantized_weight, axes[0, 1], f'{bitwidth}-bit linear quantized weight',
                vmin=quantized_min, vmax=quantized_max, cmap='tab20c')
    plot_matrix(quantized_input.t(), axes[1, 1], f'{bitwidth}-bit linear quantized input',
                vmin=quantized_min, vmax=quantized_max, cmap='tab20c')
    plot_matrix(calc_quantized_output.t(), axes[2, 1], f'quantized output from quantized_linear()',
                vmin=quantized_min, vmax=quantized_max, cmap='tab20c')
    plot_matrix(reconstructed_weight, axes[0, 2], f'reconstructed weight',
                vmin=-0.5, vmax=0.5, cmap='tab20c')
    plot_matrix(reconstructed_input.t(), axes[1, 2], f'reconstructed input',
                vmin=0, vmax=1, cmap='tab20c')
    plot_matrix(reconstructed_calc_output.t(), axes[2, 2], f'reconstructed output',
                vmin=-1.5, vmax=1.5, cmap='tab20c')

    print('* Test quantized_fc()')
    print(f'    target bitwidth: {bitwidth} bits')
    print(f'      batch size: {batch_size}')
    print(f'      input channels: {in_channels}')
    print(f'      output channels: {out_channels}')
    print('* Test passed.')
    fig.tight_layout()
    plt.show()
```

載入預訓練模型


```python
checkpoint_url = "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth"
checkpoint = torch.load(download_url(checkpoint_url), map_location="cpu")
model = VGG().cuda()
print(f"=> loading checkpoint '{checkpoint_url}'")
model.load_state_dict(checkpoint['state_dict'])
recover_model = lambda : model.load_state_dict(checkpoint['state_dict'])
```

    Downloading: "https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth" to ./vgg.cifar.pretrained.pth


    => loading checkpoint 'https://hanlab18.mit.edu/files/course/labs/vgg.cifar.pretrained.pth'



```python
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
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to data/cifar10/cifar-10-python.tar.gz


    100%|██████████| 170498071/170498071 [00:20<00:00, 8311603.99it/s]


    Extracting data/cifar10/cifar-10-python.tar.gz to data/cifar10
    Files already downloaded and verified


# 讓我們首先評估 FP32 模型的準確度和模型大小


```python
fp32_model_accuracy = evaluate(model, dataloader['test'])
fp32_model_size = get_model_size(model)
print(f"fp32 model has accuracy={fp32_model_accuracy:.2f}%")
print(f"fp32 model has size={fp32_model_size/MiB:.2f} MiB")
```

# K-Means 量化

網路量化藉由減少表示深度網路所需的權重位元數來壓縮網路。在硬體支援下，量化後的網路可以擁有更快的推論速度。

在本節中，我們將探討神經網路的 K-Means 量化，詳見論文 [Deep Compression: Compressing Deep Neural Networks With Pruning, Trained Quantization And Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf)。

![kmeans.png](data:image/png;base64,...)

一個 $n$-bit 的 K-Means 量化會將突觸（權重）劃分為 $2^n$ 個群集 (clusters)，且同一個群集中的突觸將共享相同的權重值。

因此，K-Means 量化將建立一個密碼本 (codebook)，包括：
*   `centroids`：$2^n$ 個 fp32 的群集中心。
*   `labels`：一個與原始 fp32 權重張量具有相同元素數量的 $n$-bit 整數張量。每個整數指示它屬於哪一個群集。

在推論期間，會根據密碼本生成一個 fp32 張量來進行推論：

> ***quantized_weight* = *codebook.centroids*\[*codebook.labels*\].view_as(weight)**


```python
from collections import namedtuple

Codebook = namedtuple('Codebook', ['centroids', 'labels'])
```

## 問題 1 (10 分)

請完成以下 K-Means 量化函數。


```python
from fast_pytorch_kmeans import KMeans

def k_means_quantize(fp32_tensor: torch.Tensor, bitwidth=4, codebook=None):
    """
    使用 K-Means 分群量化張量
    :param fp32_tensor:
    :param bitwidth: [int] 量化位元寬度，預設=4
    :param codebook: [Codebook] (群集質心, 群集標籤張量)
    :return:
        [Codebook = (centroids, labels)]
            centroids: [torch.(cuda.)FloatTensor] 群集質心
            labels: [torch.(cuda.)LongTensor] 群集標籤張量
    """
    if codebook is None:
        ############### 你的程式碼從這裡開始 ###############
        # 根據量化精度獲取群集數量
        # 提示：一行程式碼
        n_clusters = 2 ** bitwidth
        ############### 你的程式碼在這裡結束 #################
        # 使用 K-Means 獲取量化質心
        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
        labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
        centroids = kmeans.centroids.to(torch.float).view(-1)
        codebook = Codebook(centroids, labels)
    ############### 你的程式碼從這裡開始 ###############
    # 將密碼本解碼為 K-Means 量化張量以進行推論
    # 提示：一行程式碼
    quantized_tensor = codebook.centroids[codebook.labels]
    ############### 你的程式碼在這裡結束 #################
    fp32_tensor.set_(quantized_tensor.view_as(fp32_tensor))
    return codebook
```

讓我們透過在虛擬張量上應用上述函數，來驗證定義的 K-Means 量化的功能。


```python
test_k_means_quantize()
```

## 問題 2 (10 分)

最後一個程式碼儲存格執行了 2-bit 的 K-Means 量化，並繪製了量化前和量化後的張量。每個群集都以唯一的顏色呈現。量化後的張量中呈現了 4 種唯一的顏色。

根據此觀察，請回答以下問題。

### 問題 2.1 (5 分)

如果執行 4-bit 的 K-Means 量化，量化後的張量中將呈現多少種唯一的顏色？

**你的答案：**
16

### 問題 2.2 (5 分)

如果執行 *n*-bit 的 K-Means 量化，量化後的張量中將呈現多少種唯一的顏色？

**你的答案：**
2^n

## 對整個模型進行 K-Means 量化

類似於我們在 Lab 1 中所做的工作，我們現在將 K-Means 量化函數包裝到一個類別中，以便對整個模型進行量化。在 `KMeansQuantizer` 類別中，我們必須記錄密碼本（即 `centroids` 和 `labels`），以便在模型權重發生變化時，能夠套用或更新密碼本。


```python
from torch.nn import parameter
class KMeansQuantizer:
    def __init__(self, model : nn.Module, bitwidth=4):
        self.codebook = KMeansQuantizer.quantize(model, bitwidth)

    @torch.no_grad()
    def apply(self, model, update_centroids):
        for name, param in model.named_parameters():
            if name in self.codebook:
                if update_centroids:
                    update_codebook(param, codebook=self.codebook[name])
                self.codebook[name] = k_means_quantize(
                    param, codebook=self.codebook[name])

    @staticmethod
    @torch.no_grad()
    def quantize(model: nn.Module, bitwidth=4):
        codebook = dict()
        if isinstance(bitwidth, dict):
            for name, param in model.named_parameters():
                if name in bitwidth:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth[name])
        else:
            for name, param in model.named_parameters():
                if param.dim() > 1:
                    codebook[name] = k_means_quantize(param, bitwidth=bitwidth)
        return codebook
```

現在，讓我們使用 K-Means 量化將模型量化為 8 bits、4 bits 和 2 bits。*請注意，在計算模型大小時，我們忽略了密碼本的存儲空間。*


```python
print('請注意，計算模型大小時會忽略密碼本的存儲空間。')
quantizers = dict()
for bitwidth in [8, 4, 2]:
    recover_model()
    print(f'k-means quantizing model into {bitwidth} bits')
    quantizer = KMeansQuantizer(model, bitwidth)
    quantized_model_size = get_model_size(model, bitwidth)
    print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size/MiB:.2f} MiB")
    quantized_model_accuracy = evaluate(model, dataloader['test'])
    print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}%")
    quantizers[bitwidth] = quantizer
```

## 受訓的 K-Means 量化 (Trained K-Means Quantization)

正如我們從上一個儲存格的結果中所看到的，當將模型量化為較低的位元數時，準確度會顯著下降。因此，我們必須進行量化感知訓練 (quantization-aware training) 來恢復準確度。

在 K-Means 量化感知訓練期間，質心（centroids）也會被更新，這是在論文 [Deep Compression: Compressing Deep Neural Networks With Pruning, Trained Quantization And Huffman Coding](https://arxiv.org/pdf/1510.00149.pdf) 中提出的。

質心的梯度計算如下：
> $\frac{\partial \mathcal{L} }{\partial C_k} = \sum_{j} \frac{\partial \mathcal{L} }{\partial W_{j}} \frac{\partial W_{j} }{\partial C_k} = \sum_{j} \frac{\partial \mathcal{L} }{\partial W_{j}} \mathbf{1}(I_{j}=k)$

其中 $\mathcal{L}$ 是損失 (loss)，$C_k$ 是第 *k* 個質心，$I_{j}$ 是權重 $W_{j}$ 的標籤。$\mathbf{1}()$ 是指示函數，而 $\mathbf{1}(I_{j}=k)$ 代表 $I_{j}=k$ 時為 $1$，否則為 $0$（即 $I_{j}==k$）。

在本實驗中，**為了簡化**，我們直接根據最新權重來更新質心：

> $C_k = \frac{\sum_{j}W_{j}\mathbf{1}(I_{j}=k)}{\sum_{j}\mathbf{1}(I_{j}=k)}$

### 問題 3 (10 分)

請完成以下密碼本更新函數。

**提示**：

上述更新質心的等式實際上是使用同一群集中權重的 `mean`（平均值）作為更新後的質心值。


```python
def update_codebook(fp32_tensor: torch.Tensor, codebook: Codebook):
    """
    使用更新後的 fp32_tensor 更新密碼本中的質心
    :param fp32_tensor: [torch.(cuda.)Tensor]
    :param codebook: [Codebook] (群集質心, 群集標籤張量)
    """
    n_clusters = codebook.centroids.numel()
    fp32_tensor = fp32_tensor.view(-1)
    for k in range(n_clusters):
    ############### 你的程式碼從這裡開始 ###############
        # 提示：一行程式碼
        codebook.centroids[k] = fp32_tensor[codebook.labels == k].mean()
    ############### 你的程式碼在這裡結束 #################
```

現在，讓我們運行以下程式碼儲存格以微調 K-Means 量化模型以恢復準確度。如果準確度下降小於 0.5，我們將停止微調。


```python
accuracy_drop_threshold = 0.5
quantizers_before_finetune = copy.deepcopy(quantizers)
quantizers_after_finetune = quantizers

for bitwidth in [8, 4, 2]:
    recover_model()
    quantizer = quantizers[bitwidth]
    print(f'k-means quantizing model into {bitwidth} bits')
    quantizer.apply(model, update_centroids=False)
    quantized_model_size = get_model_size(model, bitwidth)
    print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size/MiB:.2f} MiB")
    quantized_model_accuracy = evaluate(model, dataloader['test'])
    print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}% before quantization-aware training ")
    accuracy_drop = fp32_model_accuracy - quantized_model_accuracy
    if accuracy_drop > accuracy_drop_threshold:
        print(f"        Quantization-aware training due to accuracy drop={accuracy_drop:.2f}% is larger than threshold={accuracy_drop_threshold:.2f}%")
        num_finetune_epochs = 5
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0
        epoch = num_finetune_epochs
        while accuracy_drop > accuracy_drop_threshold and epoch > 0:
            train(model, dataloader['train'], criterion, optimizer, scheduler,
                  callbacks=[lambda: quantizer.apply(model, update_centroids=True)])
            model_accuracy = evaluate(model, dataloader['test'])
            is_best = model_accuracy > best_accuracy
            best_accuracy = max(model_accuracy, best_accuracy)
            print(f'        Epoch {num_finetune_epochs-epoch} Accuracy {model_accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')
            accuracy_drop = fp32_model_accuracy - best_accuracy
            epoch -= 1
    else:
        print(f"        No need for quantization-aware training since accuracy drop={accuracy_drop:.2f}% is smaller than threshold={accuracy_drop_threshold:.2f}%")
```

# 線性量化

在本節中，我們將實作並執行線性量化。

線性量化在範圍截斷和縮放後，直接將浮點值四捨五入為最接近的量化整數。

[線性量化](https://arxiv.org/pdf/1712.05877.pdf)可以表示為：

$r = S(q-Z)$

其中 $r$ 是浮點實數，$q$ 是 $n$-bit 整數，$Z$ 是 $n$-bit 整數，而 $S$ 是浮點實數。$Z$ 是量化零點 (zero point)，$S$ 是量化縮放因子 (scale)。常數 $Z$ 和 $S$ 皆為量化參數。

## *n*-bit 整數

一個 $n$-bit 有符號整數通常以 [二補數 (two's complement)](https://en.wikipedia.org/wiki/Two%27s_complement) 表示法表示。

一個 $n$-bit 有符號整數可以編碼範圍在 $[-2^{n-1}, 2^{n-1}-1]$ 內的整數。例如，8-bit 整數的範圍在 [-128, 127]。


```python
def get_quantized_range(bitwidth):
    quantized_max = (1 << (bitwidth - 1)) - 1
    quantized_min = -(1 << (bitwidth - 1))
    return quantized_min, quantized_max
```

## **問題 4** (10 分)

請完成以下線性量化函數。

**提示**：
*   從 $r=S(q-Z)$，我們有 $q = r/S + Z$。
*   由於 $r$ 和 $S$ 都是浮點數，因此我們不能直接將整數 $Z$ 加到 $r/S$。所以 $q = \mathrm{int}(\mathrm{round}(r/S)) + Z$。
*   要將 [`torch.FloatTensor`](https://pytorch.org/docs/stable/tensors.html) 轉換為 [`torch.IntTensor`](https://pytorch.org/docs/stable/tensors.html)，我們可以使用 [`torch.round()`](https://pytorch.org/docs/stable/generated/torch.round.html#torch.round), [`torch.Tensor.round()`](https://pytorch.org/docs/stable/generated/torch.Tensor.round.html#torch.Tensor.round) 或 [`torch.Tensor.round_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.round_) 先將所有值轉換為浮點整數，然後使用 [`torch.Tensor.to(torch.int8)`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to) 將資料類型從 [`torch.float`](https://pytorch.org/docs/stable/tensors.html) 轉換為 [`torch.int8`](https://pytorch.org/docs/stable/tensors.html)。




```python
def linear_quantize(fp_tensor, bitwidth, scale, zero_point, dtype=torch.int8) -> torch.Tensor:
    """
    單個 fp_tensor 的線性量化
      從
        fp_tensor = (quantized_tensor - zero_point) * scale
      我們有，
        quantized_tensor = int(round(fp_tensor / scale)) + zero_point
    :param tensor: [torch.(cuda.)FloatTensor] 要被量化的浮點張量
    :param bitwidth: [int] 量化位元寬度
    :param scale: [torch.(cuda.)FloatTensor] 縮放因子
    :param zero_point: [torch.(cuda.)IntTensor] 張量值所需的中心點
    :return:
        [torch.(cuda.)FloatTensor] 值為整數的量化張量
    """
    assert(fp_tensor.dtype == torch.float)
    assert(isinstance(scale, float) or
           (scale.dtype == torch.float and scale.dim() == fp_tensor.dim()))
    assert(isinstance(zero_point, int) or
           (zero_point.dtype == dtype and zero_point.dim() == fp_tensor.dim()))

    ############### 你的程式碼從這裡開始 ###############
    # 步驟 1：縮放 fp_tensor
    scaled_tensor = fp_tensor / scale
    # 步驟 2：將浮點值四捨五入為整數值
    rounded_tensor = scaled_tensor.round()
    ############### 你的程式碼在這裡結束 #################

    rounded_tensor = rounded_tensor.to(dtype)

    ############### 你的程式碼從這裡開始 ###############
    # 步驟 3：平移 rounded_tensor 使零點為 0
    shifted_tensor = rounded_tensor + zero_point
    ############### 你的程式碼在這裡結束 #################

    # 步驟 4：將 shifted_tensor 限制在 bitwidth 位元範圍內
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    quantized_tensor = shifted_tensor.clamp_(quantized_min, quantized_max)
    return quantized_tensor
```

讓我們透過在虛擬張量上套用上述函數，來驗證定義的線性量化的功能。


```python
test_linear_quantize()
```

## 問題 5 (15 分)

現在我們確定線性量化的縮放因子 $S$ 和零點 $Z$。

回顧一下，線性量化可以表示為：

$r = S(q-Z)$

### 縮放因子 (Scale)

線性量化將浮點數範圍 [*fp_min*, *fp_max*] 投影到量化範圍 [*quantized_min*, *quantized_max*]。也就是說：

> $r_{\mathrm{max}} = S(q_{\mathrm{max}}-Z)$
>
> $r_{\mathrm{min}} = S(q_{\mathrm{min}}-Z)$

兩式相減，我們得到：


#### 問題 5.1 (3 分)

請選擇正確答案，並在下一個文字儲存格中刪除錯誤答案。

> $S=(r_{\mathrm{max}} - r_{\mathrm{min}}) / (q_{\mathrm{max}} - q_{\mathrm{min}})$


有不同的方法來確定浮點張量 `fp_tensor` 的 $r_{\mathrm{min}}$ 和 $r_{\mathrm{max}}$：

*   最常用的方法是直接使用 `fp_tensor` 的最小值和最大值。
*   另一個廣泛使用的方法是最小化 Kullback-Leibler (KL) 散度來確定 *fp_max*。

### 零點 (Zero Point)

一旦我們確定了縮放因子 $S$，我們就可以直接利用 $r_{\mathrm{min}}$ 與 $q_{\mathrm{min}}$ 之間的關係來計算零點 $Z$。

#### 問題 5.2 (4 分)

請選擇正確答案，並在下一個文字儲存格中刪除錯誤答案。

> $Z = \mathrm{int}(\mathrm{round}(q_{\mathrm{min}} - r_{\mathrm{min}} / S))$

### 問題 5.3 (8 分)

請完成以下用於從浮點張量 $r$ 計算縮放因子 $S$ 和零點 $Z$ 的函數。



```python
def get_quantization_scale_and_zero_point(fp_tensor, bitwidth):
    """
    獲取單個張量的量化縮放因子與零點
    :param fp_tensor: [torch.(cuda.)Tensor] 要被量化的浮點張量
    :param bitwidth: [int] 量化位元寬度
    :return:
        [float] scale
        [int] zero_point
    """
    quantized_min, quantized_max = get_quantized_range(bitwidth)
    fp_max = fp_tensor.max().item()
    fp_min = fp_tensor.min().item()

    ############### 你的程式碼從這裡開始 ###############
    # 提示：計算 scale 的一行程式碼
    scale = (fp_max - fp_min) / (quantized_max - quantized_min)
    # 提示：計算 zero_point 的一行程式碼
    zero_point = quantized_min - fp_min / scale
    ############### 你的程式碼在這裡結束 #################

    # 將 zero_point 限制在 [quantized_min, quantized_max] 範圍內
    if zero_point < quantized_min:
        zero_point = quantized_min
    elif zero_point > quantized_max:
        zero_point = quantized_max
    else: # 使用 round() 將浮點數轉換為整數
        zero_point = round(zero_point)
    return scale, int(zero_point)
```

我們現在將問題 4 中的 `linear_quantize()` 和問題 5 中的 `get_quantization_scale_and_zero_point()` 包裝成一個函數。


```python
def linear_quantize_feature(fp_tensor, bitwidth):
    """
    特徵張量的線性量化
    :param fp_tensor: [torch.(cuda.)Tensor] 要被量化的浮點特徵
    :param bitwidth: [int] 量化位元寬度
    :return:
        [torch.(cuda.)Tensor] 量化張量
        [float] 縮放因子張量
        [int] 零點
    """
    scale, zero_point = get_quantization_scale_and_zero_point(fp_tensor, bitwidth)
    quantized_tensor = linear_quantize(fp_tensor, bitwidth, scale, zero_point)
    return quantized_tensor, scale, zero_point
```

## 特殊情況：對權重張量進行線性量化

讓我們首先看看權重值的分布。


```python
def plot_weight_distribution(model, bitwidth=32):
    # bins = (1 << bitwidth) if bitwidth <= 8 else 256
    if bitwidth <= 8:
        qmin, qmax = get_quantized_range(bitwidth)
        bins = np.arange(qmin, qmax + 2)
        align = 'left'
    else:
        bins = 256
        align = 'mid'
    fig, axes = plt.subplots(3,3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            ax.hist(param.detach().view(-1).cpu(), bins=bins, density=True,
                    align=align, color = 'blue', alpha = 0.5,
                    edgecolor='black' if bitwidth <= 4 else None)
            if bitwidth <= 4:
                quantized_min, quantized_max = get_quantized_range(bitwidth)
                ax.set_xticks(np.arange(start=quantized_min, stop=quantized_max+1))
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
    fig.suptitle(f'Histogram of Weights (bitwidth={bitwidth} bits)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()

recover_model()
plot_weight_distribution(model)
```

正如我們從上面的直方圖中所看到的，權重值的分布幾乎是關於 0 對稱的（在這種情況下，分類器除外）。因此，在對權重進行量化時，我們通常使零點 $Z=0$。

從 $r = S(q-Z)$ 且 $Z=0$，我們有：

> $r_{\mathrm{max}} = S \cdot q_{\mathrm{max}}$

因此有：

> $S = r_{\mathrm{max}} / q_{\mathrm{max}}$

我們直接使用權重值的最大絕對值（幅度）作為 $r_{\mathrm{max}}$。


```python
def get_quantization_scale_for_weight(weight, bitwidth):
    """
    獲取單個權重張量的量化縮放因子
    :param weight: [torch.(cuda.)Tensor] 要被量化的浮點權重
    :param bitwidth: [integer] 量化位元寬度
    :return:
        [floating scalar] scale
    """
    # 我們只是假設權重中的值是對稱的
    # 我們也始終對權重將 zero_point 設為 0
    fp_max = max(weight.abs().max().item(), 5e-7)
    _, quantized_max = get_quantized_range(bitwidth)
    return fp_max / quantized_max
```
### 逐通道線性量化 (Per-channel Linear Quantization)

回顧一下，對於 2D 卷積，權重張量是一個 4-D 張量，形狀為 (num_output_channels, num_input_channels, kernel_height, kernel_width)。

密集的實驗表明，對不同的輸出通道使用不同的縮放因子 $S$ 和零點 $Z$ 會表現得更好。因此，我們必須獨立確定每個輸出通道子張量的縮放因子 $S$ 和零點 $Z$。


```python
def linear_quantize_weight_per_channel(tensor, bitwidth):
    """
    權重張量的線性量化，對不同的輸出通道使用不同的縮放因子和零點
    :param tensor: [torch.(cuda.)Tensor] 要被量化的浮點權重
    :param bitwidth: [int] 量化位元寬度
    :return:
        [torch.(cuda.)Tensor] 量化張量
        [torch.(cuda.)Tensor] 縮放因子張量
        [int] 零點（始終為 0）
    """
    dim_output_channels = 0
    num_output_channels = tensor.shape[dim_output_channels]
    scale = torch.zeros(num_output_channels, device=tensor.device)
    for oc in range(num_output_channels):
        _subtensor = tensor.select(dim_output_channels, oc)
        _scale = get_quantization_scale_for_weight(_subtensor, bitwidth)
        scale[oc] = _scale
    scale_shape = [1] * tensor.dim()
    scale_shape[dim_output_channels] = -1
    scale = scale.view(scale_shape)
    quantized_tensor = linear_quantize(tensor, bitwidth, scale, zero_point=0)
    return quantized_tensor, scale, 0
```

### 快速瀏覽權重的線性量化

現在，讓我們看一下在不同位元寬度的權重上應用線性量化時的權重分布和模型大小。


```python
@torch.no_grad()
def peek_linear_quantization():
    for bitwidth in [4, 2]:
        for name, param in model.named_parameters():
            if param.dim() > 1:
                quantized_param, scale, zero_point = \
                    linear_quantize_weight_per_channel(param, bitwidth)
                param.copy_(quantized_param)
        plot_weight_distribution(model, bitwidth)
        recover_model()

peek_linear_quantization()
```

## 量化推論 (Quantized Inference)

量化之後，卷積層和全連接層的推論也隨之改變。

回顧一下 $r = S(q-Z)$，我們有：

> $r_{\mathrm{input}} = S_{\mathrm{input}}(q_{\mathrm{input}}-Z_{\mathrm{input}})$
>
> $r_{\mathrm{weight}} = S_{\mathrm{weight}}(q_{\mathrm{weight}}-Z_{\mathrm{weight}})$
>
> $r_{\mathrm{bias}} = S_{\mathrm{bias}}(q_{\mathrm{bias}}-Z_{\mathrm{bias}})$

由於 $Z_{\mathrm{weight}}=0$，所以 $r_{\mathrm{weight}} = S_{\mathrm{weight}}q_{\mathrm{weight}}$。

浮點數卷積可以寫成：

> $r_{\mathrm{output}} = \mathrm{CONV}[r_{\mathrm{input}}, r_{\mathrm{weight}}] + r_{\mathrm{bias}}\\
> \;\;\;\;\;\;\;\;= \mathrm{CONV}[S_{\mathrm{input}}(q_{\mathrm{input}}-Z_{\mathrm{input}}), S_{\mathrm{weight}}q_{\mathrm{weight}}] + S_{\mathrm{bias}}(q_{\mathrm{bias}}-Z_{\mathrm{bias}})\\
> \;\;\;\;\;\;\;\;= \mathrm{CONV}[q_{\mathrm{input}}-Z_{\mathrm{input}}, q_{\mathrm{weight}}]\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}}) + S_{\mathrm{bias}}(q_{\mathrm{bias}}-Z_{\mathrm{bias}})$

為了進一步簡化計算，我們可以令：

> $Z_{\mathrm{bias}} = 0$
>
> $S_{\mathrm{bias}} = S_{\mathrm{input}} \cdot S_{\mathrm{weight}}$

使得：

> $r_{\mathrm{output}} = (\mathrm{CONV}[q_{\mathrm{input}}-Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}})$
> $\;\;\;\;\;\;\;\;= (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}}S_{\mathrm{weight}})$

由於：
> $r_{\mathrm{output}} = S_{\mathrm{output}}(q_{\mathrm{output}}-Z_{\mathrm{output}})$

我們有：
> $S_{\mathrm{output}}(q_{\mathrm{output}}-Z_{\mathrm{output}}) = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}})$

因此：
> $q_{\mathrm{output}} = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}] + q_{\mathrm{bias}})\cdot (S_{\mathrm{input}}S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}$

由於 $Z_{\mathrm{input}}$, $q_{\mathrm{weight}}$, $q_{\mathrm{bias}}$ 在推論前就已確定，令：

> $Q_{\mathrm{bias}} = q_{\mathrm{bias}} - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]$

我們有：

> $q_{\mathrm{output}} = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}}) \cdot (S_{\mathrm{input}}S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}$

同樣地，對於全連接層，我們有：

> $q_{\mathrm{output}} = (\mathrm{Linear}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} \cdot S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}$

其中：

> $Q_{\mathrm{bias}} = q_{\mathrm{bias}} - \mathrm{Linear}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]$

### 問題 6 (5 分)

請完成以下用於對偏差（bias）進行線性量化的函數。

**提示**：

從上述推導中，我們知道：

> $Z_{\mathrm{bias}} = 0$
>
> $S_{\mathrm{bias}} = S_{\mathrm{input}} \cdot S_{\mathrm{weight}}$


```python
def linear_quantize_bias_per_output_channel(bias, weight_scale, input_scale):
    """
    單個 bias 張量的線性量化
        quantized_bias = fp_bias / bias_scale
    :param bias: [torch.FloatTensor] 要被量化的浮點偏差
    :param weight_scale: [float 或 torch.FloatTensor] 權重縮放因子張量
    :param input_scale: [float] 輸入縮放因子
    :return:
        [torch.IntTensor] 量化後的 bias 張量
    """
    assert(bias.dim() == 1)
    assert(bias.dtype == torch.float)
    assert(isinstance(input_scale, float))
    if isinstance(weight_scale, torch.Tensor):
        assert(weight_scale.dtype == torch.float)
        weight_scale = weight_scale.view(-1)
        assert(bias.numel() == weight_scale.numel())

    ############### 你的程式碼從這裡開始 ###############
    # 提示：一行程式碼
    bias_scale = input_scale * weight_scale
    ############### 你的程式碼在這裡結束 #################

    quantized_bias = linear_quantize(bias, 32, bias_scale,
                                     zero_point=0, dtype=torch.int32)
    return quantized_bias, bias_scale, 0
```

### 量化全連接層 (Quantized Fully-Connected Layer)

對於量化全連接層，我們首先預先計算 $Q_{\mathrm{bias}}$。回顧一下 $Q_{\mathrm{bias}} = q_{\mathrm{bias}} - \mathrm{Linear}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]$。


```python
def shift_quantized_linear_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    平移量化 bias 以將 input_zero_point 併入 nn.Linear 中
        shifted_quantized_bias = quantized_bias - Linear(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] 量化後的偏差 (torch.int32)
    :param quantized_weight: [torch.CharTensor] 量化後的權重 (torch.int8)
    :param input_zero_point: [int] 輸入零點
    :return:
        [torch.IntTensor] 平移後的量化偏差張量
    """
    assert(quantized_bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    return quantized_bias - quantized_weight.sum(1).to(torch.int32) * input_zero_point
```

#### 問題 7 (15 分)

請完成以下量化全連接層推論函數。

**提示**：

> $q_{\mathrm{output}} = (\mathrm{Linear}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}})\cdot (S_{\mathrm{input}} S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}$


```python
def quantized_linear(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale):
    """
    量化全連接層
    :param input: [torch.CharTensor] 量化輸入 (torch.int8)
    :param weight: [torch.CharTensor] 量化權重 (torch.int8)
    :param bias: [torch.IntTensor] 平移後的量化偏差或 None (torch.int32)
    :param feature_bitwidth: [int] 輸入與輸出的量化位元寬度
    :param weight_bitwidth: [int] 權重的量化位元寬度
    :param input_zero_point: [int] 輸入零點
    :param output_zero_point: [int] 輸出零點
    :param input_scale: [float] 輸入特徵縮放因子
    :param weight_scale: [torch.FloatTensor] 權重的逐通道縮放因子
    :param output_scale: [float] 輸出特徵縮放因子
    :return:
        [torch.CharIntTensor] 量化輸出特徵 (torch.int8)
    """
    assert(input.dtype == torch.int8)
    assert(weight.dtype == input.dtype)
    assert(bias is None or bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    assert(isinstance(output_zero_point, int))
    assert(isinstance(input_scale, float))
    assert(isinstance(output_scale, float))
    assert(weight_scale.dtype == torch.float)

    # 步驟 1：基於整數的全連接（8 位元乘法與 32 位元累加）
    if 'cpu' in input.device.type:
        # 為了簡化使用 32 位元 MAC
        output = torch.nn.functional.linear(input.to(torch.int32), weight.to(torch.int32), bias)
    else:
        # 當前版本的 PyTorch 尚不支援 GPU 上的整數型 linear()
        output = torch.nn.functional.linear(input.float(), weight.float(), bias.float())

    ############### 你的程式碼從這裡開始 ###############
    # 步驟 2：縮放輸出
    #         提示：1. 縮放因子是浮點數，我們也需要將輸出轉換為浮點數
    #               2. 權重縮放因子的形狀是 [oc, 1, 1, 1]，而輸出的形狀是 [batch_size, oc]
    output = output.float() * (input_scale * weight_scale.view(1, -1) / output_scale)

    # 步驟 3：根據 output_zero_point 平移輸出
    #         提示：一行程式碼
    output = output + output_zero_point
    ############### 你的程式碼在這裡結束 #################

    # 確保所有值都落在 bitwidth 位元範圍內
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output
```

讓我們驗證定義的量化全連接層的功能。


```python
test_quantized_fc()
```

### 量化卷積 (Quantized Convolution)

對於量化卷積層，我們首先預先計算 $Q_{\mathrm{bias}}$。回顧一下 $Q_{\mathrm{bias}} = q_{\mathrm{bias}} - \mathrm{CONV}[Z_{\mathrm{input}}, q_{\mathrm{weight}}]$。


```python
def shift_quantized_conv2d_bias(quantized_bias, quantized_weight, input_zero_point):
    """
    平移量化 bias 以將 input_zero_point 併入 nn.Conv2d 中
        shifted_quantized_bias = quantized_bias - Conv(input_zero_point, quantized_weight)
    :param quantized_bias: [torch.IntTensor] 量化後的偏差 (torch.int32)
    :param quantized_weight: [torch.CharTensor] 量化後的權重 (torch.int8)
    :param input_zero_point: [int] 輸入零點
    :return:
        [torch.IntTensor] 平移後的量化偏差張量
    """
    assert(quantized_bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    return quantized_bias - quantized_weight.sum((1,2,3)).to(torch.int32) * input_zero_point
```

#### 問題 8 (10 分)

請完成以下量化卷積函數。

**提示**：
> $q_{\mathrm{output}} = (\mathrm{CONV}[q_{\mathrm{input}}, q_{\mathrm{weight}}] + Q_{\mathrm{bias}}) \cdot (S_{\mathrm{input}}S_{\mathrm{weight}} / S_{\mathrm{output}}) + Z_{\mathrm{output}}$



```python
def quantized_conv2d(input, weight, bias, feature_bitwidth, weight_bitwidth,
                     input_zero_point, output_zero_point,
                     input_scale, weight_scale, output_scale,
                     stride, padding, dilation, groups):
    """
    量化 2d 卷積
    :param input: [torch.CharTensor] 量化輸入 (torch.int8)
    :param weight: [torch.CharTensor] 量化權重 (torch.int8)
    :param bias: [torch.IntTensor] 平移後的量化偏差或 None (torch.int32)
    :param feature_bitwidth: [int] 輸入與輸出的量化位元寬度
    :param weight_bitwidth: [int] 權重的量化位元寬度
    :param input_zero_point: [int] 輸入零點
    :param output_zero_point: [int] 輸出零點
    :param input_scale: [float] 輸入特徵縮放因子
    :param weight_scale: [torch.FloatTensor] 權重的逐通道縮放因子
    :param output_scale: [float] 輸出特徵縮放因子
    :return:
        [torch.(cuda.)CharTensor] 量化輸出特徵
    """
    assert(len(padding) == 4)
    assert(input.dtype == torch.int8)
    assert(weight.dtype == input.dtype)
    assert(bias is None or bias.dtype == torch.int32)
    assert(isinstance(input_zero_point, int))
    assert(isinstance(output_zero_point, int))
    assert(isinstance(input_scale, float))
    assert(isinstance(output_scale, float))
    assert(weight_scale.dtype == torch.float)

    # 步驟 1：計算基於整數的 2d 卷積（8 位元乘法與 32 位元累加）
    input = torch.nn.functional.pad(input, padding, 'constant', input_zero_point)
    if 'cpu' in input.device.type:
        # 為了簡化使用 32 位元 MAC
        output = torch.nn.functional.conv2d(input.to(torch.int32), weight.to(torch.int32), None, stride, 0, dilation, groups)
    else:
        # 當前版本的 PyTorch 尚不支援 GPU 上的整數型 conv2d()
        output = torch.nn.functional.conv2d(input.float(), weight.float(), None, stride, 0, dilation, groups)
        output = output.round().to(torch.int32)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)

    ############### 你的程式碼從這裡開始 ###############
    # 提示：此程式碼區塊應與 quantized_linear() 非常相似

    # 步驟 2：縮放輸出
    #         提示：1. 縮放因子是浮點數，我們也需要將輸出轉換為浮點數
    #               2. 權重縮放因子的形狀是 [oc, 1, 1, 1]，而輸出的形狀是 [batch_size, oc, height, width]
    output = output.float() * (input_scale * weight_scale.view(1, -1, 1, 1) / output_scale)

    # 步驟 3：根據 output_zero_point 平移輸出
    #         提示：一行程式碼
    output = output + output_zero_point
    ############### 你的程式碼在這裡結束 #################

    # 確保所有值都落在 bitwidth 位元範圍內
    output = output.round().clamp(*get_quantized_range(feature_bitwidth)).to(torch.int8)
    return output
```

## 問題 9 (10 分)

最後，我們將所有內容整合在一起，並為模型執行訓練後 `int8` 量化 (Post-Training Quantization)。我們將逐一將模型中的卷積層和線性層轉換為量化版本。

1. 首先，我們將一個 BatchNorm 層融合成其前一個卷積層，這是量化前的標準做法。融合 BatchNorm 可減少推論期間的額外乘法運算。

我們還將驗證融合後的模型 `model_fused` 具有與原始模型相同的準確度（BN 融合是一個等價變換，不會改變網路功能）。


```python
def fuse_conv_bn(conv, bn):
    # 修改自 https://mmcv.readthedocs.io/en/latest/_modules/mmcv/cnn/utils/fuse_conv_bn.html
    assert conv.bias is None

    factor = bn.weight.data / torch.sqrt(bn.running_var.data + bn.eps)
    conv.weight.data = conv.weight.data * factor.reshape(-1, 1, 1, 1)
    conv.bias = nn.Parameter(- bn.running_mean.data * factor + bn.bias.data)

    return conv

print('Before conv-bn fusion: backbone length', len(model.backbone))
# 將 batchnorm 融合進卷積層中
recover_model()
model_fused = copy.deepcopy(model)
fused_backbone = []
ptr = 0
while ptr < len(model_fused.backbone):
    if isinstance(model_fused.backbone[ptr], nn.Conv2d) and \
        isinstance(model_fused.backbone[ptr + 1], nn.BatchNorm2d):
        fused_backbone.append(fuse_conv_bn(
            model_fused.backbone[ptr], model_fused.backbone[ptr+ 1]))
        ptr += 2
    else:
        fused_backbone.append(model_fused.backbone[ptr])
        ptr += 1
model_fused.backbone = nn.Sequential(*fused_backbone)

print('After conv-bn fusion: backbone length', len(model_fused.backbone))
# 完整性檢查，已無 BN
for m in model_fused.modules():
    assert not isinstance(m, nn.BatchNorm2d)

# 融合後準確度將保持相同
fused_acc = evaluate(model_fused, dataloader['test'])
print(f'Accuracy of the fused model={fused_acc:.2f}%')
```

    Before conv-bn fusion: backbone length 15
    After conv-bn fusion: backbone length 11
    Accuracy of the fused model=92.93%


2. 我們將使用一些樣本數據運行模型以獲取每個特徵圖的範圍，以便我們可以獲取特徵圖的範圍並計算其相應的縮放因子和零點。


```python
# 新增 hook 來記錄活化的最小/最大值
input_activation = {}
output_activation = {}

def add_range_recoder_hook(model):
    import functools
    def _record_range(self, x, y, module_name):
        x = x[0]
        input_activation[module_name] = x.detach()
        output_activation[module_name] = y.detach()

    all_hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ReLU)):
            all_hooks.append(m.register_forward_hook(
                functools.partial(_record_range, module_name=name)))
    return all_hooks

hooks = add_range_recoder_hook(model_fused)
sample_data = iter(dataloader['train']).__next__()[0]
model_fused(sample_data.cuda())

# 移除 hooks
for h in hooks:
    h.remove()
```

3. 最後，讓我們進行模型量化。我們將按以下對應關係轉換模型：
```python
nn.Conv2d: QuantizedConv2d,
nn.Linear: QuantizedLinear,
# 以下兩者只是包裝器，因為當前的 torch 模組不支援 int8 數據格式；
# 我們將暫時將它們轉換為 fp32 進行計算
nn.MaxPool2d: QuantizedMaxPool2d,
nn.AvgPool2d: QuantizedAvgPool2d,
```


```python
class QuantizedConv2d(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 stride, padding, dilation, groups,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # 當前版本的 PyTorch 尚不支援以 IntTensor 作為 nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.stride = stride
        self.padding = (padding[1], padding[1], padding[0], padding[0])
        self.dilation = dilation
        self.groups = groups

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth


    def forward(self, x):
        return quantized_conv2d(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale,
            self.stride, self.padding, self.dilation, self.groups
            )

class QuantizedLinear(nn.Module):
    def __init__(self, weight, bias,
                 input_zero_point, output_zero_point,
                 input_scale, weight_scale, output_scale,
                 feature_bitwidth=8, weight_bitwidth=8):
        super().__init__()
        # 當前版本的 PyTorch 尚不支援以 IntTensor 作為 nn.Parameter
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

        self.input_zero_point = input_zero_point
        self.output_zero_point = output_zero_point

        self.input_scale = input_scale
        self.register_buffer('weight_scale', weight_scale)
        self.output_scale = output_scale

        self.feature_bitwidth = feature_bitwidth
        self.weight_bitwidth = weight_bitwidth

    def forward(self, x):
        return quantized_linear(
            x, self.weight, self.bias,
            self.feature_bitwidth, self.weight_bitwidth,
            self.input_zero_point, self.output_zero_point,
            self.input_scale, self.weight_scale, self.output_scale
            )

class QuantizedMaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        # 當前版本的 PyTorch 尚不支援基於整數的 MaxPool
        return super().forward(x.float()).to(torch.int8)

class QuantizedAvgPool2d(nn.AvgPool2d):
    def forward(self, x):
        # 當前版本的 PyTorch 尚不支援基於整數的 AvgPool
        return super().forward(x.float()).to(torch.int8)

# 我們使用 int8 量化，這非常受歡迎
feature_bitwidth = weight_bitwidth = 8
quantized_model = copy.deepcopy(model_fused)
quantized_backbone = []
ptr = 0
while ptr < len(quantized_model.backbone):
    if isinstance(quantized_model.backbone[ptr], nn.Conv2d) and \
        isinstance(quantized_model.backbone[ptr + 1], nn.ReLU):
        conv = quantized_model.backbone[ptr]
        conv_name = f'backbone.{ptr}'
        relu = quantized_model.backbone[ptr + 1]
        relu_name = f'backbone.{ptr + 1}'

        input_scale, input_zero_point = \
            get_quantization_scale_and_zero_point(
                input_activation[conv_name], feature_bitwidth)

        output_scale, output_zero_point = \
            get_quantization_scale_and_zero_point(
                output_activation[relu_name], feature_bitwidth)

        quantized_weight, weight_scale, weight_zero_point = \
            linear_quantize_weight_per_channel(conv.weight.data, weight_bitwidth)
        quantized_bias, bias_scale, bias_zero_point = \
            linear_quantize_bias_per_output_channel(
                conv.bias.data, weight_scale, input_scale)
        shifted_quantized_bias = \
            shift_quantized_conv2d_bias(quantized_bias, quantized_weight,
                                        input_zero_point)

        quantized_conv = QuantizedConv2d(
            quantized_weight, shifted_quantized_bias,
            input_zero_point, output_zero_point,
            input_scale, weight_scale, output_scale,
            conv.stride, conv.padding, conv.dilation, conv.groups,
            feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
        )

        quantized_backbone.append(quantized_conv)
        ptr += 2
    elif isinstance(quantized_model.backbone[ptr], nn.MaxPool2d):
        quantized_backbone.append(QuantizedMaxPool2d(
            kernel_size=quantized_model.backbone[ptr].kernel_size,
            stride=quantized_model.backbone[ptr].stride
            ))
        ptr += 1
    elif isinstance(quantized_model.backbone[ptr], nn.AvgPool2d):
        quantized_backbone.append(QuantizedAvgPool2d(
            kernel_size=quantized_model.backbone[ptr].kernel_size,
            stride=quantized_model.backbone[ptr].stride
            ))
        ptr += 1
    else:
        raise NotImplementedError(type(quantized_model.backbone[ptr]))  # should not happen
quantized_model.backbone = nn.Sequential(*quantized_backbone)

# 最後，量化分類器
fc_name = 'classifier'
fc = model.classifier
input_scale, input_zero_point = \
    get_quantization_scale_and_zero_point(
        input_activation[fc_name], feature_bitwidth)

output_scale, output_zero_point = \
    get_quantization_scale_and_zero_point(
        output_activation[fc_name], feature_bitwidth)

quantized_weight, weight_scale, weight_zero_point = \
    linear_quantize_weight_per_channel(fc.weight.data, weight_bitwidth)
quantized_bias, bias_scale, bias_zero_point = \
    linear_quantize_bias_per_output_channel(
        fc.bias.data, weight_scale, input_scale)
shifted_quantized_bias = \
    shift_quantized_linear_bias(quantized_bias, quantized_weight,
                                input_zero_point)

quantized_model.classifier = QuantizedLinear(
    quantized_weight, shifted_quantized_bias,
    input_zero_point, output_zero_point,
    input_scale, weight_scale, output_scale,
    feature_bitwidth=feature_bitwidth, weight_bitwidth=weight_bitwidth
)
```

量化過程完成了！讓我們列印並視覺化模型架構，並驗證量化模型的準確度。

### 問題 9.1 (5 分)

為了運行量化後的模型，我們需要一個額外的預處理，將範圍在 (0, 1) 的輸入數據映射到 (-128, 127) 的 `int8` 範圍內。填寫下面的程式碼以完成額外的預處理。

**提示**：你應該會發現量化後的模型與其 `fp32` 對應模型具有大約相同的準確度。


```python
print(quantized_model)

def extra_preprocess(x):
    # 提示：你需要將範圍 (0, 1) 的原始 fp32 輸入轉換為範圍 (-128, 127) 的 int8 格式
    ############### 你的程式碼從這裡開始 ###############
    x_scaled = x * 255.0 - 128.0
    return x_scaled.clamp(-128, 127).to(torch.int8)
    ############### 你的程式碼在這裡結束 #################

int8_model_accuracy = evaluate(quantized_model, dataloader['test'],
                               extra_preprocess=[extra_preprocess])
print(f"int8 model has accuracy={int8_model_accuracy:.2f}%")
```

## 問題 9.2 (5 分)

解釋為什麼線性量化模型中沒有 ReLU 層。

**你的答案：**
因為在線性量化模型中，ReLU（修正線性單元）的效果已經被融合到量化引進的範圍截斷和零點平移（Clamp 限制）中了。具體來說，在 ReLU(x) = max(0, x) 中，負數會被歸零，而量化時的 quantized_feature 的 scale 和 zero_point 經過計算，以及對 output 的 get_quantized_range 裁剪（例如 [0, 127] 或 [-128, 127]）配合 ReLU 啟動函數時，我們可以直接在估計 ReLU 輸出激活範圍時將最小值（fp_min）設為大於等於 0，此時對應的 zero_point 將會將所有小於 0 的部分在 clamp 時自動截斷至 quantized_min（代表實際的 0 浮點值）。因此，ReLU 層的非線性映射已經被下一個量化層的 clip/clamp 截斷操作隱式完成了，不需要獨立的 ReLU 運算層。

# 問題 10 (5 分)

請比較基於 K-Means 的量化和線性量化的優缺點。你可以從準確度、延遲、硬體支援等角度進行討論。

**你的答案：**

1. **基於 K-Means 的量化 (K-Means Quantization)**
   * **優點**：
     * **較高的高倍率壓縮準確度**：分群是不均勻的，能夠在權重分布較為極端或密集的部分提供更精細的表示，所以在非常低位元（如 2-bit、4-bit）的非線性量化下，其準確度保留能力通常優於簡單的均勻線性量化。
   * **缺點**：
     * **硬體支援極差與高推論延遲**：在推論時，因為硬體通常不支援直接用密碼本索引 (labels) 進行乘加運算，所以在實際計算前，必須通過查表 (Look-up Table) 把索引解碼回原始的 FP32 數值再去作浮點數運算。這導致模型雖然佔用儲存空間變小，但無法在硬體上得到推論加速，延遲反而可能因查表操作而增加。

2. **線性量化 (Linear Quantization / Integer-only Inference)**
   * **優點**：
     * **極佳的硬體支援與超低延遲**：量化為均勻的整數間隔，可以直接將實數運算轉換為純整數運算（例如 INT8 乘法加上 INT32 累加），現今絕大多數的 AI 晶片、NPU、DSP、CPU (例如使用 SIMD 指令集) 都對整數運算有極強的硬體加速與平行的支援，能大幅降低推論延遲與功耗。
   * **缺點**：
     * **均勻投影的精度限制**：因為是等間隔均勻映射，若權重或特徵圖中存在極端的離群值 (outliers)，會導致大部分常態分布的權重被壓縮在極窄的數值範圍內，進而損失精度。在低位元（如 2-bit、4-bit）時精度下降極為嚴重，通常需要配合量化感知訓練 (QAT) 才能恢復。
