# Jupyter 筆記本：lab4.ipynb

# **MIT 6.5940 EfficientML.ai Lab 4: LLM Quantization with AWQ (大型語言模型量化)**

by MIT HAN Lab

原始連結：https://colab.research.google.com/drive/16H9RvSg4XIF35X3fLGQUVwAE9ccvDj14

### [Cell 0] (Markdown)

# **MIT 6.5940 EfficientML.ai Lab 4: LLM Quantization with AWQ**

### [Cell 1] (Markdown)

## 簡介 (Introduction)

本 Colab 筆記本提供了 Lab 4：大型語言模型量化 (LLM Quantization) 的程式碼與框架。你將學習如何量化一個大型語言模型，使其能夠高效率地運行。我們將實作 **AWQ (activation aware weight only quantization，激活感知的僅權重量化)** 來達成 4-bit 的僅權重量化。

在邊緣端 (Edge) 運行大型語言模型 (LLMs) 非常重要，這不僅能提升使用者體驗，也能解決隱私疑慮——因為敏感資料會保留在本地端，降低了資料外洩的風險。

然而，在邊緣裝置上部署 LLM 存在重大挑戰。邊緣裝置的功耗限制極為嚴格，這是它們與工作站或雲端伺服器最大的不同。這意味著邊緣端的記憶體頻寬 (memory bandwidth) 受限，且峰值運算吞吐量 (peak computation throughput) 有限。舉例來說，NVIDIA Jetson Orin Nano 只有 8GB DRAM，即使是最精簡的 LLaMA-2 模型在半精度 (half precision) 下也放不進去。幸運的是，AWQ 提供了一個「一鍵式 (push-the-button)」的權重量化解決方案，讓 LLM 能在記憶體受限的邊緣裝置上進行推論。

此外，透過 AWQ 4-bit 僅權重量化演算法，搭配高效率的 4-bit kernel，我們可以在 RTX 4090 上達到以下的加速效果。在下一節的實驗中，我們也會使用 TinyChatEngine 來達成實際的效能加速。

### [Cell 2] (Markdown)

### 在 RTX 4090 上的展示 (Demo on an RTX 4090):

### [Cell 3] (Markdown)

![4090_example](assets/lab4/4090_example.gif)

### [Cell 4] (Markdown)

### 在 Apple MacBook Air (M1, 2020) 上的展示:

### [Cell 5] (Markdown)

![demo.gif](https://github.com/mit-han-lab/TinyChatEngine/blob/main/assets/figures/chat_demo_m1.gif?raw=true)

### [Cell 6] (Markdown)

# AWQ (activation aware weight only quantization，激活感知的僅權重量化)

### [Cell 7] (Markdown)

![nas_overview.png](assets/lab4/nas_overview.png)

### [Cell 8] (Markdown)

大型語言模型 (LLMs) 在各式任務上都展現了優異的表現，但天文數字般的模型大小提高了服務部署的硬體門檻 (記憶體容量)，並拖慢了 token 的生成速度 (記憶體頻寬)。LLM 的模型大小與計算量正以指數方式成長，而記憶體頻寬的成長卻相當緩慢。這個差距是 LLM 的主要瓶頸。在本次實驗中，我們將探索如何使用一種新穎的量化演算法 (AWQ) 來降低 LLM 的記憶體佔用，並達成推論加速。

### [Cell 9] (Markdown)

在先前的課程中，我們已經學過量化的基本方法。
量化可分為兩種類型：

- 同時量化權重與激活值 (Quantize both weight and activation)
    - 較適合**計算受限 (computation-bounded)** 的場景：context stage、大批次 (large batch) 推論
    - 例如 SmoothQuant：W8A8 量化
- 僅權重量化 (Weight-only quantization)
    - 較適合**記憶體受限 (memory-bounded)** 的場景：decoding stage、單批次 (single batch) 推論
    - 例如本次實驗要介紹的 AWQ：W4A16 量化

### [Cell 10] (Markdown)

以 LLaMA-65B 模型為例，在單批次推論的 decoding stage，我們需要執行 GEMV 運算 $[1, 8192] \times [8192, 8192]$。以 NVIDIA A100 80G 為例，其半精度 (FP16) 效能為 312TFLOPS，記憶體頻寬約為 2000GB/s。因此其計算密度 (computation intensity) 為：

$$
\frac{\text{FLOP}}{\text{Byte}} = \frac{2\times 8192^2}{8192^2} << \frac{3.12\times 10^{11}}{2\times 10^9}
$$

這是非常典型的記憶體受限情境 (相差約 $10^2$ 倍)，這就是為什麼我們需要低位元的權重量化。

### [Cell 11] (Markdown)

## 環境設定 (Setup)

### [Cell 12] (Code)

```python
print('Installing packages...')
!pip install torch transformers==4.31.0 accelerate==0.21.0 sentencepiece==0.1.99 tokenizers==0.13.3 datasets==2.15.0 tqdm zstandard
```

### [Cell 13] (Code)

```python
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import gc
```

### [Cell 14] (Markdown)

這裡我們使用 wikitext-2 數據集進行評估。該數據集會由程式碼自動下載。

### [Cell 15] (Code)

```python
def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))
```

### [Cell 16] (Markdown)

以下程式碼用來計算模型大小。

### [Cell 17] (Code)

```python
def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB
```

### [Cell 18] (Markdown)

我們先來評估 FP32 模型的困惑度 (perplexity) 與模型大小。

### [Cell 19] (Code)

```python
model_path = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=32, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")
```

### [Cell 20] (Markdown)

均勻量化 (Uniform quantization) 是將位於 $[\beta, \alpha]$ 範圍內的實數值，映射到 $[0, 2^{b} - 1]$ 的區間內。

符號說明：

- 量化後權重 (Quantized Weight)：$w_q$

- 縮放因子 (Scale factor)：$s_q$

- 零點 (Zero Point)：$z$
\begin{equation}
s_q = \frac{\alpha - \beta}{2^{b} - 1} \tag{1},
\end{equation}
\begin{equation}
z = -\text{Round}(\beta * scale) \tag{2}
\end{equation}
\begin{equation}
w_q = \text{Clamp}(\text{Round}(\frac{w}{s_q}) + z) \tag{3},
\end{equation}

### [Cell 21] (Markdown)

### 偽量化 (pseudo quantization)
以下程式碼是偽量化的實作。

偽量化用於**模擬**量化對模型造成的影響，但並不會真的把模型權重轉成低位元格式。(也就是說：先四捨五入到最接近的量化值，再**反量化 (dequantize) 回浮點數**。)

### [Cell 22] (Code)

```python
# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_group_size,
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)
```

### [Cell 23] (Markdown)

我們來評估量化成 3-bit 後模型的困惑度與模型大小。

### [Cell 24] (Code)

```python
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")
```

### [Cell 25] (Markdown)

我們可以看到模型大小下降了，但困惑度卻顯著上升。

### [Cell 26] (Markdown)

在 LLM 的激活值 (activations) 中有一個觀察：**離群值 (outliers) 只出現在少數的通道 (channels) 中**。如果某個通道有離群值，它會**持續出現在所有的 token 上**。對於給定的某個 token，各通道之間的變異很大 (某些通道的激活值非常大，但大多數都很小)；但對於給定的某個通道，跨 token 之間的數值大小變異卻很小 (離群通道始終都是大的)。

根據 AWQ 的觀察，對應到激活值離群值的權重通道更為**顯著 (salient)**，保留這些顯著權重可以帶來顯著的效能提升。接下來，讓我們試著找出這些顯著權重並保留其原始數值，觀察困惑度的變化。

以下程式碼用來載入校準數據集 (calibration dataset)，以便取得激活值離群值來辨識顯著權重。

### [Cell 27] (Code)

```python
def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

@torch.no_grad()
def get_calib_feat(model, tokenizer):
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return input_dict
```

### [Cell 28] (Code)

```python
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
input_feat = get_calib_feat(model, tokenizer)
```

### [Cell 29] (Markdown)

### 問題 1 (Question 1, 50 分)
#### 問題 1.1 (20 分)
接下來，請在量化前後加入程式碼，以保護 1% 的顯著權重通道 (重要性最高的 1% 通道)，確保它們的數值在量化後保持不變。(**目標困惑度為 17.15**)

### [Cell 30] (Code)

```python
@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(
    model, w_bit, q_group_size, input_feat
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Find 1% of the salient weight channels according to importance (hint: use torch.topk())
            num_salient_channels = max(1, int(0.01 * importance.shape[0]))
            outlier_indices = torch.topk(importance, k=num_salient_channels, largest=True).indices
            assert outlier_indices.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the salient weight channels
            outlier = m.weight.data[:, outlier_indices].clone()

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Restore the 1% salient weight channels to their original FP16 values
            m.weight.data[:, outlier_indices] = outlier

            ############### YOUR CODE ENDS HERE #################
```

### [Cell 31] (Code)

```python
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_salient_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")
```

### [Cell 32] (Markdown)

#### 問題 1.2 (15 分)
讓我們做一個消融實驗 (ablation experiment)：**隨機**保護 1% 的權重通道，確保它們的數值在量化後保持不變，然後觀察困惑度。(**預期困惑度會超過 100**)

### [Cell 33] (Code)

```python
@torch.no_grad()
def pseudo_quantize_model_random_weight_fp16(
    model, w_bit, q_group_size, input_feat
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Randomly choose 1% of the weight channels
            num_random_channels = max(1, int(0.01 * importance.shape[0]))
            outlier_indices = torch.randperm(importance.shape[0])[:num_random_channels]
            outlier_mask = outlier_indices # Renamed for consistency with original placeholder
            assert outlier_mask.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the selected weight channels
            outlier = m.weight.data[:, outlier_mask].clone()

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Restore the 1% selected weight channels to their original FP16 values
            m.weight.data[:, outlier_mask] = outlier

            ############### YOUR CODE ENDS HERE #################
```

### [Cell 34] (Code)

```python
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_random_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")
```

### [Cell 35] (Markdown)

#### 問題 1.3 (15 分)
請提出一個可能的解釋：為什麼顯著權重通道 (salient weight channels) 如此重要？

#### 解答 1.3 (Answser 1.3)
############### YOUR ANSWER STARTS HERE #################


############### YOUR ANSWER ENDS HERE #################

### [Cell 36] (Markdown)

### 問題 2 (Question 2, 50 分)

### [Cell 37] (Markdown)

雖然將 0.1% 的權重保留為 FP16 可以在模型大小 (以總位元數衡量) 沒有明顯增加的情況下改善量化後的效能，但這種混合精度 (mixed-precision) 的資料型別會讓系統實作變得困難。我們需要想出一個方法，能在不真的把重要權重保留為 FP16 的前提下保護它們。

### [Cell 38] (Markdown)

根據 AWQ 的方法論，只要單純地**放大 (scale up)** 顯著權重通道就可以保護它們。原理如下：

- 考慮一個線性層的通道 $\mathbf{y} = \mathbf{w}x$ (源自 $\mathbf{W}x$)。我們關注的是 $Q(\mathbf{w})x$ 所帶來的量化誤差。

- $Err(Q(\mathbf{w}) x) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x$，其中 $Δ = \frac{\max(|w|)}{2^{N - 1}}$。
- 經過縮放的版本為 $Err(Q(\mathbf{w} \cdot s)(\frac{x}{s})) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x\cdot \mathbf{\frac{1}{s}}$。
- $RoundErr$ 始終約為 0.25 (在 0~0.5 之間的平均值)。
- 當 group size 相對較大時 (例如 128)，放大單一通道通常不會提高該群組內的最大值 (也就是 $Δ$ 維持不變)。
- 因此，$Err(Q(\mathbf{w} \cdot s)(\frac{x}{s})) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x\cdot \mathbf{\frac{1}{s}}$ < $Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x = Err(Q(\mathbf{w}) x)$。

### [Cell 39] (Markdown)

以下圖為例，若我們假設採用 3-bit 整數量化，那麼 $W$ 中第二列最後一行的數值 $(+1.4)$ 所造成的量化誤差應為 $Err(Q(\mathbf{w}) x) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x$ = $\frac{4}{2^{3 - 1}} * |1.4 - 1.0| * (2 + 2 + 2) = 2.4$。

如果將第二個通道放大 $2$ 倍，所產生的量化誤差會降低為 $\frac{4}{2^{3 - 1}} * |2.8 - 3.0| * (2/2 + 2/2 + 2/2) = 0.6$。

### [Cell 40] (Markdown)

![scaleup.png](assets/lab4/scaleup.png)

### [Cell 41] (Markdown)

#### 問題 2.1 (20 分)
請撰寫程式碼來放大顯著權重通道，接著對其進行量化，最後再縮放回原本的比例，並觀察困惑度的變化。(**目標困惑度為 18.93**)

### [Cell 42] (Code)

```python
@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Find 1% of the salient weight channels
            outlier_mask =
            assert outlier_mask.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # To simulate applying the scale factor, we can simply multiply it before quantization, and then divide by the scale factor after quantization.
            # Scale up the values of the salient weight channels
            m.weight.data[:, outlier_mask] *= scale_factor

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Scale back down the values of the salient weight channels


            ############### YOUR CODE ENDS HERE #################
```

### [Cell 43] (Code)

```python
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_scaleup(model, w_bit=3, q_group_size=128, input_feat=input_feat, scale_factor=2)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")
```

### [Cell 44] (Markdown)

#### 問題 2.2 (15 分)
請在程式碼中嘗試不同的縮放因子 (scale factor，例如 1、2、3、4)，並觀察困惑度的變化。

你是否觀察到困惑度先下降、之後又上升？請根據上述原理解釋為什麼會發生這種現象。

#### 解答 2.2 (Answer 2.2)
############### YOUR ANSWER STARTS HERE #################


############### YOUR ANSWER ENDS HERE #################

### [Cell 45] (Markdown)

### 問題 2.3 (15 分)
由於微調 (fine-tuning) 具有不穩定性，在一個預先定義好的搜尋空間中尋找最佳的 $s$ 會是更好的選擇。我們可以在搜尋空間中找出最佳的縮放值，以在保護顯著權重的同時，也一併考慮其他數值。在實務上可以觀察到，只考慮激活值就足以獲得良好的結果。請補上搜尋的程式碼並執行，觀察困惑度。(**目標困惑度為 17.92**)

### [Cell 46] (Markdown)

$$
𝐋(\mathbf{s})=\lVert Q(\mathbf{W}\cdot \mathbf{s})  (\mathbf{s^{-1}} \cdot \mathbf{X}) - \mathbf{W}\mathbf{X}  \rVert,  \quad\mathbf{s}= \mathbf{s_X}^{\alpha}
$$
$$
\mathbf{s}^* = \text{argmin}_{\mathbf{s}} 𝐋(\mathbf{s}),\quad \alpha^*=\text{argmin}_{\alpha} 𝐋(\mathbf{s_X}^{\alpha})
$$

### [Cell 47] (Code)

```python
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def auto_scale_block(module, name, w_bit,
                     q_group_size,
                     input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        ############### YOUR CODE STARTS HERE ###############

        # Step 1: Initialize the best_error, best_ratio and best_scales
        best_error =
        best_ratio =
        best_scales =

        ############### YOUR CODE ENDS HERE #################

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Calculate the scales by the formula: scales = s_x^ratio
            scales =
            assert scales.shape == s_x.shape

            ############### YOUR CODE ENDS HERE #################

            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:

                scales = scales.to(fc.weight.device)

                # Scale up the values of the weight channels
                fc.weight.mul_(scales)

                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)

                ############### YOUR CODE STARTS HERE ###############

                # Step 3: Scale back down the values of the weight channels


                ############### YOUR CODE ENDS HERE #################

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # attention input
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # attn out
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.self_attn.out_proj, [module.self_attn.out_proj], inp)
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # fc1
    inp = input_feat[name + '.fc1']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # fc2
    inp = input_feat[name + '.fc2']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)

@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(
    model, w_bit, q_group_size, input_feat
):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)
```

### [Cell 48] (Code)

```python
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_auto_scale(model, w_bit=3, q_group_size=128, input_feat=input_feat)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")
```

### [Cell 49] (Markdown)

## 加分題 (Bonus point)
你有想到任何不使用混合精度 (mixed precision) 的優化技巧嗎？試著實作它們來進一步改善困惑度！如果你能把困惑度進一步降低到 $x$，你可以在此獲得 $\max(0, (17.92 - x) \times 10)$ 的加分！

### [Cell 50] (Markdown)

總結來說，我們可以在不使用混合精度的情況下大幅降低困惑度。透過高效率的 kernel 實作，4-bit 模型能在推論時達到不錯的加速。透過下一節對 TinyChatEngine 的學習，我們就能像簡介中展示的 demo 一樣，在自己的筆電上運行 LLaMA-7B 模型。
