# Session Transcript: MIT 6.5940 Lecture 7 Neural Architecture Search (Part I) Q&A

## Context
Deep dive into MIT 6.5940 (EfficientML.ai) Lecture 7 (Neural Architecture Search Part I) text transcript and core Q&A.

---

### Round 1: 讀取 '.books/efficient-ml/EfficientML.ai Lecture 7 - Neural Architecture Search (Part I) (MIT 6.5940, Fall 2023, Zoom).txt'
**User:**
讀取 '.books/efficient-ml/EfficientML.ai Lecture 7 - Neural Architecture Search (Part I) (MIT 6.5940, Fall 2023, Zoom).txt'

**Gemini:**
我已經成功讀取並整理了 `.books/efficient-ml/EfficientML.ai Lecture 7 - Neural Architecture Search (Part I) (MIT 6.5940, Fall 2023, Zoom).txt` 的完整內容。

以下為本堂課程（MIT 6.5940 Lecture 7）的核心內容摘要：

---

# MIT 6.5940 Lecture 7: Neural Architecture Search (Part I) 課程摘要

本課開始進入**神經架構搜索（Neural Architecture Search, NAS）**的篇章。先前探討的剪枝（Pruning）與量化（Quantization）是在已有的模型上進行壓縮，而 NAS 的核心目標在於**如何從頭開始設計出高效且緊湊的模型**（即「用 AI 設計更好的 AI」）。

## 一、 高效神經網絡的考量維度
設計高效神經網絡需要在多個不可兼得的維度之間取得最優平衡（Trade-off）：
1. **延遲 (Latency)**：對實時交互與安全應用（如自動駕駛）至關重要。
2. **吞吐量 (Throughput)**：在數據中心端處理海量數據的關鍵指標。
3. **準確度 (Accuracy)**：首要的效能指標。
4. **能耗 (Energy/Electricity)**：訓練和部署大模型（如 H100、A100）的電費成本與環境保護考量（Green AI）。
5. **存儲與傳輸 (Storage/Bandwidth)**：影響 App Store 下載大小、OTA（Over-The-Air）更新頻寬以及分佈式訓練時的梯度通信開銷。

---

## 二、 基本運算與經典手動設計塊（Primitives & Building Blocks）
在進入自動化搜索前，課程回顧了人類手動設計模型的底層運算與 MAC（Multiply-Accumulate）/ 參數計算：

*   **全連接層與 2D 卷積**：基本的空間與通道計算。
*   **分組卷積 (Group Conv) 與深度卷積 (Depthwise Conv)**：極大地減少了計算量與參數。
*   **ResNet Bottleneck 模塊**：藉由 $1\times1$ 降維 $\rightarrow$ $3\times3$ 卷積 $\rightarrow$ $1\times1$ 升維，相比直觀的 $3\times3$ 卷積，能減少約 8.5 倍的計算量與參數。
*   **MobileNetV2 Inverted Bottleneck 模塊**：
    *   **設計理念**：深度卷積（Depthwise Conv）運算量極輕，但表達能力有限。因此先用 $1\times1$ 擴張通道數（擴張因子通常為 6），做完深度卷積後，再用 $1\times1$ 投影回原通道數。
    *   **隱患分析**：雖然模型大小與參數減少，但**峰值激活值（Peak Activation）大幅增加**（通道擴張 6 倍）。在訓練時，總激活值僅減少 10%，導致顯存與內存佔用極大，這在微控制器（MCU）等極度受限的設備上會成為瓶頸。
*   **ShuffleNet**：引入**通道重排（Channel Shuffle）**，解決分組卷積中不同組（Groups）之間無法進行信息交互的問題。
*   **Transformer 模塊（自注意力機制）**：介紹 $Q, K, V$ 投影與縮放點積注意力，其計算複雜度為 $O(N^2 D)$。

---

## 三、 神經架構搜索（NAS）的核心組成

NAS 的架構主要由三個要素組成：**搜索空間（Search Space）**、**搜索策略（Search Strategy）** 與 **性能評估（Performance Estimation）**。本課著重於前兩者。

### 1. 搜索空間 (Search Space)
*   **細胞級別搜索空間 (Cell-level Search Space)**：
    *   搜索單個 Cell（如 Normal Cell 和 Reduction Cell）的拓撲結構，再重複堆疊成網絡（如 NASNet）。
    *   **限制**：機器設計的 Cell 分支複雜，數據依賴多，導致硬件部署時內存釋放不及時，造成多餘的內存足跡（Memory Footprint），且難以在硬件上高效並行。
*   **網絡級別搜索空間 (Network-level Search Space)**：
    *   搜索網絡的深度（Depth）、寬度（Width）、輸入解析度（Resolution）、卷積核大小（Kernel Size）及網絡拓撲。
    *   **TinyNAS（針對極端微控制器 MCU 的設計）**：
        *   微控制器通常只有幾百 KB 的 SRAM 與 1-2 MB 的 Flash。
        *   **代理啟發式指標**：數據移動代價極高，而計算便宜。在相同的內存約束下，**最大化 FLOPs** 可以反映出模型具有更大的容量（Capacity），從而更容易達到高精度。TinyNAS 藉此在不經過完整訓練的情況下，大幅縮減了不合適的搜索空間。

### 2. 搜索策略 (Search Strategy)
*   **網格搜索 (Grid Search)**：暴力枚舉，代價高昂。引入**複合縮放（Compound Scaling）**概念（即同時縮放深度、寬度與分辨率，而非僅縮放單一維度）。
*   **隨機搜索 (Random Search)**：常用於基準測試與代碼正確性的初步驗證（Sanity Check）。
*   **強化學習 (Reinforcement Learning)**：利用 RNN 作為控制器生成架構，訓練子網絡獲取 Accuracy 作為 Reward，再更新控制器。缺點是運算速度非常慢。
*   **梯度下降/可微搜索 (Differentiable NAS, 如 DARTS)**：
    *   將多條候選路徑賦予權重概率（如 $\alpha, \beta, \gamma$），使架構選擇變得可微並能進行端到端訓練。
    *   **缺點**：必須把所有候選路徑的激活值保留在 GPU 內存中，容易導致內存爆炸。
    *   **可微延遲搜索**：將線下測量的硬件延遲納入 Loss 函數，使優化目標同時兼顧精度與硬件延遲。
*   **進化算法搜索 (Evolutionary Search)**：
    *   模擬生物進化，保留適應度（Fitness）最佳的個體作為下一代家長。
    *   支持**突變 (Mutation)**（隨機改變網絡深度或算子大小）與**交叉 (Crossover)**（融合兩個優秀個體的結構參數）。這是課程後續作業實作的重點。

---

### Round 2: 搜索空間 (Search Space) 是什麼
**User:**
搜索空間 (Search Space) 是什麼

**Gemini:**
在神經架構搜索（Neural Architecture Search, NAS）中，**搜索空間（Search Space）**是指**搜索算法可以選擇、評估與組合的所有可能神經網絡架構的集合**。它定義了搜索的邊界與範圍，決定了我們能「搜出什麼樣的模型」。

根據 MIT 6.5940 課程內容，搜索空間主要分為以下兩個層級，並且其設計與縮減策略對於搜索效率至關重要：

---

### 一、 搜索空間的兩大層次

#### 1. 細胞級別搜索空間 (Cell-level Search Space)
這種設計方法不直接搜索整個網絡，而是**搜索一個局部構建塊（稱為 Cell 或細胞）的內部結構**，然後通過重複堆疊這些 Cell（如 Normal Cell 和用於降低分辨率的 Reduction Cell）來拼裝成完整的神經網絡。
*   **搜索的元素包括**：
    *   **輸入來源**：決定當前節點要接收哪一個前序圖層的輸出（例如：前一層或前前層的輸出，從而自動形成殘差/跳躍連接）。
    *   **轉換算子（Operations）**：在每個輸入上套用什麼運算（例如：$3\times3$ 深度可分離卷積、$5\times5$ 卷積、最大池化、恆等映射等）。
    *   **結合方式**：如何融合不同分支的輸出（例如：元素級相加 `Add` 或通道拼接 `Concat`）。
*   **特點**：機器設計的 Cell 通常分支非常複雜且凌亂，雖能提供高精度，但由於複雜的數據依賴，在實際硬件部署時容易導致內存佔用大（內存足跡高）且不易並行化。

#### 2. 網絡級別搜索空間 (Network-level Search Space)
這種設計關注的是整個網絡的宏觀配置（Macro-architecture），即如何調整網絡在各個維度上的規模。
*   **搜索的維度包括**：
    *   **深度 (Depth)**：網絡包含多少個模塊或圖層。
    *   **寬度 (Width)**：每一層卷積或模塊輸出的通道數量（如 128、256、512 等）。
    *   **解析度 (Resolution)**：輸入圖片或特徵圖的尺寸大小（例如從 $128\times128$ 到 $256\times256$ 之間選擇）。
    *   **卷積核大小 (Kernel Size)**：每一層使用多大的卷積核（如 $3\times3$、$5\times5$ 還是 $7\times7$）。
    *   **拓撲連接 (Topology)**：例如在圖像生成或分割任務中，網絡如何進行下採樣（Downsample）與上採樣（Upsample）的路徑網格。

---

### 二、 為什麼搜索空間的「設計」與「縮減」如此重要？

1.  **空間極其龐大**：
    即使只在細胞級別進行簡單的限制（例如：5 種候選算子、2 種結合方式、10 個層級），其可能組合的數量也可以輕易達到 **$10^{11}$ 次方** 以上。在如此龐大的空間中進行暴力搜索，在計算資源上是完全不可行的。
2.  **空間縮減（TinyNAS 的啟發式設計）**：
    為了在極度受限的邊緣設備（如微控制器 MCU，僅有幾百 KB 的 SRAM）上部署模型，必須對搜索空間進行合理的限制。
    *   課程中提到的 **TinyNAS** 提出一個核心思路：**數據移動代價極高，而計算便宜**。
    *   在相同的內存約束下，FLOPs（浮點運算量）越高的模型通常具有更大的模型容量與更高的潛在準確度。
    *   因此，無須事先花費高昂代價去訓練模型，我們可以直接通過分析「內存約束下能達到的最大 FLOPs 分佈」，從而將巨大的搜索空間**縮減**為一個高質量的精簡搜索空間，這使得在受限資源下搜索出高精度模型變得高效可行。

---

### Round 3: grouped covolution layer 是什麼
**User:**
grouped covolution layer 是什麼

**Gemini:**
**分組卷積層（Grouped Convolution Layer）**是一種旨在降低神經網絡計算量與參數量的卷積運算方式。

在傳統的標準卷積中，每一個輸出通道（Output Channel）都必須與**所有**的輸入通道（Input Channels）進行計算；而分組卷積則是將輸入和輸出通道分成若干個獨立的「組（Groups）」，並在組內獨立進行卷積運算。

---

### 一、 分組卷積的工作原理

假設我們有：
*   輸入通道數：$C_{in}$
*   輸出通道數：$C_{out}$
*   分組數（Groups）：$G$

1.  **通道劃分**：將輸入通道均勻分成 $G$ 組，每組包含 $C_{in} / G$ 個通道；同樣地，輸出通道也均勻分成 $G$ 組，每組 $C_{out} / G$ 個通道。
2.  **獨立卷積**：第 1 組的輸出通道，**僅與**第 1 組的輸入通道進行卷積計算；第 2 組的輸出通道，**僅與**第 2 組的輸入通道計算，以此類推。組與組之間在計算時是完全隔離、互不干擾的。

---

### 二、 參數與計算量的節省

分組卷積最核心的優勢在於能將**參數量**與**計算量（MACs）**同時縮減為原來的 $\frac{1}{G}$。

設卷積核大小為 $K_h \times K_w$，輸出特徵圖尺寸為 $H_{out} \times W_{out}$：

*   **標準卷積**：
    *   參數數量 = $K_h \times K_w \times C_{in} \times C_{out}$
    *   計算量 (MACs) = $H_{out} \times W_{out} \times K_h \times K_w \times C_{in} \times C_{out}$

*   **分組卷積 ($G$ 個分組)**：
    *   參數數量 = $G \times \left( K_h \times K_w \times \frac{C_{in}}{G} \times \frac{C_{out}}{G} \right) = \frac{\text{標準卷積參數}}{G}$
    *   計算量 (MACs) = $G \times \left( H_{out} \times W_{out} \times K_h \times K_w \times \frac{C_{in}}{G} \times \frac{C_{out}}{G} \right) = \frac{\text{標準卷積計算量}}{G}$

*例如：如果將卷積分為 4 組 ($G=4$)，則其參數與計算量直接降為原本標準卷積的 $\frac{1}{4}$（即 25%）。*

---

### 三、 極端情況：深度卷積 (Depthwise Convolution)

當分組數 $G$ 等於輸入通道數 $C_{in}$ 時（此時每組只有 1 個通道），分組卷積就演變成了**深度卷積（Depthwise Convolution）**。
*   這是 MobileNet 等輕量化網絡的核心基礎運算，它完全消除了跨通道的空間計算，將空間特徵提取（Spatial Modeling）極簡化。

---

### 四、 分組卷積的缺點與解決方案

*   **缺點（信息孤島）**：
    由於各個分組之間的計算完全獨立，**不同分組的通道之間無法進行信息交流**（例如第 1 組的特徵永遠看不到第 2 組的特徵）。這會限制模型的特徵表達能力與準確度。
*   **解決方案**：
    1.  **$1\times1$ 卷積（Pointwise Conv）**：在分組卷積或深度卷積後，加上一層標準的 $1\times1$ 卷積，專門負責在通道維度上混合信息（如 MobileNet 的設計）。
    2.  **通道重排（Channel Shuffle）**：如 ShuffleNet 的設計，在每次分組卷積後，將不同組的通道均勻打散並重新分組，以零計算代價的方式強迫不同分組之間進行信息交換。

---

### Round 4: 給予一個分組卷積跟標準卷積的 python 範例
**User:**
給予一個分組卷積跟標準卷積的 python 範例

**Gemini:**
這裡為您提供一個使用目前主流深度學習框架 **PyTorch** 實現標準卷積與分組卷積的 Python 程式碼範例。

在此範例中，我們會建立兩個卷積層，並對比它們的**輸出尺寸**、**權重張量（Weight Tensor）的形狀**以及**參數數量**。

---

### Python 程式碼範例 (使用 PyTorch)

```python
import torch
import torch.nn as nn

# 設定超參數
batch_size = 1
in_channels = 16   # 輸入通道數 (必須能被 groups 整除)
out_channels = 32  # 輸出通道數 (必須能被 groups 整除)
height, width = 64, 64  # 特徵圖的高與寬
kernel_size = 3    # 卷積核大小 (3x3)
groups = 4         # 分組數量 (用於分組卷積)

# 1. 建立一個隨機輸入張量 (Shape: [Batch, Channels, Height, Width])
input_tensor = torch.randn(batch_size, in_channels, height, width)
print(f"輸入張量尺寸: {input_tensor.shape}\n")

# =====================================================================
# 2. 標準卷積 (Standard Convolution)
# =====================================================================
# groups=1 為預設值，代表不分組
std_conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=1,
    padding=1,
    bias=False,  # 為了方便對比權重參數，這裡不使用偏置項
    groups=1
)

# 進行前向傳播
output_std = std_conv(input_tensor)

# 計算標準卷積的參數數量
std_params = sum(p.numel() for p in std_conv.parameters())

print("--- [標準卷積 (Standard Convolution)] ---")
print(f"輸出張量尺寸: {output_std.shape}")
print(f"權重張量尺寸 (out_channels, in_channels, K_h, K_w): {std_conv.weight.shape}")
print(f"參數總數量: {std_params} (計算公式: {out_channels} * {in_channels} * {kernel_size} * {kernel_size})")
print()

# =====================================================================
# 3. 分組卷積 (Grouped Convolution)
# =====================================================================
# 設定 groups=4，代表輸入與輸出通道都將被均分為 4 個小組獨立計算
grouped_conv = nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=1,
    padding=1,
    bias=False,
    groups=groups
)

# 進行前向傳播
output_group = grouped_conv(input_tensor)

# 計算分組卷積的參數數量
grouped_params = sum(p.numel() for p in grouped_conv.parameters())

print("--- [分組卷積 (Grouped Convolution)] ---")
print(f"輸出張量尺寸: {output_group.shape}")
# 注意：在 PyTorch 中，分組卷積的權重尺寸第二維會是 in_channels // groups
print(f"權重張量尺寸 (out_channels, in_channels // groups, K_h, K_w): {grouped_conv.weight.shape}")
print(f"參數總數量: {grouped_params} (計算公式: {out_channels} * ({in_channels} / {groups}) * {kernel_size} * {kernel_size})")
print()

# =====================================================================
# 4. 對比總結
# =====================================================================
reduction_ratio = std_params / grouped_params
print("--- [對比總結] ---")
print(f"標準卷積與分組卷積的輸出尺寸是否相同? {output_std.shape == output_group.shape}")
print(f"參數縮減比例: 標準卷積的參數是分組卷積的 {reduction_ratio:.1f} 倍 (正好等於 groups 數量)")
```

---

### 程式執行輸出結果（預期）

```text
輸入張量尺寸: torch.Size([1, 16, 64, 64])

--- [標準卷積 (Standard Convolution)] ---
輸出張量尺寸: torch.Size([1, 32, 64, 64])
權重張量尺寸 (out_channels, in_channels, K_h, K_w): torch.Size([32, 16, 3, 3])
參數總數量: 4608 (計算公式: 32 * 16 * 3 * 3)

--- [分組卷積 (Grouped Convolution)] ---
輸出張量尺寸: torch.Size([1, 32, 64, 64])
權重張量尺寸 (out_channels, in_channels // groups, K_h, K_w): torch.Size([32, 4, 3, 3])
參數總數量: 1152 (計算公式: 32 * (16 / 4) * 3 * 3)

--- [對比總結] ---
標準卷積與分組卷積的輸出尺寸是否相同? True
參數縮減比例: 標準卷積的參數是分組卷積的 4.0 倍 (正好等於 groups 數量)
```

---

### 重點解析

1.  **輸出特徵圖的尺寸完全相同**：
    不論是標準卷積還是分組卷積，在設定相同的 `stride` 和 `padding` 下，輸出的張量維度完全一致（都是 `[1, 32, 64, 64]`），這代表分組卷積在模型結構中可以**無縫替換**標準卷積。
2.  **PyTorch 權重形狀的設計**：
    *   標準卷積的權重形狀為 `[32, 16, 3, 3]`。
    *   分組卷積的權重形狀為 `[32, 4, 3, 3]`。
    *   這是因為在分組卷積中，每個輸出通道不再需要與全部 16 個輸入通道做計算，而只需要與所屬分組的 4 個輸入通道（$16 \div 4 = 4$）做計算。
3.  **參數量呈線性比例下降**：
    分組卷積的參數總數為 1152，正好是標準卷積 4608 的 **$\frac{1}{4}$**（即縮減了 4 倍），這與我們設定的 `groups=4` 完全對應。
