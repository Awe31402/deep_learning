# EfficientML.ai 第二講完整筆記：神經網路基礎（Basics of Neural Networks）

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2023 — Lecture 2
> **講者**：Song Han
> **來源**：Zoom 錄影逐字稿（1:12:14）＋ 官方投影片 `Lec02-Basics.pdf`（77 頁）
> **標記說明**：`【口頭】` 表示教授在課堂上口頭補充、投影片沒有的內容；`【Q&A】` 表示學生提問與回答。

---

## 目錄

- [0. 這堂課要解決什麼問題](#0-這堂課要解決什麼問題)
- [1. 術語與模型維度](#1-術語與模型維度)
- [2. Building Blocks：每一層的維度與公式](#2-building-blocks每一層的維度與公式)
  - [2.1 Linear（全連接層）](#21-linear全連接層)
  - [2.2 1D Convolution](#22-1d-convolution)
  - [2.3 2D Convolution](#23-2d-convolution)
  - [2.4 輸出尺寸公式與 Padding](#24-輸出尺寸公式與-padding)
  - [2.5 Receptive Field（感受野）](#25-receptive-field感受野)
  - [2.6 Strided Convolution](#26-strided-convolution)
  - [2.7 Grouped Convolution](#27-grouped-convolution)
  - [2.8 Depthwise Convolution](#28-depthwise-convolution)
  - [2.9 Pooling Layer](#29-pooling-layer)
  - [2.10 Normalization Layer](#210-normalization-layer)
  - [2.11 Activation Function](#211-activation-function)
  - [2.12 Transformer 預覽](#212-transformer-預覽)
- [3. 經典 CNN 架構](#3-經典-cnn-架構)
- [4. 效率指標（本講重點）](#4-效率指標本講重點)
  - [4.1 三個目標：Smaller / Faster / Greener](#41-三個目標smaller--faster--greener)
  - [4.2 Latency vs Throughput](#42-latency-vs-throughput)
  - [4.3 Energy：記憶體才是吃電大戶](#43-energy記憶體才是吃電大戶)
  - [4.4 #Parameters（參數量）](#44-parameters參數量)
  - [4.5 Model Size（模型大小）](#45-model-size模型大小)
  - [4.6 #Activations：Total vs Peak](#46-activationstotal-vs-peak)
  - [4.7 MAC / FLOP / FLOPS / OP / OPS](#47-mac--flop--flops--op--ops)
- [5. 一頁速查表](#5-一頁速查表)
- [6. 與後續課程／作業的連結](#6-與後續課程作業的連結)

---

## 0. 這堂課要解決什麼問題

### 0.1 算力供給 vs 模型需求的缺口

這是整門課的出發點：

| | 成長速度 |
|---|---|
| **硬體供給**（GPU 記憶體、電晶體數） | 摩爾定律，每 2 年約 2× |
| **模型需求**（參數量、算力） | 每 2 年 **超過 4×** |

兩條線是指數發散的。差距不會自己收斂，所以必須靠 **模型壓縮（model compression）** 與 **高效 AI 運算（efficient AI computing）** 從中間把缺口補起來。

【口頭】教授拿當天 MIT 首頁上他們自己的 EfficientViT 當例子：高解析度的逐像素預測（street scene segmentation）記憶體與運算量都非常大，優化後在 **NVIDIA Jetson AGX Orin**（一個手掌大的 Android 級 GPU，不是桌機 GPU）上跑到 **21 FPS**，baseline 只有 1.6 FPS。另一個例子是 Meta 的 **SAM（Segment Anything）**，在完全維持精度的前提下從 12 img/s 加速到 **840 img/s**。

### 0.2 這門課的三大板塊

| 板塊 | 內容 |
|---|---|
| **第一部分：通用推論優化** | Pruning、Quantization、Neural Architecture Search、Knowledge Distillation |
| **第二部分：應用專屬優化** | LLM（2 講）、Vision Transformer（1 講）、AIGC / Diffusion、Point Cloud、Video |
| **第三部分：訓練優化** | 分散式訓練、Model / Data / Pipeline Parallelism（例如 175B 的 GPT-3） |

### 0.3 先修需求

【口頭】上一講很多學生問先修，教授重講一次：

- 機器學習：**6.036** 或同等課程（activation function、SGD 等）
- 計算機架構：**6.191 / 6.004** 或同等（page table、cache locality 等）
- **C / C++**：最後一次 lab 要用
- 課程網站 `efficientml.ai` 放投影片與作業；**Canvas 是唯一交作業的地方**；討論在 Discord
- Lab 0（PyTorch 入門）不計分，但要交，目的是熟悉繳交流程

---

## 1. 術語與模型維度

### 1.1 生物神經元 → 數學模型

整堂課的核心公式其實只有一條：

$$y_j = f\left(\sum_i w_i x_i + b\right)$$

生物與數學的對應關係：

| 生物 | 數學 / 深度學習 | 別名 |
|---|---|---|
| **Synapse（突觸）** | 權重 $w$ | **weight、parameter** |
| **Neuron / Axon（神經元）** | 特徵值 $x$、$y$ | **activation、feature** |
| **Cell Body 的閾值判斷** | activation function $f$ | — |

**這組別名要記熟，之後整學期會混用**：

- 說「GPT-3 有 175B parameters」＝ 175B 個 **weight** ＝ 175B 個 **synapse**，是同一件事。
- 說「activation 太大」＝「feature map 太大」＝「neuron 太多」，也是同一件事。

### 1.2 Width（寬）vs Depth（深）

以一個 3 層網路（layer 0 / 1 / 2，中間兩層 hidden layer）為例：

- **Width（寬度）** = hidden dimension 的大小 = 每層有幾個神經元
- **Depth（深度）** = 有幾層

「這個模型很寬」講的是 hidden size；「這個模型很深」講的是層數。

### 1.3 【口頭】思考題：寬淺 vs 窄深，哪個在 GPU 上跑得快？

教授在這裡停下來讓全班想。**前提是參數量與計算量相同。**

**答案：寬而淺的比較快。** 理由：

1. **Kernel call 數量**：每一層大致對應一次 GPU kernel call。層數越多，call 越多，**啟動開銷（launch overhead）** 越大。
   - （雖然可以用 kernel fusion 把多層融合，但那是後面才會講的優化技巧。）
2. **GPU 利用率（utilization）**：如果矩陣很小（例如 3×4），GPU 上大量運算單元閒置，**嚴重 under-utilized**。寬的層矩陣大，平行度高，利用率好。

**但天下沒有白吃的午餐**：

> 為了讓模型好收斂、達到高準確率，通常還是需要**深**的模型。

【口頭】教授說這正是 neural architecture design 有趣的地方 —— 你必須同時平衡：
- **演算法端**：高精度、好訓練、易收斂 → 傾向深
- **硬體端**：hardware-friendly、高利用率、快 → 傾向寬

這個 trade-off 是後面 NAS（Lecture 7-8）整章的主題。

---

## 2. Building Blocks：每一層的維度與公式

> 【口頭】教授反覆強調的學習方法：
> **「每學一個新的 layer，最關鍵的就是搞懂它的維度（dimension）。」**
> 因為所有效率指標 —— 參數量、activation 大小、MAC —— 全部都是從維度算出來的。

### 通用符號表

| 符號 | 意義 |
|---|---|
| $n$ | Batch size |
| $c_i$ / $c_o$ | Input / Output channels |
| $h_i, w_i$ / $h_o, w_o$ | Input / Output height, width |
| $k_h, k_w$ | Kernel height, width |
| $g$ | Groups |
| $s$ | Stride |
| $p$ | Padding |

---

### 2.1 Linear（全連接層）

最基本的層：對輸入做一次線性變換再加 bias。

**單一輸入（無 batch）：**

| Tensor | Shape |
|---|---|
| Input $X$ | $(c_i,)$ |
| Output $Y$ | $(c_o,)$ |
| Weight $W$ | $(c_o, c_i)$ |
| Bias $b$ | $(c_o,)$ |

例：input `1×5`、weight `5×3` → output `1×3`。

**加上 batch 維度：**

| Tensor | Shape |
|---|---|
| Input $X$ | $(n, c_i)$ |
| Output $Y$ | $(n, c_o)$ |
| Weight $W$ | $(c_o, c_i)$ |
| Bias $b$ | $(c_o,)$ |

**關鍵觀察**：不論 batch 多大，**weight 和 bias 的維度完全不變**。

> 【口頭】教授在這裡點出一個貫穿全課的分野：
> **weight / bias 是「input-agnostic」（與輸入無關）—— 它們是模型本身；activation 才會隨 batch 和解析度變動。**
> 這就是為什麼之後算 model size 只看參數，算記憶體瓶頸卻要看 activation。

---

### 2.2 1D Convolution

**動機**：Linear layer 中每個 output 都連到**所有** input。但如果 input feature map 很大，而 output 通常只跟 input 的**局部子集**有關呢？那就用 convolution。

**典型應用**：語音（speech）是最常見的一維訊號。

> 【口頭】這裡教授特別澄清一個很容易搞混的地方：
> 「投影片上這張圖看起來是二維的，但它其實是**一維訊號**。」
> **一個維度是 channel（通道），另一個才是 spatial（時間/空間）。**
> 每個時間步 $t_1, t_2, t_3\ldots$ 上都有一個一維向量來表示那個時刻的特徵（音量、音高等）。
> 就像影像的第一層有 RGB 三個 channel 一樣 —— **channel 維度不算「空間維度」**。

**維度：**

| Tensor | Shape |
|---|---|
| Input $X$ | $(n, c_i, w_i)$ |
| Output $Y$ | $(n, c_o, w_o)$ |
| Weight $W$ | $(c_o, c_i, k_w)$ |
| Bias $b$ | $(c_o,)$ |

一個 filter 產生一個 output channel；要 3 個 output channel 就要 3 個 filter。把 window 沿時間軸滑動，就產生完整的輸出序列。

---

### 2.3 2D Convolution

從 1D 推廣：現在 spatial 是 $(h, w)$。例如 1024×768 的影像，加上 channel 就是三維 tensor。

**維度（6 個獨立維度，是本講最需要記牢的一組）：**

| Tensor | Shape |
|---|---|
| Input $X$ | $(n, c_i, h_i, w_i)$ |
| Output $Y$ | $(n, c_o, h_o, w_o)$ |
| Weight $W$ | $(c_o, c_i, k_h, k_w)$ |
| Bias $b$ | $(c_o,)$ |

**運作方式**：一個 $k_h \times k_w$ 的 filter 與 input tensor 做卷積 → 產生**一個** output channel。$c_o$ 個 filter → $c_o$ 個 output channel。filter 在 $h$、$w$ 方向滑動，產生完整的 output feature map。

**例**：5×5 input、3×3 kernel → output 是 3×3（**少了 2 個 pixel**）。3 個 filter → output shape `3×3×3`。

---

### 2.4 輸出尺寸公式與 Padding

#### 為什麼 feature map 會變小

3×3 kernel 在 5×5 input 上只能移動 3 個位置（起點 + 往右 2 次），所以輸出是 3。

**基本公式（無 padding、stride=1）：**

$$h_o = h_i - k_h + 1 \qquad w_o = w_i - k_w + 1$$

**驗證**：4×4 input、3×3 kernel → $4 - 3 + 1 = 2$，輸出 2×2。✔

#### Padding

為了**維持 feature map 尺寸不變**，在 input 邊界補值：

$$h_o = h_i + 2p - k_h + 1$$

**例**：$h_i = w_i = 5$，$k = 3$，$p = 1$ → $h_o = 5 + 2 \times 1 - 3 + 1 = 5$。尺寸保住了。✔

**四種 padding：**

| 種類 | 做法 | 備註 |
|---|---|---|
| **Zero Padding** | 邊界補 0 | **PyTorch 預設，實務上最常用** |
| **Reflection Padding** | 以邊界為鏡面反射（`1 5 9` → `9 5 1 5 9`） | |
| **Replication Padding** | 複製最靠近的實際數值 | |
| **Constant Padding** | 補一個固定常數 | |

---

### 2.5 Receptive Field（感受野）

#### 定義

**輸出的某一個元素，會被輸入中多大的區域影響。**

【口頭】教授的比喻：「你要理解我在做什麼動作，不能只看我身體的一小部分，得看整個人才知道。」

#### 成長規律

- 單一 3×3 conv：一個 output 元素依賴 $3 \times 3$ 的輸入窗口
- **每疊一層 conv，感受野增加 $k - 1$**

$$\text{RF} = L \cdot (k - 1) + 1$$

**逐層驗證（$k = 3$）：**

| 層數 $L$ | 感受野 |
|---|---|
| 1 | 3 × 3 |
| 2 | 5 × 5（= 25 個 pixel） |
| 3 | 7 × 7 |

#### 問題與解法

> **問題**：對大張影像，要讓輸出「看見」整張圖，需要**非常多層**。
> **解法**：在網路內部做 **downsample（下採樣）**。

#### 【Q&A】感受野應該要多大？

**學生問**：是不是要大到「某個像素的感受野涵蓋整張圖」？還是那樣反而太吵、太糟？

**教授答**：

- 經典設定是 input 224×224 → 最後 feature map 7×7，這樣做分類沒問題。
- **但不是永遠如此**。反例：如果任務是「判斷 A 是否在追 B」，而 **A 在左上角、B 在右下角**，那你**必須同時看到兩個角落**才能理解它們的關係。
- **結論：取決於任務。一般來說感受野大一點是好事。**

#### 為什麼不能只靠加層數或加大 kernel

| 做法 | 代價 |
|---|---|
| **增加層數** | 更慢（見 §1.3：kernel call 增加、GPU 利用率下降） |
| **加大 kernel** | 參數暴增：3×3 是 9 個權重，**7×7 是 49 個** |
| **✅ Downsample** | 便宜又有效 → 見 §2.6 |

---

### 2.6 Strided Convolution

**Stride $s$**：filter 每次滑動幾格。

$$h_o = \left\lfloor \frac{h_i + 2p - k_h}{s} \right\rfloor + 1$$

#### 為什麼 stride 能放大感受野

投影片上的對照（$k = 3$，都是 **2 層**）：

| 設定 | 2 層後的感受野 |
|---|---|
| $s = 1$ | **5 × 5** |
| $s = 2$ | **7 × 7** |

同樣是 7×7 的感受野，$s=1$ 要 **3 層**，$s=2$ 只要 **2 層**。

【口頭】原理：stride = 2 時 feature map 解析度直接砍半（8×8 → 4×4），更多的 input pixel 被「壓縮」映射到同一個 output pixel，所以感受野擴張得更快。

> **重要**：stride **不影響參數量**（參數只跟 $c_o, c_i, k_h, k_w$ 有關），但會**大幅減少 MAC 與 activation 大小**，因為 $h_o, w_o$ 縮小了。

---

### 2.7 Grouped Convolution

**核心想法**：不再讓每個 output channel 都依賴**所有** input channel，而是**分組**，組與組之間完全不互通。

**維度：**

| Tensor | Shape |
|---|---|
| Input $X$ | $(n, c_i, h_i, w_i)$ |
| Output $Y$ | $(n, c_o, h_o, w_o)$ |
| **Weight $W$** | $(g \cdot \frac{c_o}{g}, \frac{c_i}{g}, k_h, k_w)$ |
| Bias $b$ | $(c_o,)$ |

**關鍵**：**input / output feature map 尺寸完全不變，只有 weight 變小 —— 變成原本的 $1/g$。**

#### 【口頭】教授的具體數字

以 $c_i = c_o = 16$、忽略 kernel size 為例：

| | 參數量 |
|---|---|
| $g = 1$（一般 conv） | $16 \times 16 = 256$ |
| $g = 2$ | $8 \times 8 = 64$，兩組共 $128$ |

$128 / 256 = 1/2 = 1/g$。✔

---

### 2.8 Depthwise Convolution

**定義**：grouped convolution 的極端情況，$g = c_i = c_o$。**每個 channel 有自己獨立的 filter，channel 之間完全不交互。**

**維度：**

| Tensor | Shape |
|---|---|
| Weight $W$ | $(c, k_h, k_w)$ ← **只剩三項！** |

原本是 $c_o \cdot c_i \cdot k_h \cdot k_w$，因為 $g = c$ 抵消掉一個 channel 維度，變成 $c \cdot k_h \cdot k_w$。

這是 **MobileNet 家族的基礎**。

#### ⚠️ 【口頭】本講最重要的反例：參數少 ≠ 速度快

教授在這裡花了很長時間強調，這段投影片上沒有：

> 「這其實**不是一個很有效率的設計**。」

拆解一下：

| 指標 | Depthwise conv 的影響 |
|---|---|
| **參數量** | ✅ 大幅下降 |
| **FLOPs / MACs** | ✅ 大幅下降 |
| **Activation 大小** | ❌ **完全沒變** |

**然後是連鎖反應：**

1. 減少權重 → **模型容量（capacity）流失**
2. 為了補回容量 → 必須用**很大的 expansion ratio**
3. **MobileNetV2 的 channel 數在 block 中間膨脹 6 倍**
4. Channel 變多 → **activation 變大** → **大量的記憶體資料搬移（data movement）**
5. 而 data movement **非常昂貴**（見 §4.3：比運算貴兩個數量級）

> **教授的結論**：
> **「天下沒有白吃的午餐。表面上我們減少了權重，但要付出的代價是增加 channel 數。這在 parameter 上很有效率，但不一定能轉換成硬體上的速度。」**

這個觀念是後面 **Hardware-aware NAS**（Lecture 7-8）與 **MCUNet** 的直接動機。

---

### 2.9 Pooling Layer

**目的**：把 feature map 縮小（downsample）。

- 概念類似 convolution，但是在感受野內做「池化」而非加權求和
- **stride 通常設成跟 kernel size 一樣：$s = k$**
- **對每個 channel 獨立操作**
- **沒有可學習的參數**

| 種類 | 例（2×2, stride 2） |
|---|---|
| **Max Pooling** | 取窗口內最大值 |
| **Average Pooling** | 取窗口內平均值 |

投影片範例（input `[[5,0,1,7],[2,1,3,5],[0,2,3,1],[2,8,1,3]]`）：

- Max Pool → `[[5,7],[8,3]]`
- Avg Pool → `[[2,4],[3,2]]`

#### 【口頭】Pooling vs Strided Conv 的取捨

| | 參數 | 表達力 |
|---|---|---|
| **Pooling** | **0 個** | 低 |
| **Strided Conv** | 有 | 高 |

> 「零參數也意味著零容量、零表達力。所以你得謹慎地選 —— 要用 2×2 stride-2 的 strided convolution，還是一個 downsample layer？」

這正是 neural architecture search 要自動化決定的事情之一。

---

### 2.10 Normalization Layer

**目的**：把特徵正規化，讓優化更快、更穩。

#### 公式

$$\hat{x}_i = \frac{x_i - \mu_i}{\sigma_i}, \qquad \sigma_i = \sqrt{\frac{1}{m}\sum_{k \in \mathcal{S}_i}(x_k - \mu_i)^2 + \epsilon}$$

然後接一個 **per-channel 的線性變換**（可學習的 scale $\gamma$ 與 shift $\beta$），補回被正規化「壓掉」的表達能力：

$$y = \gamma_{i_c}\hat{x}_i + \beta_{i_c}$$

【口頭】**為什麼要有 $\epsilon$**：因為要除以標準差，$\epsilon$ 是為了**防止除以零**。

#### 四種正規化的差別 —— 只差在「選哪一組 pixel $\mathcal{S}_i$」

四種方法**公式完全一樣**，唯一差別是計算 mean / std 的**集合怎麼選**：

| 種類 | 選取的維度 | 意義 |
|---|---|---|
| **Batch Norm** | $N \times H \times W$ | **每個 channel** 有自己的 mean / std |
| **Layer Norm** | $C \times H \times W$ | **每個樣本** 有自己的 mean / std |
| **Instance Norm** | $H \times W$ | 每個樣本的**每個 channel** 各自算 |
| **Group Norm** | 一組 channel × $H \times W$ | 介於 Layer Norm 與 Instance Norm 之間 |

#### 【口頭】兩個實務重點

**1. Batch Norm 可以「摺進」前一層 conv**

> 「BN 可以被 fold 進前面的 convolution layer，**省下一次實際的 kernel call**。所以它非常 implementation-friendly，也非常普及。」

**2. Layer Norm 在 LLM 中極為重要**

> 「Layer Norm 在近期的大型語言模型裡非常廣泛使用。做不同 token 之間的 attention 之前，**先做 Layer Norm**。」
> LLM 裡沒有 H、W 維度，只有 C 維度，所以 Layer Norm 就是沿 C 做正規化。

#### 【口頭】$\gamma$ 和 $\beta$ 為什麼在後面的課很關鍵

**參數量**：每個維度 2 個可學習參數（$\gamma$、$\beta$），**只是一維向量，參數量極小**。

教授預告了兩個之後會用到的地方：

| 應用 | 說明 |
|---|---|
| **PEFT（參數高效微調）** | 微調時**只調 $\gamma$ 與 $\beta$**，是成本極低的 fine-tuning 方法 |
| **量化（SmoothQuant 等）** | 量化產生的 **scaling factor 可以被吸收進前一層的 normalization**（進到 $\gamma$ 和 $\beta$ 裡），**省下額外的 kernel 成本** |

> 【口頭】教授明確說：**「這在最後一次作業、也就是 Lab 4 壓縮大型語言模型的時候會用到。做 Lab 4 的時候回來看這一張投影片。」**

---

### 2.11 Activation Function

Activation function 通常是**非線性**函數。

| 函數 | 公式 |
|---|---|
| **Sigmoid** | $y = 1/(1 + e^{-x})$ |
| **ReLU** | $y = \max(0, x)$ |
| **ReLU6** | $y = \min(\max(0, x), 6)$ |
| **Leaky ReLU** | $y = \max(\alpha x, x)$ |
| **Swish** | $y = x/(1 + e^{-x})$ |
| **Hard Swish** | $y = 0 \ (x \le -3)$；$x \ (x \ge 3)$；$x(x+3)/6$（其他） |

其他：Tanh、GELU、ELU、Mish…

#### 設計動機

| 函數 | 為什麼存在 |
|---|---|
| **ReLU6** | 【口頭】把上界 clip 在 6，**讓量化更容易**（值域已知且固定） |
| **Leaky ReLU / Hard Swish** | 【口頭】ReLU 在負區間梯度為 0；這些函數**處處有梯度** |

#### ⚠️ 【口頭】硬體友善度的陷阱

> 「關鍵的地雷在於：**ReLU 非常 hardware-friendly，但有些 activation function 極難實作、對硬體很不友善。所以除非真的必要，有時候要避開那些硬體不友善的 activation function。**」

（Hard Swish 存在的理由就是這個 —— 用分段線性去逼近 Swish，避免 exp。）

---

### 2.12 Transformer 預覽

> 【口頭】「Transformer 我們有**兩堂專門的課**（Lecture 12 起）會講，這裡只給個預覽。」

#### 為什麼重要

> 【口頭】「Transformer **不只革新了 NLP，也革新了視覺**。過去視覺大多用 CNN 架構，現在正被 Transformer 改造。所以 ViT 非常重要。」

#### 兩個階段

- **Context stage**（處理 prompt）
- **Generation stage**（逐 token 生成）

#### 一個 Transformer Block 的組成

1. **Multi-Head Attention（MHA）**
2. **Feed-Forward Network（FFN / MLP）** —— 就是全連接層

#### QKV：用搜尋引擎理解

【口頭】教授用 **YouTube 搜尋** 做比喻：

| 符號 | 對應 |
|---|---|
| **Query** | 你在搜尋框打的字 |
| **Key** | 影片的標題／描述 |
| **Value** | 影片本身的內容 |

#### Scaled Dot-Product Attention 流程

```
Q[N,d]  K[N,d]  V[N,d]
   └──Matmul──┘             ← Q × Kᵀ，得到 inner product
     Attn[N,N]
        │
    Mask (opt)
        │
     Softmax                 ← 得到 N×N 的 attention weights
     Prob[N,N]
        └────Matmul────V     ← 乘上 Value
          Out[N,d]
```

**除以 $\sqrt{d}$ 的理由**：【口頭】「不管你有多少維度，我們希望**公平**」—— 正規化掉維度大小的影響。

#### Multi-Head

多個 head 各自學不同的特徵，最後合併。投影片範例是 3 個 head。

【口頭】教授預告後面還會講 **Multi-Query Attention** 與 **Grouped-Query Attention**。

把這個 block 重複很多次，就是完整的 Transformer。

---

## 3. 經典 CNN 架構

> 【口頭】教授的定位：「對已經很熟的人，我們**專注在怎麼算維度**。」
> 這一節的重點不是背架構，而是**用 §2.4 / §2.6 的公式實際算一遍**。

### 3.1 AlexNet（2012）

**結構**：5 層 convolution + 3 層 linear。

| 層 | 輸出 shape (C×H×W) |
|---|---|
| Image | 3 × 224 × 224 |
| 11×11 Conv, ch 96, **stride 4**, pad 2 | 96 × 55 × 55 |
| 3×3 MaxPool, stride 2 | 96 × 27 × 27 |
| 5×5 Conv, ch 256, pad 2, **groups 2** | 256 × 27 × 27 |
| 3×3 MaxPool, stride 2 | 256 × 13 × 13 |
| 3×3 Conv, ch 384, pad 1 | 384 × 13 × 13 |
| 3×3 Conv, ch 384, pad 1, **groups 2** | 384 × 13 × 13 |
| 3×3 Conv, ch 256, pad 1, **groups 2** | 256 × 13 × 13 |
| 3×3 MaxPool, stride 2 | 256 × 6 × 6 |
| Linear, ch 4096 | 4096 |
| Linear, ch 4096 | 4096 |
| Linear, ch 1000 | 1000 |

**自己驗算第一層**（$h_i = 224$, $p = 2$, $k = 11$, $s = 4$）：

$$h_o = \left\lfloor \frac{224 + 2\times 2 - 11}{4} \right\rfloor + 1 = \left\lfloor \frac{217}{4} \right\rfloor + 1 = 54 + 1 = 55 \;✔$$

> 【口頭】「對高效深度學習研究來說，**搞懂 weight 和 feature map 的維度是最關鍵的基本功**。給你一個 conv layer 和各種 kernel 維度，你要能立刻算出 output channel 和 output dimension。」

這張表接下來在 §4.4 / §4.6 / §4.7 會被用三次，分別算參數量、activation、MAC。

### 3.2 VGG-16（2014）

**特色**：**同質化（homogeneous）** —— 從頭到尾都是 3×3 convolution。

- 13 層 conv + 3 層 fully-connected = **16 層 → 所以叫 VGG-16**
- 每個 conv 後面接 batch normalization
- 結構極簡單、極規律

### 3.3 ResNet-50 Bottleneck（2015）

**Bottleneck block 的三層結構：**

```
輸入 (c 個 channel)
  │
  ├─ 1×1 Conv  →  channel 縮小 4 倍   ← 「bottleneck」的由來
  ├─ 3×3 Conv  →  在窄的 channel 上做重活
  ├─ 1×1 Conv  →  channel 投影回 c
  │
  └───────── residual（跳接） ─────────┘
```

#### 為什麼要先用 1×1 把 channel 縮 4 倍？

【口頭】教授：**「因為 3×3 convolution 非常重（heavy）。所以我們先把 channel 數縮小，來降低運算量、降低參數量。」**

從 §4.7 的公式看就很清楚：conv 的 MAC 正比於 $c_o \cdot c_i$。把 $c_i$ 砍成 1/4，3×3 那層的成本就掉到約 1/4。

#### 兩種 residual 分支

| 情況 | Residual 分支 |
|---|---|
| **輸入輸出維度相同** | **Identity**（直接相加，零成本） |
| **有 downsample / stride 2**（維度對不上） | 加一個 **1×1 conv (stride 2)** 做投影 —— **不改 channel 數，但解析度減半**，好讓兩邊維度對得上 |

### 3.4 MobileNetV2 Inverted Bottleneck（2018）

**與 ResNet bottleneck 的關鍵差異**：中間那層用 **3×3 depthwise convolution**（§2.8，$g = c$ 的極端 grouped conv）。

#### 為什麼叫「Inverted（倒過來的）」Bottleneck

| | 中間層的 channel |
|---|---|
| **ResNet Bottleneck** | **縮小** 4× → 中間細，像瓶頸 |
| **MobileNetV2 Inverted Bottleneck** | **放大** 6× → **中間胖**，跟瓶頸相反 |

【口頭】「我們把 channel 數從 $n$ 變成 $n \times 6$。因為中間變大而不是變小，所以叫它 **inverted bottleneck**。」

**膨脹 6 倍的理由**：depthwise conv 參數太少、容量不足，必須靠加寬 channel 來補回表達能力（見 §2.8 的完整因果鏈）。

**Downsample 的做法**：MobileNet 的 downsample block **沒有 residual 分支**，直接用 stride = 2 的 3×3 depthwise conv。

---

## 4. 效率指標（本講重點）

> 【口頭】「這一部分超級重要。你們一定聽過好幾次 TOPS、TFLOPS、ops per second —— 大寫 S、小寫 s 差在哪？我們會詳細講。」

### 4.1 三個目標：Smaller / Faster / Greener

| 目標 | 為什麼在意 | 對應指標 |
|---|---|---|
| **Smaller（更小）** | 手機 App 從 App Store 下載不能太久；裝置存不下 | `#Parameters`、`Model Size`、`total/peak #Activations` |
| **Faster（更快）** | 即時應用 | `Latency`、`Throughput` |
| **Greener（更省電）** | 不能把手機電池吃光；Green AI | `Energy` |

**指標分成兩大類：**

```
Efficiency Metrics
├── Memory-Related（記憶體相關）
│   ├── #parameters              （參數量）
│   ├── model size               （模型大小）
│   └── total / peak #activations（總／峰值激活量）
└── Computation-Related（運算相關）
    ├── MAC                      （乘積累加）
    ├── FLOP, FLOPS
    └── OP, OPS
```

---

### 4.2 Latency vs Throughput

#### 定義

| 指標 | 定義 | 投影片例子 |
|---|---|---|
| **Latency（延遲）** | 完成**單一特定任務**的延遲 | SegFormer **638 ms** (82.4 mIoU) → EfficientViT **46 ms** (82.7 mIoU)　*Jetson AGX Orin, TensorRT, fp16, batch=1* |
| **Throughput（吞吐量）** | **單位時間內**處理資料的速率 | 低吞吐 **6.1 video/s** → 高吞吐 **77.4 video/s** |

#### ⭐ 核心觀念：兩者不能互推

教授把這個當成本講最重要的結論之一，用兩個設計做對照：

| | **Design 1** | **Design 2** |
|---|---|---|
| 結構 | 單一引擎，逐張處理，每張 50 ms | **4 個引擎平行**，每張 100 ms |
| **Latency** | **50 ms**（較低 ✅） | 100 ms |
| **Throughput** | 20 image/s | **40 image/s**（較高 ✅） |

> **「較低的 latency 不會自動轉換成較高的 throughput；較高的 throughput 也不代表較低的 latency。這兩件事互不蘊含。」**

表面上「latency 低 = 快」、「throughput 高 = 快」，但**它們是兩個獨立的「快」**。

#### 為什麼 latency 比較難優化

| | 怎麼改善 | 難度 |
|---|---|---|
| **Throughput** | Batching、平行處理 —— **多丟幾張 GPU、多開幾個核心就好** | 容易 |
| **Latency** | 平行化**外層迴圈**完全幫不上忙 —— 單一樣本的處理時間不會因為多了機器就變短 | **難** |

> 【口頭】「優化 latency 通常更有挑戰性。」

#### Latency 的組成公式

$$\boxed{\text{Latency} \approx \max(T_{\text{computation}},\; T_{\text{memory}})}$$

**運算部分：**

$$T_{\text{computation}} \approx \frac{\overbrace{\text{模型的運算次數}}^{\text{NN 規格}}}{\underbrace{\text{處理器每秒能處理的運算次數}}_{\text{硬體規格}}}$$

**記憶體部分：**

$$T_{\text{memory}} \approx T_{\text{activation 搬移}} + T_{\text{weight 搬移}}$$

$$T_{\text{weight 搬移}} \approx \frac{\overbrace{\text{Model Size}}^{\text{NN 規格}}}{\underbrace{\text{記憶體頻寬}}_{\text{硬體規格}}}
\qquad
T_{\text{activation 搬移}} \approx \frac{\text{Input Activation} + \text{Output Activation}}{\text{記憶體頻寬}}$$

#### 【口頭】為什麼是 max 而不是加總？

> 「可能是 **compute-bounded**，也可能是 **memory-bounded**。你可能有一大堆運算單元，但記憶體餵不動這麼高的算力 —— 兩邊哪一邊都可能先飽和。所以取 max。」
> **「就像水桶一樣 —— 最短的那塊木板決定你能裝多少水。」**

**注意這個公式的漂亮之處**：每一項都乾淨地拆成 **NN 規格**（分子，你能改模型）和 **硬體規格**（分母，你能換硬體）。整門課後半段的技術，本質上都在動分子。

#### 降低 Latency 的關鍵技巧：Overlap（重疊）

把 **compute** 和 **memory access** 疊在一起做，把資料搬移的時間「藏」起來：

```
時間 →
Layer 1:  [載入 weight/input] [    計算    ] [存 output]
Layer 2:                      [載入 weight/input] [    計算    ] [存 output]
                              ↑ 重疊：Layer 1 在算的同時，Layer 2 已在載資料
```

#### 【Q&A】Overlap 是跨樣本還是跨層？

**學生問**：投影片那張圖看起來像是**兩個不同樣本**在執行，是這樣嗎？還是同一個樣本內部在做載入？

**教授答**：

> 「你說得對，我講的是**一個神經網路、處理一張影像**的情境。一張影像要經過很多層，**所以我們把它們重疊起來** —— 處理這一張影像的總時間就縮短了。所以這是**同一張圖、跨層**的重疊。這裡畫的是**一層**：這是 layer 1 載入 input，這是 layer 2，我們把它們疊起來。」

**學生追問**：「load weight 是指從 CPU memory 搬到 GPU memory 嗎？」

**教授答**：

> 「那個我們還沒講。這裡指的就是**從記憶體載入權重、在該處理器上計算**。你也可以想成從 DDR / GDDR6 載到 GPU memory 再到運算單元。**這個概念是通用的** —— 從哪裡搬到哪裡，取決於你在哪裡跑。」

---

### 4.3 Energy：記憶體才是吃電大戶

> 【口頭】「在手機上跑，你不會希望手機把電池燒光。」

**核心觀念：資料搬移 → 更多記憶體存取 → 更多能耗。**

**45nm 0.9V 製程下各種操作的能耗**（Horowitz, ISSCC 2014）：

| 操作 | 能耗 (pJ) | 相對成本 |
|---|---|---|
| 32-bit int ADD | **0.1** | 1× |
| 32-bit float ADD | 0.9 | 9× |
| 32-bit Register File | 1 | 10× |
| 32-bit int MULT | 3.1 | 31× |
| 32-bit float MULT | 3.7 | 37× |
| **32-bit SRAM Cache** | **5** | 50× |
| **32-bit DRAM Memory** | **640** | **6400×** |

> ⚠️ **投影片上標紅的關鍵比較：DRAM 存取 vs Register File = 200×。**
> 【口頭】「這是**對數刻度**。存取記憶體比做算術**貴兩個數量級**。做一次加法或乘法只花幾個 pJ，但**從 DRAM 搬資料要 640 pJ**。」
> **「要盡量避免記憶體存取。運算便宜，記憶體才是把電池吸乾的元兇。」**

這條原則是整門課的物理基礎 —— 它解釋了：

- 為什麼 §2.8 的「參數少但 activation 大」是個壞交易
- 為什麼 pruning、quantization 值得做（都在減少要搬的資料量）
- 為什麼 peak activation（§4.6）比參數量更能決定實際瓶頸

---

### 4.4 #Parameters（參數量）

#### 四個公式（忽略 bias）

| 層 | 參數量 |
|---|---|
| **Linear** | $c_o \cdot c_i$ |
| **Convolution** | $c_o \cdot c_i \cdot k_h \cdot k_w$ |
| **Grouped Conv** | $c_o \cdot c_i \cdot k_h \cdot k_w \,/\, g$ |
| **Depthwise Conv** | $c_o \cdot k_h \cdot k_w$ |

**推導邏輯**（記住這條就不用背公式）：

1. Conv kernel 有 4 個維度：$k_h \times k_w \times c_i \times c_o$
2. 分 $g$ 組 → 組間互不相連 → **除以 $g$**
3. Depthwise 是 $g = c$ 的極端 → $c_i/g = 1$，**整個 $c_i$ 項消失**，只剩 3 項

#### AlexNet 逐層參數量

| 層 | 計算式 | #Params |
|---|---|---|
| 11×11 Conv, ch 96, s4, p2 | $96 \times 3 \times 11 \times 11$ | 24,848 |
| 5×5 Conv, ch 256, p2, **g2** | $256 \times 96 \times 5 \times 5 / 2$ | 307,200 |
| 3×3 Conv, ch 384, p1 | $384 \times 256 \times 3 \times 3$ | 884,736 |
| 3×3 Conv, ch 384, p1, **g2** | $384 \times 384 \times 3 \times 3 / 2$ | 663,552 |
| 3×3 Conv, ch 256, p1, **g2** | $256 \times 384 \times 3 \times 3 / 2$ | 442,368 |
| **Linear, ch 4096** | $4096 \times (256 \times 6 \times 6)$ | **37,748,736** |
| **Linear, ch 4096** | $4096 \times 4096$ | **16,777,216** |
| **Linear, ch 1000** | $1000 \times 4096$ | **4,096,000** |
| | | **合計 ≈ 61M** |

（MaxPool 層沒有參數。）

#### ⭐ 從這張表能立刻看出什麼

**三層 fully-connected 就佔了約 58.6M / 61M ≈ 96% 的參數。**

> 【口頭】「你馬上就能看出哪一層吃掉大量參數 —— **fully-connected layer**，因為每個 output 都連到每個 input，**非常冗餘**。這就是為什麼後來的 CNN 模型都把 fully-connected layer 拿掉了。」

**但**：

> 「不過最近的 Transformer **全部都是 fully-connected layer**，為的是提供足夠的容量。所以**沒有絕對的對錯** —— 就是靠實驗、靠大量 GPU 小時去試，直到找出好的配方。」

---

### 4.5 Model Size（模型大小）

$$\boxed{\text{Model Size} = \#\text{Parameters} \times \text{Bit Width}}$$

#### 常見位元寬度

| 精度 | 每個權重佔用 |
|---|---|
| FP32（全精度） | 4 bytes |
| FP16 | 2 bytes |
| INT8 | 1 byte |
| **INT4** | **0.5 byte** ← **Lab 4 要用的** |

#### 例 1：AlexNet（61M 參數）

| 精度 | 模型大小 |
|---|---|
| FP32 | 61M × 4 B = **244 MB** |
| **INT8** | 61M × 1 B = **61 MB** |

#### 例 2：7B 參數的 LLM

【口頭】教授當場帶全班算（提醒：1 billion = $10^9$，1 mega = $10^6$）：

| 精度 | 模型大小 |
|---|---|
| FP16 | **14 GB** |
| INT8 | **7 GB** |
| **INT4** | **3.5 GB** ← 作業要用的 |

> 【口頭】「所以確認你的筆電**至少有 8 GB 記憶體**才能完成作業 —— 2023 年了，這應該沒問題。」

**壓縮模型大小的核心手段就是降低位元寬度** —— 這就是 Quantization（Lecture 5-6）整章在做的事。

---

### 4.6 #Activations：Total vs Peak

#### 為什麼這一節最重要

> 【口頭】「從 ResNet 到 MobileNet，**參數量大幅下降，但 peak activation 反而增加**。這對 IoT 裝置就是問題。」

**Activation 大小 = $C \times H \times W$**（每一層 feature map 的元素數）。

#### 兩種算法

| | 定義 | 用在哪 |
|---|---|---|
| **Total #Activations** | 所有層的 activation **加總** | **訓練** —— 反向傳播要存下所有中間結果來算梯度 |
| **Peak #Activations** | 逐層計算 `input activation + output activation`，取**最大值** | **推論** —— 只需要當下這一層的記憶體 |

$$\text{Peak} \approx \max_{\text{layer } \ell} \left( \#\text{input act}_\ell + \#\text{output act}_\ell \right)$$

【口頭】為什麼是 input + output：「要計算某一層，我需要多少記憶體？**一階近似就是 input activation 加 output activation。**」

#### AlexNet 逐層 Activation

| 層 | C×H×W | #Activation |
|---|---|---|
| **Image** | 3×224×224 | **150,528** |
| **11×11 Conv, ch 96, s4, p2** | 96×55×55 | **290,400** |
| 3×3 MaxPool, s2 | 96×27×27 | 69,984 |
| 5×5 Conv, ch 256, p2, g2 | 256×27×27 | 186,624 |
| 3×3 MaxPool, s2 | 256×13×13 | 43,264 |
| 3×3 Conv, ch 384, p1 | 384×13×13 | 64,896 |
| 3×3 Conv, ch 384, p1, g2 | 384×13×13 | 64,896 |
| 3×3 Conv, ch 256, p1, g2 | 256×13×13 | 43,264 |
| 3×3 MaxPool, s2 | 256×6×6 | 9,216 |
| Linear, ch 4096 | 4096 | 4,096 |
| Linear, ch 4096 | 4096 | 4,096 |
| Linear, ch 1000 | 1000 | 1,000 |

- **Total #Activation = 932,264**
- **Peak #Activation = 150,528 + 290,400 = 440,928**（第一層 conv）

#### ⭐ 真正的瓶頸：MCUNet 的三組數據

**（1）ResNet-18 vs MobileNetV2-0.75（都是 ~70% ImageNet Top-1，8-bit 整數）**

| | 降幅 |
|---|---|
| **Param (MB)** | **4.6×** ✅ |
| **Peak Activation (MB)** | **只有 1.8×** ❌ |

> **「從 ResNet 到 MobileNetV2，#Activation 幾乎沒有改善。」**

**（2）MobileNetV2 的記憶體分佈極度不平衡**

| | 數值 |
|---|---|
| MobileNetV2 的 **Peak Memory**（在前面某個 block） | **1372 kB** |
| **MCU（微控制器）的記憶體限制** | **256 kB** |

> 【口頭】「如果**某一層**要這麼多記憶體，而你的記憶體只有這麼多，那你**根本放不進去** —— 儘管**其他所有層都放得下**。這就是 activation 不平衡的問題。」

超出 5 倍多。**這就是 TinyML 真正的牆**，也是 MCUNet 這條研究線的起點。

**（3）訓練時的瓶頸也是 activation**

> 投影片標題直接寫：**「#Activation is the memory bottleneck in training, not #Parameters.」**（ResNet-50 vs MobileNetV2-1.4×）

#### U 型分佈：為什麼前期 activation 大、後期 weight 大

【口頭】教授解釋 CNN 中 weight 與 activation 沿深度呈**互補的分佈**：

| 階段 | 解析度 $H \times W$ | Channel 數 $C$ | **Activation** | **Weight** |
|---|---|---|---|---|
| **早期層** | **大**（例如 224×224） | 少（3, 96…） | **大** 📈 | **小** 📉 |
| **後期層** | 小（13×13, 6×6） | **多**（384, 4096…） | **小** 📉 | **大** 📈 |

- **Activation** $\propto C \times H \times W$ → 早期解析度主導 → **前面大**
- **Weight** $\propto c_o \times c_i$ → 後期 channel 主導 → **後面大**

**實務意涵**：優化前段要盯 activation，優化後段要盯 weight。這兩件事需要不同的技術。

---

### 4.7 MAC / FLOP / FLOPS / OP / OPS

#### MAC = Multiply-Accumulate

> 【口頭】「MAC —— **不是麥當勞**，是 multiply and accumulate。」

$$a \;{+}{=}\; b \times c \qquad \Longrightarrow \qquad \textbf{1 MAC}$$

#### 矩陣運算的 MAC 數

| 運算 | 形狀 | MAC 數 |
|---|---|---|
| **Matrix-Vector (MatVec)** | $(m \times n) \times (n \times 1)$ | $m \cdot n$ |
| **Matrix-Matrix (GEMM)** | $(m \times n) \times (n \times k)$ | $m \cdot n \cdot k$ |

【口頭】投影片的例子：$8\times 4$ 乘 $4 \times 1$ → 每個 output 要 4 次乘加，共 8 個 output → $4 \times 8 = 32$ MAC。換成 $8 \times 4$ 乘 $4 \times 2$ → $8 \times 4 \times 2$。

> **「可以看出 GEMM 比 MatVec 的運算量大得多。」**
> （這正是為什麼 LLM 的 decode 階段（MatVec）是 memory-bound、prefill 階段（GEMM）是 compute-bound —— 後面的課會回到這裡。）

#### 四個 MAC 公式（batch size $n = 1$）

| 層 | MACs |
|---|---|
| **Linear** | $c_o \cdot c_i$ |
| **Convolution** | $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o$ |
| **Grouped Conv** | $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o \,/\, g$ |
| **Depthwise Conv** | $c_o \cdot k_h \cdot k_w \cdot h_o \cdot w_o$ |

**怎麼記（教授的拆法）：**

$$\underbrace{c_i \cdot k_h \cdot k_w}_{\text{算一個 output pixel 要幾次 MAC}} \times \underbrace{h_o \cdot w_o \cdot c_o}_{\text{要算幾個 output pixel}}$$

**跟參數量公式的關係非常好記：**

> **MAC 公式 = 參數量公式 × $h_o \cdot w_o$**（linear layer 除外，因為它沒有空間維度）

#### AlexNet 逐層 MAC

| 層 | 計算式 | MACs |
|---|---|---|
| 11×11 Conv, ch 96, s4, p2 | $96 \times 3 \times 11 \times 11 \times 55 \times 55$ | 105,415,200 |
| 5×5 Conv, ch 256, p2, g2 | $256 \times 96 \times 5 \times 5 \times 27 \times 27 / 2$ | 223,948,800 |
| 3×3 Conv, ch 384, p1 | $384 \times 256 \times 3 \times 3 \times 13 \times 13$ | 149,520,384 |
| 3×3 Conv, ch 384, p1, g2 | $384 \times 384 \times 3 \times 3 \times 13 \times 13 / 2$ | 112,140,288 |
| 3×3 Conv, ch 256, p1, g2 | $256 \times 384 \times 3 \times 3 \times 13 \times 13 / 2$ | 74,760,192 |
| Linear, ch 4096 | $4096 \times (256 \times 6 \times 6)$ | 37,748,736 |
| Linear, ch 4096 | $4096 \times 4096$ | 16,777,216 |
| Linear, ch 1000 | $1000 \times 4096$ | 4,096,000 |
| | | **合計 = 724M** |

> ⚠️ **對照 §4.4 很有意思**：FC 層佔了 **96% 的參數**，卻只佔約 **8% 的 MAC**。
> **參數量大的層 ≠ 運算量大的層。** 這也是為什麼 pruning FC 層能大幅縮小模型，卻不太能加速。

#### FLOP、FLOPS、OP、OPS

$$\boxed{\textbf{1 MAC} = \textbf{2 FLOPs}}$$

因為一個 MAC 包含**一次乘法 + 一次加法**，兩個都是浮點運算。

**AlexNet**：724M MACs × 2 = **1,448 MFLOPs = 1.4 GFLOPs**

#### ⭐ 大小寫的 s（教授特別強調）

| 寫法 | 意義 |
|---|---|
| **FLOPs**（小寫 s） | **複數** —— 浮點運算「次數」 |
| **FLOPS**（大寫 S） | **per Second** —— 每秒能做幾次浮點運算（**速度**） |

> 【口頭】「**大寫 S 是 per second，小寫 s 是複數。**」

#### OP / OPS：比 FLOP 更通用

> 【口頭】「因為不是所有數字都是浮點數。你可以有整數、可以有 4-bit、可以有半精度。**OP 更通用**。」

- 一次 4-bit 乘法 = 1 OP
- 一次 4-bit 加法 = 1 OP
- **AlexNet**：724M MACs = **1.4 GOPs**

在講量化過的模型時，用 **OPS / TOPS** 才精確；FLOPS 專指浮點。

#### 【Q&A】峰值 FLOPS 是硬體的速度上限嗎？

**學生問**：MAC 衡量運算次數，那 FLOPS 和 OPS 是不是就是硬體的速度上限？

**教授答**：**不完全是。**

> 「**不一定是速度，因為它跟 workload 有關。** 例如同一顆 GPU 跑 AlexNet 可能達到某個 FLOPS，但**跑 MobileNet 時 FLOPS 會低很多，因為 MobileNet 的平行度比較低** —— 這**不是硬體造成的**，所以實際的 FLOPS 可能遠低於峰值。」

**學生追問**：「峰值效能是理論值嗎？」

**教授答**：

> 「對，理論峰值大概是**有幾個乘法單元 ÷ 時間**算出來的理論可達值。但實際上你會遇到**其他瓶頸，例如記憶體**，所以你**幾乎不可能達到 100%**。」

**這一段扣回 §4.2 的 $\max(T_{\text{compute}}, T_{\text{memory}})$：**
Peak FLOPS 只是那條公式的分母。如果你是 memory-bound，分母再大也沒用。

---

## 5. 一頁速查表

### 5.1 維度公式

| 項目 | 公式 |
|---|---|
| Conv 輸出（無 padding、$s=1$） | $h_o = h_i - k_h + 1$ |
| Conv 輸出（有 padding） | $h_o = h_i + 2p - k_h + 1$ |
| Conv 輸出（**通用**） | $h_o = \left\lfloor \dfrac{h_i + 2p - k_h}{s} \right\rfloor + 1$ |
| 感受野（$L$ 層，kernel $k$，$s=1$） | $\text{RF} = L \cdot (k-1) + 1$ |

### 5.2 Tensor Shape

| 層 | Input | Output | Weight |
|---|---|---|---|
| **Linear** | $(n, c_i)$ | $(n, c_o)$ | $(c_o, c_i)$ |
| **1D Conv** | $(n, c_i, w_i)$ | $(n, c_o, w_o)$ | $(c_o, c_i, k_w)$ |
| **2D Conv** | $(n, c_i, h_i, w_i)$ | $(n, c_o, h_o, w_o)$ | $(c_o, c_i, k_h, k_w)$ |
| **Grouped Conv** | $(n, c_i, h_i, w_i)$ | $(n, c_o, h_o, w_o)$ | $(g \cdot \frac{c_o}{g}, \frac{c_i}{g}, k_h, k_w)$ |
| **Depthwise Conv** | $(n, c_i, h_i, w_i)$ | $(n, c_o, h_o, w_o)$ | $(c, k_h, k_w)$ |

### 5.3 #Params vs MACs（並排對照）

| 層 | **#Parameters** | **MACs**（$n=1$） |
|---|---|---|
| **Linear** | $c_o \cdot c_i$ | $c_o \cdot c_i$ |
| **Convolution** | $c_o \cdot c_i \cdot k_h \cdot k_w$ | $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o$ |
| **Grouped Conv** | $c_o \cdot c_i \cdot k_h \cdot k_w / g$ | $c_o \cdot c_i \cdot k_h \cdot k_w \cdot h_o \cdot w_o / g$ |
| **Depthwise Conv** | $c_o \cdot k_h \cdot k_w$ | $c_o \cdot k_h \cdot k_w \cdot h_o \cdot w_o$ |

> **右欄 = 左欄 × $h_o \cdot w_o$**（Linear 除外）

### 5.4 效率指標

| 指標 | 公式 |
|---|---|
| **Model Size** | $\#\text{Params} \times \text{bit width}$ |
| **#Activation（單層）** | $C \times H \times W$ |
| **Total #Activation** | $\sum_{\ell} \#\text{Activation}_\ell$ |
| **Peak #Activation** | $\max_\ell (\#\text{in}_\ell + \#\text{out}_\ell)$ |
| **FLOPs** | $\#\text{MACs} \times 2$ |
| **Latency** | $\max(T_{\text{computation}}, T_{\text{memory}})$ |
| **Throughput** | 單位時間處理的樣本數（**與 latency 互不蘊含**） |

### 5.5 AlexNet 三個數字（背起來當基準）

| 指標 | 數值 |
|---|---|
| **#Parameters** | **61 M**（FC 層佔 96%） |
| **MACs** | **724 M** = **1.4 GFLOPs / 1.4 GOPs** |
| **Total #Activation** | 932,264 |
| **Peak #Activation** | 440,928（第一層 conv） |
| **Model Size** | FP32 = 244 MB；INT8 = 61 MB |

---

## 6. 與後續課程／作業的連結

這一講看起來像「複習」，但其實每個觀念都是後面某一講的伏筆：

| 本講觀念 | 後面在哪裡用到 |
|---|---|
| **Normalization 的 $\gamma$、$\beta$** | **Lab 4（LLM 壓縮）**：量化的 scaling factor 被吸收進前一層 normalization 的 $\gamma$ / $\beta$（SmoothQuant）。教授明確說「做 Lab 4 時回來看這張投影片」。也是 PEFT 的基礎 |
| **Peak Activation vs #Params** | **MCUNet / TinyML**：MobileNetV2 峰值 1372 kB vs MCU 只有 256 kB，是整條 TinyML 研究線的起點 |
| **Depthwise 的陷阱（參數少 ≠ 快）** | **Lecture 7-8 NAS**：這正是 hardware-aware NAS、以及「MACs ≠ Latency」的直接動機 |
| **寬淺 vs 窄深的 GPU 取捨** | **Lecture 7-8 NAS**：search space 設計要平衡的就是這件事 |
| **DRAM 640 pJ vs ALU 幾 pJ** | **Lecture 3-4 Pruning、Lecture 5-6 Quantization**：兩者本質都是「減少要搬的資料」；也是 **EIE** 論文的核心論證 |
| **FC 層佔 96% 參數但只佔 8% MAC** | **Pruning**：解釋為什麼剪 FC 層能大幅縮小模型卻難加速；也是 unstructured vs structured pruning 的分野 |
| **$\text{Latency} \approx \max(T_{\text{comp}}, T_{\text{mem}})$** | 幾乎所有後續講次 —— 判斷一個優化是不是打在真正的瓶頸上，靠的就是這條公式 |
| **GEMM vs MatVec 的 MAC 差距** | **LLM 講次**：prefill（GEMM, compute-bound）vs decode（MatVec, memory-bound）的根本原因 |

---

## 附：本講一句話總結

> **所有效率指標都是從「維度」算出來的；而參數量、MAC、activation 這三個數字會指向完全不同的瓶頸 —— 分不清楚它們，就會把力氣花在錯的地方。**

---

*筆記依據 MIT 6.5940 Fall 2023 Lecture 2 逐字稿（Zoom 錄影 1:12:14）與官方投影片 `Lec02-Basics.pdf` 整理。*
