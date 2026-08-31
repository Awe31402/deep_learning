# EfficientML.ai 第三、四講完整筆記：剪枝與稀疏性（Pruning and Sparsity）

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2023
> **講者**：Song Han
> **涵蓋**：
> - **Lecture 3** — Pruning and Sparsity (Part I)，1:09:24
> - **Lecture 4** — Pruning and Sparsity (Part II)，1:17:41
>
> **來源**：Zoom 錄影逐字稿 ＋ 官方投影片 `Lec03-Pruning-I.pdf`（74 頁）、`Lec04-Pruning-II.pdf`（119 頁）
>
> **標記說明**
> - `【口頭】` = 教授課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 學生提問與回答
> - `【投影片新版】` = 投影片是後續學期更新過的版本，數字與 2023 課堂口述不同，兩者都列出

---

## 目錄

- [總覽：Pruning 的四個決策](#總覽pruning-的四個決策)
- [Lecture 3 — Pruning and Sparsity (Part I)](#lecture-3--pruning-and-sparsity-part-i)
  - [1. 動機](#1-動機)
  - [2. 什麼是 Pruning](#2-什麼是-pruning)
  - [3. Pruning Granularity（剪枝顆粒度）](#3-pruning-granularity剪枝顆粒度)
  - [4. Pruning Criterion（剪枝準則）](#4-pruning-criterion剪枝準則)
  - [5. 課堂 Demo](#5-課堂-demo)
- [Lecture 4 — Pruning and Sparsity (Part II)](#lecture-4--pruning-and-sparsity-part-ii)
  - [6. Pruning Ratio（每層剪多少）](#6-pruning-ratio每層剪多少)
  - [7. Fine-tuning / Training](#7-fine-tuning--training)
  - [8. Lottery Ticket Hypothesis（彩票假說）](#8-lottery-ticket-hypothesis彩票假說)
  - [9. 系統與硬體支援](#9-系統與硬體支援)
- [10. 一頁速查表](#10-一頁速查表)
- [11. 與其他課程／作業的連結](#11-與其他課程作業的連結)

---

## 總覽：Pruning 的四個決策

整整兩講，其實就是在回答**四個問題**。先把這張地圖記住，後面每一節都是在填其中一格：

| # | 問題 | 英文 | 在哪一講 | 本筆記章節 |
|---|---|---|---|---|
| 1 | **剪什麼形狀？** | Determine the Pruning **Granularity** | Lec 3 | §3 |
| 2 | **剪哪些？** | Determine the Pruning **Criterion**（which synapses / which neurons?） | Lec 3 | §4 |
| 3 | **每層剪多少？** | Determine the Pruning **Ratio** | Lec 4 | §6 |
| 4 | **怎麼把準確率救回來？** | **Fine-tune / Train** Pruned Neural Network | Lec 4 | §7 |

**基本流程**（Han et al., NeurIPS 2015）：

```
Train Connectivity  →  Prune Connections  →  Train Weights (fine-tune)
        ↑                                              │
        └──────────── 迭代（Iterative Pruning）─────────┘
```

---
---

# Lecture 3 — Pruning and Sparsity (Part I)

## 1. 動機

### 1.1 MLPerf：AI 運算的奧運會

【口頭】教授先介紹 **MLPerf** —— 「**AI 運算的奧林匹克**」。Nvidia、Google、AMD、Intel 等公司每年參賽，比的是「**在給定準確率下，你的硬體＋演算法跑一個特定 workload 有多快**」。

**兩個賽組（divisions）：**

| 賽組 | 規則 |
|---|---|
| **Closed Division（封閉組）** | **不能改模型**、不能改精度、不能改權重數量。純粹比硬體創新與編譯器優化 |
| **Open Division（開放組）** | **可以隨便改** —— pruning、compression、quantization 都行，**只要準確率達標就算有效提交** |

**【口頭】2023 課堂上講的數字**：同一個硬體平台，Closed Division 約 129 samples/s，Open Division 因為可以進一步優化模型，可以衝到 **4000 samples/s 以上**。

**【投影片新版】Llama 2 70B 的例子**（單張 NVIDIA H200）：

| | Closed Division | Open Division | Speedup |
|---|---|---|---|
| **Offline samples/sec** | 4,488 | **11,189** | **2.5×** |

做法（在維持 **99% 準確率**的前提下）：
- **Depth pruning（深度剪枝）**：80 層 → **32 層**
- **Width pruning（寬度剪枝）**：intermediate dimension 28,762 → **14,336**

### 1.2 Nvidia 怎麼做到 4.5× 的

【口頭】教授拆解 Nvidia 的 open division 提交，用了**三個支柱**：

| 步驟 | 結果（維持 99% 準確率） |
|---|---|
| 只做 **Quantization** | 模型 ~600 MB，speedup = **1×** |
| ＋ **Pruning** | 模型大幅縮小，**2.6×** |
| ＋ **Distillation**（把 pruning 掉的準確率救回來）＋ QAT | **4.5×** |

> 這三個支柱正好是這門課第一部分的前三章。

### 1.3 記憶體才是耗電大戶（Lecture 2 的回顧）

**45nm 製程的能耗表**（Horowitz, ISSCC 2014）：

| 操作 | 能耗 (pJ) |
|---|---|
| 32-bit int ADD | 0.1 |
| 32-bit float ADD | 0.9 |
| 32-bit Register File | 1 |
| 32-bit int MULT | 3.1 |
| 32-bit float MULT | 3.7 |
| 32-bit SRAM Cache | 5 |
| **32-bit DRAM Memory** | **640** ← **200× Register File** |

> 【口頭】「一旦你去存取 DRAM，能耗就變成**高兩個數量級**。是 **memory reference 在把我們手機的電池吸乾**。所以我們要減少記憶體 —— **weight 和 activation 都要減**。」

**這就是 pruning 的物理動機**：少一個權重 = 少一次 DRAM 存取 = 電池更耐用。

### 1.4 人腦也在剪枝

一個很漂亮的生物學類比 —— 人腦每個神經元的突觸數量：

| 階段 | 每個神經元的突觸數 |
|---|---|
| **新生兒（Newborn）** | 2,500 |
| **2–4 歲** | **15,000**（暴增） |
| **青少年期（Adolescence）** | **開始下降** ← 剪枝發生 |
| **成人（Adult）** | 穩定在約 **7,000** |

> 【口頭】教授特別點出這個弔詭之處：「有趣的是，這個數字**沒有隨著孩子成長持續增加**。青春期正是我們一生中學最多知識的時期，**但突觸數量卻在這段期間開始下降**。」

**大腦先過度連接，再剪掉冗餘 → 變得更高效。** 神經網路的 pruning 完全是同一個邏輯。

---

## 2. 什麼是 Pruning

### 2.1 定義

**把稠密（dense）的神經網路變成稀疏（sparse）的**，移除冗餘的 **synapse（權重）** 和 **neuron（神經元）**。

【口頭】「**移除（remove）就是把它設成零**。因為**零乘以任何東西都是零** —— 我們不需要計算它，也不需要儲存它。這樣就同時省下記憶體和運算。」

### 2.2 數學形式

$$\arg\min_{W_P} L(\mathbf{x}; W_P) \qquad \text{subject to} \qquad \|W_P\|_0 < N$$

- $W_P$：pruned 之後的權重（原本記作 $W$）
- $\|W_P\|_0$：**非零元素的個數**（L0 norm）
- $N$：目標的非零數量

【口頭】具體例子：「你有 6000 萬個參數，目標剪枝率 90%，那 $N$ 就是 **600 萬**。」

### 2.3 【Q&A】剪權重 vs 剪神經元，哪個比較強？

**學生問**：生物學的例子講的是**突觸**，不是移除神經元。實務上移除神經元和移除突觸，哪個效果比較好？

**教授答**：

> 「我們兩個都會講 —— 剪權重、也剪 activation，並說明它們的關係：**如果你剪掉一個 neuron，等於把所有跟這個 neuron 相關的權重全部拿掉。所以它們不是獨立的。**」

（完整答案在 §4.5。**Neuron pruning = 粗顆粒度的 weight pruning**。）

### 2.4 AlexNet 的經典實驗（Han et al., 2015）

教授講的是他自己 2015 年做的實驗，用 AlexNet（6100 萬參數）：

**第一步：只剪不微調**

- 權重分佈是一個以 0 為中心的鐘型；**把絕對值小的權重砍掉**，分佈中間出現一個空洞
- 剪掉越多，準確率掉越多
- **剪掉 80% 的參數 → 掉 4.5% 準確率**

> 【口頭】「大家設計一個新網路架構，就是為了把準確率提升個 1%、2%，結果你一下掉 4%，這太糟了。**能不能做得更好？**」

**第二步：加上 Fine-tune（微調）**

固定剪枝後的連接圖（哪些連接死了、哪些還活著），**只微調剩下的權重**：

- **剪掉 ~80% → 幾乎沒有準確率損失**
- 【口頭】「有時候準確率甚至**還會微幅上升**，因為 **overfitting 變少了**。」

**微調後的權重分佈變化**：從「中間有洞的雙峰」變得**更平坦** —— 剩下的權重有了自由度，可以往左右移動來補償被剪掉的部分。

**第三步：Iterative Pruning（迭代剪枝）**

不要一次剪到底，而是「**剪一點 → 微調 → 再剪一點 → 再微調**」：

- **可以剪到約 90%（10× 壓縮）而準確率幾乎不掉**
- 【投影片】在 AlexNet 上，把壓縮率從 **5× 推到 9×**

### 2.5 【Q&A】三個關於 Iterative Pruning 的提問

**Q1：可以從時間 0（隨機初始化）就直接開始剪嗎？**

> 教授：「**非常有挑戰性。你至少得先訓練幾個 epoch。**」（這個問題在 §8 彩票假說會完整展開。）

**Q2：這是 training accuracy 還是 test accuracy？**

> 教授：「**Test accuracy。**」

**Q3：為什麼要迭代？直接說「90% 要歸零」不行嗎？迭代會不會反而卡在局部最小值？**

> 教授：「我試過。**如果你一次給一個很高的剪枝率、移掉大量連接，剩下的參數很難把準確率救回來。**反之，如果你**一小步一小步走** —— 從 15% 到 40%，再從 40% 到 ...... —— 這樣**比直接從 15% 跳到高剪枝率好。**」

**Q4：迭代到某個程度會收斂到一個極限嗎？**

> 教授：「會。**一旦你越過某個閾值，就算再多做幾次迭代，也很難再往前推了。**」

### 2.6 各種模型能剪多少

【投影片】不同模型剪枝後的壓縮率：

| 神經網路 | 剪枝前 #Params | 剪枝後 #Params | **參數縮減** | **MAC 縮減** |
|---|---|---|---|---|
| **AlexNet** | 61 M | 6.7 M | **9×** | 3× |
| **VGG-16** | 138 M | 10.3 M | **12×** | 5× |
| **GoogleNet** | 7 M | 2.0 M | 3.5× | 5× |
| **ResNet-50** | 26 M | 7.47 M | 3.4× | **6.3×** |
| **SqueezeNet** | 1 M | 0.38 M | 3.2× | 3.5× |

#### ⭐ 從這張表要讀出的三件事

**1. 模型越大 / 越冗餘，能壓越多。**

> 【口頭】「VGG-16 這種大模型壓縮比最高。AlexNet **本身就很冗餘**，可以壓 9 倍。」

**2. 就算是已經很小、已經設計得很精簡的模型，還是有壓縮空間。**

> 【口頭】「SqueezeNet 從名字就看得出很小，**只有 1 MB**（這是我 2016 年跟 UC Berkeley 合作的）。**即使是一開始就這麼小的模型，我們還是能再壓約 3×。**」

**3. #Params 縮減 ≠ MAC 縮減。**

看 ResNet-50：參數只縮 3.4×，但 MAC 縮了 **6.3×**。反過來 AlexNet 參數縮 9×，MAC 只縮 3×。
（**原因見 Lecture 2 §4.7**：AlexNet 的參數集中在 FC 層，而 FC 層的 MAC 佔比很低。）

### 2.7 剪枝也適用於多模態任務

【口頭】教授展示 image captioning（看圖說話）模型剪枝後的輸出：

| 剪枝率 | 輸出 |
|---|---|
| Baseline | "a basketball player in a white uniform is playing with a ball" |
| **剪 90%** | "a basketball player in a **white** uniform is playing with a **basketball**" ✅ 幾乎一樣 |
| Baseline | "a man is riding a surfboard on a wave" |
| **剪 90%** | "a man in a wetsuit is riding a wave on a beach" ✅ 意思正確 |
| Baseline | "a soccer player in red is running in the field" |
| **剪 95%** | "a **man** in a **red shirt** and **black and white**..." ❌ **開始語無倫次** |

> 【口頭】「剪到 95% 就**太激進了**，描述**不再正確**。所以**存在一個閾值，我們得把它找出來。**」

### 2.8 這條研究線的歷史與產業採用

**歷史**：

> 【口頭】「這條研究線其實**始於 1980 年代**，有一篇論文叫 **Optimal Brain Damage**（LeCun et al., NeurIPS 1989）。深度學習美妙的地方就在於：**有了更多算力、更多資料、更大規模，很多原理和數學其實幾十年前就存在了，只是在現代深度學習的脈絡下重生。**」

**產業採用**：

| 產品 / 硬體 | 應用 |
|---|---|
| **EIE**（Han et al., ISCA 2016） | 第一個針對稀疏、壓縮模型的 DNN 加速器 —— **不需要解壓縮就能直接跑** |
| **NVIDIA A100 GPU** | 採用 **weight sparsity（2:4）**，峰值效能 2×，實測約 **1.5× 加速** |
| **AMD（原 Xilinx）Vitis AI** | 來自新創 **DeePhi Tech**（教授參與創立），用剪枝把參數減少 **5–15×**，準確率影響極小 |

---

## 3. Pruning Granularity（剪枝顆粒度）

### 3.1 核心取捨

整節就一句話：

> **越規則（structured）→ 硬體越好加速，但能剪的越少。
> 越隨意（unstructured）→ 能剪的越多，但硬體越難加速。**

### 3.2 二維權重矩陣（8×8）的兩個極端

| | **Fine-grained / Unstructured** | **Coarse-grained / Structured** |
|---|---|---|
| 做法 | 零的位置**完全隨機**，任意 row / column 都可能 | 例如**整行歸零**（8 行剪掉 3 行，剩 5 行） |
| **彈性** | **最高** ✅ | 低 ❌ |
| **硬體加速** | **難** ❌ —— 不規則，而**平行運算討厭不規則** | **容易** ✅ —— 剪完還是一個稠密的小矩陣，可以直接用廠商的 GEMM 函式庫 |

> 【口頭】「剪完之後**矩陣仍然是稠密的、只是變小了** —— 你還是可以用廠商提供的 dense matrix multiplication 函式庫，就當作是**一個比較小的矩陣乘法**。」

### 3.3 卷積層有四個維度 → 五種顆粒度

Convolution 的權重有 **4 個維度**（$c_o, c_i, k_h, k_w$），比 FC 層的 2 個維度多，所以**選擇更多**：

```
Fine-grained  →  Pattern-based  →  Vector-level  →  Kernel-level  →  Channel-level
   最不規則                                                              最規則
   剪最多                                                                剪最少
   最難加速                                                              最好加速
```

| 顆粒度 | 定義 |
|---|---|
| **Fine-grained** | 任何維度的任何單一元素都可以歸零 |
| **Pattern-based** | 強制某種**規律的圖樣**（例如 N:M sparsity） |
| **Vector-level** | 整個 row（一條向量）全零或全不零 |
| **Kernel-level** | 整個 $k_h \times k_w$ 的 kernel 全零或全不零 |
| **Channel-level** | 整個 channel 全部保留或全部剪掉 |

### 3.4 Fine-grained Pruning（細顆粒度）

**優點**：剪枝率最高（AlexNet 9×、VGG-16 12×、ResNet-50 3.4×）。

**缺點**：

> 【口頭】「就像上一講講的，**參數數量的減少，不會直接轉換成速度提升**。speedup 還取決於它**是否容易被平行化與硬體加速**。」

### 3.5 Pattern-based Pruning：N:M Sparsity ⭐

**定義**：**在每連續 M 個元素中，剪掉 N 個。**

**經典案例：2:4 sparsity**（= 50% 稀疏度），**由 NVIDIA Ampere 架構原生支援，最高 2× 加速。**

#### 【Q&A】為什麼加上 pattern 就能加速？

**學生問**：這個 pattern-based 的好處是什麼？

**教授答**（一步步引導全班）：

> 「加上 pattern 之後**就不再是完全不規則的了**。看這個矩陣，你看得出什麼規律嗎？
> 你可以把它**壓縮**成：每一行**從 8 個元素變成剛好 4 個** —— **它仍然是稠密的**。
> 但你**得付出一些額外的儲存成本來記 index**。」

**Index 要幾個 bit？**（教授的課堂互動）

> 「在這 4 個元素中，每個留下來的元素要指出**自己坐在哪個位置**。有幾種可能？**4 種**（位置 0、1、2、3）。所以需要幾個 bit？—— **2 bits。**」

**壓縮後的儲存格式**（投影片，$R \times C$ 矩陣）：

$$\underbrace{R \times \tfrac{C}{2} \text{ 個非零值}}_{\text{Non-zero data values}} \;+\; \underbrace{R \times \tfrac{C}{2} \times 2\ \text{bits}}_{\text{2-bit indices（metadata）}}$$

> 【口頭】「所以我們**需要這些索引作為實際的額外開銷，但它相對很小**。這就是 Ampere 架構用的方式，給你 **2× 的峰值效能**。」

#### 準確率如何？

【投影片】**「Usually maintains accuracy（通常能維持準確率）」**，在多種任務上驗證過：

| 模型 | Dense | 2:4 Sparse |
|---|---|---|
| ResNet-50 | 76.1 | **76.2** |
| ResNeXt-50-32x4 | 77.6 | 77.7 |
| ResNeXt-101-32x16 | 79.7 | 79.9 |
| DenseNet-121 | 75.5 | 75.3 |
| Wide ResNet-50 | 78.5 | 78.6 |
| Inception v3 | 77.1 | 77.1 |
| VGG-16 | 74.0 | 74.1 |

【口頭】教授也提到涵蓋 SSD（偵測）、Mask R-CNN、Transformer、BERT-large，「**橫跨視覺與語言模型，準確率都維持得相當好**」。

### 3.6 Channel Pruning（通道剪枝）：最規則的極端

**做法**：整個 channel 拿掉 → 剪完就是一個維度變小的稠密張量。

| | |
|---|---|
| ✅ **最容易加速** | 【口頭】「**在你的手機上、沒有任何加速器、甚至一台 Raspberry Pi，你都能立刻拿到記憶體和運算的縮減。**」 |
| ❌ **剪得少** | 【口頭】「不好的地方是**有時只能剪掉 30%**。這就是取捨 —— fine-grained 可以到 **9 倍**，這個只有 **30%**。」 |

> 【口頭】「這在**業界被廣泛採用**。如果你去一家手機公司，**這大概就是他們最常用的技術**，因為它實在太容易加速了。」

#### Uniform vs Non-uniform Shrink

同樣的**整體剪枝率**，兩種分配方式：

| 做法 | 說明 |
|---|---|
| **Uniform Shrink** | 每一層都縮同樣的比例（例如全部砍 25%） |
| **Non-uniform**（每層不同比例） | 有些層剪多、有些層剪少 |

**Latency vs Accuracy 曲線**：理想點在**左上角**（低延遲、高準確率）。

> **結論：聰明地為每層選擇不同的稀疏率，比均勻縮減得到更好的 Pareto 曲線。**

**那要怎麼決定每層剪多少？** → 這正是 **Lecture 4 §6** 的主題。

---

## 4. Pruning Criterion（剪枝準則）

> 【口頭】「有這麼多 synapse、這麼多 neuron 都可以當候選，**我們要怎麼選？**」

### 4.1 直覺：從一個三權重的感知器開始

假設一個單層感知器：

$$y = f(10 x_0 - 8 x_1 + 0.1 x_2)$$

> 【口頭】「憑直覺，如果只能移除一個權重，你會移除哪一個？」
> **答案：$0.1$。**

**這就是 magnitude-based pruning 的起點**，也正是教授 2015 年論文用的方法。

### 4.2 Magnitude-based Pruning（基於數值大小）

#### Element-wise（細顆粒度）

$$\text{Importance} = |W|$$

**投影片範例**：權重矩陣 $\begin{bmatrix} 3 & -2 \\ 1 & -5 \end{bmatrix}$

| 重要性 | 剪枝後 |
|---|---|
| $\vert 3\vert =3,\ \vert {-2}\vert =2$ <br> $\vert 1\vert =1,\ \vert {-5}\vert =5$ | $\begin{bmatrix} 3 & 0 \\ 0 & -5 \end{bmatrix}$（剪掉最小的兩個：2 和 1） |

#### Row-wise，L1-norm（粗顆粒度）

要剪掉整行時，怎麼推廣？**把整行的絕對值加起來。**

$$\text{Importance} = \sum_{i \in S} |w_i|$$

**同一個矩陣**：

| 行 | L1-norm | 結果 |
|---|---|---|
| Row 0：$3, -2$ | $\vert 3\vert  + \vert {-2}\vert  = \mathbf{5}$ | **剪掉** ❌（較小） |
| Row 1：$1, -5$ | $\vert 1\vert  + \vert {-5}\vert  = \mathbf{6}$ | 保留 ✅ |

#### Row-wise，L2-norm

$$\text{Importance} = \sqrt{\sum_{i \in S} |w_i|^2}$$

| 行 | L2-norm |
|---|---|
| Row 0：$3, -2$ | $\sqrt{3^2 + (-2)^2} = \sqrt{13}$ ← 較小，**剪掉** |
| Row 1：$1, -5$ | $\sqrt{1^2 + (-5)^2} = \sqrt{26}$ |

> 【口頭】「**結果一樣** —— L1 和 L2 都剪掉第一行。」

#### 通式：Lp-norm

$$\|W^{(S)}\|_p = \left( \sum_{i \in S} |w_i|^p \right)^{1/p}$$

#### ⭐ 【口頭】為什麼這麼簡單的方法反而是業界標準

> 「這看起來超級簡單，但有趣的是，**這就是業界過去五年採用的方法**。有很多花俏的技術，但我們總是要在**有效性 vs. 整合難易度**之間找平衡。做研究，跟真正能落地成產品，這兩件事有非常實在的關係。」

**教授對「有效性」的定義**：

$$\text{Effectiveness} = \frac{\text{準確率 / 效能}}{\text{複雜度}}$$

> 「如果程式碼超級複雜，你需要一個 predictor、需要一堆複雜的東西，**那就很難做進產品裡。**」

### 4.3 Scaling-based Pruning（基於縮放因子）

**適用於 filter / channel pruning。**

**做法**：

1. 為每個 filter（也就是每個 output channel）配一個**可訓練的 scaling factor**
2. 這個 factor 乘在該 channel 的輸出上
3. 訓練時**鼓勵 scaling factor 趨近於零**
4. 剪枝時，**scaling factor 絕對值小的 channel 就剪掉**

**投影片範例**：

| Filter | Scaling Factor | 結果 |
|---|---|---|
| Filter 0 | 1.17 | 保留 ✅ |
| Filter 1 | **0.10** | **剪掉** ❌ |
| Filter 2 | **0.29** | **剪掉** ❌ |
| Filter 3 | 0.82 | 保留 ✅ |
| ⋮ | ⋮ | |
| Filter N-1 | 0.56 | 保留 ✅ |

#### ⭐ 免費的午餐：直接重用 BatchNorm 的 $\gamma$

$$z_o = \gamma \cdot \frac{z_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \beta$$

> 【口頭】「**幸運的是，這個 scaling factor 已經在那裡了 —— 就在 batch normalization 層裡**（我們上一講講過 BN）。所以我們可以**直接重用 BN 的 scaling factor $\gamma$** 來決定要剪哪個 channel。」

這就是 **Network Slimming**（Liu et al., ICCV 2017）的核心想法：**不用新增任何參數**。

（↔ 這直接扣回 **Lecture 2 §2.10** 講的 $\gamma$、$\beta$。）

### 4.4 Second-Order-based Pruning（二階導數）

**目標**：讓「剪掉某個 synapse 所造成的 loss 增加」最小。

#### 泰勒展開

$$\delta L = L(\mathbf{x}; W) - L(\mathbf{x}; W_P = W - \delta W) = \sum_i g_i \delta w_i + \frac{1}{2}\sum_i h_{ii}\delta w_i^2 + \frac{1}{2}\sum_{i \ne j} h_{ij}\delta w_i \delta w_j + O(\|\delta W\|^3)$$

其中

$$g_i = \frac{\partial L}{\partial w_i}, \qquad h_{i,j} = \frac{\partial^2 L}{\partial w_i \partial w_j}$$

#### Optimal Brain Damage 的三個假設（LeCun et al., 1989）

| # | 假設 | 消掉哪一項 |
|---|---|---|
| 1 | **目標函數 $L$ 近似二次** | 消掉 $O(\Vert \delta W\Vert ^3)$ 三階項 |
| 2 | **訓練已經收斂** → 梯度為 0 | 消掉**一階項** $\sum_i g_i \delta w_i$ |
| 3 | **刪除每個參數造成的誤差彼此獨立** | 消掉**交叉項** $\sum_{i \ne j} h_{ij}\delta w_i \delta w_j$ |

**只剩下對角二階項：**

$$\delta L_i \approx \frac{1}{2} h_{ii} w_i^2$$

**重要性分數（Saliency）：**

$$\boxed{\text{importance}_{w_i} = |\delta L_i| = \frac{1}{2} h_{ii} w_i^2}$$

（$h_{ii}$ 是非負的。$|\delta L_i|$ **越小的 synapse 越先剪**。）

#### ⚠️ 為什麼實務上不用

> 【投影片】**「Hessian Matrix H is difficult to compute.」**
> 【口頭】「找出精確的數學解在**計算上相當昂貴**，因為要算二階導數。所以人們推導出**各種近似方法**，不去算完整的 Hessian 矩陣。」

（這條線在 LLM 時代又復活了 —— SparseGPT / GPTQ 都是基於 Hessian 近似。）

### 4.5 Neuron Pruning：其實就是粗顆粒度的 Weight Pruning

> 【投影片】**「Neuron pruning is coarse-grained weight pruning.」**

#### 對 Linear Layer

剪掉一個 output neuron ⇒ **權重矩陣的一整行（row）被移除**。

【口頭】原本 5 個 input neuron、4 個 output neuron；剪掉一個 output neuron，**所有跟它相連的權重全部消失**。

#### 對 Convolution Layer

剪掉一個 channel ⇒ **對應的整組 filter 全部移除**。

【口頭】例：「第 3 個 channel 被剪掉，第 6 個 channel 也被剪掉 → 對應的權重就整個被丟掉。」

> **所以 neuron pruning 和 weight pruning 高度相關 —— 前者只是後者的粗顆粒度版本。**（這正是 §2.3 那個 Q&A 的完整答案。）

### 4.6 APoZ：用「零的比例」選 Neuron

**Average Percentage of Zero activations（平均零激活比例）**

**動機**：**ReLU 會在輸出 activation 產生大量的零。** 一個 channel 如果**經常是零**，把它整個歸零的影響就很小。

> 【投影片】**「The smaller APoZ is, the more importance the neuron has.」**
> （APoZ 越小 → 這個 neuron 越重要 → **APoZ 最大的那個 channel 該剪掉**）

#### 逐步算一次（投影片的完整例子）

**設定**：batch = 2 張圖，channel = 3，每張 feature map 是 4×4。

**Channel 0**：

- 圖 1 的 channel 0 有 **5 個零**
- 圖 2 的 channel 0 有 **6 個零**
- 總元素數 = $2 \times 4 \times 4 = 32$

$$\text{APoZ}_0 = \frac{5 + 6}{2 \cdot 4 \cdot 4} = \frac{11}{32}$$

**三個 channel 全部算完：**

| Channel | 零的個數（圖1 + 圖2） | APoZ | |
|---|---|---|---|
| **Channel 0** | 5 + 6 = 11 | $11/32$ | 最重要 ✅ |
| **Channel 1** | 5 + 7 = 12 | $12/32$ | |
| **Channel 2** | 6 + 8 = 14 | **$14/32$** | **← 剪掉** ❌ |

> 【口頭】「因為 channel 2 有最多的零，**你把它們全部歸零，改變最小**。所以我們要剪掉**零比例最大**的 channel —— 把它們全設成零，對最終結果的影響最小。」

### 4.7 Regression-based Pruning（回歸式剪枝）

**跟前面所有方法的根本差異**：

| | 看什麼 |
|---|---|
| 前面的方法 | **整個網路最終的 loss** $L(\mathbf{x}; W)$ |
| **Regression-based** | **單一層輸出的重建誤差（reconstruction error）** |

#### ⭐ 為什麼這在 LLM 時代特別重要

> 【口頭】「這在**近期的大型語言模型浪潮中非常有用**。知道為什麼嗎？因為它們**運算成本太高了** —— 175B 參數的模型，要跑完整個網路、反向傳播回每一層去重建權重，**極其困難**。
> 所以人們想出一個更聰明的方法：**我可以在本地做 —— 只看一層，試著讓這一層前後的變化最小。**」

#### 數學形式

原本的輸出（$X$ 是 $b \times c_i$ 的輸入，$W$ 是權重）：

$$Z = XW^T = \sum_{c=0}^{c_i - 1} X_c W_c^T$$

【口頭】「把它拆成**外積（outer product）的和**：$X_c$ 是 $b \times 1$，$W_c^T$ 是 $1 \times c_o$，外積得到 $b \times c_o$ 的**中間輸出**。$c_i$ 個 channel 就有 $c_i$ 個這樣的部分矩陣，加起來就是 $Z$。」

**剪枝問題就變成**：給每個部分矩陣一個**可學習的係數 $\beta_c$**，$\beta_c = 0$ 代表 channel $c$ 被剪掉。

$$\arg\min_{W, \beta} \|Z - \hat{Z}\|_F^2 = \left\| Z - \sum_{c=0}^{c_i-1} \beta_c X_c W_c^T \right\|_F^2 \qquad \text{s.t.} \qquad \|\beta\|_0 \le N_c$$

- $\beta$：長度 $c_i$ 的 channel 選擇係數向量
- $N_c$：保留的非零 channel 數

#### 求解方法：交替固定

> 【投影片】
> 1. **固定 $W$，解 $\beta$** → 做 channel selection（選要剪誰）
> 2. **固定 $\beta$，解 $W$** → 最小化重建誤差（調整剩下的權重去補償）
>
> **反覆迭代。**

【口頭】「解 $W$ 的意思是**我們可以調整權重，讓它在剪枝之後仍然能 fit**。」

#### 缺點

> 【口頭】「你可以想見這有缺點 —— **你可能沒有整個神經網路的全局視野，只是在重建單獨一層而已。所以永遠沒有白吃的午餐。**」

#### 【Q&A】這跟 PCA 是不是很像？

**學生問**：這是不是很類似 PCA？

**教授答**：

> 「**不完全是 PCA。** PCA 也是在最小化原始維度的誤差…… 但這裡用的是**迭代式的方法**，而且**重建的過程真的在微調權重**。」

**另一個相關的釐清**（【口頭】，學生問到 low-rank approximation）：

> 「**Low-rank approximation 跟 pruning 是正交（orthogonal）的技術。**
> 你可以想成 LoRA（LLM 微調中很流行的方法）是 pruning 的一個特例，**但不完全相同** —— 因為如果一個權重矩陣是 low-rank，把它們乘起來**維度還是一樣的**。
> 所以人們通常把 **pruning、quantization、low-rank、distillation** 當成**不同但可疊加**的技術：你可以在 low-rank 矩陣上再剪枝，也可以在剪枝過的矩陣上再量化。」

---

## 5. 課堂 Demo

【口頭】教授在 Google Colab 上現場跑了一個 **MNIST 手寫數字辨識** 的剪枝 demo。

> 「**這就是你們作業一（Lab 1）可以參考的起始程式碼** —— 基本上就是實作這個 pruning 流程、這個 fine-tuning 流程。」

### 5.1 結果

| 剪枝率 | 剪枝後準確率 | 微調 2 epoch 後 |
|---|---|---|
| 0%（原始模型） | **99.0%** | — |
| **70%** | 96.22%（小幅下降） | — |
| **80%** | **76%**（大幅崩壞，數字 8 被誤判成 5） | **98.89%** ✅ **幾乎完全恢復** |
| **90%** | 20% | ~88.5%（相當不錯） |
| **95–99%** | **~10%**（等同亂猜） | **25%，救不回來** ❌ |

> 【口頭】「就算移除了 **80% 的參數**，剩下的 20% 仍然強大到足以**完全恢復準確率**。」
> 「99% 的時候它把所有東西都預測成同一個數字。**準確率 10% 左右 —— 跟隨機猜一樣**（10 個類別）。」
> 「所以**這大概就是極限了。**」

### 5.2 【重要】Demo 沒有真的變快

**學生問**：Colab 沒有 Ampere 架構的 GPU，如果換成比較新的卡，是不是就能看到加速？

**教授答**：

> 「這個要付費才能用 Ampere GPU，而且**你得用 2:4 稀疏才行，那樣可以有 2× 加速**。
> **這裡示範的是不規則的隨機稀疏（irregular random sparsity）。**
> **我們做的事情只是把那些被剪掉的參數 mask 掉，然後微調剩下的權重 —— 其他部分完全一樣。所以你還是在做全部的運算。**
> 要拿到實測的加速，**下一講**會講那些能真正加速稀疏網路的技術。」

⚠️ **這是整個 Lecture 3 最容易誤解的一點**：**軟體層的 mask ≠ 實際加速。** Lecture 4 §9 就是在補這一塊。

### 5.3 【Q&A】Pruning vs Dropout

**學生問**：這跟 Dropout 有什麼不同？

**教授答**：

| | **Dropout** | **Pruning** |
|---|---|---|
| **目的** | **防止 overfitting** | **減少推論的運算量** |
| **什麼時候** | **訓練時** | **推論時**（模型固定） |
| **隨機性** | 每次迭代**隨機丟掉不同的權重** | 剪掉的是**固定的一組** |
| **最終模型** | 依賴**完整（dense）的模型**來做預測 | 就是那個**稀疏模型** |

> 【口頭】「**Dropout 是為了訓練，pruning 是為了推論。**」

### 5.4 【口頭】Google Colab 的實務建議

- **免費版就夠**完成 Lab 1 到 Lab 3
- **Lab 4 玩大型語言模型**時，建議付費升級 Colab Pro（約 $10/月）拿更快的 GPU
- Lab 0 先跑一次熟悉環境

---
---

# Lecture 4 — Pruning and Sparsity (Part II)

> 【口頭】開場回顧：「上一講我們剪掉了 90% 的參數，微調之後準確率仍然維持得很好。這一講繼續談 **pruning 的策略**。」
>
> **Lab 1（剪枝作業）已經上線**。【口頭】「點作業連結時，**記得選『Open as Google Colab』，不要選『Open as text editor』**，不然會看到一堆亂七八糟的標籤。這個 lab **不需要買 Colab Pro** 就能在合理時間內完成。」

---

## 6. Pruning Ratio（每層剪多少）

**問題**：一個網路有很多層，**該均勻剪，還是有些層多剪、有些層少剪？**

【投影片】Latency vs Accuracy 曲線：

- 起點是完整網路（右上）
- **Uniform shrink**（每層縮同樣比例）→ latency 下降，但**準確率也掉得很多**
- **Non-uniform（每層不同比例）** → **明顯更好的 latency-accuracy 折衷曲線**

### 6.0 【Q&A】先釐清「channel」和「layer」

**學生問**：不同的縮減比例，是指**同一層內的不同 channel**，還是**不同的層**？

**教授答**：

> 「是**不同的層**。這是一個 5 層的網路，我們在縮減它們 —— **這一層剪多一點、那一層剪少一點。**」

**學生追問**：那 channel 呢？channel 不是在層裡面嗎？

**教授答**：

> 「對，一層裡面有多個 channel，你是**剪掉其中一定數量的 channel**。例如**這裡原本是 100 個 channel，現在只剩 30 個**。好問題。」

---

### 6.1 方法一：Sensitivity Analysis（敏感度分析）

**核心觀察**：

> 【口頭】「**某些層你只要剪一點點，準確率就會急遽下降；某些層（例如 fully-connected layer）就算剪掉很多，準確率還是維持得住。**」

#### 做法（投影片：VGG-11 on CIFAR-10）

```
對每一層 Lk：
  1. 只剪這一層（其他層保持不動）
  2. 用一系列剪枝率試過去：10%, 20%, ..., 90%（step size ≈ 10%）
  3. 每個點測一次準確率
  4. 畫出「剪枝率 vs 準確率」曲線
把所有層的曲線疊在同一張圖上
```

**曲線長相**：都是往右下走（剪越多，準確率越低），但**斜率差很多**。

- **斜率陡的層** = **敏感** → 少剪
- **斜率平的層** = **不敏感** → 多剪（例如某層剪掉 90% 準確率只掉一點點）

#### 選 Ratio 的方法：畫一條水平閾值線

> 【口頭】「這純粹是**工程上的啟發式（engineering heuristic）**：我們設一個閾值（例如『準確率損失不超過 X%』），然後**每一層各自取它曲線跟這條線的交點**當作剪枝率。」

- 藍線（很平）→ 交點在很右邊 → **剪很多**
- 紅線（很陡）→ 一開始就低於閾值 → **剪很少**

> 【口頭】「**step size 越細，你得到的結果就越準確。**」

#### ⚠️ 兩個 Catch（教授在課堂上引導學生找出來的）

**Catch 1：忽略了層與層之間的交互作用**

**學生問**：這張圖是**只動那一層、不碰其他層**做出來的吧？那看起來每層剪 50% 都很安全，但**我們不知道全部一起剪之後會怎樣**。

**教授答**：

> 「**非常好的觀察，這正是第一個 catch。**我們**假設不同的層是獨立的、彼此不互動**。但實際上它們**可能會互動** —— 如果你先把 layer 1 剪掉 50%，其他層的敏感度曲線可能就**不長這樣了**。
> **我們之所以這樣做，是因為要在實驗成本（GPU hours、要評估幾個點）和準確度之間取捨。**」

【投影片】直接寫明：**「Sensitivity analysis ignores the interaction between layers → sub-optimal」**

**Catch 2：完全沒有考慮「每層有多大」** ⭐

> 【口頭】「還有另一個 catch。記得我們的目標是要在**運算量、準確率、模型大小**之間取得平衡。這裡少了什麼？」

**學生答**：「我們不知道每層的絕對大小 —— 剪掉小層的 90%，可能還比不上大層的 10%。」

**教授**：「**完全正確。**」

> 「綠色曲線那一層**可能是一個超小的層，只有 10 個參數** —— 就算你剪掉 80%，也只是拿掉 8 個參數。
> 反過來，藍色曲線如果是**一個巨大的層**，就算只剪 20%，可能就已經減少了大量的權重。」

#### 【口頭】為什麼這個問題在 LLM 時代變輕鬆了

| | 各層大小 |
|---|---|
| **Transformer / LLM** ✅ | **非常同質（homogeneous）** —— 例如 Llama 7B 有幾十層，**每層結構相同、維度相同**（QKV projection、expansion 都一樣） |
| **CNN** ❌ | 各層大小**差異極大** |

> 【口頭】「所以我們要確保不只考慮準確率，**也要考慮模型大小**。為了簡化，這裡我們假設各層大小差不多（差個兩三倍還好，不能差一百倍）。」

#### 【Q&A】那 20% 是剪哪 20%？

**學生問**：曲線上「20%」那個點，是**哪** 20% 被拿掉？

**教授答**：

> 「這是上一講講過的 —— **基於 magnitude**。先計算所有權重的絕對值、**排序**，然後**剪掉最小的那 20%**。這就是我們用來排序的啟發式。」

#### 【口頭】重要的實務忠告：不要用 test set 做 sensitivity analysis

**學生問**：這樣大量搜尋解空間，會不會有 **overfit 到測試集**的風險？

**教授答**：

> 「**經驗法則是：做 sensitivity analysis 時要用另外一個 validation set，而不是拿你的目標 test set 去 overfit。你不應該用 test set 做 sensitivity analysis。非常好的觀點。**」

（教授另外補充：pruning 本質上是**減少**參數量，而參數多才容易 overfit，所以 pruning **本身**傾向降低而非增加 overfitting 風險。）

#### 這個方法會在作業裡實作

> 【口頭】「**我們會在作業裡實作這個。**但歡迎你提出更進階的剪枝演算法 —— 不只根據準確率敏感度，**還把參數量也納入考慮**。如果你做到了，我們可以給你**額外加分**。」

---

### 6.2 方法二：AMC（AutoML for Model Compression）

#### 動機：人工調參不 scale

【口頭】教授講了自己的親身經歷：

> 「我 2015 年寫論文的時候只有 AlexNet 和 VGG，**ResNet 是 2015/2016 才出來的**。2017 年寫博論時，我覺得應該把這個新架構也納進來 ——
> **ResNet-50 有 50 層，AlexNet 只有 8 層**，設計空間大了非常多。
> 我花了**一整個暑假**（在 Stanford）剪 ResNet-50，**光是猜每層要剪多少，就花了超過一週。**」

> 「我們負擔不起僱一大堆工程師來做這件事。所以我到 MIT 之後就和學生設計**自動化的方法** —— 我們希望**不只是機器學習專家和硬體專家**能做模型壓縮，**也要讓非專家能一鍵完成**。」

**目標：Push-the-button solution** —— 你指定「目標準確率損失」和「目標模型大小」，丟一個模型進去，**吐出一個更小的模型**。

#### 把剪枝變成強化學習問題

【投影片】AMC 的 RL 設定：

| RL 要素 | 內容 |
|---|---|
| **State（狀態）** | 該層的特徵：**layer index、channel 數、kernel size、FLOPs** …… |
| **Action（動作）** | **一個連續數值** —— 該層的剪枝率 $a \in [0, 1)$ |
| **Agent** | **DDPG**（因為它支援**連續**動作輸出） |
| **Reward（獎勵）** | $R = -\text{Error}$（滿足限制時）；$-\infty$（不滿足限制時） |

> 【口頭】對 reward 的補充：「當然 **error 越低越好**，所以前面有個負號。我們也用 **FLOPs 的 log** 去調整它 —— 我們希望**低 error，同時低 FLOPs**。」

**為什麼用 $\log(\text{FLOPs})$**：

> 【口頭】「人們用實驗刻畫過**運算量與準確率的關係**，發現**大致是對數關係**。所以我們把它放進 reward function。」

**也可以直接優化 latency**：用**預先建好的 lookup table（LUT）**把每層配置對應到實測延遲。

#### 【Q&A】關於 RL 設定的三個提問

**Q：Critic 是某種 value function 或 Q function 嗎？**
> 教授：對，它回饋給 actor。

**Q：決策是 layer-by-layer 做的，那前面層的結果會不會影響後面？**
> 教授：「你是在**最後一步**才拿到 reward。你得**把整場遊戲玩完** —— 所有層的剪枝率都選完了，才拿到準確率當 reward。」
> （教授的比喻：「就**像玩一個遊戲** —— 你有很多層，每個動作就是選一個剪枝率，不是左右滑動，而是從一個連續空間選。」）

**Q：可以只用一些簡單的層級特徵嗎？**
> 教授：「我覺得那會是個不錯的啟發式候選。」

#### 結果

**vs 人工（教授本人）**：

| | 壓縮率 | 花費 |
|---|---|---|
| **人工專家（教授本人）** | 剪到 **29%** | **一週**的努力 |
| **AMC agent** | 剪到 **20%**（同樣準確率） | **4 張 GPU、不到一天** |

**在手機上的實測**（Samsung Galaxy S7 Edge，Qualcomm Snapdragon SoC，TF-Lite，單核，batch=1）：

| Model | MAC | Top-1 | Latency | **Speedup** | Memory |
|---|---|---|---|---|---|
| **1.0 MobileNet**（baseline） | 569 M | 70.6% | 119.0 ms | 1× | 20.1 MB |
| **AMC (50% FLOPs)** | 285 M | **70.5%** | 64.4 ms | **1.8×** | 14.3 MB |
| **AMC (50% Time)** | 272 M | 70.2% | **59.7 ms** | **2.0×** | 13.2 MB |
| 0.75 MobileNet（均勻縮減） | 325 M | **68.4%** ❌ | 69.5 ms | 1.7× | 14.8 MB |

> **關鍵對比**：AMC 跟 0.75 MobileNet 速度差不多，但 **AMC 的準確率高了 2.1%**。

#### ⭐ AMC 學到的規律：鋸齒狀（Zigzag）的剪枝策略

【投影片圖 14】ResNet-50 各層的剪枝率呈**明顯的鋸齒狀**：

| 位置 | Agent 的決定 | 原因 |
|---|---|---|
| **波谷（Crests，剪得兇）** | **3×3 convolution** → **剪很多** | 「3×3 conv **有更多冗餘**」 |
| **波峰（Peaks，剪得少）** | **1×1 convolution** → **剪很少** | 「1×1 conv **冗餘較少**」 |

**【口頭】教授的解釋**：

> 「3×3 conv **權重多很多，而且有重疊**。而且**更划算** —— **你剪掉一個 channel，在 3×3 就等於移除 9 個權重；在 1×1 只移除 1 個權重。**
> **這正是我們剛剛講的第二個 catch！我們的 agent 其實相當聰明，它部分解決了『沒有把層的大小納入考量』這個問題。**」

#### 【Q&A】為什麼剪 25% 卻快了 70%？

**學生問**：0.75 MobileNet 就是每層縮到 3/4，只少 25%，**為什麼會有將近 70% 的加速？是不是太多了？**

**教授引導**（先排除錯誤答案）：

> 學生猜「因為塞得進 GPU？」
> 教授：「那是一個原因，但**不太可能** —— 這模型本來就很小，只有 4 MB。」

**正確答案**：

> 「記得**卷積層 FLOPs 的計算有六項**：$h_o \cdot w_o \cdot k_h \cdot k_w \cdot c_i \cdot c_o$。
> **input channel 和 output channel 都變成原本的 3/4**，所以剩下的運算量是 $0.75 \times 0.75 \approx 0.56$ ——
> **它是二次方的（quadratic），不是線性的！** 這就是為什麼我們拿到約 70% 的加速。」

$$0.75 \times 0.75 = 0.5625 \quad \Longrightarrow \quad \frac{1}{0.5625} \approx 1.78\times$$

（↔ 直接應用 **Lecture 2 §4.7** 的 conv MAC 公式。）

#### 【Q&A】AMC 用的是哪種顆粒度？

**學生問**：agent 決定剪什麼樣的 pattern？

**教授答**：

> 「這裡我們用**最硬體友善的方式** —— 因為是跑在 **CPU** 上，所以我們走**最右邊的、整個 channel** 的方式，這樣可以直接被加速。
> **但這是正交的** —— 你也可以用同樣的 agent 去做 fine-grained pruning。」

---

### 6.3 方法三：NetAdapt（漸進式、以資源為目標）

> 【口頭】「這是我很喜歡的一個方法，來自我同事 Vivian（Vivienne Sze）的組。」

**與 AMC 的差異**：AMC 是 RL；**NetAdapt 是規則式、迭代式、漸進式（rule-based, iterative, progressive）**。

**目標**：找出每層的剪枝率，**滿足一個全域資源限制**（例如「總延遲要降到 X ms」）。

#### 演算法（投影片）

```
每一次迭代，目標是把延遲降低 ΔR（手動指定，例如 1 ms）：

  對每一層 Lk（圖中的 A 到 Z）：
    ① 剪這一層，剪到「整個網路的延遲減少剛好達到 ΔR」
       （用預先建好的 lookup table 查延遲）
    ② 短期微調（short-term fine-tune，10k iterations）
    ③ 量測微調後的準確率 → Acc_k

  在 Acc_A, Acc_B, ..., Acc_Z 之中，
  選擇「準確率最高」的那一層，真的把它剪掉

重複，直到總延遲減少量滿足限制
最後做一次「長期微調（long-term fine-tune）」→ 得到最終模型
```

#### 【口頭】直觀解釋

> 「原本跑 10 ms，我現在想跑 9 ms，所以 $\Delta R = 1$ ms。
> **第一層可能要剪得非常薄才能省下 1 ms；第二層只要剪一點點就夠了；最後一層又得剪得很兇。**
> 有這麼多種選擇都能達到同樣的延遲 —— **我們選那個給出最高準確率的。**
> 然後重複：從 9 ms 再降到 8 ms……最後找到模型，再做長期微調把準確率救回來。」

#### 結果

【投影片】latency vs Top-1 accuracy：相較於「乘一個固定係數（例如 0.75）」的均勻縮減，NetAdapt 找到的架構
**快約 1.7×，同時準確率還高 0.3%**。

---

### 6.4 三個方法總結

| 方法 | 類型 | 優點 | 缺點 |
|---|---|---|---|
| **Sensitivity Analysis** | 手工啟發式 | **超好實作** | ❶ 忽略層間交互 ❷ **忽略每層大小** |
| **AMC** | AutoML / RL（DDPG） | 全自動、一鍵、可直接優化 latency；**自動學到 3×3 剪多、1×1 剪少** | 需要 RL 訓練成本（幾張 GPU、一天） |
| **NetAdapt** | 規則式、漸進迭代 | 直接以**實測資源**為目標；概念簡單 | 每次迭代要試所有層 × 短期微調，**成本高** |

---

## 7. Fine-tuning / Training

### 7.1 為什麼需要

【投影片】剪枝率越高，準確率掉得越兇 —— **尤其是高剪枝率時掉得非常明顯**。微調就是把它救回來。

### 7.2 ⭐ Learning Rate 要調小

【口頭】教授的課堂提問：「微調一個剪枝過的模型時，learning rate 該**調大還是調小**？」

**答案：調小。**

> 「因為這個模型**基本上已經收斂了**。相較於一個未訓練、或從頭訓練的模型，這個已經差不多收斂。
> 所以通常你會用**原本 learning rate 的 1/10，甚至 1/100**。」

### 7.3 Iterative Pruning（迭代剪枝）

【投影片】**「Consider pruning followed by a fine-tuning is one iteration.」**（一次「剪枝 + 微調」算一輪迭代）

**做法**：逐輪提高目標稀疏度。

```
20% → 微調 → 40% → 微調 → 60% → 微調 → 70% → 微調 → 90% → 微調
```

**效果**（AlexNet）：**壓縮率從 5× 提升到 9×**（相較於一次到位的激進剪枝）。

#### ⚠️ 【口頭】但產品裡不見得這樣做

> 「這**從實作和工程角度需要更多心力** —— 你得為每一步重新啟動剪枝、重新做 sensitivity analysis、重新訓練。
> 所以（在產品裡）我們**實際上只實作了單次（one-pass）剪枝**，因為受限於使用者體驗。
> **但如果是做研究，我高度推薦用迭代的做法。**
> 這就是**學術研究**和**要做進產品、要能服務多種客戶與多樣網路結構**之間的差別。」

### 7.4 ⚠️ 產品化的另一個難題：Bypass Layer（跳接層）

> 【口頭】「我們在做剪枝產品時遇到的另一個挑戰是 **bypass layer**。
> 因為 bypass 有**依賴關係** —— `tensor_a + tensor_b`，**兩個張量必須大小一樣**，否則沒辦法做 element-wise add。
> **ResNet-50、MobileNet 都有這種層。**
> 所以我們得做一些 **tracing 來找出這些依賴關係**，才能正確地套用剪枝。」

> **「所以你會發現作業裡比較簡單，因為那裡沒有 bypass layer。如果你想挑戰自己，可以試著剪一個有 bypass layer 的網路。」**

### 7.5 Regularization（正規化）

**目的**（投影片）：

- **懲罰非零參數**（penalize non-zero parameters）
- **鼓勵參數變小**（encourage smaller parameters）

【口頭】為什麼要鼓勵變小：「**如果它很小，那在下一輪迭代剪枝時，它就很可能會被剪掉。**」

**兩個常用的正規化：**

| | 公式 |
|---|---|
| **L1 Regularization** | $L' = L(\mathbf{x}; W) + \lambda\Vert W\Vert $ |
| **L2 Regularization** | $L' = L(\mathbf{x}; W) + \lambda\Vert W\Vert ^2$ |

**實際用在哪：**

| 方法 | 用哪個正規化 |
|---|---|
| **Magnitude-based Fine-grained Pruning** | 對**權重**用 **L2** |
| **Network Slimming**（§4.3） | 對 **channel scaling factor** 用 **smooth-L1** |

> 【口頭】「當然有很多研究在做不同的啟發式、不同的正規化函數，**但實務上如果你要做產品、做新創，就用這兩個。**」

---

## 8. Lottery Ticket Hypothesis（彩票假說）

### 8.1 問題

> 【口頭】「大家會問：**我們得先訓練到收斂，然後再丟掉一堆東西。能不能一開始就從一個稀疏的模型開始訓練，直接把稀疏模型訓練到收斂，並且達到跟稠密模型一樣的準確率？**」

### 8.2 【口頭】先講為什麼直覺上很難

**學生的猜測**：「我覺得答案是不行，因為完整的維度比較容易 model 資料，你可以之後再降維；但直接從低維開始就很難。」

**教授**：「**非常好的觀點。**」

> 「對這種大型神經網路來說，**冗餘（redundancy）對於收斂到一個好的局部極小值非常關鍵**。
> 因為我們是用**非凸優化**（SGD 之類）去處理一個**高度非凸**的問題（神經網路）。
> **所以過參數化（over-parameterization）和冗餘相當關鍵。**」

### 8.3 論文的發現

> 【口頭】「但這篇論文很有意思：**對某些情況 —— 例如 CIFAR、MNIST 這種相對簡單的任務 —— 從一個稀疏網路開始訓練是可能的。**」

#### ⚠️ 但有一個關鍵的 catch

**你仍然必須先訓練到收斂，才能知道要剪哪些。**

流程：

```
1. 訓練 dense 網路到收斂
2. 基於 magnitude 剪枝  →  得到一個「稀疏性圖樣（sparsity pattern）」
3. 【關鍵】只繼承這個「圖樣」，權重重新隨機初始化
4. 從頭訓練這個稀疏網路  →  可以收斂到好的準確率
```

> 【口頭】「一開始你**不知道**該剪第一個還是第二個權重 —— **只有在你訓練完、拿到 magnitude 之後，才知道要剪哪個。**
> 然後你用那個**特定的圖樣**，但用**隨機初始化的權重**（所以投影片上這裡的顏色是淺藍色，那裡是深色）。
> **只繼承稀疏圖樣，不繼承權重。**」

### 8.4 限制與現況

| 資料集 | 可行嗎 |
|---|---|
| **MNIST、CIFAR** | ✅ 可以 |
| **ImageNet** | ❌ **非常有挑戰性** |

> 【口頭】「**如何擺脫這個訓練過程、直接知道要剪哪個，仍然是非常活躍的研究領域。**
> 有人發現**大概訓練三、四個 epoch 就足以找出稀疏圖樣** —— 但**那仍然不是零。**
> **能不能一開始就從非常稀疏、剪枝過的狀態出發？** 這還是個開放問題。」

---

## 9. 系統與硬體支援

> 【口頭】「上一講有個很好的問題：**我們把模型剪到只剩 10% 的參數，但延遲完全一樣，沒有任何加速。那要怎麼拿到加速？**」

**第一個答案**：用 **coarse-grained pruning**（整行移除）→ 剪完仍是稠密矩陣 → 直接用現成 GEMM 函式庫。
**但代價**：剪枝率低很多。

**那 fine-grained 的高剪枝率要怎麼變現？** → 需要**系統與硬體支援**。

【投影片】三條工作線：

| 系統 | 利用哪種稀疏 |
|---|---|
| **EIE** | **Weight sparsity + Activation sparsity** |
| **NVIDIA Sparse Tensor Core** | **M:N weight sparsity（2:4）** |
| **TorchSparse & PointAcc** | **Activation sparsity**（點雲的空間稀疏） |

---

### 9.1 EIE：Efficient Inference Engine

> 【投影片】**「The First DNN Accelerator for Sparse, Compressed Model」**（Han et al., ISCA 2016）

#### 三個原則

| 原則 | 意義 | 收穫 |
|---|---|---|
| **`0 × A = 0`**（Sparse Activation） | activation 是零 → 不用乘權重 | **3× less computation** |
| **`W × 0 = 0`**（Sparse Weight） | 權重是零 → 不用乘 activation | **10× less computation**、**5× less memory footprint** |
| **`2.09, 1.92 ⇒ 2`**（Weight Sharing） | 神經網路**不需要精確** → 可以量化、近似 | **8× less memory footprint**（4-bit weights） |

#### 兩種稀疏的關鍵差異 ⭐

| | **Weight Sparsity** | **Activation Sparsity** |
|---|---|---|
| **稀疏度** | ~**90%**（剪掉 90%，留 10%） | ~**60–70%**（ReLU 之後） |
| **靜態 or 動態** | **靜態（static）** —— 不管輸入是什麼，稀疏圖樣**完全一樣** | **動態（dynamic）** —— **執行時才知道**，不同輸入圖樣完全不同 |
| **省運算** | ✅ ~10× | ✅ ~3× |
| **省記憶體** | ✅ ~5× | ❌ **很難省**（因為是動態的） |

#### 【Q&A】為什麼 weight 省 10× 運算，卻只省 5× 記憶體？

**教授問全班**：「少掉的那一半去哪了？」

**答案**：**Index（索引）。**

> 「為了表示稀疏矩陣，**你需要索引來告訴你自己在哪裡** —— 我是非零沒錯，但我在第一行還是第二行？
> **這就是稀疏表示法的額外開銷。** 具體多少取決於你用的精度。」

#### 【Q&A】這些節省可以相乘嗎？

**學生問**：10×、3×、8× 這些是獨立的嗎？可以相乘成 30× 嗎？

**教授答**：**可以相乘。**

> 「**不是要 A 和 B 都是零才跳過 —— 只要 A 或 B 任一個是零就可以跳過。所以你可以把它們相乘。**
> 而 weight sharing 減少的是**每個權重的 bit 數**，pruning 減少的是**權重的個數** —— 這兩個是不同的軸，**所以也可以相乘。**」

**實測數字**（FC 層，AlexNet）：

- 權重剩下原本的 **9%**
- activation 剩下原本的 **35%**
- $0.09 \times 0.35 \approx 0.03$ → **約 33× 的運算縮減**

**反例（NeuralTalk / RNN-LSTM）**：

> 【口頭】「**LSTM 不用 ReLU 當 activation**，所以 activation 密度是 **100%** —— **節省只來自權重剪枝。**」

#### EIE 的微架構（PE 內部）

```
Activation Queue（只存非零的 activation）
        ↓
Pointer Read（column pointer 定位）
        ↓
Sparse Matrix SRAM（只存非零權重 + 相對索引）
        ↓
Weight Decoder（4-bit index → 16-bit 實際權重）  ← weight sharing
        ↓
Arithmetic Unit（乘加）
        ↓
Activation Read/Write + ReLU（產生下一層的非零 activation）
```

#### 稀疏矩陣的儲存格式（教授逐格講解）

三個陣列：

| 陣列 | 內容 |
|---|---|
| **Virtual weight** | 只存**非零**的權重值 |
| **Relative index** | **這個非零元素與前一個非零元素之間跳過了幾個零** |
| **Column pointer** | 指出**哪個元素是新的一 column 的開始** |

【口頭】走了一遍例子：`W0,0` → 下一個非零跳過 1 個 → 相對索引 = 1；再下一個跳過 2 個 → 相對索引 = 2……

**平行化的方式**：8 行的權重矩陣分給 4 個 **PE（Processing Element）**，每個 PE 負責 2 行，4 個 PE 平行算。遇到零的 activation 直接跳過，非零的才廣播給各 PE。

#### EIE 的歷史地位

> 【口頭】「這篇論文被列為 **ISCA 50 年來最令人興奮的五篇論文之一**。」

| 年份 | 論文 |
|---|---|
| 1995 | （早期經典） |
| 2017 | **Google TPU** —— Datacenter Performance Analysis of a TPU |
| — | Wattch: Power analysis |
| — | Transactional Memory |
| **2016** | **EIE** —— 第一個利用權重稀疏與剪枝的專用加速器 |

> 「有趣的是，這兩篇（TPU 和 EIE）**相對很新**，但在 ISCA 50 年裡，**機器學習加速器這個主題近年累積了非常大的動能。**」

#### ⭐ Retrospective：EIE 十年後的優缺點檢討

【口頭】教授今年寫了一篇 retrospective paper（委員會邀請作者在多年後回顧）。

**優點（Pros）：**

| # | 內容 |
|---|---|
| 1 | **專用硬體確實能讓稀疏運算變快** —— 即使密度只有 ~50% 也划算 |
| 2 | **weight sparsity 和 activation sparsity 都有機會** —— 跳過零不只**省能耗**，還**省下計算週期** |
| 3 | **激進量化到 4-bit（配合 weight sharing）省記憶體，解碼成 16-bit 再算** |

**關於第 3 點的關鍵洞見**：

> 【口頭】「一般來說，**記憶體和運算用同樣的精度** —— 8-bit 存、8-bit 算。
> 但這裡是說：**你可以用 4-bit 存、用 16-bit 算。因為記憶體很貴、運算很便宜。**
> 所以我們負擔得起用 16-bit 運算，但**儲存必須用更少的 bit**。
> **這正是 Lab 4 和 Lab 5 要實作的** —— 用這個方法把 Llama 2 部署到筆電上。」

> 「因為 **Llama 2 這類大型語言模型的即時文字生成是 memory-bound 的**。所以下一講會講 **W4A16（4-bit 權重、16-bit activation）**。」

**缺點（Cons）：**

| # | 內容 |
|---|---|
| 1 | **不適合向量處理器陣列** —— 平行化能力受限（要靠**結構化稀疏**才能解決 → 見 §9.2） |
| 2 | **控制流程開銷大** —— 「你想做一個算術運算，卻得做一堆冗餘操作」，還要存索引 |
| 3 | **當年只支援 FC 層**，不支援卷積層 |
| 4 | **假設所有權重都塞得進 SRAM** |

**關於第 3 點的意外轉折**：

> 【口頭】「**當時不支援卷積層。但有趣的是，去年 LLM 大爆發之後，它們全都是 FC 層 —— 所以這反而不算是缺點了。**」

**關於第 4 點**：

> 「這個假設對 **TinyML 和很多視覺應用來說很完美** —— 所有權重都能放上晶片的 SRAM。
> **TPU 其實就設計了 28 MB 的 SRAM**，因為當時的權重是 1400 萬個參數，用 2 bytes 剛好 28 MB。
> **但對大型語言模型，到今天為止仍然很困難** —— **70 億參數就要 7 GB 的 SRAM，那還是太貴了。**」

#### ⭐ 教授歸納的「高效運算通用原則」

> 【口頭】「**Be lazy（要偷懶）。**」

| 原則 | 說明 | 應用 |
|---|---|---|
| **遇到零就跳過** | 不要在零上做運算，避免冗餘 | Pruning、ReLU sparsity |
| **快速拒絕 / 延後工作** | 如果是零，直接拒絕，不要做 | **生成式 AI 的空間稀疏** —— 編修照片時只改了 10% 的像素，何必重新生成其他的？ |
| **Token-level sparsity** | 「**不是每個字都同樣重要** —— 你把一個句子遮掉 20%，還是看得懂意思」 | Transformer |
| **Progressive Quantization** | **先用 4-bit 算，看 softmax 分佈** —— 如果已經很尖銳（很確定），就不用算剩下的 4 bit；如果很平坦（不確定），才用完整 8-bit | LLM |
| **Temporal sparsity（時間稀疏）** | 影片的 frame 1 和 frame 2 之間可能只有幾個像素變了 —— **相減之後有大量的零** | 影片理解 |
| **Spatial sparsity（空間稀疏）** | 自駕車頂上旋轉的 LiDAR 打出很多光束，**不是每個地方都有物體** | 點雲（→ §9.3） |
| **Mixture-of-Experts** | 「**GPT-4 廣泛使用 MoE** —— 你有一個巨大的模型，但**每次不是用到所有參數，只用其中一部分。**」 | LLM |

> 「所以我們預期**未來的 AI 模型會是稀疏的**，而且會有**結構化稀疏 + 專用加速器的協同設計**。」

---

### 9.2 NVIDIA Sparse Tensor Core：M:N Sparsity

**支援的 GPU**：**A100 及之後**（H100 等）。

**規則**：**每連續 4 個元素中，最多 2 個非零**（2:4 sparsity）。

#### 壓縮格式

原本 $R \times C$ 的矩陣壓縮成：

$$\underbrace{R \times \tfrac{C}{2}\ \text{個非零值}}_{\text{Non-zero data values}} \;+\; \underbrace{R \times \tfrac{C}{2}\ \text{個 2-bit 索引}}_{\text{2-bit indices（metadata）}}$$

【口頭】索引的計算：「這 4 個位置是 0、1、2、3，**每個非零元素要 2 bits 記錄自己的相對位置**。」

#### 計算流程（教授逐步講解）

```
權重矩陣 W (M × K)，2:4 稀疏      激活矩陣 (K × N)，稠密
        │                                    │
        ① 只選出非零權重                     │
           （原本 8 個 → 只剩 4 個）          │
        │                                    │
        └──── ② 用 2-bit 索引 ──────────────►│
                                             ③ 用 MUX 選出「對應同一個
                                                稀疏圖樣位置」的 activation
                                                （8 個中選 4 個）
                                             │
                              ④ 只做 4 個 MAC 的內積
                                 （稠密版本要 8 個）
```

> 【口頭】「所以我們**不用做全部 8 次運算，只選那些非零的權重 —— 只有 4 個**。
> 然後在這 8 個 activation 中，**用索引選出對應非零權重的那些**，這**可以用 MUX 輕鬆實作**。
> 最後只對 4 個項目做點積，而不是 8 個。」

#### 實測加速

【投影片，A100 Tensor Core，INT8 GEMM，cuSPARSELt vs cuBLAS】：

> **「Larger GEMMs achieve nearly a 2× speedup with Sparse Tensor Cores.」**

- 加速比隨 GEMM 維度（K）增大而上升：從 K=1280 到 K=20480
- 【口頭】「**矩陣越大，加速比越高，大約 1.8 到 1.9 倍**，比稠密版本快。」

**為什麼**：越大的 GEMM **arithmetic intensity 越高**，越接近理論的 2× 上限。

#### 可以疊加量化嗎？

> 【口頭】「**可以。你可以在稀疏模型之上再量化到 INT8 —— 稀疏和量化可以一起用。**」

---

### 9.3 TorchSparse / PointAcc：利用 Activation Sparsity

**應用場景**：**點雲（point cloud）**、物理粒子模擬、LiDAR。

【口頭】「我手機上就有 LiDAR 感測器 —— **它打出很多光束，打到物體才收集到點，沒有物體的地方就是零。**」

#### 問題：一般卷積會讓稀疏度消失

> 【口頭】「**問題是，你卷積越多次，它就變得越稠密（the more you convolve, the denser you get）。**」

原本空間中只有 5 個點，一般卷積會把它們「暈開」，稀疏性逐層流失。

#### 解法：Sparse Convolution

**強制輸出維持與輸入相同的稀疏圖樣。**

| | 一般 Convolution | **Sparse Convolution** |
|---|---|---|
| 輸出 | 每個窗口都產生一個輸出 | **只在「輸出非零」的位置才算** |
| 稀疏度 | 逐層下降（越來越稠密） | **不管做幾層卷積都維持** |

> 【口頭】「假設有 9 個位置，稠密卷積要**全部 100% 都算**；稀疏卷積**只有其中一小部分需要算** —— 例如只有 2 個。」

#### 新問題：嚴重的負載不均衡

> 【口頭】「不同的權重跟不同數量的輸入互動 —— 例如權重 $(-1,-1)$ 只跟 2 個輸入互動，權重 $(-1, 0)$ 只跟 1 個互動，另一個跟 5 個互動。
> **這高度不平衡。那要怎麼平行化這種圖樣？**」

#### ⭐ TorchSparse 的核心原則：用運算換規律性

> 【口頭】**「一個重要的原則：用運算量去換規律性（trade computation for regularity）。」**
>
> 「**我付出一些冗餘去計算那些零，但換到規律性，讓我更容易在 GPU 上平行化。**」

**三個做法的比較：**

| 做法 | 運算開銷 | 規律性 | 評價 |
|---|---|---|---|
| **Baseline（完全不浪費）** | ✅ 零開銷 —— 只算非零 | ❌ **非常不規則，極難平行化** | |
| **全部 padding 成稠密** | ❌ **大量浪費**（補一堆零） | ✅ **超規律** —— 整塊丟給 GPU | |
| **TorchSparse（折衷）** ⭐ | 適中 | 適中 | **把大小相近的矩陣分組（group），稍微 pad 一點，再一起發射 kernel** |

#### 【Q&A】所以還是在對零做運算？

**學生問**：所以卷積和矩陣乘法有加速，但輸出的零本身沒辦法避免？

**教授答**：

> 「對，**我確實在零上做了很多運算，浪費了不少 cycle。**
> **但我們讓它不會擴散（dilate）的方法，就是強制輸出維持跟輸入相同的稀疏圖樣。** 所以就算你在這裡算出了什麼，它還是會是零 —— **這樣下一層的輸入仍然是稀疏的。**」

---

## 10. 一頁速查表

### 10.1 四個決策

| 決策 | 選項 | 建議 |
|---|---|---|
| **Granularity** | Fine-grained ↔ Pattern ↔ Vector ↔ Kernel ↔ Channel | **通用硬體用 Channel；A100 用 2:4；有專用加速器才用 Fine-grained** |
| **Criterion** | Magnitude / Scaling / Second-order / APoZ / Regression | **業界標準 = Magnitude（L1/L2）**；LLM 場景考慮 Regression |
| **Ratio** | Uniform / Sensitivity / AMC / NetAdapt | **作業用 Sensitivity；要自動化用 AMC** |
| **Fine-tune** | One-pass / Iterative | **研究用 Iterative（5×→9×）；產品用 One-pass** |

### 10.2 重要性公式

| 準則 | 公式 |
|---|---|
| **Magnitude（element-wise）** | $\text{Importance} = \vert W\vert $ |
| **Magnitude（L1, row-wise）** | $\sum_{i \in S} \vert w_i\vert $ |
| **Magnitude（L2, row-wise）** | $\sqrt{\sum_{i \in S} \vert w_i\vert ^2}$ |
| **Lp-norm 通式** | $\left(\sum_{i\in S}\vert w_i\vert ^p\right)^{1/p}$ |
| **Scaling-based** | $\vert \gamma\vert $（直接用 BatchNorm 的 $\gamma$） |
| **Second-order（OBD）** | $\frac{1}{2} h_{ii} w_i^2$ |
| **APoZ（越大越該剪）** | $\dfrac{\text{該 channel 的零個數}}{n \cdot H \cdot W}$ |
| **Regression-based** | $\arg\min_{W,\beta} \left\Vert Z - \sum_c \beta_c X_c W_c^T\right\Vert _F^2,\ \Vert \beta\Vert _0 \le N_c$ |

### 10.3 正規化與微調

| 項目 | 建議 |
|---|---|
| **Learning rate** | 原本的 **1/10 ~ 1/100** |
| **L1 Regularization** | $L' = L + \lambda\Vert W\Vert $ |
| **L2 Regularization** | $L' = L + \lambda\Vert W\Vert ^2$ |
| Magnitude fine-grained pruning | 對權重用 **L2** |
| Network Slimming | 對 scaling factor 用 **smooth-L1** |

### 10.4 該記住的數字

| 數字 | 意義 |
|---|---|
| **9× / 12× / 3.4×** | AlexNet / VGG-16 / ResNet-50 的參數壓縮率 |
| **2:4** | NVIDIA Ampere 的 N:M 稀疏，**~2× 加速**，2-bit index |
| **~90%** | Weight sparsity（靜態） |
| **~60–70%** | Activation sparsity（ReLU 後，**動態**） |
| **10× / 3× / 8×** | EIE 的權重運算 / activation 運算 / weight sharing 記憶體節省 |
| **~33×** | EIE 在 AlexNet FC 層的實際運算縮減（$0.09 \times 0.35$） |
| **640 pJ vs 0.1 pJ** | DRAM 存取 vs int ADD（**200× vs register**） |
| **2,500 → 15,000 → 7,000** | 人腦每神經元突觸數（新生兒 → 幼兒 → 成人） |
| **0.75 × 0.75 = 0.56** | 為什麼「每層砍 25%」會有 ~1.8× 加速（**二次方**） |

---

## 11. 與其他課程／作業的連結

| 本講觀念 | 連到哪裡 |
|---|---|
| **§4.3 重用 BatchNorm 的 $\gamma$** | **Lecture 2 §2.10** —— 正規化層的 $\gamma$、$\beta$ |
| **§6.2 為什麼砍 25% channel 能快 1.8×** | **Lecture 2 §4.7** —— conv MAC 的六項公式（$c_i \cdot c_o$ 是二次的） |
| **§2.6 #Params 縮減 ≠ MAC 縮減** | **Lecture 2 §4.4/§4.7** —— AlexNet FC 層佔 96% 參數但只佔 8% MAC |
| **§1.3 DRAM 640 pJ** | **Lecture 2 §4.3** —— pruning 的物理動機 |
| **§5 課堂 Demo** | **Lab 1** —— 直接就是作業的起始程式碼 |
| **§6.1 Sensitivity Analysis** | **Lab 1** —— 要實作的部分（改良可拿額外加分） |
| **§9.1 W4A16（4-bit 存、16-bit 算）** | **Lecture 5-6 Quantization** ＋ **Lab 4 / Lab 5** —— 把 Llama 2 部署到筆電 |
| **§4.4 Second-order / Hessian** | LLM 剪枝（SparseGPT、GPTQ 都基於 Hessian 近似） |
| **§4.7 Regression-based（只看單層）** | **LLM 講次** —— 175B 模型沒辦法端到端反傳，只能逐層重建 |
| **§6.2 AMC 是 RL** | **Lecture 7-8 NAS** —— 同一套 AutoML 思路，搜的東西不同 |
| **§9.1「未來模型會是稀疏的」+ MoE** | **LLM 講次** —— GPT-4 的 Mixture-of-Experts |
| **§3.6 Channel pruning 業界最愛** | **Lecture 7-8 Hardware-aware NAS** —— 同樣是「規則 = 好加速」的邏輯 |

---

## 附：兩講的一句話總結

> **剪枝的四個決策裡，只有「剪哪些」有漂亮的數學（magnitude、Hessian、回歸）；
> 其他三個 ——「剪什麼形狀」「每層剪多少」「怎麼救回來」—— 全都被硬體現實綁死：
> 越規則的稀疏越好加速但剪得越少，而軟體層的 mask 完全不會讓模型變快。**

---

*筆記依據 MIT 6.5940 Fall 2023 Lecture 3 / Lecture 4 逐字稿（Zoom 錄影）與官方投影片 `Lec03-Pruning-I.pdf`、`Lec04-Pruning-II.pdf` 整理。標記【投影片新版】處，投影片為後續學期更新版本，與 2023 課堂口述數字不同，兩者皆已列出。*
