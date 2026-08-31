# EfficientML.ai 第九講完整筆記：知識蒸餾（Knowledge Distillation）

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2023 — Lecture 9
> **講者**：Song Han
> **來源**：Zoom 錄影逐字稿（1:00:11）＋ 官方投影片 `Lec09-Knowledge-Distillation.pdf`（84 頁）
>
> **標記說明**
> - `【口頭】` = 教授課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 學生提問與回答
> - `【投影片新增】` = 投影片為後續學期更新版本，2023 課堂逐字稿沒有這段

---

## 目錄

- [0. 為什麼需要知識蒸餾](#0-為什麼需要知識蒸餾)
- [1. KD 的直覺與形式定義](#1-kd-的直覺與形式定義)
- [2. What to Match：六種可以對齊的東西](#2-what-to-match六種可以對齊的東西)
- [3. Self / Online Distillation：能不能不要老師？](#3-self--online-distillation能不能不要老師)
- [4. KD 用在不同任務](#4-kd-用在不同任務)
- [5. Network Augmentation：反過來對付 underfitting](#5-network-augmentation反過來對付-underfitting)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與其他課程／作業的連結](#7-與其他課程作業的連結)

---

## 0. 為什麼需要知識蒸餾

### 0.1 KD 在這門課的位置

前三章都在**改模型本身**；KD 不改模型，它改的是**怎麼訓練**：

| 技術 | 動什麼 |
|---|---|
| **Pruning**（Lec 3-4） | 減少**權重的數量** |
| **Quantization**（Lec 5-6） | 減少**每個權重的位元數** |
| **NAS**（Lec 7-8） | **一開始就設計**一個小而精的架構 |
| **KD**（Lec 9） | **不動架構，改善「怎麼訓練」** ⭐ |

> 【口頭】**「Distillation 這個 teacher-student 框架，可以幫助上面所有三種場景的訓練。」**
> - Dense 網路當老師 → **pruned 網路**當學生
> - 全精度模型當老師 → **量化模型**當學生
> - NAS 搜出來的小架構，訓練時也可以配一個老師
>
> **這就是為什麼 distillation 被排在 pruning、quantization、NAS 之後。**

### 0.2 硬體落差

| | **Cloud AI** | **Tiny AI** |
|---|---|---|
| **算力（fp32）** | **19.5 TFLOPS** | **MFLOPs**（百萬級） |
| **記憶體** | **80 GB** | **256 kB** |
| 代表模型 | ResNet、ViT-Large | MCUNet、MobileNetV2-Tiny |

> **神經網路必須夠小才跑得動邊緣裝置。但問題是 —— 我們知道怎麼訓練大模型，卻不知道怎麼訓練小模型。**

### 0.3 ⭐ 核心問題：小模型會 underfit

**這是整堂課的出發點，而且跟一般直覺相反。**

| 模型 | ImageNet Top-1 | Top-5 |
|---|---|---|
| **ResNet-50** | **> 76%** | — |
| **MobileNetV2-Tiny** | **< 48%** | **只有 52%** |

**訓練曲線的差異**（投影片 p4）：

| | ResNet-50 | MobileNetV2-Tiny |
|---|---|---|
| 訓練準確率 | ~82% | ~52% |
| 驗證準確率 | ~76% | ~48% |
| **狀態** | **有 overfitting 空間**（train > val 明顯） | **underfitting** —— **連訓練集都學不好** |

> **「Tiny models underfit large datasets.」**
> 【口頭】**「大模型我們知道怎麼訓練，但小模型呢？我們能不能請大模型當老師來幫忙？」**

⚠️ **這個「小模型 underfit」的觀察會在 §5（NetAug）再回來，並導出一個非常反直覺的結論。**

---

## 1. KD 的直覺與形式定義

### 1.1 基本架構

```
              ┌──────────────────┐
        ┌────►│  Teacher Network │──► Logits ──┐
        │     │  （大、通常固定） │             │
  Input ┤     └──────────────────┘             ├──► Distillation Loss
        │     ┌──────────────────┐             │
        └────►│  Student Network │──► Logits ──┘
              │  （小）          │      │
              └──────────────────┘      └──► Classification Loss（跟真實標籤比）
```

**學生同時受到兩種監督：**

| 損失 | 來源 |
|---|---|
| **Classification Loss** | 真實標籤（跟平常一樣） |
| **Distillation Loss** ⭐ | **老師的輸出** —— 這是**額外**多出來的監督訊號 |

論文：Distilling the Knowledge in a Neural Network [Hinton et al., NeurIPS Workshops 2014]

### 1.2 ⭐ 直覺：貓狗二分類的具體數字

投影片 p6 用一張貓的圖片走完全程（2 類：Cat / Dog）：

| | **Logit (Cat)** | **Logit (Dog)** | **P(Cat)** | **P(Dog)** |
|---|---|---|---|---|
| **Teacher（大模型）** | **5** | 1 | **0.982** | 0.017 |
| **Student（小模型）** | **3** | 2 | **0.731** | 0.269 |

**Softmax 怎麼算的**：

$$P(\text{Cat}) = \frac{\exp(5)}{\exp(5) + \exp(1)} = 0.982$$

> **關鍵觀察：學生模型「比較沒自信」（less confident）** —— 老師 98.2%，學生只有 73.1%。
>
> **KD 要做的事：把學生的機率分佈推向老師的機率分佈。**

### 1.3 Temperature（溫度）

$$\boxed{p(z_i, T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}}$$

- $i, j = 0, 1, \ldots, C-1$，$C$ 是類別數
- $T$ 是**溫度**，**預設是 1**

**同一組 logit，不同溫度**（投影片 p8）：

| | Logits | **P (T=1)** | **P (T=10)** |
|---|---|---|---|
| **Cat** | 5 | **0.982** | **0.599** |
| **Dog** | 1 | 0.017 | **0.401** |

$$T=1: \frac{\exp(5/1)}{\exp(5/1)+\exp(1/1)} = 0.982 \qquad T=10: \frac{\exp(5/10)}{\exp(5/10)+\exp(1/10)} = 0.599$$

> **「A larger temperature smooths the output probability distribution.」**
> （溫度越高，輸出機率分佈越平滑。）

**為什麼要平滑**：老師在 $T=1$ 時輸出接近 one-hot（0.982 / 0.017），這樣傳給學生的資訊跟真實標籤差不多、幾乎沒有額外價值。**調高溫度後，那些「非正確類別之間的相對大小」（所謂 dark knowledge）才凸顯出來** —— 例如「這張圖比較不像狗，但更不像卡車」這種資訊。

### 1.4 形式定義

> **KD 的目標：讓老師與學生的類別機率分佈對齊（align the class probability distributions）。**

---

## 2. What to Match：六種可以對齊的東西

> 【口頭】「我們可以匹配 output logits、intermediate weights、intermediate features、gradients、sparsity pattern、relational information —— **有很多地方可以讓老師和學生對齊。**」

投影片 p11 的完整清單：

| # | 匹配什麼 | 小節 |
|---|---|---|
| 1 | **Output logits**（輸出對數機率） | §2.1 |
| 2 | **Intermediate weights**（中間層權重） | §2.2 |
| 3 | **Intermediate features**（中間層特徵圖） | §2.3 |
| 4 | **Gradients**（梯度／注意力圖） | §2.4 |
| 5 | **Sparsity patterns**（稀疏圖樣） | §2.5 |
| 6 | **Relational information**（關係資訊） | §2.6 |

---

### 2.1 匹配 Output Logits（最簡單、最常用）

**兩種損失可選：**

| 損失 | 公式 |
|---|---|
| **Cross entropy loss** | $\mathbb{E}(-p_t \log p_s)$ |
| **L2 loss** | $\mathbb{E}(\Vert p_t - p_s \Vert_2^2)$ |

論文：Hinton et al. 2014；Do Deep Nets Really Need to be Deep? [Ba and Caruana, NeurIPS 2014]

---

### 2.2 ⭐ 匹配 Intermediate Weights：投影器（Projector）

#### 【Q&A】能直接算兩邊權重的距離嗎？

**教授問全班**：「這裡的 catch 是什麼？可以直接計算老師權重和學生權重的 L2 距離嗎？」

**學生答**：「不行，因為學生是比較小的模型，**兩者不是一對一的**。」

**教授**：**「完全正確 —— 而讓它小本來就是我們的目的。」**

#### 解法：加一個線性投影層

```
Student 的中間層 ──► [Projector: 1×1 Conv 或 FC] ──► 維度對齊 ──► 跟 Teacher 比
                          ↑ 這一塊是可學的，跟整個網路一起 end-to-end 訓練
```

**具體維度**（教授的例子）：

> 「老師的 channel 是 **512**，學生的 channel 是 **256**，那從學生投影到老師的投影器維度是多少？
> —— **256 × 512**。就是一個簡單的 FC 層，用來對齊 channel 數。」

#### 【Q&A】那不就只是在做線性組合嗎？

**學生問**：所以我們其實只是在**用大模型權重的線性組合**去近似？

**教授答**：**「對，就是這樣 —— 這就是線性投影在做的事。」**

【口頭】補充：**投影器是跟主網路一起 end-to-end 訓練的**；實務上還有一些技巧（例如加 clipping）。

---

### 2.3 匹配 Intermediate Features（中間特徵圖）

**直覺**（投影片 p16）：

> **「老師和學生應該有相似的『特徵分佈』，而不只是相似的『輸出機率分佈』。」**

**做法**：最小化兩邊 feature map 的 **Maximum Mean Discrepancy（MMD）**。

**注意**：不只是最後一層的輸出特徵，**中間每一層的特徵圖都可以加 KD loss**。

論文：Like What You Like: Knowledge Distill via Neuron Selectivity Transfer [Huang and Wang, arXiv 2017]

---

### 2.4 ⭐ 匹配 Gradients：中間注意力圖（Attention Map）

#### 定義

CNN 特徵圖 $x$ 的「注意力」定義為 **loss 對該特徵圖的梯度**：

$$\text{Attention} = \frac{\partial L}{\partial x}$$

#### 直覺

> **「如果 $\dfrac{\partial L}{\partial x_{i,j}}$ 很大，代表位置 $(i,j)$ 一個微小的擾動，就會顯著影響最終輸出。
> 因此網路正在把更多注意力放在位置 $(i,j)$。」**

【口頭】教授用一張狐狸的圖示範：注意力圖確實**亮在狐狸身上**，而不是背景 —— 「這很合理」。

#### 通道要怎麼壓成一張圖（Reduction Function）

梯度有很多 channel（例如 64 個），但注意力圖只要一張，所以要做 reduction。可選的方式：

| 方法 |
|---|
| **Sum**（各通道相加） |
| **Sum of squares**（平方和） |
| **Sum of power 4**（四次方和） |
| **Max** |

> 【口頭】「**不管用哪一種 reduction function 都可以試**，各種函數都行。」

#### ⭐ 一個很有說服力的驗證

【口頭】教授比較不同架構的注意力圖：

| 模型 | 準確率 | 注意力圖 |
|---|---|---|
| **Network-in-Network** | **62%** | **跟其他兩個差很多** ❌ |
| **ResNet-34** | 73% | **彼此非常相似** ✅ |
| **ResNet-101** | 77% | **彼此非常相似** ✅ |

> **「這些高效能模型的注意力圖彼此非常相似，跟那個表現差的網路則差異很大。」**
>
> **推論：好模型「看的地方」是一致的 → 所以「讓學生的注意力圖對齊老師」是個合理的監督訊號。**

**維度問題**：梯度的維度跟權重一樣，所以老師和學生的梯度維度也不同 → **一樣用 FC 投影層對齊**（同 §2.2）。

論文：Paying More Attention to Attention [Zagoruyko and Komodakis, ICLR 2017]

---

### 2.5 匹配 Sparsity Patterns（稀疏圖樣）

**直覺**：

> **「老師和學生在 ReLU 之後應該有相似的稀疏圖樣。」**

**指示函數**：

$$\rho(x) = \mathbb{1}[x > 0]$$

**也就是說**：

| 老師的該神經元 | 學生的該神經元應該 |
|---|---|
| **啟動（> 0）** | **也要 > 0** |
| **是 0** | **也要是 0** |

（↔ 這直接連回 **Lecture 3-4**：ReLU 後約 60–70% 的 activation 是零，而零可以跳過。）

論文：Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons [Heo et al., AAAI 2019]

---

### 2.6 ⭐⭐ 匹配 Relational Information（關係資訊）

前面五種都是「一個點對一個點」地比。**關係資訊比的是「點與點之間的關係」。**

分成兩種：

#### (A) 不同「層」之間的關係（FSP Matrix）

**做法**：用**內積**萃取關係。

【口頭】教授在白板上帶著算了一遍：

```
同一個 stage 的前後兩個特徵圖（解析度相同，channel 數可以不同）：

  輸入端： 112 × 112 × 64   ← 64 channels
  輸出端： 112 × 112 × 128  ← 128 channels
             │
             ├─ 在 X, Y 空間維度上做 reduction（矩陣乘法）
             ▼
       得到一個 64 × 128 的矩陣  ← 這就是「關係」
```

**老師也算出同樣形狀的矩陣（$C_{\text{in}} \times C_{\text{out}}$），然後兩邊對齊。**

> **重點：這不是在看「一張特徵圖」，而是在看「兩張特徵圖之間的關係」。**

⚠️ **前提**：兩張特徵圖必須**在同一個 stage、解析度相同**（channel 數可以不同）。

#### (B) 不同「樣本」之間的關係（RKD）

**這是更漂亮的一種。**

```
【傳統 KD（Individual KD）】          【關係型 KD（Relational KD）】

 x₁ ─► Teacher ─► t₁ ─┐               x₁ ─► t₁ ─┐
                      │ 比對                     ├─► ψ(t₁,…,tₙ) ─┐
 x₁ ─► Student ─► s₁ ─┘               x₂ ─► t₂ ─┤                │
                                      xₙ ─► tₙ ─┘                │ 比對
 x₂ ─► t₂ vs s₂（同樣逐一比）                                     │
 x₃ ─► t₃ vs s₃                       x₁ ─► s₁ ─┐                │
                                      x₂ ─► s₂ ─┼─► ψ(s₁,…,sₙ) ─┘
                                      xₙ ─► sₙ ─┘
```

**關係向量的定義**（投影片 p25）：

$$\psi(s_1, s_2, \ldots, s_n) = \left(\Vert s_1 - s_2\Vert_2^2,\ \Vert s_1 - s_3\Vert_2^2,\ \ldots,\ \Vert s_{n-1} - s_n\Vert_2^2\right)$$

**長度 $n(n-1)/2$ 的向量，記錄的是「所有樣本兩兩之間的距離」。**

#### 🔑 為什麼這樣更聰明

> 【口頭】**「老師和學生的特徵可能只是整體平移了一點點，但它們彼此之間形成的關係非常相似。」**
>
> 傳統 KD 要求學生的特徵**位置**要跟老師一樣（太嚴格）；
> RKD 只要求學生特徵之間的**相對結構**跟老師一樣（更寬鬆、更合理）。

論文：Relational Knowledge Distillation [Park et al., CVPR 2019]

---

## 3. Self / Online Distillation：能不能不要老師？

### 3.1 問題

**傳統 KD 的架構**：老師是**大的、而且固定不動的（large and fixed）**。

> 【口頭】**「我們最後只需要一個小的學生模型。但為了訓練它，我們得先訓練一個大模型 —— 這顯然是額外的開銷。能不能把這個開銷去掉？」**

**投影片 p28 的討論題**：

> **Discussion: What is the disadvantage of fixed large teachers? Does it have to be the case that we need a fixed large teacher in KD?**

**兩條解法：**

| 方向 | 做法 |
|---|---|
| **Self-Distillation** | **完全不要老師** —— 自己當自己的老師 |
| **Online Distillation** | **讓老師也一起進步** —— 老師和學生同時從零開始訓練 |

---

### 3.2 Self-Distillation：Born-Again Neural Networks

**核心**：**只有一個架構，但分成好幾個「世代」迭代訓練。**

```
Step 0:  T   ← 從零初始化，只用【真實標籤】訓練
              （不用訓練到完全收斂）
           │
Step 1:  S₁  ← 新的模型（同架構、不同隨機種子）
              監督 = 【真實標籤】+ 【T 的輸出】
           │
Step 2:  S₂  ← 監督 = 【真實標籤】+ 【S₁ 的輸出】
           │
  ...      ⋮   可以重複很多代
           │
Step k:  Sₖ
```

> 【口頭】教授的比喻：**「祖父監督父親，父親監督小孩」** —— 可以一路傳下去。

**兩個結果：**

| 觀察 | 說明 |
|---|---|
| **準確率單調上升** | $T < S_1 < S_2 < \cdots < S_k$ |
| **可以再 ensemble** | 因為**架構完全相同**，可以把不同世代的模型**直接合併**（甚至直接把權重相加）。**推論時零額外開銷。** |

> 【口頭】**「你也可以做加權平均，因為後面世代的品質可能比前面的高。」**

論文：Born Again Neural Networks [Furlanello et al., ICML 2018]

---

### 3.3 Online Distillation：Deep Mutual Learning

**核心**：**老師不必比學生大 —— 兩個模型同時從零開始，互相學習。**

> 【口頭】教授的比喻：**「這比較像是你的同學、或坐你旁邊的同桌 —— 互相學習，同時也向最終的標籤學習。」**

**兩邊的損失完全對稱：**

$$\mathcal{L}(S) = \underbrace{\text{CrossEntropy}(S(I), y)}_{\text{向標籤學}} + \underbrace{\text{KL}(S(I)\ \Vert\ T(I))}_{\text{向對方學}}$$

$$\mathcal{L}(T) = \text{CrossEntropy}(T(I), y) + \text{KL}(T(I)\ \Vert\ S(I))$$

> 【口頭】「不只可以在最終輸出上做，**中間的特徵圖也可以匹配**（同 §2）。」

#### 【Q&A】為什麼這樣會比「直接訓練一個稍大的模型」好？

**學生問**：這個直覺是什麼？為什麼比用同樣總算力去訓練一個略大的單一模型好？

**教授答**：

> **「因為我們並不想要一個大模型 —— 我們要的就是一個小模型。**我們只是想加一些額外的監督，讓它訓練得比『從零單獨訓練這個小模型』更好。
>
> 而且這兩個模型雖然都是隨機初始化，**其中一個可能拿到比較好的初始化、另一個比較差 —— 這種隨機性是有影響的。**加上 KL 這一項，就能拿到額外的監督、有效容量也大一點，**還能避免被一個糟糕的初始化卡死，因為現在你有兩個模型、兩組初始化。**」

#### ⭐ 實驗結果（投影片 p33，CIFAR-10 / CIFAR-100 Top-1）

| Net 1 | Net 2 | 獨立訓練 (1 / 2) | **DML (1 / 2)** | 提升 (1 / 2) |
|---|---|---|---|---|
| **CIFAR-10** | | | | |
| ResNet-32 | ResNet-32 | 92.47 / 92.47 | **92.68 / 92.80** | +0.21 / +0.33 |
| WRN-28-10 | ResNet-32 | 95.01 / 92.47 | **95.75 / 93.18** | +0.74 / +0.71 |
| MobileNet | ResNet-32 | 93.59 / 92.47 | **94.24 / 93.32** | +0.65 / +0.85 |
| MobileNet | MobileNet | 93.59 / 93.59 | **94.10 / 94.30** | +0.51 / +0.71 |
| WRN-28-10 | WRN-28-10 | 95.01 / 95.01 | **95.66 / 95.63** | +0.65 / +0.62 |
| **CIFAR-100** | | | | |
| ResNet-32 | ResNet-32 | 68.99 / 68.99 | **71.19 / 70.75** | **+2.20 / +1.76** |
| MobileNet | ResNet-32 | 73.65 / 68.99 | **76.13 / 71.10** | **+2.48 / +2.11** |
| MobileNet | MobileNet | 73.65 / 73.65 | **76.21 / 76.10** | **+2.56 / +2.45** |
| WRN-28-10 | MobileNet | 78.69 / 73.65 | **80.28 / 77.39** | +1.59 / **+3.74** |

#### 三個要讀出來的重點

1. **所有差值都是正的** —— 不論兩個網路是**同架構**還是**不同架構**，DML 都比獨立訓練好。
2. **大模型也變好了**（Net 1 也有提升）—— 不是只有小模型受益。**「Deep mutual learning can improve both student (net 2) and teacher (net 1) models.」**
3. **CIFAR-100 的提升遠大於 CIFAR-10**（+2~3% vs +0.2~0.8%）—— 任務越難、額外監督越有價值。

#### ⚠️ 教授自己提出的保留

> 【口頭】**「我希望作者能提供更多在 ImageNet 上的消融實驗，這樣結論會更紮實。」**（目前只有 CIFAR 級別的資料集。）

論文：Deep Mutual Learning [Zhang et al., CVPR 2018]

---

### 3.4 兩者結合：Be Your Own Teacher

**做法**：**深度監督（deep supervision）+ 蒸餾**。

```
Input ─► Block1 ─┬─► Block2 ─┬─► Block3 ─┬─► Block4 ─► Classifier 4/4（最終）
                 │           │           │                    │
                 ▼           ▼           ▼                    │
            Classifier   Classifier  Classifier               │
               1/4          2/4         3/4                   │
                 ▲           ▲           ▲                    │
                 └───────────┴───────────┴────────────────────┘
                        用【更深層的預測】去監督【更淺層的預測】
```

**直覺**：

> **「後面 stage 的標籤更可靠，所以拿它們來監督前面 stage 的預測。」**

**每個 classifier 都受三種監督**：真實標籤 + 深層的蒸餾 + 深層的特徵。

#### ⭐ Early Exit 的取捨

| 只跑到 | 準確率 | 加速 |
|---|---|---|
| **Classifier 1/4** | **最低** | **最好**（後面的 block 都不用跑） |
| Classifier 4/4 | 最高 | 沒有加速 |

【口頭】教授把它連到 **GoogLeNet 的 early exit**：跑完第一個 block 就 softmax 出一個預測。

#### 實驗結果（投影片 p35，CIFAR-100）

| Network | **Baseline** | Cls 1/4 | Cls 2/4 | Cls 3/4 | Cls 4/4 | **Ensemble** |
|---|---|---|---|---|---|---|
| VGG19(BN) | 64.47 | 63.59 | **67.04** | 68.03 | 67.73 | **68.54** |
| ResNet-18 | 77.09 | 67.85 | 74.57 | **78.23** | 78.64 | **79.67** |
| ResNet-50 | 77.68 | 68.23 | 74.21 | 75.23 | **80.56** | **81.04** |
| ResNet-101 | 77.98 | 69.45 | 77.29 | **81.17** | 81.23 | **82.03** |
| ResNet-152 | 79.21 | 68.84 | 78.72 | **81.43** | 81.61 | **82.29** |
| WideResNet44-8 | 79.93 | 72.54 | **81.15** | 81.96 | 82.09 | **82.61** |
| PyramidNet101-240 | 81.12 | 69.23 | 78.15 | 80.98 | 82.30 | **83.51** |

#### 這張表最有意思的地方

> 【口頭】**「對 VGG19 只要跑到 classifier 2/4 就能追上、甚至超過 baseline 的準確率；ResNet 大概到 3/4 就可以。所以你根本不必跑完整個網路。」**

- **VGG19**：Cls 2/4 = **67.04** > baseline **64.47** ✅（只跑一半就贏）
- **ResNet-101**：Cls 3/4 = **81.17** > baseline **77.98** ✅（跑 3/4 就贏 3 個百分點）
- **Ensemble 一律最高** —— 所有模型都超過 baseline

#### ⚠️ 代價

> 【口頭】**「當然，用後段特徵圖去監督前段、用後段預測去監督前段，這確實增加了一些複雜度。」**

論文：Be Your Own Teacher [Zhang et al., ICCV 2019]

---

## 4. KD 用在不同任務

> 【口頭】**「不管是什麼架構、什麼任務，關鍵就是『找出要匹配什麼』。」**

### 4.1 物件偵測（Object Detection）

**比分類多了兩個麻煩：**

| 問題 | 說明 |
|---|---|
| **① 類別不平衡** | 大部分區域都是**背景** —— 「可能有一堆河流、樹木，但只有很少的貓和狗」 |
| **② Bounding box 是回歸問題** | 位置可以是任意實數，**不是分類問題**，沒辦法直接算兩個機率分佈的距離 |

#### 解法 1：加權的 Cross Entropy（處理不平衡）

**用不同的權重 $w_c$ 對待前景與背景類別** —— 前景更重要、背景較不重要。

#### 解法 2：⭐ 老師只當「上界」（Bounded Regression Loss）

> **「把老師的預測當成學生要達到的『上界』。一旦學生的品質超過老師某個 margin，這個 loss 就變成零。」**

【口頭】**「我們允許學生超越老師，但只允許超過某個幅度，之後就停止 —— 因為如果老師已經比學生弱了，再向它學就沒意義了。」**

#### 解法 3：⭐ 把回歸問題轉成分類問題（Localization Distillation）

```
Bounding box = 4 個實數 (x₁, y₁, x₂, y₂)
        │
        ▼
把 y 軸切成 6 個 bin、x 軸切成 6 個 bin
        │
        ▼
「目標落在哪個 bin」= 一個 1-of-6 的分類問題
        │
        ▼
現在可以用 §2.1 的方法算兩個機率分佈之間的蒸餾損失了 ✅
```

**特徵層**一樣可以用 1×1 conv 對齊形狀後匹配。

論文：Learning Efficient Object Detection Models with KD [Chen et al., NeurIPS 2017]；Localization Distillation for Dense Object Detection [Zheng et al., CVPR 2022]

---

### 4.2 語義分割（Semantic Segmentation）

**任務**：逐像素預測（教授提到當週 ICCV 的 best paper 提名 **SAM / Segment Anything**）。

**兩層做法：**

| 做法 | 說明 |
|---|---|
| **Feature imitation** | 跟分類、偵測一樣 —— 匹配中間特徵圖（pair-wise loss） |
| ⭐ **Adversarial loss** | **多加一個 discriminator 網路** |

#### Discriminator 在做什麼

```
Teacher 的 score map ──┐
                       ├──► Discriminator ──► 「這張分割圖是老師畫的還是學生畫的？」
Student 的 score map ──┘
```

- **學生**被訓練成要**騙過** discriminator（讓它分不出來）
- **discriminator** 被訓練成要**分得出來**
- **兩者一起進步** → 最後學生的分割圖就跟老師極度接近

> 【口頭】「就是用 GAN 的方法訓練一個判別網路，**讓學生強到可以騙過判別器。**」

論文：Structured Knowledge Distillation for Semantic Segmentation [Liu et al., CVPR 2019]

---

### 4.3 ⭐ GAN — GAN Compression

> 【口頭】**「這是我學生的工作，叫 GAN Compression。」**

#### 三項損失加起來

| 損失 | 作用 |
|---|---|
| **cGAN Loss** | 一般的 GAN 損失：generator 騙 discriminator，discriminator 分辨真假 |
| **Reconstruction Loss** | **paired cGAN**：用 ground truth 監督（例如馬→斑馬有配對資料）<br>**unpaired cGAN**：**用老師產生的圖片**去監督學生產生的圖片 |
| **Distillation Loss** | 中間特徵圖用 **1×1 conv 投影**後匹配（同 §2.2） |

#### ⭐ 跟 NAS 結合

> 【口頭】**「這也是另一個把 NAS 跟 distillation 結合的例子** —— 訓練時我們有一個 **candidate generator pool**，可以選 channel 或部分 channel，**就跟 Once-for-All 一樣是彈性的**，可以抽子網路而不是用完整網路。」

（↔ 直接連回 **Lecture 7-8 §9** 的 OFA。）

#### 實測數字（NVIDIA Jetson Nano GPU，horse2zebra）

| | **原始 CycleGAN** | **GAN Compression** |
|---|---|---|
| **MACs** | 56.8 G | **4.81 G（11.8× 少）** |
| **FPS** | 1.6 | **3.9（2.5× 快）** |
| **FID**（越低越好） | 24.2 | 26.6 |

> ⚠️ 【口頭】教授在課堂上口述的是「**56 GOps → 3 GOps，約 16 倍**」，跟投影片標示的 **56.8G → 4.81G（11.8×）** 略有出入 —— **以投影片數字為準**。

#### 互動式修圖 demo（edges2shoes）

【口頭】教授現場播放：在 Jetson Nano 上畫鞋子的邊緣、即時生成鞋子的圖片。

> 「壓縮前 **1.6 FPS**，我們試了很多次想把它變成一隻 Nike 鞋、換個 logo，**但實在太慢了**。
> 壓縮後幀率提升到接近 **4 FPS**，就可以比較順地擦掉、加新東西 —— **在手機上做互動式修圖。**」

論文：GAN Compression [Li et al., CVPR 2020]

---

### 4.4 NLP — MobileBERT 的 Attention Transfer

**Transformer 的結構回顧**（教授順帶點出的一個細節）：

```
Multi-Head Attention (QKV projection + output projection)
        │
    Normalization  ← 投影片畫的是 post-norm
        │              【口頭】「但近期的大型語言模型常用 pre-norm，
        │                       也就是在 MHA 之前先做 normalization」
    Feed-Forward Network (FFN，本質就是 FC 層)
```

**匹配什麼**：**attention map**（＋ feature map）。

**投影片 p52 的視覺證據**：

| | Attention Map 長相 |
|---|---|
| Teacher | （基準） |
| **Student，沒有 attention transfer** | 跟老師**差很多** ❌ |
| **Student，有 attention transfer** | 跟老師**明顯接近** ✅ |

論文：MobileBERT [Sun et al., ACL 2020]

---

### 4.5 【投影片新增】LLM / VLM — Minitron

> ⚠️ **這兩張投影片（p53–54）是後續學期更新的內容，2023 秋季的課堂逐字稿沒有這一段。**

**現況**：

> **「Pruning and distillation becomes common practice to obtain small LLMs.」**
> （剪枝＋蒸餾已經成為得到小型 LLM 的標準做法。）

**Minitron 的五步流程**（Muralidharan et al., NeurIPS 2024）：

```
1. Trained LLM          ← 拿一個訓練好的大模型
       ↓
2. Estimate importance  ← 估計 embedding / head / channel / layer 的重要性
       ↓
3. Rank                 ← 排序
       ↓
4. Trim                 ← 剪掉（structured pruning）
       ↓
5. Distillation         ← ⭐ 剪完之後用【蒸餾】重新訓練
       ↑                    （不是從頭 retrain，而是拿原模型當老師）
       └──── Iterative ────┘
```

**投影片點出的一個重要轉變**：

> **在 LLM 上，傳統的重要性指標（例如 weight magnitude）已經失效。**
> 近期的 LLM 結構化剪枝改用：**gradient / Taylor、cosine similarity、在校準資料集上的 perplexity**。
>
> 【投影片註記】但因為 LLM 太大，**計算梯度資訊在記憶體和算力上都極為昂貴** —— Minitron 的主要目標之一就是**避開這個昂貴步驟**。

（↔ 這正好對照 **Lecture 3 §4.2**：magnitude-based pruning 在 CNN 時代是業界標準，**到了 LLM 就不管用了**。）

**應用實例**：Llama 3.2（Meta, 2024）也用了 pruning + distillation。

---

## 5. Network Augmentation：反過來對付 underfitting

> 【口頭】**「我們都聽過 data augmentation，那什麼是 network augmentation？」**

### 5.1 ⭐ 核心洞見：問題方向反了

$$\boxed{\textbf{Data Augmentation 對付 overfitting；Network Augmentation 對付 underfitting}}$$

| | 大模型 | **小模型** |
|---|---|---|
| **問題** | **Overfitting**（資料不夠） | **Underfitting**（**容量不夠**） |
| **該用** | Data augmentation、Dropout | ？ |

### 5.2 傳統做法在小模型上「不只無效，而且有害」

#### 傳統做法回顧

| 方法 | 做什麼 |
|---|---|
| **Data Augmentation** | Cutout、Mixup、AutoAugment（顏色/旋轉/裁切的自動組合） |
| **Dropout** | 訓練時隨機移除神經元；**推論時全部裝回來，跑完整模型** |
| **Spatial Dropout / DropBlock** | 更粗顆粒 —— 移除整個 channel 或整個區塊 |

#### ⭐ 實驗結果（投影片 p63–65）

**大模型 ResNet-50（4.1G MACs），ImageNet Top-1：**

| | Baseline | Mixup | AutoAugment | DropBlock |
|---|---|---|---|---|
| Top-1 | ~76 | ↑ | ↑ | **> 78** ✅ |

**小模型 MobileNetV2-Tiny（23.5M MACs），ImageNet Top-1：**

| | Baseline | Mixup | AutoAugment | DropBlock | **NetAug** |
|---|---|---|---|---|---|
| Top-1 | **> 52** | ↓ | ↓ | **< 50** ❌ | **最高** ✅ |

> 【口頭】**「這些技術用在只有 2350 萬 MACs 的 MobileNetV2-Tiny 上，準確率反而更低了 —— 從 52% 掉到 50% 以下。」**
>
> **「傳統的 data augmentation 和 dropout 這些方法，對小型神經網路的訓練不但沒用，甚至有害 —— 因為它們容量根本不夠。」**

**原因**：這些技術全都是**在對付 overfitting**（讓模型更難學）。但小模型的問題是 **underfitting** —— **它已經學不動了，你還讓它更難學。**

### 5.3 NetAug 的做法：反過來「擴增網路」

```
【訓練時】
                        ┌── 增廣模型 A（多一些 channel）
  Input ──► Tiny Model ─┼── 增廣模型 B（多更多 channel）
            (紅色權重)   └── 增廣模型 C
                              ↑ 黑色的是【只屬於增廣模型】的額外權重

  · 紅色權重是【共享】的 —— 目標小模型與所有增廣模型完全共用
  · loss = 小模型自己的 loss + 各增廣模型的 loss（加權）
  · 增廣模型的梯度會【流回】共享的紅色權重
  · 每一步隨機取樣不同大小的增廣模型

【推論時】
  只用中間那個 Tiny Model —— 【零額外開銷】
```

**損失函數**（投影片 p67）：

$$\mathcal{L}_{\text{aug}} = \underbrace{\mathcal{L}(W_t)}_{\text{base：小模型本身}} + \underbrace{\alpha_1 \mathcal{L}([W_t, W_1]) + \cdots + \alpha_i \mathcal{L}([W_t, W_i]) + \cdots}_{\text{aug：各個增廣模型}}$$

#### ⭐ 跟 Dropout 恰好相反

> 【投影片 p67】**「Contrary to dropout methods that encourage subsets of the neural network to produce predictions, NetAug encourages the tiny neural network to work as a sub-model of a set of larger models.」**

| | Dropout | **NetAug** |
|---|---|---|
| 方向 | 讓網路的**子集**能做預測 | 讓小網路成為一組**更大模型的子模型** |
| 對付 | overfitting | **underfitting** |
| 訓練時 | 隨機**移除**神經元 | 隨機**增加**神經元 |

#### 跟 Once-for-All 的關係

> 【口頭】**「就跟 Once-for-All 網路一樣 —— 它們共享權重。」**

差別在目的：
- **OFA**：訓練 supernet，**目標是抽出子網路來部署**（子網路是產品）
- **NetAug**：訓練增廣模型，**目標只是幫小模型訓練得更好**（增廣模型是鷹架，用完就丟）

**訓練開銷**：投影片註明 **+16.7%**；**推論開銷：零**。

### 5.4 結果

#### (A) 訓練曲線：只幫得到小模型

| 模型 | 訓練準確率 | **驗證準確率** |
|---|---|---|
| **MobileNetV2-Tiny** | **+1.6%** | **+1.3%** ✅ |
| **ResNet-50** | **+1.6%** | **−0.3%** ❌ |

> 【口頭】**「NetAug 只提升了大模型的『訓練』準確率，卻沒有提升它的『驗證』準確率 —— 因為它防的是 underfitting，而大模型本身容量就夠，根本沒這個問題。」**

**這是一個很乾淨的因果驗證**：如果 NetAug 是靠某種泛化正則化在起作用，大模型的驗證準確率也該上升。它沒有 → 說明機制確實是「補容量」。

#### (B) ⭐ NetAug 與 KD 正交（投影片 p69）

| 模型 | Baseline | KD | **NetAug** | **NetAug + KD** |
|---|---|---|---|---|
| MobileNetV2 w0.35 r160 | 基準 | ↑ | ↑ | **最高** ✅ |
| MobileNetV3 w0.35 r160 | 基準 | ↑ | ↑ | **最高** ✅ |
| ProxylessNAS w0.35 r160 | 基準 | ↑ | ↑ | **最高** ✅ |

> **「NetAug is orthogonal to KD.」**（兩者可以疊加）

#### (C) 遷移學習：NetAug 贏得更明顯（投影片 p70）

**相對 baseline 的提升（%）：**

| 方法 | ImageNet | Food101 | Flowers102 | Cars | Pets | Pascal VOC |
|---|---|---|---|---|---|---|
| **KD** | ~持平 | −1.1 ❌ | −2.2 ❌ | — | — | — |
| **4× 訓練輪數** | ~持平 | −0.2 ❌ | 0.9 | 0.4 | — | — |
| **NetAug** | ~持平 | **1.5** ✅ | **2.9** ✅ | **1.6** ✅ | **1.2** ✅ | **2.0** ✅ |

> **「NetAug 提供比 KD 和 4× 訓練排程更好的遷移學習表現 —— 儘管它們在 ImageNet 上的表現差不多。」**

#### (D) 【口頭】「那我多訓練幾輪不就好了？」

> **「因為擔心 underfitting，那我們就訓練久一點如何？**
> **實際上我們發現訓練更久沒有幫助，有時候甚至有負面效果。」**
>
> （對照上表：4× 訓練輪數在 Food101 上是 **−0.2%**。）

#### (E) 遷移到物件偵測（投影片 p71）

**YOLOv3 + MobileNetV2 w0.35**，同樣的 AP50 之下：

| 資料集 | MACs 節省 |
|---|---|
| **Pascal VOC** | **−41% MACs** |
| **COCO** | **−38% MACs** |

論文：Network Augmentation for Tiny Deep Learning [Cai et al., ICLR 2022]

---

## 6. 一頁速查表

### 6.1 KD 的核心公式

| 項目 | 公式 |
|---|---|
| **帶溫度的 Softmax** | $p(z_i, T) = \dfrac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$ |
| **Cross entropy 蒸餾損失** | $\mathbb{E}(-p_t \log p_s)$ |
| **L2 蒸餾損失** | $\mathbb{E}(\Vert p_t - p_s\Vert_2^2)$ |
| **Attention map** | $\dfrac{\partial L}{\partial x}$（跨 channel reduction：sum / sum² / sum⁴ / max） |
| **Sparsity indicator** | $\rho(x) = \mathbb{1}[x > 0]$ |
| **RKD 關係向量** | $\psi(s_1,\ldots,s_n) = (\Vert s_1-s_2\Vert_2^2,\ldots,\Vert s_{n-1}-s_n\Vert_2^2)$，長度 $n(n-1)/2$ |
| **Deep Mutual Learning** | $\mathcal{L}(S) = \text{CE}(S(I), y) + \text{KL}(S(I) \Vert T(I))$ |
| **NetAug** | $\mathcal{L}_{\text{aug}} = \mathcal{L}(W_t) + \sum_i \alpha_i \mathcal{L}([W_t, W_i])$ |

### 6.2 六種 What to Match

| 匹配對象 | 需要投影器？ | 關鍵直覺 |
|---|---|---|
| **Output logits** | ❌ | 最簡單、最常用 |
| **Intermediate weights** | ✅ **必須**（維度不同） | 用大模型權重的線性組合去近似 |
| **Intermediate features** | ✅ | 特徵分佈也要像，不只輸出分佈 |
| **Gradients / Attention** | ✅ | 好模型「看的地方」一致 |
| **Sparsity patterns** | — | ReLU 後誰該是零 |
| **Relational info** | — | **比關係，不比絕對位置**（更寬鬆、更合理） |

### 6.3 有沒有老師

| 方法 | 老師 | 特點 |
|---|---|---|
| **傳統 KD** | 大、**固定** | 要先付出訓練大模型的成本 |
| **Self-Distillation（BAN）** | **沒有** —— 上一代的自己 | 同架構可直接 ensemble，**零推論開銷** |
| **Online Distillation（DML）** | **同儕**，同時從零訓練 | **雙方都變好**；避免壞初始化 |
| **Be Your Own Teacher** | **深層監督淺層** | 支援 early exit；VGG19 跑一半就贏 baseline |

### 6.4 不同任務的關鍵技巧

| 任務 | 特有的問題 | 解法 |
|---|---|---|
| **Detection** | 前景/背景不平衡 | **加權 cross entropy** |
| **Detection** | bbox 是回歸問題 | **切 6×6 bin，轉成分類問題** |
| **Detection** | 老師可能比學生弱 | **老師當上界，超過 margin 後 loss 歸零** |
| **Segmentation** | — | **加 discriminator 做 adversarial loss** |
| **GAN** | — | cGAN loss + reconstruction loss + distillation loss；**結合 OFA 抽 generator 子網路** |
| **NLP** | — | **匹配 attention map** |
| **LLM** | magnitude 指標失效 | **Taylor / cosine / perplexity 選重要性；剪完用蒸餾 retrain** |

### 6.5 該記住的數字

| 數字 | 意義 |
|---|---|
| **19.5 TFLOPS / 80 GB vs MFLOPs / 256 kB** | Cloud AI vs Tiny AI 的落差 |
| **ResNet-50 > 76% vs MobileNetV2-Tiny < 48%** | ImageNet Top-1，小模型的 underfitting |
| **5, 1 → 0.982, 0.017**（老師） | 貓狗範例，T=1 |
| **3, 2 → 0.731, 0.269**（學生） | 學生「比較沒自信」 |
| **T=10 → 0.599, 0.401** | 溫度把分佈平滑掉 |
| **512 → 256×512 投影器** | 對齊老師與學生的 channel |
| **62% vs 73% / 77%** | Network-in-Network vs ResNet-34/101 —— 注意力圖差異的分界 |
| **DML CIFAR-100：+2.2 ~ +3.7%** | 任務越難，互學越有價值 |
| **VGG19 Cls 2/4 = 67.04 > baseline 64.47** | Be Your Own Teacher：跑一半就贏 |
| **56.8G → 4.81G（11.8×）；1.6 → 3.9 FPS（2.5×）；FID 24.2 → 26.6** | GAN Compression on Jetson Nano |
| **NetAug：MbV2-Tiny +1.3% val，ResNet-50 −0.3% val** | 只幫得到小模型 —— 機制驗證 |
| **NetAug 訓練 +16.7%，推論 +0%** | 開銷 |
| **YOLOv3+MbV2：−41% / −38% MACs** | Pascal VOC / COCO |

---

## 7. 與其他課程／作業的連結

| 本講觀念 | 連到哪裡 |
|---|---|
| **§0.1 KD 可以疊在 pruning / quantization / NAS 之上** | **Lecture 3-8 全部** —— KD 是唯一「不改模型、只改訓練」的技術 |
| **§0.3 小模型 underfit** | **Lecture 7-8 NAS §0** —— 「能不能一開始就設計小而精的模型」的另一面 |
| **§2.2 投影器（1×1 conv / FC 對齊維度）** | **Lecture 2 §2.3** —— 1×1 conv 就是純 channel 混合 |
| **§2.5 Sparsity pattern（ReLU 後 60–70% 是零）** | **Lecture 4 §9.1 EIE** —— activation sparsity 的來源 |
| **§4.1 Detection 的類別不平衡加權** | **Lecture 3 §4** —— 同樣是「重要性加權」的思路 |
| **§4.3 GAN Compression 用 OFA 抽 generator 子網路** | **Lecture 7-8 §9 Once-for-All** |
| **§4.5 Minitron：magnitude 在 LLM 失效** | **Lecture 3 §4.2** —— magnitude-based pruning 是 CNN 時代的業界標準 |
| **§4.5 剪枝後用蒸餾 retrain** | **Lecture 4 §7** —— 原本是 fine-tune，現在換成蒸餾 |
| **§5.3 NetAug 與 OFA 的權重共享** | **Lecture 7-8 §9.3** —— 同樣是共享權重 + 隨機取樣子模型 |
| **§5.4 NetAug 與 KD 正交** | 兩者可疊加，訓練小模型的標準組合 |
| **下一講** | **Lecture 10：MCUNet** —— TinyML 的演算法-系統協同設計 |

---

## 附：這一講的一句話總結

> **前面三章都在問「模型能多小」；這一講問的是「小模型怎麼才學得會」。
> 答案是同一件事的兩面：訓練時給它額外的監督（老師、同儕、上一代的自己，或一組更大的自己），
> 推論時全部拆掉 —— 因為小模型的病不是 overfitting，是容量不足。**

---

*筆記依據 MIT 6.5940 Fall 2023 Lecture 9 逐字稿（Zoom 錄影 1:00:11）與官方投影片 `Lec09-Knowledge-Distillation.pdf` 整理。標記【投影片新增】處為後續學期更新的內容，2023 課堂逐字稿未涵蓋。*
