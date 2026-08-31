# EfficientML.ai 第五、六講完整筆記：量化（Quantization）

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2023
> **講者**：Song Han
> **涵蓋**：
> - **Lecture 5** — Quantization (Part I)，1:15:25
> - **Lecture 6** — Quantization (Part II)，1:10:32
>
> **來源**：Zoom 錄影逐字稿 ＋ 官方投影片 `Lec05-Quantization-I.pdf`（70 頁）、`Lec06-Quantization-II.pdf`（82 頁）
>
> **標記說明**
> - `【口頭】` = 教授課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 學生提問與回答

---

## 目錄

- [總覽：三種量化方法](#總覽三種量化方法)
- [Lecture 5 — Quantization (Part I)](#lecture-5--quantization-part-i)
  - [1. 動機：為什麼要量化](#1-動機為什麼要量化)
  - [2. 整數表示法](#2-整數表示法)
  - [3. 浮點數表示法](#3-浮點數表示法)
  - [4. 什麼是量化](#4-什麼是量化)
  - [5. K-Means-based Quantization](#5-k-means-based-quantization)
  - [6. Linear Quantization](#6-linear-quantization)
- [Lecture 6 — Quantization (Part II)](#lecture-6--quantization-part-ii)
  - [7. Quantization Granularity（量化顆粒度）](#7-quantization-granularity量化顆粒度)
  - [8. Dynamic Range Clipping（動態範圍裁切）](#8-dynamic-range-clipping動態範圍裁切)
  - [9. Rounding（捨入）](#9-rounding捨入)
  - [10. QAT：Quantization-Aware Training](#10-qatquantization-aware-training)
  - [11. Binary and Ternary Quantization](#11-binary-and-ternary-quantization)
  - [12. Mixed-Precision Quantization（HAQ）](#12-mixed-precision-quantizationhaq)
- [13. 一頁速查表](#13-一頁速查表)
- [14. 與其他課程／作業的連結](#14-與其他課程作業的連結)

---

## 總覽：三種量化方法

> 【口頭】開場的一句話定位（承接 Pruning）：
> **「Pruning 減少的是『權重的數量』；Quantization 減少的是『每個權重的位元數』。」**

$$\text{總儲存量} = \underbrace{\#\text{weights}}_{\text{Pruning 處理這個}} \times \underbrace{\text{bits per weight}}_{\text{Quantization 處理這個}}$$

**整整兩講，就是在講三種方法。先把這張表記住：**

| | **K-Means-based** | **Linear** | **Binary / Ternary** |
|---|---|---|---|
| **Storage（儲存）** | **整數權重 + 浮點 codebook** | **整數權重** | **Binary / Ternary 權重** |
| **Computation（運算）** | **浮點運算** ❌ | **整數運算** ✅ | **位元運算（XNOR + popcount）** ✅✅ |
| **省什麼** | **只省儲存與記憶體存取** | **儲存 + 運算都省** | 兩者都極省 |
| **適合什麼** | **memory-bound**（LLM 即時生成）；**Lab 4/5 用這個** | 通用推論；**Lab 2 用這個** | 極端場景，工業界少用 |

**Lecture 5** 講「怎麼表示數字」+ 前兩種方法；**Lecture 6** 講「怎麼做好」（顆粒度、裁切、捨入、QAT）+ 第三種方法。

---
---

# Lecture 5 — Quantization (Part I)

## 1. 動機：為什麼要量化

### 1.1 三個省

| 省什麼 | 怎麼省 |
|---|---|
| **儲存（Storage）** | $\#\text{params} \times \text{bit width}$ —— 位元少了，模型就小 |
| **記憶體存取（Memory reference）** | 搬的資料少了 |
| **運算（Arithmetic）** | 位元少了，加法乘法都便宜 |

### 1.2 ⭐ 位元數與能耗：加法是線性，乘法是平方

**45nm 0.9V 製程的能耗表**（Horowitz, ISSCC 2014）：

| 操作 | 能耗 (pJ) | |
|---|---|---|
| **8-bit int ADD** | **0.03** | ↕ **30×** |
| **32-bit int ADD** | **0.1** | |
| 16-bit float ADD | 0.4 | |
| 32-bit float ADD | 0.9 | |
| **8-bit int MULT** | **0.2** | ↕ **16×** |
| **32-bit int MULT** | **3.1** | |
| 16-bit float MULT | 1.1 | |
| **32-bit float MULT** | **3.7** | |

#### 教授的課堂推導

> 【口頭】「**加法**：8-bit 到 32-bit，能耗差多少？—— **能耗跟位元數是線性關係**（0.03 → 0.1，約 3×，投影片標 30×）。
> **乘法**：8-bit 到 32-bit 是 $O(n)$ 還是 $O(n^2)$？—— **是平方！** 位元數是 4 倍，但**能耗差了將近 16 倍**（0.2 → 3.1）。
> 因為做乘法時，你要做的工作量是**二次方**的。」

$$\boxed{\text{ADD 能耗} \propto n \qquad \text{MULT 能耗} \propto n^2}$$

> 【口頭】「16-bit 到 32-bit 浮點也大約是 4 倍，**但不完全是 4 倍**，因為浮點運算的細節不一樣。」

**這條 $n^2$ 規律在 Lecture 6 §11 會再回來** —— 它解釋了為什麼「越量化越有 diminishing return」。

---

## 2. 整數表示法

### 2.1 Unsigned Integer

$n$ 位元表示 $0$ 到 $2^n - 1$：

$$\text{value} = \sum_{i=0}^{n-1} b_i \cdot 2^i$$

投影片例：`00110001` → $2^5 + 2^4 + 2^0 = 49$

### 2.2 Signed Integer — Sign-Magnitude（原碼）

**第一個 bit 當符號位**，其餘照 unsigned 算。

投影片例：`10110001` → 符號=1（負），$-（2^5+2^4+2^0）= -49$

#### ⚠️ 缺點：浪費一個 slot

> 【口頭】「這裡我們**浪費了一個 slot**，因為 `00000000` 和 `10000000`（1 後面全 0）**都代表零**。」

### 2.3 Signed Integer — Two's Complement（二補數）

**第一個 bit 不再只是符號 —— 它代表 $-2^{n-1}$：**

$$\text{value} = -b_{n-1} \cdot 2^{n-1} + \sum_{i=0}^{n-2} b_i \cdot 2^i$$

投影片例：`10110001` → $-2^7 + 2^6+2^5+2^4+2^3+2^2+2^1+2^0$ …… 逐位加總

**兩個關鍵值：**

| 表示 | 值 |
|---|---|
| `00000000` | **0**（只有一種表示法 ✅） |
| `10000000` | **$-2^{n-1}$**（最小值） |

**這就是為什麼 $n$-bit 的範圍是 $[-2^{n-1},\ 2^{n-1}-1]$** —— 例如 INT4 是 $[-8, 7]$。

### 2.4 Fixed-Point Number（定點數）

不是整數，例如 `3.0625`。**用小數點切開：**

- 小數點**左邊**：$2^0, 2^1, 2^2 \ldots$
- 小數點**右邊**：$2^{-1}, 2^{-2} \ldots$

**兩種等價算法**（投影片）：

$$\underbrace{(-2^7 + 2^6+2^5+2^4+2^3+2^2+2^1+2^0)}_{= 49} \times 2^{-4} = 49 \times 0.0625 = \mathbf{3.0625}$$

> 【口頭】「你可以**用原本的整數表示法算完，再乘以 $2^{-4}$ 位移**（因為我們把小數點移了 4 位），兩種方法算出來的值一樣。」

---

## 3. 浮點數表示法

> **這一節是本講最重要的部分**，也是 FP8 / FP4 / BF16 這些名詞的底層原理。

### 3.1 IEEE 754 FP32 的三個部分

```
┌─┬────────────┬──────────────────────────┐
│S│  Exponent  │  Fraction (significand)  │
│1│    8 bit   │          23 bit          │   = 32 bits
└─┴────────────┴──────────────────────────┘
```

**Normal numbers（Exponent ≠ 0）的公式：**

$$(-1)^{\text{sign}} \times (1 + \text{Fraction}) \times 2^{\text{Exponent} - 127}$$

#### 兩個要問的問題

**Q1：為什麼是 $1 + \text{Fraction}$？**

> 【口頭】「因為我們**免費送你一個 1** —— 反正一定會用到這個表示法。但這裡有個 catch，就是等下要講的 subnormal number。」

**Q2：Bias 為什麼是 127？**

$$\text{Exponent Bias} = 2^{8-1} - 1 = 127$$

> 【口頭】「8 bits 能表示 0 到 255，**我們用中間點當 bias**。127 是 **7 個 bit 能表示的最大數**。
> **最好記的方法：不管你有幾個 exponent bit，就是「少一位元能表示的最大數」。**
> 這樣你才能同時有**正的**和**負的**實際指數。」

#### 完整範例：把 0.265625 編碼成 FP32

$$0.265625 = (1 + 0.0625) \times 2^{-2}$$

| 欄位 | 值 |
|---|---|
| **Sign** | `0`（正數） |
| **Exponent** | $-2 + 127 = 125$ |
| **Fraction** | $0.0625$ |

【口頭】Fraction 的每一位：$0.5,\ 0.25,\ 0.125,\ 0.0625 \ldots$ —— **每一位是前一位的一半。**

### 3.2 ⭐ Subnormal Numbers（次正規數）：怎麼表示 0

**問題**：用 $(1 + \text{Fraction}) \times 2^{\ldots}$，**永遠不可能是 0**（因為有那個免費的 1）。

**解法**：**當 Exponent 全為 0 時，改用另一條公式：**

$$(-1)^{\text{sign}} \times \underbrace{\text{Fraction}}_{\text{沒有 }+1} \times 2^{1 - 127}$$

> 【口頭】兩個要注意的地方：
> 1. **不再是 $1 + \text{Fraction}$，就只有 $\text{Fraction}$** → 所以 Fraction 全 0 時 = **真正的 0** ✅
> 2. 指數不是 $0 - 127$ 而是**強制成 $1 - 127$**，**這樣數值才連續**（不會在 normal 和 subnormal 交界處斷掉）

#### 🔑 Subnormal 區域是「線性」的

> 【口頭】「這個區間有什麼特別？**Exponent 被固定成 0，你只剩一個自由度 —— fraction。而 fraction 是線性的，不是指數的。**
> **所以每個 centroid 之間的距離是固定的 —— 非常像整數表示法。**」

#### 兩個極值（教授逐步推導）

**最小的正 subnormal 值**：exponent 全 0，fraction 最低位設 1

$$2^{-23} \times 2^{1-127} = 2^{-23} \times 2^{-126} = \mathbf{2^{-149}}$$

> 【口頭】「**take home：在電腦系統裡你沒辦法表示無限小的數。** 這就是 FP32 能表示的最小數。**所以做除法時要非常小心數值穩定性。**」

**最大的 subnormal 值**：exponent 全 0，fraction 全 1

$$(2^{-1} + 2^{-2} + \cdots + 2^{-23}) \times 2^{1-127} = (1 - 2^{-23}) \times 2^{-126}$$

### 3.3 特殊值：Infinity 與 NaN

**當 Exponent 全為 1（`FFH` = 255）時：**

| Fraction | 代表 |
|---|---|
| **全 0** | $\pm\infty$ |
| **非全 0** | **NaN（Not a Number）** |

> 【口頭】「你在 Python 或 MATLAB 裡看到的 NaN —— 我們都很討厭它 —— 就是這樣表示的。
> **可以看到我們浪費了非常多 slot**：只要 fraction 非零，通通都是 NaN。
> **等下講 FP8 的時候我們負擔不起這種浪費，所以會為這個 case 定義新規則。**」

### 3.4 FP32 完整規則表

| Exponent | Fraction = 0 | Fraction ≠ 0 | 公式 |
|---|---|---|---|
| **`00H` = 0** | $\pm 0$ | **subnormal** | $(-1)^s \times \text{Fraction} \times 2^{1-127}$ |
| **`01H`…`FEH` = 1…254** | **normal** | **normal** | $(-1)^s \times (1 + \text{Fraction}) \times 2^{\text{Exp}-127}$ |
| **`FFH` = 255** | $\pm\text{INF}$ | **NaN** | — |

**數線分佈**（投影片）：

```
±0    2^-149  ......  (1-2^-23)·2^-126 │ 2^-126  ............  (1+1-2^-23)×2^127
      └────── subnormal（等距） ───────┘└──── normal（間距越來越大）────┘
```

> 【口頭】「**subnormal 區間每個 centroid 的距離是固定的**，因為那裡是線性的。
> **normal 區間的距離越來越大**，因為有 $2^{\text{Exponent}}$ —— **而這正是我們要的，擴大 dynamic range。**」

### 3.5 ⭐ 核心口訣：Exponent → Range；Fraction → Precision

$$\boxed{\textbf{Exponent 寬度 → Dynamic Range}\qquad \textbf{Fraction 寬度 → Precision}}$$

| 格式 | Exponent | Fraction | Total |
|---|---|---|---|
| **IEEE FP32**（Single Precision） | **8** | **23** | 32 |
| **IEEE FP16**（Half Precision） | **5** | **10** | 16 |
| **Google BF16**（Brain Float） | **8** ⭐ | **7** | 16 |

#### 為什麼 BF16 這麼重要

> 【口頭】「大約五、六年前，人們開始想能不能用更少位元訓練神經網路。
> FP16 只有 5 個 exponent bit，**dynamic range 比 FP32 小非常多**。
> **Google 想出一個更聰明的表示法** —— 用 **跟 FP32 一樣的 8 個 exponent bit**（保住 dynamic range），但**只用 16 bit 的總成本**（省一半儲存），代價是 fraction 只剩 7 bit。
>
> **實務上用 BF16 訓練大型語言模型，通常比 FP16 更容易收斂**，能避免訓練時那些奇怪的 spike。**現在 BF16 已經被廣泛使用。**」

#### 🔑 為什麼訓練需要大的 dynamic range

> 【口頭】「訓練深度神經網路時，**特別是最開始的幾次迭代，神經網路是高度湍流（turbulent）的**，算梯度時數值可能變得相當大。**這就是為什麼大 dynamic range 對訓練很有幫助。**」

> 【口頭】**「如果這堂課你只帶走一件事，那就是：exponent bits 決定 dynamic range，而 dynamic range 對訓練至關重要。」**

### 3.6 課堂測驗（教授帶全班算）

#### 測驗 1：這個 FP16 是多少？

$$\texttt{1100011100000000} \qquad \text{Exponent Bias} = 15$$

```
┌─┬───────┬────────────┐
│1│ 10001 │ 1100000000 │
└─┴───────┴────────────┘
 S  5-bit    10-bit
```

| 步驟 | 計算 |
|---|---|
| **Bias** | 5 個 exponent bit → bias = $2^{5-1}-1 = \mathbf{15}$ |
| **Exponent** | `10001` = $17$ → $17 - 15 = \mathbf{2}$ |
| **Fraction** | `11000...` = $0.5 + 0.25 = \mathbf{0.75}$ |
| **Sign** | `1` → 負數 |

$$(-1)^1 \times (1 + 0.75) \times 2^2 = -1.75 \times 4 = \mathbf{-7}$$

#### 測驗 2：把 2.5 編碼成 BF16

| 步驟 | 計算 |
|---|---|
| **Bias** | 8 個 exponent bit → bias = $2^{8-1}-1 = \mathbf{127}$ |
| **拆解** | $2.5 = 1.25 \times 2^1$ |
| **Exponent binary** | $1 + 127 = 128 = \texttt{10000000}$ |
| **Fraction binary** | $0.25 = \texttt{0100000}$ |
| **Sign** | $0$ |

$$\texttt{0 | 10000000 | 0100000}$$

### 3.7 FP8：NVIDIA H100 的兩種格式

> 【口頭】「最近有個新的數字格式出來，**進一步降低精度、減少記憶體佔用、讓訓練更快更便宜** —— 那就是 FP8。」

**8 bits = 1 個 sign + 7 個要分配的 bit。** 怎麼分？NVIDIA 在 H100 選了兩種：

| 格式 | Exponent | Fraction | 用在哪 | 特殊值處理 |
|---|---|---|---|---|
| **E4M3** | 4 | 3 | **推論 / 訓練的 forward** | **沒有 INF**；`S.1111.111` 保留給 **NaN** |
| **E5M2** | 5 | 2 | **訓練的 backward（算梯度）** | **有 INF**（`S.11111.00`）；NaN 是 `S.11111.XX` |

#### 兩者能表示的最大值

| 格式 | 最大 normal 值 | 為什麼 |
|---|---|---|
| **E4M3** | `S.1111.110` = **448** | 因為 `111` 保留給 NaN，所以 `110` 才是最大 |
| **E5M2** | `S.11110.11` = **57,344** | 有更多 exponent bit → **dynamic range 大得多** |

#### 🔑 為什麼 forward 用 E4M3、backward 用 E5M2

> 【口頭】**「訓練時我們要更大的 dynamic range，推論時我們要更高的 precision。這裡有取捨，沒有白吃的午餐 —— 你要嘛得到更大的 dynamic range 但更低的精度，要嘛得到更小的 dynamic range 但更高的精度。」**
>
> **E4M3 dynamic range 較小、精度較高 → 用在 forward（推論）**
> **E5M2 dynamic range 較大 → 用在 backward 算梯度**

【口頭】教授的旁白：「H100 比 A100 快了大概 **2 到 3 倍**，全世界都在追那些 GPU。矽谷最大的傳聞就是哪家公司哪家新創募了多少錢、然後買了多少張 H100 —— **現在 GPU 甚至變成公司募資的抵押品**，這實在有點瘋狂。」

### 3.8 INT4 和 FP4

> 【口頭】「對更少位元的追求從未停止 —— **我們在 Lab 4 部署 LLM 到筆電時就會用 4-bit。**」

#### INT4

$$[-8, 7]，\text{共 16 個值，等距}$$

```
-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7
└──────────── 間距全部相同 ────────────┘
```

#### FP4 的四種切法（投影片完整列出）

| 格式 | 能表示的正值 | 特性 |
|---|---|---|
| **FP4 (E1M2)** | 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5 | **完全線性** —— 就是 INT4 乘 0.5。**沒有意義** |
| **FP4 (E2M1)** | 0, 0.5, 1, 1.5, 2, 3, 4, 6 | ✅ **右邊間距變大** → 有 dynamic range。**沒有 INF、沒有 NaN** |
| **FP4 (E3M0)** | 0, 0.25, 0.5, 1, 2, 4, 8, 16 | **純對數表示法**（沒有 mantissa）。間距成倍數增長 |

#### 【口頭】為什麼不用 E1M2

> 「E1M2 的 centroid **也是線性的**，間距都是 0.5 —— **基本上跟 INT4 一模一樣，只是縮放了 0.5**。而且你還**浪費了兩個 slot 在 0 上面**（正負零），**這樣就表示不了 -8**。
> **所以通常我們不用 E1M2，那基本上就是 INT4。我們要用的是 E2M1。**」

**E2M1 的兩個範例計算**：

| 位元 | 計算 |
|---|---|
| `0 00 1` | subnormal（exp=0）：$0.5 \times 2^{1-1} = \mathbf{0.5}$ |
| `0 11 1` | normal：$(1+0.5) \times 2^{3-1} = \mathbf{6}$ |

（注意 subnormal 那項的指數是**強制的 $1 - \text{bias}$**，不是 $0 - \text{bias}$，這跟 §3.2 一致。）

---

## 4. 什麼是量化

### 4.1 定義

> **Quantization is the process of constraining an input from a continuous or otherwise large set of values to a discrete set.**
> （量化就是把輸入從連續或很大的值域，限制到一個離散集合。）

**Quantization Error（量化誤差）** = 原值與量化後的值之差。

### 4.2 兩個直覺例子

| 例子 | 說明 |
|---|---|
| **連續訊號 → 離散訊號** | 原本可以是任意值，現在只能從 4 或 5 個選擇裡挑 |
| **原始影像 → 16 色影像** | 每個像素只能從 16 個顏色裡挑 |

> 【口頭】「這叫 **Palettization（調色盤化）**。**Apple 的 Neural Engine 就有這個技術。**」

### 4.3 三種方法的定位（本講的主軸）

| | **K-Means-based** | **Linear** | **Binary/Ternary** |
|---|---|---|---|
| **Storage** | 整數權重 + **浮點 codebook** | 整數權重 | Binary/Ternary 權重 |
| **Computation** | **浮點運算** | **整數運算** | 位元運算 |

**Naive baseline**：權重和 activation 都用 FP32 / FP16，儲存和運算都是浮點。

---

## 5. K-Means-based Quantization

### 5.1 做法（投影片的 4×4 例子，從頭到尾）

**原始權重（32-bit float）：**

```
 2.09  -0.98   1.48   0.09
 0.05  -0.14  -1.08   2.12
-0.91   1.92   0     -1.03
 1.87   0      1.53   1.49
```

**步驟 1：K-Means 分群**

相近的數值分成一群，用同一個顏色標記。例如 `2.09`、`2.12`、`1.92` 分在一起。

**步驟 2：建 Codebook（4 個 centroid）**

| Index | Centroid |
|---|---|
| 3 | **2.00** |
| 2 | **1.50** |
| 1 | **0.00** |
| 0 | **-1.00** |

**步驟 3：只存 index（2-bit int）**

```
3  0  2  1
1  1  0  3
0  3  1  0
3  1  2  2
```

**步驟 4：重建權重**

```
 2.00  -1.00   1.50   0.00
 0.00   0.00  -1.00   2.00
-1.00   2.00   0.00  -1.00
 2.00   0.00   1.50   1.50
```

**量化誤差**（原值 − 重建值）：

```
 0.09   0.02  -0.02   0.09
 0.05  -0.14  -0.08   0.12
 0.09  -0.08   0     -0.03
-0.13   0      0.03  -0.01
```

> 【口頭】「量化誤差其實**相當小**。」

### 5.2 ⭐ 省了多少儲存

| | 計算 | 大小 |
|---|---|---|
| **原始** | 32 bit × 16 | **512 bit = 64 B** |
| **Indexes** | 2 bit × 16 | 32 bit = 4 B |
| **Codebook** | 32 bit × 4 | 128 bit = 16 B |
| **合計** | 4 B + 16 B | **20 B** |

$$\frac{64}{20} = \mathbf{3.2\times\ \text{smaller}}$$

#### 通式：矩陣夠大時 codebook 可以忽略

假設 $N$-bit 量化、參數量 $M \gg 2^N$：

| | 大小 |
|---|---|
| 原始 | $32M$ bit |
| Indexes | $NM$ bit |
| Codebook | $2^{N+5}$ bit |

$$\text{當 } M \gg 2^N \text{，codebook 可忽略} \quad \Longrightarrow \quad \boxed{\frac{32}{N}\times \text{ smaller}}$$

> 【口頭】「這只是個 4×4 的玩具例子，codebook 佔比很大。但**當矩陣遠大於 codebook 時，我們就可以忽略 codebook 那部分**，直接看 32 比 $N$。」

### 5.3 Fine-tuning（微調量化後的權重）

**做法**（投影片逐步）：

```
1. 照常反向傳播，得到每個權重位置的 gradient
2. 用「跟權重完全相同的分群圖樣」把 gradient 分組
   （同一個顏色 = 同一個 centroid）
3. 每一組的 gradient 加總（或取平均）→ reduce
4. centroid ← centroid − lr × (該組的 reduced gradient)
```

**投影片的實際數字**：centroid `2.00` 微調後變成 **`1.96`**，`1.50` 變成 **`1.48`**，`-1.00` 變成 **`-0.97`**。

#### 【Q&A】微調後要重算 centroid 嗎？

**學生問**：加完 gradient 之後會重新計算 centroid 嗎？

**教授答**：

> 「**是的，這就是在重算 centroid。**你可以看到分佈有些微妙的變化 —— **有些 centroid 往鄰居的方向移動了一點點**，那就是被 gradient 改變 centroid 造成的。」

### 5.4 權重分佈的變化（三個階段）

【口頭】教授把 Lecture 3 的 pruning 串起來講：

| 階段 | 分佈 |
|---|---|
| **Pruning 後 + 微調** | **雙峰分佈**（中間接近 0 的被剪掉） |
| **量化前** | 仍然是**連續**的 —— 可以是任意值 |
| **量化後** | **離散** —— 只有 8 個或 16 個選擇。有些 centroid 底下權重多，有些少 |
| **微調後** | centroid 位置**微幅移動** |

### 5.5 需要幾個 bit？

【口頭】教授 2016 年的實驗結果：

| 層 | 需要幾 bit |
|---|---|
| **Convolution 層** | **4 bit 大致就夠** |
| **Fully-Connected 層** | **更耐操 —— 連 2 bit 都不太掉準確率** |

> 【口頭】「這是 **2016 年**的結果，但直到**去年 Qualcomm 才推出 Snapdragon 8 Gen 2 支援 4-bit 權重量化** —— **七年後，4-bit 仍然廣泛適用。**」

**低於 4 bit（2-3 bit）**：conv 層的準確率就會開始掉了。

### 5.6 Huffman Coding（可選的額外壓縮）

**想法**：不同權重用**不同數量**的 bit。

| 權重 | 用幾 bit |
|---|---|
| **不常出現的** | **更多** bit |
| **常出現的** | **更少** bit |

**適用場景**：

> 【口頭】「如果你想做一個內建 AI 的手機 App，要上架到 App Store，**不希望使用者下載太久**，這是進一步榨出最後幾個百分比儲存空間的好方法。」

**⚠️ 但實務上不好用**：

> 【口頭】「**執行時解碼是有成本的，所以實作上並不容易。**」

### 5.7 Deep Compression：把 Pruning + Quantization + Huffman 串起來

> 【口頭】「這是我 2016 年寫的論文，**拿了 ICLR 的 best paper**，現在**差不多就是業界標準**了。」

**三階段管線**（投影片）：

```
原始網路
   │
   ├─ 【Pruning：減少權重數量】
   │    Train Connectivity → Prune Connections → Train Weights（迭代）
   │    → 9x–13x 壓縮，準確率不變
   │
   ├─ 【Quantization：減少每個權重的位元數】
   │    Cluster the Weights → Generate Code Book
   │    → Quantize with Code Book → Retrain Code Book
   │    → 27x–31x 壓縮，準確率不變
   │
   └─ 【Huffman Encoding：無損編碼】
        Encode Weights + Encode Index
        → 35x–49x 壓縮，準確率不變
```

**實測結果**（投影片完整表）：

| Network | 原始大小 | 壓縮後 | **壓縮率** | 原始準確率 | 壓縮後準確率 |
|---|---|---|---|---|---|
| **LeNet-300** | 1070 KB | 27 KB | **40×** | 98.36% | **98.42%** |
| **LeNet-5** | 1720 KB | 44 KB | **39×** | 99.20% | **99.26%** |
| **AlexNet** | 240 MB | 6.9 MB | **35×** | 80.27% | **80.30%** |
| **VGGNet** | 550 MB | 11.3 MB | **49×** | 88.68% | **89.09%** |
| **GoogleNet** | 28 MB | 2.8 MB | 10× | 88.90% | 88.92% |

> ⭐ **注意：壓縮後的準確率全部持平或略高。**

【口頭】「**Huffman coding 因為實作複雜度，後來沒有被廣泛使用；但 pruning 和 quantization 非常廣泛。**」

### 5.8 已經很小的模型還能壓嗎？—— SqueezeNet

**學生的自然疑問**：越有效率的網路，壓縮率越低（因為本來就小）。**那能不能一開始就設計一個緊湊的模型？**

> 【口頭】「**這就是第三章 Neural Architecture Search 要回答的問題。**」

**SqueezeNet**（教授與 UC Berkeley 合作）：

**基本 building block**：
```
輸入
 │
 ├─ 1×1 conv（squeeze，降 channel 數）
 │
 ├─┬─ 1×1 conv ─┐  （expand，兩路平行）
 │ └─ 3×3 conv ─┤
 │              │
 └─ concatenate ┘
```

【口頭】「用**較小的 kernel 跟 3×3 kernel 平行**，並用**較少的 channel 數**，之後再擴張回原本的 channel 數，然後 concatenate。」

**結果**（投影片）：

| Network | 方法 | Size | **Ratio** | Top-1 | Top-5 |
|---|---|---|---|---|---|
| AlexNet | — | 240 MB | 1× | 57.2% | 80.3% |
| AlexNet | SVD | 48 MB | 5× | 56.0% | 79.4% |
| AlexNet | **Deep Compression** | 6.9 MB | **35×** | 57.2% | 80.3% |
| **SqueezeNet** | — | **4.8 MB** | **50×** | 57.5% | 80.3% |
| **SqueezeNet** | **Deep Compression** | **0.47 MB** | **510×** | **57.5%** | **80.3%** |

> ⭐ **重點：SqueezeNet 本身就已經比 AlexNet 小 50×，但套上 Deep Compression 之後還能再壓 10×，總共 510×，而且準確率完全不掉。**
> 【口頭】**「這說明即使是非常非常緊湊的模型，仍然有進一步壓縮的空間。」**

### 5.9 ⭐ 執行時發生什麼事：只省儲存，不省運算

**執行流程**（投影片）：

```
     儲存                          運算
┌──────────────┐
│ 2-bit index  │──decode──┐
│  (uint)      │          │
│ codebook     │          ▼
│  (float)     │      float weights ──┐
└──────────────┘                      │
                       float inputs ──┴─► Conv ─► + bias ─► ReLU ─► float outputs
                                                  (全部都是浮點！)
```

> 【投影片明確寫出】
> **「K-Means-based Weight Quantization only saves storage cost of a neural network model.
> All the computation and memory access are still floating-point.」**

#### 那為什麼還有用？

> 【口頭】**「這對 memory-bound 的工作負載超級有用** —— 例如即時的 LLM 生成。
> 一個 70 億參數的模型用 FP16，那是 **14 GB 的記憶體 —— 光是生成一個 token，你就得存取 14 GB 的記憶體。**
> **這是嚴重的 memory-bound，所以節省記憶體佔用、節省儲存非常關鍵。**」

#### 【Q&A】在 GPU 上做線上解碼會不會太慢？

**教授答**：

> 「這對加速器來說**很有效** —— 因為你**只有 16 個 entry，可以直接放進暫存器**，索引暫存器其實非常便宜。**在 GPU 上也可以做這種線上解碼。**」

> **「Lab 5 我們就用 4-bit 表示權重、解碼成 16-bit，activation 也是 16-bit，然後在 GPU 上做 16×16 bit 的運算來利用 tensor core。但權重是用 4-bit 儲存的。這樣大約比 FP16 快 3 到 4 倍。」**

---

## 6. Linear Quantization

> 【口頭】「跟 K-Means 的差別：**整數權重 + 整數運算**，而不是整數權重 + 浮點運算。」

### 6.1 核心公式

$$\boxed{r = S \cdot (q - Z)}$$

| 符號 | 意義 | 型別 |
|---|---|---|
| $r$ | 原始的實數（real） | **浮點** |
| $q$ | 量化後的值（quantized） | **整數** |
| $Z$ | **Zero point** —— 把哪個 $q$ 對應到 $r=0$ | **整數** |
| $S$ | **Scale** —— 縮放因子 | **32-bit 浮點** |

> 這是一個 **affine mapping（仿射映射）**：整數 → 實數。

### 6.2 跟 K-Means 的差別

| | K-Means | **Linear** |
|---|---|---|
| Centroid 位置 | **任意（arbitrary）** —— 靠 codebook 查 | **等距（equally spaced）** |
| 彈性 | **高** ✅ | 低 |
| 解碼 | 要查表 | **直接用線性映射算** ✅ |

### 6.3 Bit width 決定 $q_{\min}$、$q_{\max}$

| Bit Width | $q_{\min}$ | $q_{\max}$ |
|---|---|---|
| 2 | $-2$ | $1$ |
| 3 | $-4$ | $3$ |
| 4 | $-8$ | $7$ |
| **$N$** | $\mathbf{-2^{N-1}}$ | $\mathbf{2^{N-1}-1}$ |

### 6.4 ⭐ 推導 Scale 與 Zero Point（兩個未知數、兩條方程式）

**兩條端點方程式：**

$$r_{\max} = S(q_{\max} - Z) \qquad r_{\min} = S(q_{\min} - Z)$$

**相減消掉 $Z$：**

$$r_{\max} - r_{\min} = S(q_{\max} - q_{\min}) \quad \Longrightarrow \quad \boxed{S = \frac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}}$$

> 【口頭】「直覺上，**$S$ 就是「浮點範圍的長度」比上「整數範圍的長度」。**」

**再代回去求 $Z$：**

$$r_{\min} = S(q_{\min} - Z) \quad \Longrightarrow \quad \boxed{Z = \text{round}\left(q_{\min} - \frac{r_{\min}}{S}\right)}$$

（$Z$ 必須是整數，所以要 round。）

### 6.5 完整範例（投影片的同一個 4×4 矩陣，2-bit）

**原始權重：**

```
 2.09  -0.98   1.48   0.09
 0.05  -0.14  -1.08   2.12   ← r_max = 2.12
-0.91   1.92   0     -1.03
 1.87   0      1.53   1.49
```

$r_{\max} = 2.12$，$r_{\min} = -1.08$；2-bit → $q_{\min} = -2$，$q_{\max} = 1$

**算 Scale：**

$$S = \frac{2.12 - (-1.08)}{1 - (-2)} = \frac{3.20}{3} = \mathbf{1.07}$$

**算 Zero Point：**

$$Z = \text{round}\left(-2 - \frac{-1.08}{1.07}\right) = \text{round}(-2 + 1.009) = \mathbf{-1}$$

**量化後的矩陣：**

```
 1  -2   0  -1
-1  -1  -2   1
-2   1  -1  -2
 1  -1   0   0
```

**驗證重建**（教授在課堂上算的兩個）：

| 原值 $r$ | 量化值 $q$ | 重建 $S(q-Z)$ | 誤差 |
|---|---|---|---|
| **2.09** | $1$ | $(1-(-1)) \times 1.07 = \mathbf{2.14}$ | 0.05 |
| **-0.98** | $-2$ | $(-2-(-1)) \times 1.07 = \mathbf{-1.07}$ | **0.09** |

（量化方向：$q = \text{round}(r/S + Z)$。例如 $-0.98$：$\text{round}(-0.916 - 1) = -2$。）

> 【口頭】「**量化誤差非常小。**」

### 6.6 ⭐ 整數矩陣乘法的推導

考慮 $Y = WX$。把三者都用 $r = S(q-Z)$ 代入：

$$S_Y(q_Y - Z_Y) = S_W(q_W - Z_W) \cdot S_X(q_X - Z_X)$$

**移項：**

$$q_Y = \frac{S_W S_X}{S_Y}(q_W - Z_W)(q_X - Z_X) + Z_Y$$

**展開四項：**

$$\boxed{q_Y = \frac{S_W S_X}{S_Y}\left(\underbrace{q_W q_X}_{\text{①}} - \underbrace{Z_W q_X}_{\text{②}} - \underbrace{Z_X q_W}_{\text{③}} + \underbrace{Z_W Z_X}_{\text{④}}\right) + Z_Y}$$

#### 每一項的處理方式（教授逐項解說）

| 項 | 處理 |
|---|---|
| **① $q_W q_X$** | **這是重活（heavy lifting）** —— 全部是 8-bit / 4-bit 整數乘法，**累加用 32-bit 整數以防溢位** |
| **③ $Z_X q_W$** | **可以事先算好** —— 權重是已知的 |
| **④ $Z_W Z_X$** | **可以事先算好** —— activation 的 zero point 可以事先校準 |
| **② $Z_W q_X$** | ⚠️ **必須在執行時算** —— 這是唯一麻煩的一項 |

#### 🔑 關鍵簡化：權重是對稱的，所以 $Z_W = 0$

> 【口頭】「**權重的分佈通常是以 0 為中心、對稱的**，所以我們可以**強制 $Z_W = 0$** —— 也就是讓 0 直接對應到 0。」

**Symmetric quantization 的 scale：**

$$S = \frac{|r|_{\max}}{q_{\max}}$$

（不再用 $r_{\min}$、$r_{\max}$，改用絕對值的最大值。）

**$Z_W = 0$ 之後，② 和 ④ 兩項直接消失：**

$$q_Y = \frac{S_W S_X}{S_Y}\left(q_W q_X - Z_X q_W\right) + Z_Y$$

> 【口頭】「**只剩下這一項 —— 量化權重乘量化 activation，這才是重活，全部用 8-bit 或 4-bit 整數算。其他都是預先算好的。**」

#### $\frac{S_W S_X}{S_Y}$ 這個浮點數怎麼辦？

> 【口頭】「**這個 scaling factor 永遠落在 0 到 1 之間**，所以可以用「**一個 32-bit 整數 + 一個位元位移**」來近似：

$$\frac{S_W S_X}{S_Y} \approx \text{(32-bit integer)} \times 2^{-n}$$

> **$2^{-n}$ 用位移（bit shift）就能實作，非常便宜。這樣整條路徑就是純整數運算。**」

### 6.7 加上 Bias

考慮 $Y = WX + b$。

**兩個強制條件（教授的推導）：**

| 條件 | 理由 |
|---|---|
| **$Z_b = 0$** | bias 分佈也近似常態、對稱 |
| **$S_b = S_W \cdot S_X$** | **為了讓 bias 項能跟主項合併** |

**最後化簡成：**

$$q_Y = \frac{S_W S_X}{S_Y}\left(q_W q_X + q_{\text{bias}}\right) + Z_Y, \qquad q_{\text{bias}} = q_b - Z_X q_W$$

> 【口頭】「$q_b$ 是常數，$Z_X$ 已校準，$q_W$ 也事先知道，**所以 $q_{\text{bias}}$ 整個可以預先算好**。」

⚠️ **注意位寬**：

> 【口頭】「**$q_b$ 和 $q_{\text{bias}}$ 是 32-bit** —— 但幸運的是，這對應到的是整數加法，而矩陣乘法的累加**本來就用 32-bit 防溢位**，所以剛好匹配。」

### 6.8 Convolution 層

> 【口頭】「**推導完全一樣，只是把矩陣乘法換成卷積。**」

$$q_Y = \frac{S_W S_X}{S_Y}\left(\text{Conv}(q_W, q_X) + q_{\text{bias}}\right) + Z_Y$$

**完整的整數推論流程：**

```
量化權重 (int8/int4)  ──┐
                        ├─► 整數卷積 ──► 32-bit 累加 ──┐
量化輸入 (int8)      ──┘                              │
                                                      ├─► + q_bias (int32)
                                                      ├─► × scale（整數 + 位移）
                                                      ├─► + Z_Y
                                                      └─► 量化輸出 (int8)
```

> 【口頭】**「準確率維持得很好，但延遲大幅降低。這就是你們在 Lab 2 要實作的 —— 就是這些公式，直接翻譯成程式碼，應該不會太難。」**

---
---

# Lecture 6 — Quantization (Part II)

> 【口頭】課務公告：Lab 1 因為實驗室網站從 `hanlab18.mit.edu` 搬到 `hanlab.mit.edu`，導致作業裡一個 `.pth` 檔的連結壞掉。TA 已修好並**多給兩天期限**。

**這一講的四個主題：**

| # | 主題 | 屬於哪一類 |
|---|---|---|
| 1 | **Quantization Granularity**（顆粒度） | **PTQ**（Post-Training Quantization） |
| 2 | **Dynamic Range Clipping**（裁切） | **PTQ** |
| 3 | **Rounding**（捨入） | **PTQ** |
| 4 | **QAT**（Quantization-Aware Training） | 需要微調 |

**PTQ vs QAT 的定位**：

> 【口頭】「**PTQ 是在量化後直接恢復準確率、不做任何微調，計算上非常有效率。**
> 但有時候 PTQ 沒辦法完全恢復準確率，**這時候就需要 QAT —— 透過微調模型把準確率救回來。**」

---

## 7. Quantization Granularity（量化顆粒度）

> 【口頭】「就像 pruning 一樣 —— 還記得 pruning 有一整條顆粒度的光譜嗎？**量化也有。**」

**四個層級（由粗到細）：**

```
Per-Tensor  →  Per-Channel  →  Per-Vector (Group)  →  Shared Micro-exponent (MX)
   最粗                                                        最細
最硬體友善                                                   最準確
```

> 【口頭】「你可以想像**哪個更硬體友善、哪個更可能有高準確率**。
> **下面的（細顆粒）更準確**，因為你可以為每個 sub-vector 用更精細的 scaling factor。
> **上面的（粗顆粒）比較不準，但更硬體友善**（更規則），**也更省儲存** —— 因為細顆粒每個 sub-vector 都要一個 scale 和 bias，粗顆粒只要一個。」

### 7.1 Per-Tensor Quantization

**整個 tensor 共用一組 $S$、$Z$**，$r_{\max}$ 取全 tensor 絕對值的最大。

#### ⚠️ 問題：各 channel 的範圍差很多

> 【口頭】「以 CNN 為例，這裡有 32 個 channel —— **你觀察到什麼？各 channel 的範圍差非常多。**
> 有些非常小、有些非常大，有些擠在一個很窄的區間裡。
> **用單一個 scaling factor、單一個 max 值去涵蓋所有 channel，實在很不公平。**」

**後果**：

> 【投影片】用單一 scale **會導致準確率大幅下降，尤其是小模型**。
> 原因：**權重範圍差異大 + outlier weights**。

**解法** → Per-Channel Quantization。

### 7.2 Per-Channel Quantization

**每個 channel（row）有自己的 $S$、$Z$。**

#### 完整對照範例（投影片的同一個 4×4 矩陣，2-bit 對稱量化）

**Per-Tensor**：$r_{\max} = 2.12$（全矩陣最大），$S = 2.12 / 1 = 2.12$

**Per-Channel**：每一行各自算 $r_{\max}$ 和 $S$

| Row | $r_{\max}$（該行絕對值最大） | $S$ |
|---|---|---|
| Row 0：`2.09 -0.98 1.48 0.09` | **2.09** | 2.09 |
| Row 1：`0.05 -0.14 -1.08 2.12` | **2.12** | 2.12 |
| Row 2：`-0.91 1.92 0 -1.03` | **1.92** | 1.92 |
| Row 3：`1.87 0 1.53 1.49` | **1.87** | 1.87 |

**重建誤差**：Per-Channel **明顯小於** Per-Tensor。

#### 🔑 教授特別點出的「那一個不同的值」

> 【口頭】「你們有些人可能注意到，**兩個量化結果裡只有一個值不同**。是什麼造成這個微妙差異？」

以 `-1.03` 為例：

| 方法 | 計算 | 結果 |
|---|---|---|
| **Per-Tensor** | $\lvert -1.03 \rvert / 2.12 = 0.486 < 0.5$ | **round 成 0** ❌ |
| **Per-Channel** | $\lvert -1.03 \rvert / 1.92 = 0.536 > 0.5$ | **round 成 -1** ✅ |

> 【口頭】「**顯然把它當成 -1，比直接無視、當成 0 要好得多。**」

#### ⚠️ 代價：要存更多 scaling factor

**學生問**：這是免費的午餐嗎？你儲存了更多資訊吧？

**教授答**：

> 「**對，你得存更多資訊。**現在你要存 **4 個 scaling factor，每個都是 32-bit —— 這不便宜。**
> 這只是個簡單例子，我只有 4 個 channel。**在大型語言模型裡，channel 動輒 10,000 個 —— 那就要存 10,000 個 32-bit 浮點數。**」

### 7.3 Group Quantization：LLM 逼出來的解法

> 【口頭】「Per-channel 量化**對視覺模型效果超好** —— 這是我們去年講的內容，如果只有這樣我們就可以下課了。
> 但**大型語言模型在去年秋天大爆發**，我們實驗後發現：**如果你想壓到更低位元（4-bit），per-channel 還不夠。你需要更細的顆粒度 —— 用更小的 group size。**」

**做法**：在 channel 內部再切成更小的 sub-vector（**常見大小：32、64、128**），**每個 sub-vector 有自己的 scale**。

#### 多層次縮放（Multi-level scaling）的通式

$$\boxed{r = (q - z) \cdot s^{(l_0)} \cdot s^{(l_1)} \cdots}$$

| 符號 | 意義 |
|---|---|
| $r$ | 實數 |
| $q$ | 量化值 |
| $z$ | zero point（$z=0$ 就是對稱量化） |
| $s^{(l_0)}, s^{(l_1)}$ | **不同層級**的 scale factor |

> 【口頭】設計上的權衡：
> **「粗顆粒的 scale 被整個 tensor 共享 → overhead 很低，所以可以用高精度（FP32 或 FP16）。
> 細顆粒的 scale（每 16 個元素一個）→ 你不會想用超高精度，所以通常用低精度。
> 這永遠是準確率與效率之間的平衡 —— 顆粒度越細，準確率越高，但冗餘越多、overhead 越大。」**

### 7.4 VS-Quant：Per-Vector Scaled Quantization

$$r = S(q - Z) \quad\longrightarrow\quad r = \gamma \cdot S_q (q - Z)$$

| 符號 | 型別 | 顆粒度 |
|---|---|---|
| $\gamma$ | **浮點**、粗顆粒 scale | per-tensor / per-channel |
| $S_q$ | **整數**、細顆粒 scale | per-vector |

> 【投影片】設計理念：
> **「在細顆粒度用比較便宜的整數 scale，在粗顆粒度用比較貴的浮點 scale」** —— 在準確率與硬體效率之間取得平衡。

#### ⭐ 有效位寬（Effective Bit Width）的算法

**教授的課堂練習**：4-bit 量化，每 16 個元素共用一個 **4-bit** 的 per-vector scale，有效位寬是多少？

$$4 + \frac{4}{16} = \mathbf{4.25\ \text{bits}}$$

> 【口頭】「每個元素 4 bit，每 16 個元素共用 4 bit 的 scaling factor —— **所以每個元素等於多攤到 1/4 bit**，總共 4.25 bit。」

### 7.5 MX Data Type（Microsoft 的 Shared Micro-exponent）

> 【口頭】「這是**今年課程全新的內容**。稍微更複雜一點，一樣有兩層 scaling factor。」

**兩層結構：**

| 層級 | Group Size | 資料型別 |
|---|---|---|
| **L0（細）** | **每 2 個值** | **E1M0** —— **1 bit 的 exponent，被 2 個值共用** |
| **L1（粗）** | **每 16 個值** | **E8M0** —— **8 bit 的 exponent，擴大 dynamic range** |

#### ⭐ 完整對照表（投影片）

| 方法 | Data Type | L0 Group Size | L0 Scale 型別 | L1 Group Size | L1 Scale 型別 | **有效位寬** |
|---|---|---|---|---|---|---|
| **Per-Channel Quant** | INT4 | Per Channel | FP16 | — | — | **4** |
| **VSQ** | INT4 | 16 | UINT4 | Per Channel | FP16 | **4 + 4/16 = 4.25** |
| **MX4** | S1M2 | 2 | E1M0 | 16 | E8M0 | **3 + 1/2 + 8/16 = 4** |
| **MX6** | S1M4 | 2 | E1M0 | 16 | E8M0 | **5 + 1/2 + 8/16 = 6** |
| **MX9** | S1M7 | 2 | E1M0 | 16 | E8M0 | **8 + 1/2 + 8/16 = 9** |

#### 「MX9」這個名字怎麼來的（教授逐項拆）

> 【口頭】「MX9 你有 **1 個 sign bit + 7 個 mantissa bit = 8**；
> 然後有 **1 個 exponent 被 2 個值共用 → 等效 1/2**；
> 然後有 **8 個 bit 被 16 個值共用 → 等效 8/16 = 1/2**。
> 加起來 $8 + 0.5 + 0.5 = \mathbf{9}$ —— **這就是為什麼叫 MX9。**」

#### 跟 FP8 / FP16 的根本差異

> 【口頭】「上一講講的 FP16、FP8（H100 的 Transformer Engine）—— **每個數字都有自己的 exponent。**
> **這裡的差別是：2 個元素共用 exponent，然後 16 個元素再共用另一個高精度 exponent。**」

#### 【Q&A】這些格式的取捨是什麼？

**學生問**：MX4 / MX6 / MX9 之間的取捨，是不是「你的權重裡有多少 scale 變化」對上「同一個 scale 下有多少精度變化」？

**教授答**：

> 「**這是個更好的描述。**你可以看 exponent bit —— **8 bit 和 1 bit 都一樣，所以 dynamic range 幾乎相同。差別在 precision，也就是 mantissa 的位元數。**」

#### 【口頭】這是一個開放的研究題目

> 「**這在大型語言模型裡正變得越來越流行。怎麼共享、不共享什麼、階層要怎麼設計、每一層用什麼精度 —— 這是一個相當大的設計空間。
> 如果你想找 open project，這會是個好題目：設計你自己的資料型別，在準確率與硬體效率之間找到好的取捨，特別是針對現代大模型的工作負載。**」

---

## 8. Dynamic Range Clipping（動態範圍裁切）

### 8.1 為什麼只有 activation 需要

| | 何時知道範圍 |
|---|---|
| **權重（Weight）** | ✅ **靜態** —— 不管輸入是什麼，權重都一樣 |
| **Activation** | ❌ **動態** —— **不同的圖片可能有完全不同的資料** |

> **所以問題是：怎麼決定 activation 的範圍？**

### 8.2 Type 1：訓練時 —— Exponential Moving Average（EMA）

$$\hat{r}^{(t)}_{\min,\max} = \alpha \cdot r^{(t)}_{\min,\max} + (1 - \alpha) \cdot \hat{r}^{(t-1)}_{\min,\max}$$

> 【口頭】「訓練時我們追蹤 **exponential moving average** —— 例如每個 epoch 檢查這個 batch 的 $r_{\min}$、$r_{\max}$，用這個組合去更新範圍。
> **這樣觀察到的範圍會在數千個訓練步驟間被平滑掉。**」

### 8.3 Type 2：部署前 —— Calibration（校準）

**做法**：在訓練好的 FP32 模型上，跑幾個 **calibration batch**。

⚠️ **教授強調的重點**：

> 【口頭】**「這個 calibration 資料集要跟訓練集和測試集分開。」**
> （↔ 這跟 Lecture 4 §6.1 那個「不要用 test set 做 sensitivity analysis」是同一條原則。）

**簡單做法**：取各 batch 的 min/max 的**平均值**。

**⚠️ 為什麼不直接取 max**：

> 【投影片】**「spending dynamic range on the outliers hurts the representation ability.」**
> （把 dynamic range 花在極端值上，會傷害表示能力。）

### 8.4 ⭐ 方法一：假設分佈，解析求解（Laplace / Gaussian）

**目標**：最小化原輸入 $X$ 與量化重建 $Q(X)$ 之間的均方誤差：

$$\min_{|r|_{\max}} \mathbb{E}\left[(X - Q(X))^2\right]$$

**假設 activation 服從 Laplace$(0, b)$ 分佈**，最佳裁切值可以數值求解：

| Bit width | 最佳 $\lvert r \rvert_{\max}$ |
|---|---|
| **2 bit** | $2.83\,b$ |
| **3 bit** | $3.89\,b$ |
| **4 bit** | $5.03\,b$ |

（$b$ 可以從 calibration 的輸入分佈估出來。）

#### 直覺（教授的解釋）

> 【口頭】「**如果你裁切太多**，那些大的 activation 會有很大的量化誤差。
> **如果你完全不裁切**，centroid 會散得很開 —— **你在浪費 centroid**，因為那些位置其實幾乎沒有數值。」

### 8.5 ⭐ 方法二：最小化 KL Divergence（TensorRT 用的方法）

> 【口頭】「這是一個**超級廣泛使用**的方法 —— **NVIDIA TensorRT 工具箱就是用這個。**」

**適用**：分佈**不是**高斯或 Laplace 的時候。

$$D_{KL}(P \Vert Q) = \sum_{i}^{N} P(x_i) \log \frac{P(x_i)}{Q(x_i)}$$

| 分佈 | 意義 |
|---|---|
| $P$ | **裁切前**的分佈 |
| $Q$ | **裁切+編碼後**的分佈 |

> 【投影片】**「KL divergence measures the amount of information lost when approximating a given encoding.」**
> （KL 散度衡量的是「用某個編碼去近似時，損失了多少資訊」。）

#### 裁切之後分佈長什麼樣（教授看圖解說）

> 【口頭】「以 ResNet-152 的某一層為例，KL divergence 決定裁在這個點。
> **所有在裁切線右邊的點，全部被 round 到裁切線那個值** —— 把它們加起來，數量相當可觀，**所以那裡會出現一個明顯的尖峰。**」

【口頭】不同網路、不同層裁切的位置**都不一樣**（GoogleNet、ResNet-152 各層都不同）。

### 8.6 ⭐ 方法三：最小化 MSE（Newton-Raphson 法）

**MSE 對裁切值的曲線是 U 形的**（教授逐步解說為什麼）：

```
MSE
 │╲                              ╱
 │ ╲                           ╱
 │  ╲                        ╱
 │   ╲______________________╱
 │            最佳點
 └────────────────────────────────► 裁切值（大 → 小）
   裁太鬆                        裁太緊
```

| 區段 | 為什麼 MSE 高 |
|---|---|
| **裁切值太大（不裁）** | **浪費大量 centroid** 在幾乎沒有數值的區域 |
| **裁切值太小（裁太兇）** | **大量數值被壓到裁切邊界**，造成很大的誤差 |

> 【口頭】「**所以存在一個最佳點，而且可以解析地求解。**用這個方法，ResNet 和 BERT **即使激進地量化到只有 4 bit，也能有不錯的準確率 —— 相當令人印象深刻。**」

---

## 9. Rounding（捨入）

### 9.1 Round-to-Nearest 不是最佳解

> 【投影片】**「Rounding-to-nearest is not optimal.」**

**為什麼**：權重之間**不是獨立的**。單獨看每個權重「就近捨入」是局部最佳，但**整層的輸出重建誤差**未必最小。

### 9.2 AdaRound

> 【投影片】**「What is optimal? Rounding that reconstructs the original activation the best, which may be very different from round-to-nearest.」**
> （最佳的捨入，是「最能重建原始 activation」的那一種 —— 它可能跟就近捨入差很多。）

**做法**：把「往上捨入還是往下捨入」變成一個**可學習的決策**，目標是**最小化該層輸出的重建誤差**。

（↔ 這跟 Lecture 3 §4.7 的 regression-based pruning 是同一個思路：**只看單層的重建誤差，不做端到端反傳。**）

### 9.3 相關方法

投影片另外列出 **Weight Equalization** 與 **Bias Correction**（Data-Free Quantization, ICCV 2019）—— 屬於 PTQ 的其他技巧。

---

## 10. QAT：Quantization-Aware Training

> 【口頭】「通常你直接量化模型，準確率會掉。**特別是當我們激進地量化到 4-bit 這種低精度時，微調是非常必要的。**」

### 10.1 ⭐ 核心機制：保留一份 FP32 副本

> 【投影片三句話總結】
> 1. **A full precision copy of the weights is maintained throughout the training.**（訓練全程保留一份全精度權重）
> 2. **The small gradients are accumulated without loss of precision.**（小梯度得以無損累積）
> 3. **Once the model is trained, only the quantized weights are used for inference.**（訓練完後推論只用量化權重）

### 10.2 🔑 為什麼一定要留 FP32 副本（教授講了三次的重點）

**教授的具體數字例子**：

> 假設某個權重量化後是 **2.0**，每次迭代的 gradient × lr = **0.1**。

| | 沒有 FP32 副本 | **有 FP32 副本** |
|---|---|---|
| 第 1 次迭代 | $2.0 + 0.1 = 2.1$ → **round 回 2** ❌ | FP32 累積成 **2.1** |
| 第 2 次 | 又從 2 開始 → round 回 2 ❌ | **2.2** |
| 第 3 次 | 2 ❌ | **2.3** |
| 第 5 次 | **永遠是 2** ❌ | **2.5 → round 成 3** ✅ |

> 【口頭】**「所以保留浮點權重來累積這些微小變化很重要 —— 它們最終會導致量化結果真的改變。」**
> **「這就像每天學一點點，過一段時間之後就會有相當大的改變。」**

#### 【Q&A】那是在累積誤差嗎？

**學生問**：這樣誤差會不會累積？

**教授答**：

> 「**我不會說是誤差在累積，是「改變」在累積。**是**正確性**在累積。」

### 10.3 計算圖：Simulated / Fake Quantization

```
        ┌── FP32 權重 W（保留在背後，用來累積梯度）
        │
        ▼
    Q(W) = S_W · q_W        ← 權重量化節點
        │
        ▼
Q(X) ─► Layer N ─► Y ─► Q(Y) ─► Layer N+1
        (Conv → BatchNorm → ReLU，全部仍然跑全精度)
                  ▲
                  └── activation 量化節點
```

#### 為什麼叫「Fake / Simulated」Quantization

> 【口頭】「因為**背後仍然保留浮點權重**，而且**所有 data path 仍然是 FP16 或 FP32** —— 那個值雖然是 1，**但它仍然是用浮點數表示的 1**。
> 這樣做反向傳播時，你才能捕捉到那些微小的變化。
> 而 activation 量化節點的作用，是**確保權重和 activation 的數值落在量化的邊界上**，才傳給下一層。」

### 10.4 【Q&A】為什麼要在量化點取梯度，而不是在連續點？

這是課堂上被反覆追問的一題。

**學生問**：為什麼在量化值上做 forward 比較好？直接用全精度不行嗎？

**教授答**：

> 「**這模擬的是「用整數跑推論」的情境。**我們算的 loss 是**那個**loss，然後用它來算梯度。
> **因為你部署到硬體上時，用的就是這個值去 forward、去算 loss。你不會用 2.2 —— 實際部署時我們用的是量化後的值。**」

**學生追問**：那 gradient 可能是正的也可能是負的？

**教授答**：

> 「**完全有可能是 -0.1。**因為一旦我們改變下一個 centroid，可能就走過頭了，梯度就會反向。**對更深的網路，它會累積成正確的方向。**」

#### 【口頭】業界的標準流程

> 「**現在的標準流程是：用浮點訓練到收斂 → 用 PTQ 量化 → 再用 QAT 微調。**
> 有人試過從頭開始就在量化模型上訓練（大約 2016 年很流行），**但現在大家都是先浮點訓練再量化，這樣更合理。**」

### 10.5 ⭐ Straight-Through Estimator（STE）

**問題**：量化函數 $Q(\cdot)$ 是**階梯函數**。

$$\frac{\partial Q(W)}{\partial W} = \begin{cases} 0 & \text{幾乎所有地方（平的）} \\ \infty & \text{在跳階處} \end{cases}$$

> 【口頭】「**梯度幾乎處處為零，所以梯度根本流不過去。**」

**解法：STE —— 假裝那個導數是 1，直接把梯度傳過去。**

$$g_W = \frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial Q(W)}$$

$$g_Y = \frac{\partial L}{\partial Y} \approx \frac{\partial L}{\partial Q(Y)}$$

> 【投影片】**「Straight-Through Estimator (STE) simply passes the gradient through.」**

#### 【Q&A】等等，我搞混了 —— 反向不是用浮點權重嗎？

**學生問**：我們反向傳播時用的是浮點權重，那為什麼梯度幾乎處處為零會是問題？

**教授答**（釐清兩個空間）：

> 「**我們不是那樣做的，就忽略那部分。這裡我們是直接把梯度傳過來 —— 所以你把它當成浮點來看，就不會有那個問題。**
>
> **要分清楚 $q_W$ 和 $Q(W)$ 這兩個不同的空間：**
> - **$q_W$ 是整數空間**，範圍是 -128 到 +127
> - **$Q(W)$ 是實數空間**，範圍像是 -2.5 到 +2.14
>
> **我們是把梯度傳到這個（實數）空間 —— 假設這兩者之間的梯度是 1。**」

（**這正是 §10.3 那張圖裡 $Q(W) = S_W q_W$ 這個記號的用意** —— 它是整數，但**用浮點表示**，且**已經包含了量化的影響**。）

### 10.6 QAT 的效果（投影片完整表）

**INT8 Linear Quantization，ImageNet Top-1：**

| Network | **FP32** | **PTQ Asym.<br>Per-Tensor** | **PTQ Sym.<br>Per-Channel** | **QAT Asym.<br>Per-Tensor** | **QAT Sym.<br>Per-Channel** |
|---|---|---|---|---|---|
| **MobileNetV1** | 70.9% | **0.1%** 💀 | 59.1% | **70.0%** | **70.7%** ✅ |
| **MobileNetV2** | 71.9% | **0.1%** 💀 | 69.8% | 70.9% | **71.1%** ✅ |
| **NASNet-Mobile** | 74.9% | 72.2% | 72.1% | 73.0% | 73.0% |

#### ⭐ 這張表要讀出三件事

1. **Per-tensor PTQ 對 MobileNet 是災難** —— **0.1%，完全崩壞**（跟隨機猜差不多）。這正是 §7.1 講的「小模型對 per-tensor 特別敏感」。
2. **Per-channel 就能救回大半** —— 59.1% / 69.8%。
3. **QAT 幾乎完全追平 FP32** —— 70.7% vs 70.9%。

> 【口頭】「per-tensor 的改善**特別明顯 —— 從 0.1% 到 70%**。per-channel 也從 59% 提升到 70.7%。**經過 QAT 之後，準確率基本上追上了原始的準確率。**」

---

## 11. Binary and Ternary Quantization

> 【口頭】「我們能不能把量化精度推到**只有 1 bit**？」

### 11.1 Binarization（二值化）

**只看符號位：**

| 條件 | 結果 |
|---|---|
| $r \ge 0$ | **+1** |
| $r < 0$ | **-1** |

**收穫：**

| | 節省 |
|---|---|
| **記憶體** | **32× 更小** |
| **運算** | **2× 更少** —— 因為其中一個運算元只有 $\pm 1$，**不需要做乘法，只要加減** |

#### 兩種二值化方式

| 方式 | 做法 |
|---|---|
| **Deterministic Binarization** | 直接看符號：$\ge 0$ → +1，$< 0$ → -1 |
| **Stochastic Binarization** | 以機率 $p = \sigma(r)$ 取 +1，否則 -1 |

**$\sigma$ 是 hard sigmoid**：

$$\sigma(r) = \begin{cases} 0 & r \le -1 \\ \text{線性插值} & -1 < r < 1 \\ 1 & r \ge 1 \end{cases}$$

### 11.2 ⭐ 加上 Scaling Factor（BWN vs BinaryConnect）

**問題**：直接二值化，**準確率崩得很慘**。

**解法**：乘上一個 scaling factor

$$\alpha = \frac{1}{n}\Vert W \Vert_1 \qquad (\text{絕對值的平均})$$

$$W \approx \alpha \cdot \text{sign}(W)$$

**投影片的具體數字**（同一個 4×4 矩陣）：

| | 重建誤差 $\Vert W - \hat{W}\Vert_F^2$ |
|---|---|
| **沒有 scale**（$\hat{W} = \text{sign}(W)$） | **9.28** |
| **有 scale**（$\hat{W} = \alpha \cdot \text{sign}(W)$，$\alpha = 1.05 = \frac{1}{16}\Vert W\Vert_1$） | **9.24** ✅ |

**準確率的天壤之別**（AlexNet-based, ImageNet Top-1 Delta）：

| 方法 | 準確率變化 |
|---|---|
| **BinaryConnect**（無 scale） | **-21.2%** 💀 |
| **BWN**（Binary Weight Network，**有 scale**） | **+0.2%** ✅✅ |

> 【口頭】**「這個 $\alpha$ 在二值化時相當關鍵。」**

### 11.3 ⭐ 權重與 activation 都二值化：XNOR + popcount

**核心觀察**（投影片的真值表）：

| $W$ | $X$ | $Y = WX$ | $b_W$ | $b_X$ | **XNOR$(b_W, b_X)$** |
|---|---|---|---|---|---|
| 1 | 1 | **1** | 1 | 1 | **1** |
| 1 | -1 | **-1** | 1 | 0 | **0** |
| -1 | -1 | **1** | 0 | 0 | **1** |
| -1 | 1 | **-1** | 0 | 1 | **0** |

> **乘法的結果，跟 XNOR 的結果完全對應：$1 \leftrightarrow 1$，$-1 \leftrightarrow 0$。**

#### 從 XNOR 還原內積（教授的逐步推導）

> 【口頭】「**假設全部都是 -1**，那 $n$ 個加起來就是 $-n$。
> **每當 XNOR 結果是 1，實際值就比 -1 多了 2** —— 所以只要**數有幾個 1，乘以 2，加到 $-n$ 上就好。**」

$$y_i = -n + 2 \cdot \sum_j (W_{ij} \text{ xnor } x_j)$$

**用 popcount 實作**（popcount = 回傳陣列中 1 的個數）：

$$\boxed{y_i = -n + \text{popcount}(W_i \text{ xnor } x) \ll 1}$$

（$\ll 1$ 就是左移一位 = 乘以 2。）

#### 投影片的完整驗算

$W = [1, 1, -1, -1]$，$X = [1, -1, -1, 1]$，$n = 4$

$$b_W = \texttt{1100},\quad b_X = \texttt{1001}$$

$$y = -4 + \text{popcount}(\texttt{1100} \text{ xnor } \texttt{1001}) \ll 1$$
$$= -4 + \text{popcount}(\texttt{1010}) \ll 1$$

【投影片的算法】$= -4 + \text{popcount}(\texttt{1000}) \ll 1 = -4 + 1 \times 2 = \mathbf{-2}$

> 【口頭】「**popcount、位移、加法 —— 這些全部都是非常便宜的硬體操作。完全沒有乘法。**」

#### 三種組合的節省（投影片）

| Input | Weight | 運算 | **記憶體** | **運算** |
|---|---|---|---|---|
| **R**（實數） | **R**（實數） | $+\ \times$ | 1× | 1× |
| **R** | **B**（二值） | $+\ -$ | **~32× less** | **~2× less** |
| **B** | **B** | **xnor, popcount** | **~32× less** | **~58× less** ⭐ |

### 11.4 二值化的準確率代價（投影片完整表）

| Network | Quantization | W bit | A bit | **ImageNet Top-1 Delta** |
|---|---|---|---|---|
| AlexNet | **BWN**（權重二值 + scale） | 1 | 32 | **+0.2%** ✅ |
| AlexNet | **BNN**（權重與 activation 都二值，**無 scale**） | 1 | 1 | **-28.7%** 💀 |
| AlexNet | **XNOR-Net**（兩者都二值，**兩者都有 scale**） | 1 | 1 | **-12.4%** |

> 【口頭】「XNOR-Net 比 BNN 好一些，但**還是追不上「只二值化權重」的版本。只二值化權重差不多能完全保住準確率。**
> **不過那些 scaling factor 在 BNN 之上帶來了相當大的改進。**」

### 11.5 ⭐ Ternary Weight Networks（TWN）：把 0 加回來

**動機**（教授明確連回 pruning）：

> 【口頭】「**在 pruning 那一講我們發現「零」是非常重要的值 —— 零乘以任何東西都是零，可以直接跳過。**
> 但這裡我們需要 **2 bit** 來表示（**所以浪費了一個 slot**）。」

**量化規則：**

$$q = \begin{cases} r_t & r > \Delta \\ 0 & |r| \le \Delta \\ -r_t & r < -\Delta \end{cases}$$

其中

$$\Delta = 0.7 \times \mathbb{E}(|r|), \qquad r_t = \mathbb{E}_{|r| > \Delta}(|r|)$$

#### 投影片的完整數字（同一個 4×4 矩陣）

$$\Delta = 0.7 \times \frac{1}{16}\Vert W \Vert_1 = \mathbf{0.73}$$

$$r_t = \frac{1}{11}\Vert W_{W \ne 0}\Vert_1 = \mathbf{1.5}$$

**量化結果：**

```
 1  -1   1   0
 0   0  -1   1
-1   1   0  -1
 1   0   1   1
```

（`2.09 > 0.73` → +1；`0.09` 在 $\pm 0.73$ 之間 → 0；`-0.98 < -0.73` → -1）

#### 【口頭】0.7 是哪來的？

> 「**為什麼是 0.7？基本上就是個 heuristic（經驗值）。**」

**準確率**（ResNet-18, ImageNet Top-1）：

| | 準確率 |
|---|---|
| **Full Precision** | **69.6%** |
| **1 bit (BWN)** | 60.8% |
| **2 bit (TWN)** | **65.3%** ✅ |

### 11.6 Trained Ternary Quantization（TTQ）

**改進**：不再用固定的 $r_t$，而是**引入兩個可學習的參數** $w_p$（正）和 $w_n$（負）：

$$q = \begin{cases} w_p & r > \Delta \\ 0 & |r| \le \Delta \\ -w_n & r < -\Delta \end{cases}$$

> 【口頭】「**正值和負值可以不一樣**，這樣能進一步提升準確率。」

**流程**（投影片）：

```
全精度權重 ─normalize─► 正規化權重 ─quantize─► 中間三值權重 ─trained scale─► 最終三值權重
                                                                 (Wn, Wp 可學習)
```

**準確率**（ResNet-18）：

| | Full Precision | BWN (1 bit) | TWN (2 bit) | **TTQ** |
|---|---|---|---|---|
| **Top-1** | 69.6% | 60.8% | 65.3% | **66.6%** ✅ |

### 11.7 ⭐⭐ 為什麼低位元有 Diminishing Return

這是教授在本講最重要的總結之一：

> 【口頭】**「最重要的一件事是：運算便宜，記憶體存取昂貴。
> 所以當你量化得越來越深，會有 diminishing return（報酬遞減）。**
>
> **從 8 bit 到 2 bit，記憶體少了 4 倍；但運算減少了多少？—— 16 倍。**
> **因為運算是位元數的二次方。**
>
> **激進地降低精度，對記憶體只有線性回報，對運算是二次回報。
> 而因為「運算比較便宜、記憶體比較貴」，所以你會得到報酬遞減。**」

$$\boxed{\text{記憶體節省} \propto \frac{1}{n} \quad(\text{線性}) \qquad \text{運算節省} \propto \frac{1}{n^2}\quad(\text{二次})}$$

> 【口頭】**「目前為止，我們發現 4-bit 是個相當不錯的甜蜜點（sweet spot）。」**

【口頭】業界現況：「**binary/ternary 現在業界不太用，因為低於 4 bit 有報酬遞減。但 popcount 和 XNOR 是很有趣的想法，有人靠它開了新創，後來被 Xilinx 收購。**」

---

## 12. Mixed-Precision Quantization（HAQ）

> 【口頭】「就像 pruning 時不同層有不同的敏感度，**量化也是一樣。**」

### 12.1 動機

| | |
|---|---|
| **Uniform Quantization** | 每一層都用 8-bit —— **同質、簡單** |
| **Mixed-Precision** | **不敏感的層 → 更激進地量化；敏感的層 → 保守一點** |

#### 設計空間有多大

> 【口頭】「假設權重有 1 到 8 bit（8 種選擇），activation 也有 8 種，**每層就有 64 種選擇**。$N$ 層就是 $64^N$ —— **這是個巨大的空間。**」

### 12.2 HAQ：Hardware-Aware Automated Quantization

**跟 AMC（Lecture 4 §6.2）同一套 RL 架構**：

```
Actor（提出動作：每層的權重/activation 位元數）
   │
   ▼
【Hardware Simulator】← 這是 HAQ 的關鍵
   │  回饋真實的 latency 和 energy
   ▼
Critic（產生 reward）
   │
   └─► 回饋給 Actor，提出下一輪動作
```

> 【口頭】「我們用 RL 的 actor-critic，**並且有一個硬體模擬器**，把這個混合精度模型映射到硬體上，**拿到延遲與能耗的直接回饋**來產生 state 和 reward。」

**支援的硬體**（投影片）：BitFusion (Edge)、BISMO (Cloud)、BISMO (Edge)

### 12.3 結果

**HAQ 一致地贏過 Uniform Quantization**，而且**支援三種目標**：

| 目標 | HAQ 都能優化 |
|---|---|
| **Model Size Constrained** | ✅ |
| **Latency Constrained** | ✅ |
| **Energy Constrained** | ✅ |

### 12.4 ⭐ 有趣的發現：Edge 和 Cloud 學到不同的策略

> 【口頭】「非常有趣的是 —— **在 edge 裝置和 cloud 裝置上，我們的 RL agent 找到完全不同的量化方案。**
> 例如 **depthwise 層被分配較少的位元**（因為 depthwise 層是嚴重 memory-bound、運算量很少），
> 而 **1×1 convolution 被分配更多位元。**」

（↔ 這正好呼應 **Lecture 2 §2.8** 講的 depthwise conv 的記憶體特性，以及 **Lecture 4 §6.2** AMC 學到 3×3 vs 1×1 的鋸齒圖樣。）

### 12.5 ⚠️ 實務上的 catch

> 【口頭】「**唯一的 catch 是：這需要更多工程投入。你得處理編譯器、工具鏈，才能有效利用混合精度量化 —— 這並不簡單。**
> **所以實務上業界的做法是：所有 conv 層用一種精度、所有 FC 層用另一種精度**，來平衡工程複雜度與效能。」

---

## 13. 一頁速查表

### 13.1 數字格式

| 格式 | Sign | Exponent | Fraction | Total | Bias | 備註 |
|---|---|---|---|---|---|---|
| **FP32** | 1 | **8** | **23** | 32 | 127 | 基準 |
| **FP16** | 1 | **5** | **10** | 16 | 15 | range 小 |
| **BF16** | 1 | **8** | **7** | 16 | 127 | **訓練首選**（range 同 FP32） |
| **FP8 E4M3** | 1 | 4 | 3 | 8 | 7 | **forward / 推論**；無 INF；max = **448** |
| **FP8 E5M2** | 1 | 5 | 2 | 8 | 15 | **backward / 梯度**；有 INF；max = **57,344** |
| **INT4** | — | — | — | 4 | — | $[-8, 7]$，等距 |
| **FP4 E2M1** | 1 | 2 | 1 | 4 | 1 | 常用的 FP4；無 INF/NaN |

**通用規則：**

$$\text{Bias} = 2^{E-1} - 1 \qquad\qquad \boxed{\textbf{Exponent → Range;\ Fraction → Precision}}$$

**FP32 三種情況：**

| Exponent | Fraction = 0 | Fraction ≠ 0 | 公式 |
|---|---|---|---|
| $= 0$ | $\pm 0$ | subnormal | $(-1)^s \cdot \text{Frac} \cdot 2^{1-\text{bias}}$ |
| $1 \sim 2^E-2$ | normal | normal | $(-1)^s (1+\text{Frac}) \cdot 2^{\text{Exp}-\text{bias}}$ |
| $= 2^E-1$ | $\pm\infty$ | NaN | — |

### 13.2 三種量化方法

| | K-Means | Linear | Binary/Ternary |
|---|---|---|---|
| **Storage** | int index + float codebook | int | 1–2 bit |
| **Compute** | **float** | **int** | **xnor + popcount** |
| **壓縮率** | $32/N \times$ | $32/N \times$ | 32× |
| **用在** | **memory-bound（LLM）；Lab 4/5** | **通用；Lab 2** | 研究 |

### 13.3 Linear Quantization 公式

| 項目 | 公式 |
|---|---|
| **基本映射** | $r = S(q - Z)$ |
| **Scale（非對稱）** | $S = \dfrac{r_{\max} - r_{\min}}{q_{\max} - q_{\min}}$ |
| **Zero point** | $Z = \text{round}\left(q_{\min} - \dfrac{r_{\min}}{S}\right)$ |
| **Scale（對稱，$Z=0$）** | $S = \dfrac{\lvert r\rvert_{\max}}{q_{\max}}$ |
| **$N$-bit 範圍** | $q \in [-2^{N-1},\ 2^{N-1}-1]$ |
| **矩陣乘法（一般）** | $q_Y = \frac{S_W S_X}{S_Y}(q_Wq_X - Z_Wq_X - Z_Xq_W + Z_WZ_X) + Z_Y$ |
| **矩陣乘法（$Z_W=0$）** | $q_Y = \frac{S_W S_X}{S_Y}(q_Wq_X - Z_Xq_W) + Z_Y$ |
| **加 bias（$S_b = S_WS_X$, $Z_b=0$）** | $q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X + q_{\text{bias}}) + Z_Y$ |

### 13.4 顆粒度與有效位寬

| 方法 | 有效位寬 | 說明 |
|---|---|---|
| **Per-Tensor** | $N$ | 最粗，小模型會崩 |
| **Per-Channel** | $N$（+ 每 channel 一個 FP16） | 視覺模型夠用 |
| **VSQ** | $4 + 4/16 = \mathbf{4.25}$ | 兩層 scale |
| **MX4** | $3 + 1/2 + 8/16 = \mathbf{4}$ | L0: 每 2 值共用 E1M0；L1: 每 16 值共用 E8M0 |
| **MX6** | $5 + 1/2 + 8/16 = \mathbf{6}$ | |
| **MX9** | $8 + 1/2 + 8/16 = \mathbf{9}$ | |

### 13.5 動態範圍的三種決定方式

| 方法 | 時機 | 做法 |
|---|---|---|
| **EMA** | 訓練時 | 跨數千步平滑 min/max |
| **Laplace 解析解** | 校準時 | $\lvert r\rvert_{\max} = 2.83b / 3.89b / 5.03b$（2/3/4 bit） |
| **最小化 KL Divergence** | 校準時 | **TensorRT 用這個**；適合非高斯分佈 |
| **最小化 MSE（Newton-Raphson）** | 校準時 | U 型曲線的最低點 |

### 13.6 二值 / 三值

| 方法 | W | A | 有 scale | Top-1 Delta |
|---|---|---|---|---|
| **BinaryConnect** | 1 | 32 | ❌ | **-21.2%** |
| **BWN** | 1 | 32 | ✅ | **+0.2%** |
| **BNN** | 1 | 1 | ❌ | **-28.7%** |
| **XNOR-Net** | 1 | 1 | ✅✅ | **-12.4%** |

| 方法（ResNet-18） | Top-1 |
|---|---|
| Full Precision | 69.6% |
| BWN (1 bit) | 60.8% |
| **TWN (2 bit)** | **65.3%** |
| **TTQ (2 bit, 可學習 scale)** | **66.6%** |

**XNOR 內積：**

$$y_i = -n + \text{popcount}(W_i \text{ xnor } x) \ll 1$$

### 13.7 該記住的數字

| 數字 | 意義 |
|---|---|
| **0.03 / 0.1 / 0.2 / 3.1 pJ** | 8-bit ADD / 32-bit ADD / 8-bit MULT / 32-bit MULT |
| **ADD $\propto n$，MULT $\propto n^2$** | 為什麼低位元有 diminishing return |
| **127 / 15 / 7** | FP32 / FP16 / FP8-E4M3 的 exponent bias |
| **448 / 57,344** | FP8 E4M3 / E5M2 的最大值 |
| **$2^{-149}$** | FP32 能表示的最小正數 |
| **3.2× / $32/N$×** | K-Means 量化的壓縮率（玩具例 / 通式） |
| **35× / 49×** | Deep Compression 在 AlexNet / VGGNet 的壓縮率 |
| **510×** | SqueezeNet + Deep Compression（0.47 MB） |
| **$S = 1.07,\ Z = -1$** | 4×4 範例矩陣的 2-bit linear quantization 參數 |
| **70.9% → 0.1% → 70.7%** | MobileNetV1 的 FP32 → per-tensor PTQ → QAT per-channel |
| **4 bit** | 教授說的 sweet spot |

---

## 14. 與其他課程／作業的連結

| 本講觀念 | 連到哪裡 |
|---|---|
| **§1.2 位元數與能耗（$n$ vs $n^2$）** | **Lecture 2 §4.3**、**Lecture 3 §1.3** —— 同一張 Horowitz 能耗表的延伸 |
| **§5.7 Deep Compression = Pruning + Quantization** | **Lecture 3–4** —— 兩者可疊加，9–13× 再乘 27–31× |
| **§5.9 K-Means 只省儲存不省運算（W4A16）** | **Lab 4 / Lab 5** —— Llama 2 部署到筆電；**Lecture 4 §9.1** EIE 的「4-bit 存、16-bit 算」 |
| **§6 Linear Quantization 全部公式** | **Lab 2** —— 直接翻譯成程式碼 |
| **§7.2 Per-Channel 重用 scaling factor** | **Lecture 2 §2.10** BatchNorm 的 $\gamma$；**Lecture 3 §4.3** scaling-based pruning |
| **§7.3–7.5 Group Quantization / MX** | **LLM 講次** —— 4-bit 以下必須用多層 scale；教授說這是好的 final project 題目 |
| **§8.3 Calibration set 要跟 test set 分開** | **Lecture 4 §6.1** —— 同一條「不要 overfit test set」原則 |
| **§9.2 AdaRound（只看單層重建誤差）** | **Lecture 3 §4.7** —— regression-based pruning 的同一思路 |
| **§10 QAT 的 FP32 副本 + STE** | **Lab 2**；on-device training 講次 |
| **§11.7 4-bit 是甜蜜點** | **Lab 4** 用 4-bit；Qualcomm Snapdragon 8 Gen 2 支援 4-bit |
| **§12 HAQ 是 RL** | **Lecture 4 §6.2 AMC** —— 同一套 AutoML 架構，搜的是位元數而非剪枝率 |
| **§12.4 depthwise 層分配較少位元** | **Lecture 2 §2.8** —— depthwise 是 memory-bound 的根源 |
| **§3.5 BF16 的 dynamic range** | 訓練講次 —— 為什麼大模型訓練用 BF16 而非 FP16 |

---

## 附：兩講的一句話總結

> **量化的一切都繞著兩個乘法轉：儲存 = 權重數 × 位元數，而能耗 = 加法的線性 × 乘法的平方。
> 前者告訴你為什麼要量化，後者告訴你為什麼停在 4-bit —— 因為記憶體貴、運算便宜，
> 而低位元對記憶體只有線性回報。**

---

*筆記依據 MIT 6.5940 Fall 2023 Lecture 5 / Lecture 6 逐字稿（Zoom 錄影）與官方投影片 `Lec05-Quantization-I.pdf`、`Lec06-Quantization-II.pdf` 整理。*
