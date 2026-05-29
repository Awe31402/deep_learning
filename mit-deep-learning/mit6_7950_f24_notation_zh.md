# 數學符號對照表
**6.S898 / 6.7960 深度學習 — 2024年秋季**  
*MIT 開放式課程 (MIT OpenCourseWare)*

在深度學習中，我們會遇到許多不同的領域，每個領域都有其專屬的符號表記。在本書/本課程的大部分內容中，我們將遵循以下約定，並在偏離這些規則時特別註明。為了定義這些約定，我們提供了使用範例，您可以從中推導出其規律。

---

## 一般符號 (General Notation)

* **純量 (Scalar)：** $x, y, z$。
* **向量 (Vector)：** $\mathbf{x}, \mathbf{y}, \mathbf{z}$。我們使用粗體小寫字母來表示向量、矩陣和張量。
* **向量的索引 (Index of a vector)：** $x_i$, $x_j$, $y_i$ 或 $x[i]$, $x[j]$, $y[i]$。
* **矩陣 (Matrix)：** $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$。我們使用粗體大寫字母來表示矩陣（在某些情況下也代表張量）。
* **矩陣的索引 (Index of a matrix)：** $X_{ij}$, $Y_{jk}$, $Z_{ii}$ 或 $X[i, j]$, $Y[j, k]$, $Z[i, i]$。
* **矩陣索引約定 (Matrix Indices Convention)：** 對於已索引的矩陣元素 $X_{ij}$ 或 $X[i, j]$，$i$ 表示列索引（rows），$j$ 表示行索引（columns）。我們使用非粗體字型，因為 $X_{ij}$ 和 $X[i, j]$ 是純量。
* **矩陣的切片 (Slice of a matrix)：** $\mathbf{X}_i$ 或 $\mathbf{X}[i, :]$；$\mathbf{X}[:, j]$。以下是一個範例：

  $$\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \quad \mathbf{X}[2, :] = \begin{bmatrix} 3 & 4 \end{bmatrix}$$

* **張量 (Tensor，即多維陣列)：** 通常，我們會使用小寫粗體變數來表示張量，例如 $\mathbf{x}$。這是因為張量可以有任意數量的維度（它們可以是一維、二維、三維等等）。此外，我們通常會定義與張量維度無關的運算子（它們適用於任何 $N$ 維陣列）。然而，在某些章節中，我們會使用大寫字母來區分不同形狀的張量，並在這種情況下進行說明。
* **張量的索引或切片 (Index or slice of a tensor)：** $x[c, i, j, k]$, $\mathbf{x}[:, :, k]$
* **一組含有 $N$ 個數據點的集合 (A set of $N$ datapoints)：** $\{x^{(i)}\}_{i=1}^N$, $\{\mathbf{x}^{(i)}\}_{i=1}^N$, $\{\mathbf{X}^{(i)}\}_{i=1}^N$
* **點積 (Dot product)：** $\mathbf{x}^\top\mathbf{y}$
* **矩陣乘積 (Matrix product)：** $\mathbf{AB}$
* **Hadamard 乘積 (Hadamard product，即逐元素相乘)：** $\mathbf{x} \odot \mathbf{y}$, $\mathbf{A} \odot \mathbf{B}$
* **兩個純量的乘積 (Product of two scalars)：** $ab$ 或 $a * b$

---

## 機器學習 (Machine Learning)

* **損失函數 (Loss function)：** $\mathcal{L}$（通常是單一數據點的損失）
* **所有數據點的總成本 (Total cost over all datapoints)：** $J$
* **通用可學習參數 (Generic learnable parameters)：** $\theta$

---

## 神經網路 (Neural Nets)

* **參數 (Parameters)：** $\theta$；這些包括權重 $\mathbf{W}$ 和偏置 $\mathbf{b}$，以及網路的任何其他可學習參數。
* **數據 (Data)：** $\mathbf{x}$ — 「數據」可以指網路的輸入、隱藏層的激活值、網路的輸出等。任何正在被處理的信號表示都被視為「數據」。有時我們希望區分原始輸入、隱藏單元和輸出，在這種情況下，我們將分別使用 $\mathbf{x}$、$\mathbf{h}$ 和 $\mathbf{y}$。當我們需要區分激活前（pre-activation）和激活後（post-activation）的隱藏單元時，我們將使用 $\mathbf{z}$ 表示激活前，$\mathbf{h}$ 表示激活後。
* **批次表示 (Batch Representations)：** 在描述張量批次時（如程式碼中常見的情形），您可能會遇到 $\mathbf{x}_l[b, c, n, m]$ 來表示網路第 $l$ 層的激活值，其中 $b$ 索引批次元素，$n$ 和 $m$ 作為空間坐標，$c$ 索引通道。
* **深度網路第 $l$ 層的神經元值 (Neuron values on layer $l$ of a deep net)：** $\mathbf{x}_l$
* **層上的神經元索引 (Neuron indices on layers)：** $\mathbf{x}_l[n]$ 指第 $l$ 層的第 $n$ 個神經元。對於具有空間特徵圖的神經網路（例如卷積神經網路），每一層都是神經元的陣列，我們將使用諸如 $\mathbf{x}_l[n, m]$ 的表示法來索引位於 $n, m$ 位置的神經元。
* **數據點的神經表示 (Neural representation of datapoint)：** 數據集中第 $i$ 個數據點在第 $l$ 層的神經表示寫為 $\mathbf{x}_l^{(i)}$。
* **層邊界 (Layer boundaries)：** 當我們描述神經網路中的特定層或模組，並希望說明其輸入和輸出，而又不想一直追蹤層索引時，我們也會使用 $\mathbf{x}_{\text{in}}$ 和 $\mathbf{x}_{\text{out}}$。
* **通道順序 (Channel Ordering)：** 對於具有多個通道的信號（包括神經網路特徵圖），張量的第一個維度是用來索引通道。例如在 $\mathbf{x} \in \mathbb{R}^{C \times N \times M \times \dots}$ 中，$C$ 是信號的通道數。
* **Transformer 的例外 (Transformers Exception)：** 對於 Transformer，我們稍微偏離了前一點，以符合標準符號：一組 Token（將在 Transformer 章節中定義）由一個 $[N \times d]$ 矩陣表示，其中 $d$ 是 Token 的維度。

### 圖形中的節點表示 (Node Representations in Graphs)

我們將使用圓圈來表示神經元，它們是圖（graph）中的純量節點。神經元之間的邊（edge）表示 $\mathbb{R} \to \mathbb{R}$ 的轉換（通常由與邊相關的單個權重參數化）。

在許多神經網路架構中，我們網路的節點不是單個神經元，而可能是多維的神經元向量，我們將其繪製為正方形或長方形。在這些情況下，維度為 $C_1$ 的節點與維度為 $C_2$ 的節點之間的邊表示 $\mathbb{R}^{C_1} \to \mathbb{R}^{C_2}$ 的轉換（這可能由多個權重參數化）。

我們使用的三種節點符號如下所示：

#### 概念圖 (Conceptual Diagram)
```mermaid
graph TD
    %% Scalar representation (horizontal link)
    subgraph Scalar Nodes [純量節點 (左至右)]
        s1((純量)) -->|純量權重| s2((純量))
    end

    %% Vector representation (horizontal link)
    subgraph Vector Nodes [向量節點 (左至右)]
        v1[向量] -->|矩陣權重| v2[向量]
    end

    %% Token representation (vertical link)
    subgraph Token Nodes [Token 節點 (下至上)]
        t1["向量 (token)"]
        t2["向量 (token)"]
        t2 -->|矩陣權重| t1
    end
```

#### ASCII 藝術圖可視化 (ASCII Art Visualization)
```text
  [純量節點]               [向量節點]               [Token 節點]
                                                 +----------------+
                                                 | 向量 (token)   |
                                                 +----------------+
                                                         ^
   ( ) ----> ( )           [ ] ----> [ ]                 |
  純量     純量          向量     向量           +----------------+
                                                 | 向量 (token)   |
                                                 +----------------+
```

Token 是以特定方式使用的神經元向量，將在 Transformer 章節中定義。我們給它一個特殊的符號（長方形），以將其與其他類型的神經元向量區分開來。

如上所示，有時我們繪製網路時，層是從左到右移動，有時是從下到上。兩者表示相同的含意，每個圖中的方向純粹是為了視覺上的清晰而選擇。

---

## 機率 (Probabilities)

我們通常不區分隨機變數（random variables）和那些變數的實現值（realizations）；我們指的是哪一個應該能從上下文中看出來。當進行區分很重要時，我們將使用非粗體大寫字母來表示隨機變數，並使用小寫字母來表示實現值。

假設 $X, Y$ 是離散隨機變數，而 $x, y$ 是這些變數的實現值。$X$ 和 $Y$ 可以在集合 $\mathcal{X}$ 和 $\mathcal{Y}$ 中取值。

* $a = p(X = x \mid \dots)$ 是實現值 $X = x$ 的機率，可能基於某些觀測值進行條件化（$a$ 是一個純量）。
* $f = p(X \mid \dots)$ 是 $X$ 上的機率分佈，可能基於某些觀測值進行條件化（$f$ 是一個函數：$f : \mathcal{X} \to \mathbb{R}$）。如果 $\mathcal{X}$ 是離散的，$f$ 是*機率質量函數 (probability mass function)*。如果 $\mathcal{X}$ 是連續的，$f$ 是*機率密度函數 (probability density function)*。
* $p(x \mid \dots)$ 是 $p(X = x \mid \dots)$ 的簡寫。
* 依此類推，遵循這些規律。
* 假設我們已經定義了一個具名分佈，例如 $p_\theta$；那麼單獨指代 $p_\theta$ 就是 $p_\theta(X)$ 的簡寫。

對於 continuous 隨機變數（連續隨機變數），上述所有表記均適用，除了它們指的是機率密度和機率密度函數，而不是機率和機率分佈。在提到連續分佈時，我們有時會使用「機率分佈」一詞，在這些情況下，這應該被理解為指代機率密度函數。

---

## 矩陣微積分約定 (Matrix Calculus Conventions)

在這些筆記中，我們採用以下矩陣微積分的約定。這些約定使方程式更簡單，這也意味著在實際將這些方程式寫入程式碼時能有更簡單的實現。本節中的所有內容都只是定義。這沒有對錯之分。我們本可以使用其他約定，但我們會發現這些是非常有用的約定。

向量表示為形狀為 $[N \times 1]$ 的行向量（column vectors）：

$$\mathbf{x} \triangleq \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_N \end{bmatrix} \tag{1}$$

如果 $y$ 是純量且 $\mathbf{x}$ 是 $N$ 維向量，則梯度 $\frac{\partial y}{\partial \mathbf{x}}$ 是一個形狀為 $[1 \times N]$ 的列向量（row vector）：

$$\frac{\partial y}{\partial \mathbf{x}} \triangleq \begin{bmatrix} \frac{\partial y}{\partial x_1} & \frac{\partial y}{\partial x_2} & \cdots & \frac{\partial y}{\partial x_N} \end{bmatrix} \tag{2}$$

如果 $\mathbf{y}$ 是 $M$ 維向量且 $\mathbf{x}$ 是 $N$ 維向量，則梯度（在這種情況下也稱為雅可比矩陣 Jacobian）的形狀為 $[M \times N]$：

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} \triangleq \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_1}{\partial x_2} & \cdots & \frac{\partial y_1}{\partial x_N} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial y_M}{\partial x_1} & \frac{\partial y_M}{\partial x_2} & \cdots & \frac{\partial y_M}{\partial x_N} \end{bmatrix} \tag{3}$$

最後，如果 $\mathbf{W}$ 是一個 $[N \times M]$ 維矩陣，且 $\mathcal{L}$ 是純量，則梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ 表示為 $[M \times N]$ 維矩陣（注意維度與您預期的相比進行了轉置；這使得後面的數學計算更簡單）：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \triangleq \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial W_{11}} & \cdots & \frac{\partial \mathcal{L}}{\partial W_{N1}} \\ \vdots & \ddots & \vdots \\ \frac{\partial \mathcal{L}}{\partial W_{1M}} & \cdots & \frac{\partial \mathcal{L}}{\partial W_{NM}} \end{bmatrix} \tag{4}$$

我們有時會繪製矩陣和向量以幫助將運算過程可視化。例如，如果 $N = 3$ 且 $M = 4$：

### 梯度中轉置的可視化 (Visualizing Transposition in Gradients)

#### 權重矩陣 $\mathbf{W}$ (Weight Matrix)
形狀為 $[N \times M]$ ($3 \times 4$) 的權重矩陣 $\mathbf{W}$：
```text
      M = 4 行 (columns)
   +---+---+---+---+
   |   |   |   |   |  N = 3
   +---+---+---+---+  列 (rows)
   |   |   |   |   |
   +---+---+---+---+
   |   |   |   |   |
   +---+---+---+---+
```

#### 梯度矩陣 $\frac{\partial J}{\partial \mathbf{W}}$ (Gradient Matrix)
那麼，純量成本 $J$ 對 $\mathbf{W}$ 的梯度（即 $\frac{\partial J}{\partial \mathbf{W}}$）具有轉置後的形狀 $[M \times N]$ ($4 \times 3$)：
```text
     N = 3 行 (columns)
   +---+---+---+
   |   |   |   |  M = 4
   +---+---+---+  列 (rows)
   |   |   |   |
   +---+---+---+
   |   |   |   |
   +---+---+---+
   |   |   |   |
   +---+---+---+
```

---

## 不會嚴格遵循的約定 (Conventions That Will Not Be Strictly Adhered To)

* 我們通常會使用 $\mathbf{x}$ 作為函數的輸入，並使用 $\mathbf{y}$ 作為輸出。
* $f, g$ 和 $h$ 通常是函數。相應的函數空間為 $\mathcal{F}, \mathcal{G}, \mathcal{H}$。

---

## 雜項 (Miscellaneous)

* **維度 (Dimension)** 一詞在計算科學中有兩種用法。
  1. 作為多元數據結構中的坐標，例如「向量的第 $i$ 個維度」或「128 維特徵空間」。
  2. 作為多維陣列的形狀，例如「4D 張量」。
  
  我們將在本書中使用這兩種含義，希望具體含義能從上下文中看得很清楚。

---

### 課程與資源資訊
* **來源：** MIT 開放式課程 ([ocw.mit.edu](https://ocw.mit.edu))
* **課程：** [6.7960 深度學習 (2024年秋季)](https://ocw.mit.edu)
* **引用與條款：** 有關引用這些材料或使用條款的資訊，請造訪 [MIT 開放式課程使用條款](https://ocw.mit.edu/terms)。
