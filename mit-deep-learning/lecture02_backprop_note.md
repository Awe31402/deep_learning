# MIT 6.7960 第二講完整筆記：How to Train a Neural Net（反向傳播與可微分程式設計）

> **課程**：MIT 6.7960 Deep Learning, Fall 2024 — Lecture 2
> **講者**：Sara Beery
> **來源**：
>   - 官方投影片：`lecture_notes_pdf/mit6_7960_f24_lec2.pdf`（81 頁）
>   - 課堂錄影逐字稿：[Lecture 2 錄影](https://youtu.be/vidCX_dMCu0)（1:19:00）
>   - 投影片大量引用 *Foundations of Computer Vision*（Torralba, Isola, Freeman）第 14 章
>
> **標記說明**（同 [Lecture 1 筆記](lecture01_intro_note.md)）
> - `【口頭】` = 課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 課堂提問／教授反問
> - 未標記者 = 投影片本身的內容
>
> **公告**：Pset 1 已發佈，**9/24 截止**；Office Hour 本週開始；PyTorch tutorial 本週。

---

## 目錄

- [0. 本講地圖](#0-本講地圖)
- [1. ⭐ 最佳化的三個層級](#1--最佳化的三個層級)
- [2. Gradient Descent 與 SGD](#2-gradient-descent-與-sgd)
- [3. Momentum（動量）](#3-momentum動量)
- [4. ⭐⭐ Loss Landscape：六種地形](#4--loss-landscape六種地形)
- [5. 兩個救場工具：Evolution Strategies 與 Gradient Clipping](#5-兩個救場工具evolution-strategies-與-gradient-clipping)
- [6. ⭐ 好的 loss / activation 該有什麼性質：ReLU vs GeLU](#6--好的-loss--activation-該有什麼性質relu-vs-gelu)
- [7. Computation Graph（計算圖）](#7-computation-graph計算圖)
- [8. 矩陣微積分複習](#8-矩陣微積分複習)
- [9. ⭐⭐ Backpropagation 的核心把戲](#9--backpropagation-的核心把戲)
- [10. ⭐⭐ 通用層的 backward：L 與 g](#10--通用層的-backwardl-與-g)
- [11. ⭐ Linear Layer 速查表](#11--linear-layer-速查表)
- [12. ⭐⭐ 整個 MLP 的 backprop](#12--整個-mlp-的-backprop)
- [13. ⭐ Backprop through DAGs：只需要兩個操作](#13--backprop-through-dags只需要兩個操作)
- [14. ⭐ Differentiable Programming 與 Software 2.0](#14--differentiable-programming-與-software-20)
- [15. ⭐ 優化參數 vs 優化輸入](#15--優化參數-vs-優化輸入)
- [16. ⭐⭐ 手算範例：跑完一次 backprop](#16--手算範例跑完一次-backprop)
- [17. 一頁速查表](#17-一頁速查表)
- [18. 其他 Q&A 雜項](#18-其他-qa-雜項)

---

## 0. 本講地圖

投影片第 3 頁與第 71 頁前後呼應，六個段落：

1. Gradient descent、SGD 複習
2. **Computation graphs（計算圖）**
3. **Backprop through chains（鏈）**
4. **Backprop through MLPs**
5. **Backprop through DAGs**
6. **Differentiable programming**

【口頭】教授開場：

> 「這對有些人可能是複習，但**我們談『訓練神經網路』這件事的視角，可能跟你之前看過的不太一樣。**」

那個「不一樣的視角」就是：**不要把神經網路想成網路，把它想成一張計算圖；backprop 只是在圖上重用共享項的演算法。**

---

## 1. ⭐ 最佳化的三個層級

我們對 cost function $J(\theta)$ 到底知道多少？投影片把它分成三級：

| 我們能算什麼 | 叫什麼 | 實務 |
|---|---|---|
| 只能算 $J(\theta)$ | **Black box optimization** | — |
| 能算 $J(\theta)$ 和 $\nabla_\theta J(\theta)$ | **First order optimization** | **⭐ 幾乎都用這個** |
| 還能算 Hessian $H_\theta(J(\theta))$ | **Second order optimization** | 【口頭】「實務上我們很少真的這樣做」 |

【口頭】「通常我們就只專注在一階——**在某個點上取梯度的線性近似**。」

---

## 2. Gradient Descent 與 SGD

### 2.1 目標

$$\theta^\star = \arg\min_\theta \underbrace{\sum_{i=1}^{N} L\big(f_\theta(x^{(i)}), y^{(i)}\big)}_{J(\theta)}$$

【口頭】「所謂『學習』，講白了就是**在數值空間裡把那些值扭來扭去**，直到在我們手上有的資料上得到接近最佳的東西。」

### 2.2 演算法（投影片 Algorithm 10.1）

```
輸入：目標函數 J、初始參數 θ⁰、learning rate η、步數 K
for k = 0 … K-1:
    θ^(k+1) = θ^k − η ∇_θ J(θ^k)
輸出：θ* = θ^K
```

**兩個超參數**：**learning rate** 和 **步數 K**。步長 = learning rate × 梯度大小（所以步長和梯度大小成正比）。

【口頭】重要提醒：如果 loss landscape 非凸、有多個 minima，**你找到的不保證是全域最小值**，而且**那個局部最小值有多糟，我們通常也不知道**。

### 2.3 ⭐ Stochastic Gradient Descent (SGD)

| batch size | 等於什麼 |
|---|---|
| **1** | 每看一筆就更新一次 |
| **N（全部資料）** | 就是標準的 gradient descent |
| 中間值 | **SGD** |

**優點**
- **更快**：小樣本就能近似總梯度
- ⭐ **隱式正則化（implicit regularizer）**——【口頭】「那個雜訊其實可以**把你從局部最小值裡彈出來**」

**缺點**
- **高變異、更新不穩定**

【口頭】⭐ 教授從自己的研究經驗補了一個很實際的觀察：

> 「我做的資料類別**極度不平衡**。這時候抽樣對梯度穩定性影響很大——**有些類別可能連續好幾個 batch 都沒出現，一出現梯度就整個大位移。**所以這種不穩定性，其實跟你的資料集分佈**均不均勻**高度相關。」

【口頭】為什麼不能用 full-batch？「時間、計算複雜度。我剛入行時有 ImageNet 等級（百萬張）就算幸運了，現在影像資料集是十億級，文字更大。」

---

## 3. Momentum（動量）

**物理直覺**：把重物放在斜坡上，它滾下去會**越滾越快**。

**做法**：讓梯度步伐**偏向延續上一步的方向**。

投影片 Algorithm 1.3（GD + momentum）：

$$v^{k+1} = \mu v^k - \eta \nabla_\theta J(\theta^k), \qquad \theta^{k+1} = \theta^k + v^{k+1}$$

- $\mu$（有些頁寫成 $\alpha$）是 **momentum 強度，是超參數**
- **可能幫忙，也可能幫倒忙**

【口頭】三種情況的對照（投影片動畫）：

| momentum | 結果 |
|---|---|
| **0** | 穩穩地滾下去，但慢 |
| **0.5** | 快很多，大約**一半的步數**就到 |
| **太高** | **衝過頭**，開始來回震盪，反而更久 |

【口頭】**Adam** 就是這類帶 momentum 的常用 optimizer。

**推薦資源**（投影片有連結）：Gabriel Goh 的 [Distill 互動文章 "Why Momentum Really Works" (2017)](https://distill.pub/2017/momentum/)——【口頭】「有點舊了，但建立直覺超好用。」

---

## 4. ⭐⭐ Loss Landscape：六種地形

投影片連續丟三個問題給全班（每次同一組 6 張圖）：

### 4.1 【Q&A】三個問題

| 問題 | 答案 |
|---|---|
| **哪些是可微的？** | 只有其中兩個（含**平的那個**——平的也有導數，只是等於 0） |
| **哪些在 PyTorch 裡有定義好的梯度？** | ⭐ **全部**。因為 PyTorch 有 autograd |
| **哪些難最佳化？** | 多局部極小的、平的、平+不連續的、梯度爆炸的 |

⭐ 【口頭】**最關鍵的一句話**：

> **「PyTorch 可以讓任何東西變成可微的。但那不代表它會好最佳化。」**
>
> 「那個平的函數在 PyTorch 裡是可微的，但它**找不到局部最小值，因為根本不存在**。計算上可以算出梯度，但**梯度還是零**。」

【口頭】另一個直覺：**想成流水**。水不管從哪裡出發都會流到中間那個低點——但這只能幫你判斷「有沒有唯一極小」，篩掉多極小的情況而已。

### 4.2 六種地形對照

| 地形 | 性質 | 問題 |
|---|---|---|
| **Convex（凸）** | 單一極小、梯度處處指向它、接近時**優雅地趨近 0** | 沒問題（理論家最愛） |
| **Discontinuous（不連續）** | 兩側**單邊導數都有良好定義** | ⭐ **對 PyTorch 完全不是問題** |
| **Vanishing gradient（梯度消失）** | 幾乎平但沒完全平 | 進展極慢，**batch 的雜訊可能蓋過訊號** |
| **Zero gradient（零梯度）** | 完全平 | 梯度**完全沒有資訊**，永遠到不了低損失區 |
| **Exploding gradient（梯度爆炸）** | 越接近極小值梯度越大 → ∞ | 更新不穩、**衝過頭**，幾乎打不中極小值 |
| **Multiple local minima** | 多個局部極小 | **初始化位置很重要**，不保證全域最小 |

【口頭】⭐ 實務上怎麼處理「不保證凸」這件事：

> **「跑一堆不同的 random seed，然後挑表現看起來比較好的模型。而且*不同 seed 之間表現的變異數，本身就告訴你這個 loss landscape 有多不穩定*。」**

【Q&A】學生問那些圖到底在畫什麼 → 教授澄清：橫軸是**單一參數**，圖上的軌跡是 **optimizer step 0 到 100** 的移動路徑（假設從上面開始，用一個標準 learning rate）。

---

## 5. 兩個救場工具：Evolution Strategies 與 Gradient Clipping

### 5.1 Evolution Strategies（ES，投影片 Algorithm 10.4）

**它不是梯度法，但「像」梯度法**：在參數空間裡找一個局部降低損失的方向。

```
for k = 0 … K-1:
    for i = 1 … M:
        εᵢ ~ N(0, I)
        sᵢ = J(θ + εᵢ)
    θ^(k+1) = θ − η · (1/M) Σᵢ sᵢ εᵢ
```

**做法**：在目前參數附近**隨機取樣小擾動**，往「損失比較低的那些擾動」的方向移動。

⭐ **它能成功最小化一個「零梯度（不可微）」的函數**——這正是投影片舉的例子。

【口頭】直覺：

> 「即使我們從一個沒有梯度訊號的地方出發，**因為我們在擾動函數的取值，它可能就這樣被彈到一個比較好的地方。**」
> 「你幾乎可以把它想成 loss 上的一種**正則化**。」

### 5.2 Gradient Clipping（投影片 Algorithm 10.5）

**針對梯度爆炸**。做法極簡單：

$$v = \nabla_\theta J(\theta^k), \qquad \theta^{k+1} = \theta^k - \eta\big[\mathrm{clip}(v_1, -m, m), \ldots, \mathrm{clip}(v_M, -m, m)\big]^\top$$

**梯度超過某個大小 $m$ 就縮回 $m$。**

投影片自己的評語：**「有用，而且是很常用的 hack。」**

---

## 6. ⭐ 好的 loss / activation 該有什麼性質：ReLU vs GeLU

投影片列了三個性質，然後用兩個 activation 對照：

| 性質 | **ReLU** | **GeLU** |
|---|---|---|
| **Everywhere continuous（處處連續）** | ✅ | ✅ |
| **Everywhere differentiable（處處可微）** | ⚠️ **Almost!**（原點有折角） | ✅ |
| **Everywhere smooth（處處平滑）** | ❌ | ✅ |

**GeLU** = Gaussian Error Linear Unit（[arXiv:1606.08415](https://arxiv.org/abs/1606.08415)）：$x \cdot \Phi(x)$，$\Phi$ 是高斯的累積分佈函數。

⭐ 投影片自己的謹慎結論（教授口頭也強調）：

> **「我們可以同意這三個性質大概會讓最佳化更容易。就算我們在實驗上或理論上都還不確定訓練神經網路到底需要哪些性質——但趨勢確實在往『三個都滿足』的函數移動。」**

【Q&A】相關的三個提問：

- **「所以 GeLU 比 ReLU 好？」** → 教授：**「我沒有這樣說，那不一定是真的。」** 我只是在說趨勢在往那邊走，社群**還沒有**任何理論保證。
- **「loss function 可以是負的嗎？」** → 可以，看你怎麼定義。
- **「activation 可以非單調（non-monotonic）嗎？」** → 理論上存在，但不常用。⭐ 而且 GeLU 自己就有一小段往下凹。**「所以問題就變成：平滑重要，還是單調重要？這件事還沒定論。」**

【Q&A】**超參數怎麼調？** → 教授給了一個很實際的答案：

> 「沒有普世的理想值，一定要針對資料集/架構去實驗找。**但一個普遍不錯的策略：找一篇用了它的論文，看他們用什麼值，就從那裡開始。**因為人家已經幫你調過了。有時候換到新資料集會差很遠，但當起點通常不會太糟。」

---

## 7. Computation Graph（計算圖）

**定義**（投影片）：

> 一張由**函數轉換（nodes）**組成的圖，串起來就完成某個有用的計算。深度學習主要處理的是**有向無環圖（DAG）**形式的計算圖，而且**每個節點都可微**。

**DAG 拆開來看**：
- **Directed（有向）**：任一條邊上，資訊只往一個方向流
- **Acyclic（無環）**：沒有迴圈

【口頭】教授舉了一個非 ML 的例子建立直覺：

> 「一棵**決策樹**就是最簡單版本的『函數轉換圖』。我要走到教室 → 進哪棟樓 → 進了 45 號樓（對）還是 32 號樓（錯）→ 搭電梯還是走樓梯。**每個決策都是把輸入轉成輸出的函數轉換。**」

【口頭】而在 ML 裡，**每個 block 可以是一層，也可以是一整個神經網路**——這個視角本身沒有對大小的限制。

**一個 MLP 畫成計算圖**：

```
x → [linear, W₁] → z → [relu] → h → [linear, W₂] → ŷ → [loss] → J
```

### 7.1 Forward pass 的通用寫法

$$x_{out} = f(x_{in}, \theta)$$

⭐ 注意投影片這裡刻意的設計：**把參數當成「一個無參數轉換」的另一個輸入**。這正好呼應 Lecture 1 講的 data vs parameters 對稱性。

### 7.2 Learning 要算什麼

要算 **cost $J$ 對所有模型參數的梯度**。

投影片的關鍵一句：**「by design，每一層對它的輸入都是可微的（而輸入包含資料和參數）」** → 所以這件事**算得出來**。

---

## 8. 矩陣微積分複習

【口頭】「這只是備忘，因為接下來會有一堆矩陣乘矩陣。」

設 $x$ 是 $[n \times 1]$ 的行向量，$y = f(x)$：

| 情況 | 結果的形狀 |
|---|---|
| $y$ 是**純量** | $\dfrac{\partial y}{\partial x}$ 是 **$[1 \times n]$ 的列向量** |
| $y$ 是 $[m \times 1]$ 向量 | **Jacobian**：$[m \times n]$ 矩陣（m 列 n 行） |
| $y$ 純量、$X$ 是 $[n \times m]$ 矩陣 | 結果是 $[m \times n]$ 矩陣（⭐ **注意行列被轉置了**） |

投影片自己引用 Wikipedia 誠實承認：**vector-by-matrix、matrix-by-vector、matrix-by-matrix 這三種導數「記號沒有廣泛共識」**——【口頭】「這門課不會碰到。」

### 8.1 【Q&A】Chain rule 的形狀

$z = h(x) = f(g(x))$，寫成 $z = f(u)$、$u = g(x)$：

$$\left.\frac{\partial z}{\partial x}\right|_{x=a} = \left.\frac{\partial z}{\partial u}\right|_{u=g(a)} \cdot \left.\frac{\partial u}{\partial x}\right|_{x=a}$$

設 $|z| = m$、$|u| = p$、$|x| = n$：

| 項 | 形狀 |
|---|---|
| $\partial z / \partial x$ | $[m \times n]$ |
| $\partial z / \partial u$ | $[m \times p]$ |
| $\partial u / \partial x$ | $[p \times n]$ |

【口頭】教授現場帶全班一格一格填形狀，然後自嘲：「這頁寫得超容易混淆，我明年要重寫。」

---

## 9. ⭐⭐ Backpropagation 的核心把戲

這是整堂課最重要的一頁（投影片第 36 頁）。

考慮一條鏈：$x_0 \to f_1 \to x_1 \to f_2 \to \cdots \to f_L \to x_L \to L \to J$

用 chain rule 分別寫出兩個參數的梯度：

$$\frac{\partial J}{\partial \theta_1} = \frac{\partial J}{\partial x_L}\frac{\partial x_L}{\partial x_{L-1}} \cdots \frac{\partial x_3}{\partial x_2}\frac{\partial x_2}{\partial x_1}\frac{\partial x_1}{\partial \theta_1}$$

$$\frac{\partial J}{\partial \theta_2} = \underbrace{\frac{\partial J}{\partial x_L}\frac{\partial x_L}{\partial x_{L-1}} \cdots \frac{\partial x_3}{\partial x_2}}_{\text{和上面完全一樣}}\frac{\partial x_2}{\partial \theta_2}$$

⭐ **關鍵觀察**：兩式**灰框裡的項是共享的**，所以**只需要算一次**。

> **Backpropagation 就是一個「把共享項在計算圖上傳遞下去」的演算法。**
> （投影片直接註記：**aka dynamic programming（動態規劃）**）

【口頭】

> 「它基本上就是一個**效率的把戲**，但正是這個把戲讓超大模型的梯度變得**計算上可行**。」

### 9.1 兩個 pass

| Pass | 做什麼 |
|---|---|
| **Forward pass** | 資料往前送，算出每層輸出 $x_0, x_1, \ldots, x_L$，算 loss |
| **Backward pass** | 把**誤差訊號（梯度）**從輸出/loss 往回送到輸入與參數，得到 $g_L, g_{L-1}, \ldots$ 以及每層的參數梯度 |

---

## 10. ⭐⭐ 通用層的 backward：L 與 g

投影片引入**兩個簡寫**，這是整章的骨架。一個層 $x_{out} = f(x_{in}, \theta)$：

| 符號 | 定義 | 是什麼 | 形狀 |
|---|---|---|---|
| $\mathsf{L}$ | $\dfrac{\partial x_{out}}{\partial [x_{in}, \theta]}$ | **層輸出對層輸入的梯度**（**純局部**） | 矩陣 |
| $g$ | $\dfrac{\partial J}{\partial x}$ | **cost 對 activation 的梯度** | 列向量 |

細分成兩個 $\mathsf{L}$：

$$\mathsf{L}^x = \frac{\partial x_{out}}{\partial x_{in}} \ \ [|x_{out}| \times |x_{in}|], \qquad \mathsf{L}^\theta = \frac{\partial x_{out}}{\partial \theta} \ \ [|x_{out}| \times |\theta|]$$

### 10.1 有了 L 和 g，參數更新超簡單

$$\frac{\partial J}{\partial \theta} = \underbrace{g_{out}}_{\text{來自後面}} \underbrace{\mathsf{L}^\theta}_{\text{純局部}}, \qquad \theta^{i+1} = \theta^i - \eta \left(\frac{\partial J}{\partial \theta}\right)^\top$$

### 10.2 那 L 和 g 怎麼來？

| | 怎麼得到 |
|---|---|
| **$\mathsf{L}$** | ⭐ **完全是局部的過程**：來自該層的導數函數 $f'$，我們**假設它已經給定**（PyTorch 幫你定義好） |
| **$g$** | 比較 tricky——它需要 $x_{out}$ 到 $J$ 之間所有的層。**但只要有了 $g_l$，$g_{l-1}$ 只是再一次矩陣乘法** |

**遞迴關係（＝誤差訊號的反向傳播）**：

$$\boxed{g_{in} = g_{out}\,\mathsf{L}^x}$$

### 10.3 完整的 backward 函數

一個層的 **backward** 有 **3 個輸入、2 個輸出**：

| 輸入 | 輸出 |
|---|---|
| $x_{in}$、$\theta$、$g_{out}$ | $g_{in} = g_{out}\mathsf{L}^x$、$\dfrac{\partial J}{\partial \theta} = g_{out}\mathsf{L}^\theta$ |

（另一頁用「隱藏層 $l$」的畫法表達同一件事：**訓練時每層有三個輸入、三個輸出**，實線是前向、虛線是反向。）

### 10.4 完整演算法（三步驟）

| 步驟 | 內容 |
|---|---|
| **1. Forward pass** | 對每個訓練樣本，**由上而下**算出所有層的輸出 $x_l = f_l(x_{l-1}, \theta_l)$ |
| **2. Backward pass** | **迭代地**算 loss 的導數 |
| **3. Parameter update** | 算對權重的梯度，更新權重 |

…然後 repeat。

【口頭】什麼時候停？

> 「通常用 **validation set**，看那上面的變化有沒有**趨於平緩（plateau）**。之後的課會談更多機制。」

### 10.5 Batch 版本

$$J = \frac{1}{N}\sum_{i=1}^{N} J_i(x^i, \theta), \qquad \frac{\partial J}{\partial \theta} = \frac{1}{N}\sum_{i=1}^{N}\frac{\partial J_i(x^i, \theta)}{\partial \theta}$$

**總 cost 的梯度 = 每筆 cost 梯度的平均**。【口頭】「因為導數可以移進求和裡面，很直接。」

---

## 11. ⭐ Linear Layer 速查表

線性層 $x_{out} = f(x_{in}, W) = W x_{in}$，$W$ 是 $[|x_{out}| \times |x_{in}|]$。

| 方向 | 公式 | 為什麼 |
|---|---|---|
| **Forward** | $x_{out} = W x_{in}$ | — |
| **Backprop 到輸入** | $g_{in} = g_{out}\mathsf{L}^x = g_{out} W$ | 因為 $\dfrac{\partial x_{out,i}}{\partial x_{in,j}} = W_{ij}$，所以 $\mathsf{L}^x$ **就是 $W$ 本身** |
| **Backprop 到權重** | $\dfrac{\partial J}{\partial W} = x_{in}\,g_{out}$ | 因為 $\dfrac{\partial x_{out,i}}{\partial W_{ij}} = x_{in,j}$（改動 $W_{ij}$ **只影響輸出的第 i 個分量**） |
| **更新** | $W^{k+1} = W^k - \eta\left(\dfrac{\partial J}{\partial W}\right)^{\!\top}$ | — |

⭐ **這裡有個實作上的重點**：投影片指出，天真的做法是先把那個巨大的稀疏矩陣 $\mathsf{L}^\theta$（大小 $[M \times MN]$，只有 $i = k$ 的位置非零）**造出來**再做矩陣乘法——**不要這樣做**，直接用上面的簡化式，把那些乘以零的運算全部省掉。

【口頭】一句話總結：

> **「前向和反向就只是乘同一個權重矩陣，只是順序不同而已。挺漂亮的。」**

---

## 12. ⭐⭐ 整個 MLP 的 backprop

投影片拿一個具體的 MLP 走完全程（省略 bias）：$x$ 四維、$z$ 和 $h$ 三維、$\hat y$ 二維。

**Forward**：
```
x --linear(W₁)--> z --relu--> h --linear(W₂)--> ŷ --L2 loss--> J
z = W₁x,  h = relu(z),  ŷ = W₂h,  J = ‖ŷ − y‖²₂
```

### 12.1 一個換記號的技巧

**把梯度從列向量轉置成行向量**。根據矩陣恆等式 $(AB)^\top = B^\top A^\top$：

$$g_{in}^\top = (g_{out}W)^\top = W^\top g_{out}^\top$$

### 12.2 ⭐ 這揭露了一個漂亮的對稱性

> **線性層的 backward，就是和 forward 一模一樣的運算，只是權重轉置。**

$$\text{forward: } W x \qquad\Longleftrightarrow\qquad \text{backward: } W^\top g$$

### 12.3 ⭐ 那 ReLU 的 backward 是什麼？

**不是 ReLU！**

它變成一個**對角的「閘門（gating）矩陣」**，對角線上的 $a, b, c$ 由 **forward pass 時的 activation** 決定（正的位置是 1、負的位置是 0）。

【口頭】它在做什麼：

> **「確保你不會把梯度傳給那些落在 ReLU 零區的分量——那些應該被 mask 掉的部分，你不想送梯度過去。」**

【Q&A】「為什麼那個矩陣是對角的？」→ **「這樣它才是對每個元素獨立作用。」**

### 12.4 ⭐⭐ 兩個很有份量的結論

**(1) 反向傳播圖本身也是一個神經網路。**

投影片第 52 頁把三層 MLP 的完整計算圖（forward + backward）畫在一起，標題就寫著：**"It's just another neural net!"**——backward 那半邊是一串 $W_3^\top, W_2^\top, W_1^\top$ 的線性層。

**(2)【口頭】Phil 的直覺（教授說她很喜歡這個）**：

> 「不管 loss landscape 多複雜，**只要我們取的是一階近似，我們就是在這個彎曲的曲面上擬合一個平面，然後往那個平面的方向移動。**所以它照定義就必須是線性的——**我們是在一個平面上移動**，即使底下真正的 loss 一點都不平。」

### 12.5 【Q&A】記憶體的不對稱性（很好的問題）

學生問：forward pass 可以算完一層就把前一層的 embedding 丟掉；**但 backward 需要每一層的 $x_{in}$**，所以要全部留在記憶體裡？

**教授：對。**

> **「你確實需要把中間表示（intermediate representations）存下來，才能有效率地算 backprop。」**

⭐ 這就是訓練比推論吃記憶體的根本原因（也正是 EfficientML 那門課一直在對付的東西）。

【Q&A】追問「過 ReLU 回來時要記住 pre-activation 嗎？」→ **要**，你要存下每個分量 activation 的值，才能建出那個稀疏 gating 矩陣。

【Q&A】「為什麼看不到 bias 項？」→ **只是為了簡化。**實務上 bias 到處都是。

---

## 13. ⭐ Backprop through DAGs：只需要兩個操作

【口頭】真實的模型常常**不是一條鏈**——共享權重、拆權重、多頭 concat、拆成不同目標…

⭐ **投影片的核心主張：把前面所有東西推廣到任意 DAG，只需要兩個操作。**

| 操作 | Forward | **Backward** |
|---|---|---|
| **Merge（合併）** | 把多路合起來（加法、concat…） | **把梯度按對應的輸入變數拆開送回去** |
| **Branch（分支）** | 把資訊拆開或複製：$x^a = x$、$x^b = x$ | ⭐ **把各路的梯度加起來（sum）** |

### 13.1 ⭐ Parameter sharing → sum gradients

參數共享（例如把同一組 ImageNet 預訓練權重用在網路的多個地方）**在圖上就是一個 branch**——只是分支的是 $\theta$ 而不是 $x$：

$$\theta^a = \theta, \quad \theta^b = \theta \qquad\Longrightarrow\qquad \frac{\partial J}{\partial \theta} = \frac{\partial J}{\partial \theta^a} + \frac{\partial J}{\partial \theta^b}$$

> **Parameter sharing ⟹ 梯度相加。**

【Q&A】課末有學生問「MLP 的 backprop 和 DAG 的 backprop 差在哪？」→ 教授：**「沒有差別。**我們前面所有詳細例子都是鏈狀的 DAG。現在只是說，**任何 DAG 都是『有時候合併、有時候分支』的鏈**，而合併和分支的梯度傳遞規則超簡單。」

【Q&A】「為什麼 branch 是 sum？」→ 教授當場承認講不清楚：**「我現在沒辦法很簡單地講完，來 office hour 我們再談。」**（投影片註記說 lecture notes 有更詳細的推導。）

---

## 14. ⭐ Differentiable Programming 與 Software 2.0

### 14.1 為什麼深度網路受歡迎（投影片只給兩個理由）

1. **容易最佳化**（可微）
2. **可組合的「積木式程式設計」（block based programming）**

> 具備這兩個性質的一般化模型，有個新興的名字：**differentiable programming**。

投影片引了兩則（現在有點舊的）推文：
- **Yann LeCun**：深度學習作為流行語已經沒用了 → **"Vive Differentiable Programming"**
- **Tom Dietterich**：深度學習本質上是一種**新的程式設計風格**，而整個領域正在摸索**這種風格裡可重用的構件（reusable constructs）是什麼**——已知的有 convolution、pooling、LSTM、GAN、VAE、memory unit、routing unit…

### 14.2 Neural Module Networks（Andreas et al. 2017）

投影片舉的例子：輸入「Where is the dog?」→ 送給一個 **parser**（**這個可能完全沒有學習，就是標準 parser，沒有可微參數**）→ 再交給一個 CNN 去解析影像中的物件（**這個有梯度更新**）。

⭐ **重點：同一張圖裡，有些節點是學來的，有些不是。**

### 14.3 Software 2.0（Andrej Karpathy）

| | 意思 |
|---|---|
| **Software 1.0** | 傳統軟體，一切都被明確定義——只是空間裡的一個小紅點 |
| **Software 2.0** | **定義一個「可能的程式」的空間**，然後在那個空間裡**最佳化**出你需要的系統 |

所以一個實際系統會混合兩種節點：

| **Programmed by a human** | **Programmed by backprop** |
|---|---|
| 【口頭】例如**自然影像的 normalization 值**——我們直接寫死，不去學 | 用「調整行為去 match 訓練樣本」的方式編出來 |

### 14.4 ⭐⭐ 這一切的重點

> **Backprop 讓你能對「任何純量 cost」，去最佳化計算圖裡的任何一個節點（函數）或任何一條邊（變數）。**

投影片一頁一頁展示三種問法：
1. 當那個**黃色函數的權重**改變時，cost 怎麼變？
2. 當那個**函數節點本身**改變時，cost 怎麼變？
3. ⭐ 當**輸入資料**改變時，cost 怎麼變？

【Q&A】學生問「哪些該讓人寫、哪些該讓模型學，有沒有指標可以判斷？」→ 教授的回答值得記：

> 「**只要人在編寫系統的一部分，那就是在對系統加約束。**這個約束可能有用，也可能沒用。」
> 「我剛入行時大家做**feature engineering**——把一堆明確的約束寫進資料處理，再送進一個很簡單的網路甚至 SVM。結果那些瓶頸**不見得是最佳的**：**我們最好的點子，比不上一個更大、更端到端學出來的模型。**」

---

## 15. ⭐ 優化參數 vs 優化輸入

| | 在問什麼 |
|---|---|
| **優化參數**（標準做法） | 改**參數**時，總 cost 增加或減少多少？$\partial J / \partial \theta$ |
| **⭐ 優化輸入** | **固定參數**，改**影像像素**時，「chameleon」這個分數增加或減少多少？$\partial y_j / \partial x$ |

### 15.1 ⭐ 一個實作上的關鍵細節

【口頭】

> **對 softmax 之後的機率做優化時，「提高某類機率」最容易的方式往往是「把其他類壓低」，而不是「把目標類拉高」。所以優化 pre-softmax 的 logits 通常穩定得多。**

### 15.2 Unit visualization（單元視覺化）

**做一張讓「cat」輸出神經元最大化的影像**：

$$x^{k+1} = x^k + \eta \left.\frac{\partial\big(y_j(x) + R(x)\big)}{\partial x}\right|_{x = x^k}, \qquad \arg\max_x\ y_j + R(x)$$

（$R(x)$ 是正則項。）

也可以換成**第 $l$ 層第 $j$ 個神經元**的值 $h_j^l$ ——這就變成一種**探測模型在看什麼**的工具。

參考：Olah et al., [Distill "Feature Visualization" (2017)](https://distill.pub/2017/feature-visualization/)。

### 15.3 Deep Dream

同樣的想法，[Google 2015 "Inceptionism"](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)。【口頭】「有點年紀了，但我覺得這些**又迷幻又漂亮**。」

### 15.4 ⭐ CLIP + GAN：整堂課的收尾範例

**CLIP** 訓練一個文字編碼器和一個影像編碼器，讓語意相近的文字與影像在**共同 embedding 空間**裡靠得很近。

**CLIP + GAN 的玩法**：

```
輸入：一個隱向量 z （⭐ 只有這個在被優化）
z → [影像生成器（固定）] → 影像 → [CLIP 影像編碼器（固定）] → e₁
   提示詞 "What is the answer to the ultimate question of
   life, the universe, and everything?" → [CLIP 文字編碼器（固定）] → e₂
目標：最大化 e₁ · e₂
```

⭐ 【Q&A】教授特別澄清（有學生問）：

> **「這裡所有參數都是固定的**（CLIP 兩個編碼器、影像生成器都不 back-prop 更新）。**唯一被優化的就是那個丟進生成器的隱向量 z。**但你**還是需要把訊號 back-prop 穿過那些模型**才能傳到 z。」

【口頭】結語：

> 「這些梯形全都是神經網路。你可以把它們插在一起，把在某種方式下訓練出來的元件拿去用在另一種方式上。**你可以拿任何模組對任何其他模組做優化——the world's your oyster。**」

---

## 16. ⭐⭐ 手算範例：跑完一次 backprop

投影片最後 9 頁是一個完整可驗算的練習（第 72–80 頁）。

### 16.1 設定

一個 2→2→1 的小網路：

```
x₀ --W₀--> x₁ --tanh--> x₂ --W₁--> x₃ --L2 loss--> L
```

$$W_0 = \begin{bmatrix} 0.2 & 1 \\ 1 & -3 \end{bmatrix}, \qquad W_1 = \begin{bmatrix} -1 & 1 \end{bmatrix}, \qquad x_0 = \begin{bmatrix} 1.0 \\ 0.1 \end{bmatrix}, \qquad y = 0.5$$

$$L = \tfrac{1}{2}\|x_3 - y\|_2^2, \qquad \eta = -0.2 \ \text{（因為更新式用「＋」）}$$

### 16.2 Forward pass

| 步驟 | 計算 | 結果 |
|---|---|---|
| $x_1 = W_0 x_0$ | $[0.2(1.0)+1(0.1),\ 1(1.0)-3(0.1)]$ | $[0.3,\ 0.7]^\top$ |
| $x_2 = \tanh(x_1)$ | $[\tanh 0.3,\ \tanh 0.7]$ | $[0.291,\ 0.604]^\top$ |
| $x_3 = W_1 x_2$ | $-0.291 + 0.604$ | $0.313$ |
| $L$ | $\tfrac12(0.313 - 0.5)^2$ | $\approx 0.017$ |

### 16.3 Backward pass

| 項 | 公式 | 數值 |
|---|---|---|
| $\dfrac{\partial L}{\partial x_3}$ | $x_3 - y$ | $-0.1869$ |
| $\dfrac{\partial L}{\partial x_2}$ | $\dfrac{\partial L}{\partial x_3}W_1$ | $[0.1869,\ -0.1869]$ |
| $\dfrac{\partial L}{\partial x_1}$ | $\dfrac{\partial L}{\partial x_2}\big(1 - \tanh^2(x_1)\big)$ | $[0.171,\ -0.1186]$ |
| $\dfrac{\partial L}{\partial W_1}$ | $\dfrac{\partial L}{\partial x_3}\,x_2$ | $[-0.054,\ -0.113]$ |
| $\dfrac{\partial L}{\partial W_0}$ | $\dfrac{\partial L}{\partial x_1}\,x_0$ | $\begin{bmatrix} 0.171 & 0.0171 \\ -0.1186 & -0.01186\end{bmatrix}$ |

⭐ **兩個投影片特別畫線的重點**：

1. **$1 - \tanh^2(x_1)$ 是一個對角矩陣**，因為 **tanh 是逐點（pointwise）運算**——和 §12.3 的 ReLU gating 矩陣是同一件事。
2. ⭐ **注意兩個相乘項的順序**。投影片直說：「記號把細節藏起來了，但**你可以把所有 index 寫出來確認順序是對的——或者就檢查維度對不對。**」

### 16.4 更新（$W^{k+1} = W^k + \eta\left(\partial L / \partial W\right)^\top$，$\eta = -0.2$）

$$W_0^{k+1} \approx \begin{bmatrix} 0.17 & 1.0 \\ 1.02 & -3.0 \end{bmatrix}, \qquad W_1^{k+1} \approx \begin{bmatrix} -0.989 & 1.02 \end{bmatrix}$$

【口頭】教授當時擔心時間不夠，說：**「這個例子在投影片裡，解答也在最後，你們可以自己走一遍。」**——所以這 9 頁是**留作練習的**。

---

## 17. 一頁速查表

| 概念 | 一句話 |
|---|---|
| **三個最佳化層級** | Black box（只有 J）／**First order（J + 梯度，實務用這個）**／Second order（+ Hessian） |
| **GD 兩個超參數** | learning rate、步數 |
| **SGD 的好處** | 快 + **雜訊是隱式正則化，能把你彈出局部極小** |
| **SGD 的壞處** | 高變異；⭐ **資料越不平衡越不穩** |
| **Momentum** | 偏向延續上一步；太高會震盪；**Adam 是代表** |
| **⭐ PyTorch 的迷思** | **它能讓任何東西可微，但不代表好最佳化**（平的地方梯度還是零） |
| **六種地形** | 凸／不連續（PyTorch 無所謂）／梯度消失／零梯度／梯度爆炸／多局部極小 |
| **不同 random seed 的變異** | **告訴你 loss landscape 有多不穩定** |
| **Evolution Strategies** | 隨機擾動取樣，往低損失方向走；**能處理零梯度函數** |
| **Gradient Clipping** | 梯度超過 m 就縮回；**有用且常用的 hack** |
| **好 activation 的三性質** | 處處連續、處處可微、**處處平滑**（ReLU 缺第三個，GeLU 三個都有；趨勢往 GeLU 走，但**沒有理論保證**） |
| **Computation graph** | 節點是函數轉換；深度學習用的是**每個節點可微的 DAG** |
| **⭐⭐ Backprop 的把戲** | chain rule 展開後**前面的項是共享的，只算一次**＝動態規劃 |
| **$\mathsf{L}$** | 層輸出對層輸入的梯度，**純局部**，由 $f'$ 給定 |
| **$g$** | cost 對 activation 的梯度，**靠遞迴** $g_{in} = g_{out}\mathsf{L}^x$ 往回傳 |
| **參數更新** | $\partial J/\partial\theta = g_{out}\mathsf{L}^\theta$ |
| **Linear layer** | forward $Wx$；backward $g_{out}W$；權重梯度 $x_{in}g_{out}$ |
| **⭐ 對稱性** | **線性層的 backward = forward，只是把權重轉置** |
| **ReLU 的 backward** | **不是 ReLU**，是由 forward activation 決定的**對角 gating 矩陣** |
| **⭐ 記憶體不對稱** | backward 需要**每一層的中間表示**，所以訓練比推論吃記憶體 |
| **任意 DAG** | 只需兩個操作：**merge（拆梯度）** 和 **branch（加梯度）** |
| **Parameter sharing** | 在圖上就是 branch ⟹ **梯度相加** |
| **Differentiable programming** | 深網路受歡迎只因兩點：**好最佳化 + 可組合** |
| **Software 2.0** | 定義一個程式空間，然後在裡面優化出你要的系統 |
| **⭐ 最大的解放** | **對任何純量 cost，你能優化圖上任何節點或任何邊**——包含輸入像素 |
| **優化輸入的小抄** | **對 pre-softmax logits 優化比對 softmax 後的機率穩定** |

---

## 18. 其他 Q&A 雜項

課末幾個零散但有用的問答：

| 問題 | 回答 |
|---|---|
| **loss 和 cost 可以互換嗎？** | 【口頭】教授誠實承認自己用得有點鬆：「照我們的定義，**loss 是單筆的，cost 是整體的**。但這有點語意之爭，**大致互換著想不會出事。**」 |
| **embedding 是什麼？** | 【口頭】「embedding 和 representation 我們常常混著用。就是**用一個訓練好的網路（有時叫 encoder），把高維/複雜的輸入映射成低維表示**。常見長度像 **1024 或 2048**。」 |
| **PyTorch 動態圖：如果我塞一個非 torch 的 UDF 進去？** | 【口頭】原則上**任何網路內的運算都必須定義成 torch 運算**（也就是你得替它定義梯度）。⭐ **實務建議：把那種東西丟到 preprocessing / data loader 裡**——前處理不需要可微（例如 data augmentation）。「我試過把簡單統計模型塞進 ML，成功率參差；常常是『技術上可行，但學起來根本不可行』。」 |
| **切換「優化參數 / 優化輸入」麻煩嗎？** | 【口頭】「幸好有 PyTorch。**這只是決定你要 freeze 哪邊的梯度而已，不難。**」 |
| **CLIP 那個 cosine similarity 就是 attention block 嗎？** | 【口頭】**不是。**cosine similarity 是 attention 的一個**元件**，但完整的 attention block 還包含**投影到共享空間**、以及**投影回你要使用資訊的空間**。 |
| **branch 上標 a、b 到底是什麼？** | 【口頭】**刻意開放式的**——可以是單純複製，也可以是把 embedding 向量切開，**任何可微的分支操作都算**。 |

---

## 19. 與其他筆記的連結

| 主題 | 去哪看 |
|---|---|
| 本講的前置（MLP、非線性、cross-entropy、tensor） | [Lecture 1 筆記](lecture01_intro_note.md) §5 |
| Backprop 的手工推導 | [`backpropagation.md`](backpropagation.md)、[`backprop_derivation.md`](backprop_derivation.md) |
| 線性層 forward/backward 的比較 | [`linear_backward_comparison.md`](linear_backward_comparison.md) |
| Gradient descent 基礎與變體 | [`gradient_descent.md`](gradient_descent.md)、[`13_7_General_steepest_descent.md`](13_7_General_steepest_descent.md) |
| ⭐ **backward 要存中間 activation** 造成的記憶體壓力，工程上怎麼解 | EfficientML [Lecture 11 TinyEngine 筆記](../efficient_ml/lecture11_tinyengine_note.md) |
| 下一講（Lecture 3）| Approximation theory——**兩層為什麼理論上能逼近任何函數，以及為什麼實務上還是要深** |
