# MIT 6.7960 第四講完整筆記：Architectures for Grids（卷積神經網路）

> **課程**：MIT 6.7960 Deep Learning, Fall 2024 — Lecture 4
> **講者**：**Sara Beery**（同 Lecture 2 的講者）
> **來源**：
>   - 官方投影片：`mit6_7960_f24_lec4.pdf`（84 頁）
>   - 課堂錄影逐字稿：[Lecture 4 錄影](https://youtu.be/bxVkZ4M-hIE)（1:23:30）
>
> **標記說明**（同 [Lecture 1 筆記](lecture01_intro_note.md)）
> - `【口頭】` = 課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 課堂提問／教授反問
> - 未標記者 = 投影片本身的內容
> - `⭐` = 重點；`⭐⭐` = 核心
>
> **本講定位**：**架構系列的第一講**。之後接 [Lecture 5 Graph Nets](lecture05_gnn_note.md) → Transformer → RNN／memory。

---

## 目錄

- [0. ⭐⭐ 本講地圖與核心主張](#0--本講地圖與核心主張)
- [1. ⭐ MLP 的三好三壞](#1--mlp-的三好三壞)
- [2. ⭐⭐ 為什麼要別的架構：假設空間那張圖](#2--為什麼要別的架構假設空間那張圖)
- [3. ⭐⭐ 一維實驗：ReLU-net vs 正確的假設](#3--一維實驗relu-net-vs-正確的假設)
- [4. ⭐ 預告：SIREN](#4--預告siren)
- [5. ⭐⭐ 從「切 patch」推導出卷積](#5--從切-patch推導出卷積)
- [6. ⭐⭐ Translation invariance / equivariance](#6--translation-invariance--equivariance)
- [7. ⭐⭐ 卷積是什麼](#7--卷積是什麼)
- [8. ⭐⭐ Fully-connected → Locally connected → Convolutional](#8--fully-connected--locally-connected--convolutional)
- [9. ⭐⭐ Toeplitz 矩陣：卷積就是「被限制的線性層」](#9--toeplitz-矩陣卷積就是被限制的線性層)
- [10. ⭐⭐ 卷積層的五種看法](#10--卷積層的五種看法)
- [11. ⭐ 疊起來會怎樣：整個 CNN 就是一個非線性濾波器](#11--疊起來會怎樣整個-cnn-就是一個非線性濾波器)
- [12. ⭐⭐ 多通道：輸入、輸出、filter bank](#12--多通道輸入輸出filter-bank)
- [13. ⭐⭐ 兩題隨堂小考（很多人會答錯）](#13--兩題隨堂小考很多人會答錯)
- [14. Feature maps 長什麼樣](#14-feature-maps-長什麼樣)
- [15. ⭐ Pooling：為什麼要池化](#15--pooling為什麼要池化)
- [16. ⭐ Downsampling、Stride、Dilation](#16--downsamplingstridedilation)
- [17. ⭐⭐ Receptive Field（感受野）](#17--receptive-field感受野)
- [18. 怎麼實作 conv](#18-怎麼實作-conv)
- [19. ⭐ 架構動物園：Encoder-Decoder / U-Net / ResNet](#19--架構動物園encoder-decoder--u-net--resnet)
- [20. 時間維度：1D conv、3D conv、video](#20-時間維度1d-conv3d-convvideo)
- [21. ⭐⭐ 不想要 shift invariant 怎麼辦：Positional Encoding](#21--不想要-shift-invariant-怎麼辦positional-encoding)
- [22. ⭐ Neural Fields：SIREN 與 NeRF](#22--neural-fieldssiren-與-nerf)
- [23. ⭐⭐ 結語](#23--結語)
- [24. 精選 Q&A](#24-精選-qa)
- [25. 一頁速查表](#25-一頁速查表)
- [26. 與其他筆記的連結](#26-與其他筆記的連結)

---

## 0. ⭐⭐ 本講地圖與核心主張

【口頭】Beery 開場說明整堂課的五段（也是投影片 p.83 的回顧）：

| 段落 | 內容 |
|---|---|
| **1. Why build better architectures?** | 為什麼要更好的架構、什麼叫「更好」 |
| **2. Convolutional layers** | 深入卷積層 |
| **3. Pyramids** | 空間金字塔、receptive field |
| **4. Architecture zoo** | 過去十年進展的架構元件 |
| **5. Neural fields and positional encodings** | 為 Transformer 那一講鋪路 |

### 0.1 ⭐⭐ 一句話的主旨

> 【口頭】「今天要開始討論**把結構（structure）直接編進模型架構裡**的各種方式——也就是**把『我們想在哪個模型空間裡做最佳化』這個假設，直接蓋進架構**。」

⭐⭐ 這句話跟 [Lecture 5](lecture05_gnn_note.md) §0.1 Isola 說的「架構設計的重點是加限制」是同一件事的兩種講法。

---

## 1. ⭐ MLP 的三好三壞

（投影片 p.3）

MLP 的積木：**linear comb. of neurons → neuron-wise nonlinearity → linear comb. of neurons**

| ✅ 好處 | 說明 |
|---|---|
| **Universal** | 萬能逼近器（← [Lecture 3](lecture03_approximation_note.md)） |
| **Simple（elegant theory）** | ⭐【口頭】「這是**少數我們真的有漂亮理論**的模型，因為它的複雜度剛好能對上我們的數學工具箱。」 |
| ⭐ **Embarrassingly parallel** | 【口頭】兩個層次：**(a)** pointwise 非線性對每個輸出神經元**完全獨立**；**(b)** dense 層的乘法可以**跨整個 batch 平行**（每筆資料獨立） |

| ❌ 壞處 | 說明 |
|---|---|
| ⭐⭐ **Weak inductive biases** | 「沒有把什麼結構或直覺烤進模型裡，它對『你想學的函數長什麼樣』幾乎不做假設」 |
| ⭐⭐ **Sample inefficient / data hungry** | 【口頭】「可能需要**比地球上存在的資料還多**，才能真的學到那個函數並泛化」 |
| **Dense 層很貴** | 【口頭】iPhone 拍的高解析照片攤平就是**幾千維**，1000×1000 的 dense 乘法計算成本極高 |

---

## 2. ⭐⭐ 為什麼要別的架構：假設空間那張圖

（投影片 p.4–6，本講**最重要**的一張概念圖）

### 2.1 三層結構

| 元素 | 意義 |
|---|---|
| **灰色大方框** | **All mappings $\mathcal{X}\to\mathcal{Y}$** —— 所有可能的函數 |
| **綠色橢圓** | **Fits the data** —— 能擬合你手上訓練資料的那些模型 |
| 🟢 **綠點** | **True solution**（真解） |
| ✖️ **藍叉** | **Learned solution**（實際學到的） |

### 2.2 ⭐【Q&A】加更多資料會怎樣？

學生答對了：**綠色橢圓會變小**。

> 「符合**全部**資料的模型變少了，所以你優化的空間縮小，學到的東西會更接近真解。」

### 2.3 ⭐⭐ 但如果拿不到更多資料呢？

> **定義一個 hypothesis space $\mathcal{F}$。**

投影片 p.6 用一顆**白色光暈的圓**畫出 $\mathcal{F}$：像是「在所有模型的空間上打一盞聚光燈」。

⭐⭐ 投影片原文（三句，很值得記）：

> **Less data, better architecture.**
>
> **We can pin down truth *either* by adding more data, or by using a more constrained architecture.**

【口頭】Beery 補：「**兩個一起做當然最理想。**」

---

## 3. ⭐⭐ 一維實驗：ReLU-net vs 正確的假設

（投影片 p.7–8，同一個一維函數 $y=f(x)$，三種資料量：1 點 / 少量 / 大量）

### 3.1 用 5 層 ReLU-net（p.7）

| 資料量 | 結果 |
|---|---|
| **1 個點** | ⭐ 學到一條**平的直線**——完全沒用 |
| **少量點** | 在有資料的那一小段稍微像一點，整體還是很差 |
| **大量點** | ⭐⭐ **在有資料的分布內擬合得非常好，但分布外（out of distribution）泛化極差** |

### 3.2 ⭐⭐ 用正確的假設 $y = ax + \sin(bx^2)$（p.8）

這是**恰好正確**的模型族（只有兩個參數 $a, b$）：

| 資料量 | 結果 |
|---|---|
| **1 個點** | 還是不夠——好模型也定不出參數在哪 |
| ⭐⭐ **很少的點** | **就已經擬合得很漂亮** |
| **大量點** | 幾乎完美，而且**分布外也泛化得很好** |

> ⭐⭐ 【口頭】「中間那張跟右邊那張的**差別其實不大**。有了精確正確的假設，只要一點點資料就能得到很好的擬合。」

### 3.3 ⭐⭐ 投影片的兩句結論（p.8）

> **Architectures enable us to generalize *outside the training distribution*.**
>
> **A good architecture is one that can represent the true function and is otherwise minimal**（而且要**好用梯度搜尋、好平行化、GPU 上跑得快**）。

---

## 4. ⭐ 預告：SIREN

（投影片 p.9–10；Sitzmann*, Martel*, Bergman, Lindell, Wetzstein, NeurIPS 2020）

**任務**：擬合一張影像 —— 也就是一個函數 $(x,y) \to l$（座標 → 亮度）

| 模型 | 結果 |
|---|---|
| ReLU-net | 糊成一團 |
| TanH-net | 幾乎全灰 |
| ReLU + P.E. | 有格狀雜訊 |
| RBF ReLU | 模糊團塊 |
| ⭐ **SIREN**（sin 當 activation） | **最接近原圖** |

⭐【口頭】為什麼？
> 「**Fourier basis 是表示影像的好基底，而 Fourier basis 的基本積木就是正弦波。**」

⚠️ **投影片自己的誠實註記**（p.9 括號）：

> **這個結果可能來自更好的逼近能力，但也可能來自更好的最佳化能力；這兩個效應在實驗上通常是耦合的。**

⭐【口頭】p.10 換成 **5 層 sin-net** 重跑一維實驗：只用幾個點，輸出就開始出現**週期性**，往真函數的方向靠。

> 【Q&A】「為什麼輸出會有週期性？」→ 學生：「因為是 sin。」→ 「對，sin 是週期的。**我們只是把『分布應該是週期的』這個 inductive bias 加進架構裡而已。**」

---

## 5. ⭐⭐ 從「切 patch」推導出卷積

（投影片 p.12–22，一整串鳥群照片的推導。這段是本講最漂亮的鋪陳）

### 5.1 【Q&A】暖身：這張圖裡有什麼？

學生答「鳥」、「天空」、「大自然」。

⭐【口頭】Beery 的評語：
> 「你們可以看到 **generality 和 specificity 的取捨**——『喔太難了，那我預測一個粗粒度的』，這是合理的退路。」
>
> **「但重點是：單一標籤通常不足以表示影像裡的語意概念。」**（想想 Times Square 的照片，或你們現在坐的這間教室——是講堂？是學生？是投影機？是 MIT？**全部都是。**）

### 5.2 步驟一：把影像切成 patch，每塊各自分類（p.13–17）

得到一張**低解析度的標籤圖**（Sky / Sky / … / Bird）。有空間資訊了，不錯。

### 5.3 ⭐ 問題（p.18）

> **What if objects don't fit neatly into these patches?**
> **How to increase the resolution of the output map?**

【口頭】「有些 patch 只有鳥翅膀的尖端在角落。整塊要算 bird 嗎？還是看多數 pixel？很模糊。」

### 5.4 ⭐ 錯誤的解法：把 patch 變小（p.19）

> **Smaller patches increase resolution, but not easy to recognize content in each small patch.**

【口頭】極端情況是**一個 pixel**，那根本沒有足夠 context 判斷語意。

### 5.5 ⭐⭐ 正確解法（投影片斜體）

> **Instead: we will use large but *overlapping* patches.**

⭐⭐ 【口頭】核心直覺：
> **「你想在預測某個 pixel 時能『看看它周圍』，但你也想做得很密（dense）。」**

### 5.6 做法（p.20–21）

- 用一個夠大的 context window（大到能看見整隻鳥）
- **滑動**這個窗
- **只預測正中央那個 pixel 的類別**，但用的是它周圍的 context
- **訓練資料** = {patch, 中心 pixel 的標籤}

（投影片 p.20 畫了四個同樣大小的 patch，紅色小框標出中心 pixel，答案分別是 Bird / Bird / Sky / Sky）

### 5.7 ⭐ 這個問題叫 semantic segmentation（p.22）

> **對每個 pixel 分類**（而不是對整張圖分類）。顏色代表 one-hot code。

【口頭】延伸：**instance segmentation** —— 還要把同一類的不同個體分開。

---

## 6. ⭐⭐ Translation invariance / equivariance

（投影片 p.23）

因為**每個 patch 都用完全相同的方式處理**，所以模型天然對輸入的平移有對稱性：

$$f(\mathrm{translate}(x)) = \mathrm{translate}(f(x))$$

> 投影片：**Translation invariance: process each patch in the same way.**
> 左下角：**An *equivariant* mapping.**

### 6.1 ⭐⭐【Q&A】為什麼這對影像是個好的 inductive bias？

學生的答案 + Beery 的補充：

> 「如果你拍一張鳥的照片，然後**把相機稍微移一下**，你拍的那個東西**還是同一個東西**。」
>
> ⭐⭐ **「所以現實世界的『類別』本身就有這種對稱性。而這個對稱性，我們是透過模型結構把它蓋進假設空間裡的。」**

⚠️ 【口頭】誠實的但書：
> **「當然我們這裡跳過了 pose（姿態），那是更複雜的因素。」**

---

## 7. ⭐⭐ 卷積是什麼

（投影片 p.24–25）

### 7.1 從 patch 到 kernel（p.24）

> **W computes a weighted sum of all pixels in the patch**
> ⭐⭐ **W is a convolutional kernel applied to the full image!**

### 7.2 定義（p.25）

> **Convolution: Linear, shift-invariant transformation**

$$x_{\mathrm{out}}[n,m] = b + \sum_{k_1,k_2=-K}^{K} w[k_1,k_2]\, x_{\mathrm{in}}[n+k_1,\ m+k_2]$$

投影片示範：小丑魚 + 一個上暗下亮的 filter → 輸出把**水平邊緣**highlight 出來。

⭐【口頭】怎麼讀這個輸出：
> 「filter 在『找』這個 pattern（這說法很擬人化）。當你有**同樣的暗→亮轉變**時，輸出值最大。」
> 「所以**暗到亮的邊界是很白（高值）**，**亮到暗的邊界是很黑（低值）**——因為那是跟 filter 最反向的地方。」

### 7.3 ⭐【口頭】一個訊號處理的小警告

> 「如果你修過訊號處理，這其實**不完全是**訊號處理裡 convolution 的定義（那個通常會**翻轉**）。這裡在神經網路裡我們把它叫 convolution，訊號處理文獻裡會叫 **cross correlation**。」
>
> ⭐ **「其中一個原因是在這些網路裡，『學會翻轉符號』很容易，所以沒差那麼多。」**

（【Q&A】有人問「一定是五角星嗎？」→「⭐ 大家什麼都用。我們課上跟講義都用五角星，**我覺得它比較漂亮**。」）

---

## 8. ⭐⭐ Fully-connected → Locally connected → Convolutional

（投影片 p.26–29，三張圖一路推）

| 投影片 | 名稱 | 說明 |
|---|---|---|
| p.26 | **Fully-connected (fc) layer** | 每個輸出依賴**全部**輸入。大起來很貴 |
| p.27 | **Locally connected network** | ⭐ **Often, we assume output is a local function of input.** |
| p.28–29 | **Convolutional neural network** | ⭐⭐ **If we use the same weights (weight sharing) to compute each local function, we get a convolutional neural network.** |

$$\mathbf{z} = \mathbf{w} \star \mathbf{x} + b$$

⭐ 【口頭】好處：「你又拿到那個 embarrassing parallelism——同一個權重矩陣套用在很多小 patch 上，可以平行做。」

---

## 9. ⭐⭐ Toeplitz 矩陣：卷積就是「被限制的線性層」

（投影片 p.30–34，⭐⭐ 本講**第二重要**的觀念）

### 9.1 兩者的矩陣形式對照

| | 式子 | 矩陣長相 |
|---|---|---|
| **Fully-connected linear layer**（p.30） | $\mathbf{x}_{\mathrm{out}} = \mathbf{W}\mathbf{x}_{\mathrm{in}} + \mathbf{b}$ | **稠密**，每格都是自由參數 |
| ⭐⭐ **Convolutional layer**（p.31） | $\mathbf{x}_{\mathrm{out}} = \mathbf{w}\star\mathbf{x}_{\mathrm{in}} + b$ | **超稀疏**：只有對角線及其上下一條有值，而且**整條對角線的值完全一樣** |

【口頭】三個權重 $w[-1], w[0], w[1]$：
> 「把它們變成線性層，就是對角線上、對角線本身、對角線下這三條。你就是**把那個梯子沿著對角線滑出去**。」

### 9.2 ⭐⭐ Toeplitz 矩陣（p.32）

$$\begin{pmatrix} a & b & c & d & e \\ f & a & b & c & d \\ g & f & a & b & c \\ h & g & f & a & b \\ i & h & g & f & a \end{pmatrix}$$

投影片的兩個 bullet：

> - **Constrained linear layer**
> - ⭐ **Fewer parameters ⟹ easier to learn, less overfitting**

### 9.3 ⭐⭐ 最漂亮的推論（p.33–34）

> ⭐⭐ **Conv layers can be applied to arbitrarily-sized inputs**
> **(generalizes beyond the training data due to an architectural structure!)**

【口頭】「因為除了對角線附近都是 0，**你可以直接把那個矩陣放大**，模型本身完全不變。」

⚠️ **對照 MLP**：
> 「如果你拿一個 MLP 套在影像上，然後想套到更大的影像——**你做不到，你根本沒有那個尺寸的權重。**」

（p.35 示範：同一個 filter 套在變色龍、熊、小丑魚上，都能得出邊緣圖。）

---

## 10. ⭐⭐ 卷積層的五種看法

（投影片 p.36：**Five views on convolutional layers**）

| # | 看法 | 一句話 |
|---|---|---|
| **1** | ⭐⭐ **Equivariant with translation** | $f(\mathrm{translate}(x)) = \mathrm{translate}(f(x))$ —— 這是**內建的假設** |
| **2** | **Patch processing** | 把顯式的平行性蓋進計算裡，計算效率好 |
| **3** | **Image filter** | 突顯輸入資料裡某些結構（而且這個結構是**學出來的**） |
| **4** | **Parameter sharing** | 同一組參數用在整個輸入上 |
| **5** | ⭐ **A way to process variable-sized tensors** | 對輸入尺寸的內建泛化 |

---

## 11. ⭐ 疊起來會怎樣：整個 CNN 就是一個非線性濾波器

（投影片 p.37：**What happens when you stack convolutional layers?**）

結構：**conv → 非線性 → conv**

> ⭐⭐ 投影片底部：**The whole CNN acts like a (nonlinear) convolutional filter!**

【口頭】兩個要點：
1. 輸出 $\mathbf{x}_L[i]$ **仍然只依賴輸入的局部結構**
2. 而且**那個函數對所有位置都一樣**（雖然中間有非線性）

⭐ **但**：疊深會讓 **receptive field 變大**（$\mathbf{x}_L[i]$ 對應的輸入 patch 比單層大很多）→ 這就是 **spatial pyramid**（→ §17）。

【Q&A】「第二層會用同一個 kernel 嗎？」→ **「不會，所以圖上顏色不一樣。你當然可以，但通常不會。權重是學出來的，而且在同一層內共享。」**

【Q&A】「一層只有幾個參數，那不是很難學嗎？」→ 【口頭】**「對！所以卷積層通常要嘛**平行放很多個不同的 filter**，要嘛**加很多層**。而實驗上（也有不錯的理論）顯示，只要這個 inductive bias 是**合適的**，你就能學得更有效率——用更少資料得到更好的擬合。」

---

## 12. ⭐⭐ 多通道：輸入、輸出、filter bank

（投影片 p.38–43）

### 12.1 Multichannel **inputs**（p.39）

$$\mathbf{x}_{\mathrm{out}} = \sum_c \mathbf{w}[c,:] \star \mathbf{x}_{\mathrm{in}}[c,:] + b[c]$$

⭐ **每個輸入通道學一組不同的權重，然後把所有通道的卷積結果加起來**，得到**一個**輸出值。

### 12.2 Multichannel **outputs**（p.40）

> **Filter bank of C filters**

$$\mathbf{x}_{\mathrm{out}}[0,:] = \mathbf{w}[0,:]\star\mathbf{x}_{\mathrm{in}} + b[0],\quad \ldots,\quad \mathbf{x}_{\mathrm{out}}[C,:] = \mathbf{w}[C-1,:]\star\mathbf{x}_{\mathrm{in}} + b[C-1]$$

⭐ **每個 filter 各自產生一個輸出維度**。

### 12.3 ⭐⭐ 通用形式：Multi-Input, Multi-Output（p.41）

$$\mathbf{x}_{\mathrm{out}}[c_2,:,:] = \sum_{c_1=1}^{C_{\mathrm{in}}} \mathbf{w}[c_1, c_2, :, :] \star \mathbf{x}_{\mathrm{in}}[c_1,:,:] + b[c_2]$$

$$\mathbf{x}_{\mathrm{in}} \in \mathbb{R}^{C_{\mathrm{in}}\times H\times W} \ \longrightarrow\ \mathbf{x}_{\mathrm{out}} \in \mathbb{R}^{C_{\mathrm{out}}\times H\times W}$$

- $c_1$ = **Input channel**，$c_2$ = **Output channel**
- $\mathbf{w}$ 的形狀：$C_{\mathrm{out}}$ 個 filter，每個是 $C_{\mathrm{in}}\times K\times K$

### 12.4 ⭐ Filter sizes 公式（p.46）

從 $\mathbf{x}_l \in \mathbb{R}^{C_l\times N\times M}$ 映到 $\mathbf{x}_{(l+1)} \in \mathbb{R}^{C_{(l+1)}\times N\times M}$，filter 空間範圍 $K_1\times K_2$：

| 量 | 公式 |
|---|---|
| ⭐⭐ **每個 filter 的參數量** | $K_1 \times K_2 \times C_l$ |
| ⭐⭐ **filter 的個數** | $C_{(l+1)}$ |

---

## 13. ⭐⭐ 兩題隨堂小考（很多人會答錯）

（投影片 p.44–45）

**設定**：$\mathbf{x}_{\mathrm{in}} \in \mathbb{R}^{3\times128\times128}$ → filter bank with **3×3 filters** → $\mathbf{x}_{\mathrm{out}} \in \mathbb{R}^{96\times128\times128}$

### Q1：每個 **filter** 有幾個參數？(a) 9 (b) 27 (c) 96 (d) 864

【Q&A】學生第一時間答 **9**。Beery：「為什麼是 9？」→「因為 3×3。」→ ⭐ **「但我們有幾個輸入通道？」**

> **答案：(b) 27** （$3\times3\times3$）

⭐⭐ 【口頭】為什麼容易搞錯（這是本講最實用的一段）：
> **「這是慣例問題。我們講 filter size 的時候，通常只講 filter 的『空間範圍』（spatial extent），因為預設我們不在 channel 空間上做卷積。」**
>
> 「當然你可以——例如 hyperspectral 輸入，你可能真的想在光譜帶上做卷積，那你可以蓋進模型。但標準做法是**每個輸入通道各自獨立學權重**，所以我們只講空間範圍。」

### Q2：這個 bank 裡有幾個 filter？(a) 3 (b) 27 (c) 96 (d) can't say

> **答案：(c) 96** —— 有 96 個輸出通道，就需要 96 個 filter，每個 filter 27 個參數。

---

## 14. Feature maps 長什麼樣

（投影片 p.42–43, 47）

> - **Each layer can be thought of as a set of C feature maps aka channels**
> - **Each feature map is an N×M image**

p.43 用 **AlexNet** 展示：conv1 的 feature maps 看得出**邊緣、某些結構**；conv2 之後**開始複雜、常常很難詮釋**。

⭐【口頭】p.47 那張圖的重點：
> 「Layer 1 的每個 filter 是 **RGB 三通道**的；Layer 2 的每個 filter 是**跨 Layer 1 全部 $C_1$ 個 feature map** 的。**只要一兩層深度，表示就會迅速變得極度複雜。**」

### 14.1 ⭐【Q&A】怎麼確保每張 feature map 都不一樣？

> 「這**不是天生就有的機制**。有些人用 **dropout** 之類的 regularization。」
>
> ⭐ 「但通常模型**自己就會**學得不一樣，因為那給它更多容量。這裡沒有任何東西**逼**它們保持相同——那它為什麼不去最大化它能有的變異呢？」
>
> ⚠️ 「不過視資料、架構、最佳化而定，**有時候會發生 feature collapse**，某些東西就被忽略掉了。那時你可以再蓋一些機制進去避免。」

---

## 15. ⭐ Pooling：為什麼要池化

（投影片 p.48–53）

### 15.1 兩種 pooling

| 名稱 | 式子 |
|---|---|
| **Max pooling** | $y_j = \max_{j\in\mathcal{N}(j)} h_j$ |
| **Mean pooling** | $y_j = \frac{1}{\vert\mathcal{N}\vert}\sum_{j\in\mathcal{N}(j)} h_j$ |

順序：**conv（w, b）→ pointwise 非線性 → pool**

⭐【口頭】做 pooling 的動機之一：**維度會長得很快很大**。

### 15.2 ⭐ 為什麼？(1) 空間上：對小平移穩定（p.50–52）

> **Pooling across spatial locations achieves stability w.r.t. small translations**
>
> ⭐ **large response regardless of exact position of edge**

【口頭】「邊緣稍微移動，取 max 之後**還是會辨識出那裡有一條邊**。」

### 15.3 ⭐⭐ 為什麼？(2) 通道上：得到別種不變性（p.53）

> **Pooling across feature channels (filter outputs) can achieve other kinds of invariances**
>
> ⭐⭐ **large response for any edge, regardless of its orientation**

【口頭】對**不同方向的邊緣 filter** 做 max pooling → 得到一個「**對邊緣方向不變**」的偵測器。

### 15.4 ⭐【Q&A】pooling 和 conv 的關係

（課末問答）
> **「Pooling 基本上就跟跑一個卷積層一樣，只是你不學權重，你就用 max 函數或 average 函數。所以它是一個受限的、不學習的卷積層。」**

---

## 16. ⭐ Downsampling、Stride、Dilation

（投影片 p.54–60）

### 16.1 標準 building block（p.54–55）

$$f(\mathbf{x}) = f_L(\ldots f_2(f_1(\mathbf{x})))$$

每個 block：**Filter → ReLU → Pool**，或 **Filter → ReLU → Downsample**，最後 **Classify**。

⭐【口頭】為什麼要 downsample：
> 「特別是做分類這種**只想要一個低秩表示**（一個字、一個 one-hot）的任務，我們**不需要在網路的每一層都保有那麼高維的輸入結構**。」

### 16.2 ⭐ Strided operations（p.58–59）

> **Strided operations combine a given operation (convolution or pooling) and downsampling into a single operation.**

【口頭】「不要每次只滑一格，把 **stride** 調大。」

⚠️ 實務建議：
> 「通常我們會讓 stride **有重疊**，這樣不會丟掉原始輸入的 context。比如 kernel 是 7×7，stride 可能是 5 這樣。」

### 16.3 ⭐ Dilated filter（p.60）

> **Covers a large receptive field with fewer parameters.**

【口頭】「本質上就是**把 filter 裡某些值設成 0**，然後把這件事烤進模型。你用更少參數拿到更大的空間 context——**彈性變少，但 context window 變大**，這是可以優化的取捨。」

---

## 17. ⭐⭐ Receptive Field（感受野）

（投影片 p.61–63）

### 17.1 定義

⭐⭐ 【口頭】Beery 給的定義最清楚：

> **「Receptive field 就是說：初始輸入的哪些部分，會對輸出的這個部分產生任何影響。」**
>
> ⚠️ **「在那個 receptive field 之外，模型完全沒有輸入資訊。就算外面剛好有極有用的線索，模型也拿不到。」**

### 17.2 ⭐ 怎麼變大 receptive field（【Q&A】）

| 方法 | 說明 |
|---|---|
| **1. 加大 kernel / filter size** | 直接 |
| ⭐ **2. 加深** | 疊層數也會變大 |
| **3. Dilation** | §16.3 |

投影片 p.62 用兩個小圖示範：**conv → relu → conv** 之後，$x_2[3]$ 的感受野明顯比 $x_1[5]$ 大。

### 17.3 ⭐ 視覺化證據（p.63）

AlexNet / VGG16 / ResNet18 的中間層 activation：

| 深度 | 樣子 |
|---|---|
| **早期層（conv1）** | ⭐ **非常細緻**（因為卷積很局部） |
| **後期層（conv5 / conv13 / block8）** | ⭐ **非常瀰散（diffuse）** |

【口頭】「因為 receptive field 變大，每個輸出維度都在匯集非常複雜的資訊。所以你得到**語意上更有意義、但對細微紋理變化較不敏感**的 feature activation。」

【Q&A】「那些圖是同一個像素尺寸嗎？」→ 「**我們把它們縮放成一樣大了。** 實際上很看架構——ResNet 到 block 8 空間範圍低很多，但通道多很多。」

### 17.4 ⭐【Q&A】怎麼選架構參數？（很實用的一段）

> 「架構的搜尋空間**非常大**。光是線性層 + 卷積層，加上不同輸入大小、深度、通道數，就已經是巨大的搜尋空間了。有一整個領域叫 **neural architecture search**，就是在架構空間上做最佳化——但訓練起來**很貴**。」
>
> ⭐⭐ **十年來的經驗趨勢**：
> - **原始 AlexNet**：filter 很大（空間範圍大）、層數比較少
> - **現在**：filter **小**（5 或 7），但**疊得很深** ⟹ 最後 receptive field 還是很大

### 17.5 ⭐【Q&A】圖片複雜度有差嗎？（極實用）

> 「實驗上**有**。像 CIFAR 這種很簡單的小圖，網路本身可以簡單很多。」
>
> ⭐⭐ 「但如果**細粒度的特徵細節真的很重要**（例如我們做物種辨識），那**比什麼都重要的往往是輸入影像的尺寸**。」
>
> 「我們常常會在前處理把影像縮小再送進網路。但我們發現，**把輸入尺寸放大、然後把計算量縮小一點，有時候是更好的取捨**——如果你真的在乎細粒度特徵，你不會想丟掉那些細節。」

### 17.6 ⭐⭐【Q&A】手工設計 filter vs 學出來的 filter（本講最好的一段故事）

學生問：訊號處理是精心手工設計 filter，這裡是學出來的，取捨是什麼？

> 【口頭】「你講的其實是十、十五年前這個領域發生的**巨大典範轉移**。以前大部分工作叫 **feature engineering**——你手工打造這些結構，再組成一個大系統。」
>
> ⭐ **「當我們開始看到『端到端學出來的 filter』預測力更好時，對很多研究者來說在情感上簡直是一場災難。你會想：不會吧，我一輩子都在學怎麼精雕細琢這些 filter，我不想只是把一堆資料塞進一個結構然後它就更好。」**
>
> ⭐⭐ **取捨的結論**：
> - **資料不夠** ⟹ **手工設計、加更多知識、更多 inductive bias 比較好**（可以加更多限制，用更少資料優化得更好）
> - **資料很多** ⟹ 學出來的比較好，因為**我們對「什麼是最佳 filter」其實沒有自己以為的那麼懂**
>
> ⭐ 「設計**單一個** filter 很直覺。但你要怎麼**手工設計一個 128 層網路裡第 7 個卷積層的某個通道**？系統複雜到我們很難建立直覺。」

---

## 18. 怎麼實作 conv

（投影片 p.64）

**基本實作三步驟**：

| 步驟 | 做什麼 |
|---|---|
| **1. `im2col`** | $N\times M\times C \longrightarrow N_{\mathrm{patches}} \times (K\times K\times C)$ |
| **2. `bmm`** | batch `matmul` with kernel |
| **3. `col2im`** | $N_{\mathrm{patches}} \times (K\times K\times C) \longrightarrow N\times M\times C$ |

**或**：fft signal processing stuff…

⭐ **好用的函式庫**：`timm` — https://github.com/rwightman/pytorch-image-models

【Q&A】「一定要真的建那個稀疏矩陣嗎？」→ **「不一定。為了效率跟平行性，我們通常就真的是 patch-wise 去算。」**

---

## 19. ⭐ 架構動物園：Encoder-Decoder / U-Net / ResNet

（投影片 p.65–72）

### 19.1 分類網路（p.66）

**Filter → ReLU → Downsample**，重複，最後 Classify → `"heron"`。

### 19.2 Encoder-Decoder（p.67）

| 部分 | 做什麼 |
|---|---|
| **Encoder** | Convolution → Nonlinearity → **Subsample**，一路降到低維的 $\mathbf{z}$ |
| **Decoder** | Convolution → Nonlinearity → **Upsample**，把 $\mathbf{z}$ 還原成影像 |

⭐ 訓練訊號：**讓 decode 出來的輸出盡量吻合 encode 的輸入**。

【口頭】用途：
- **VAE（variational autoencoder）**的基礎
- **Masked autoencoder**（不過那個用 attention 而非 conv 來跨空間共享資訊）
- ⭐ **兩半可以拆開單獨用**：encoder 拿去做影像表示，decoder 拿去**生成**影像（呼應 [Lecture 2](lecture02_backprop_note.md) 的 differentiable programming）

### 19.3 ⭐ 中間那個瓶頸：**a blessing and a curse**

> ⭐⭐ 【口頭】「它**逼**模型學到語意上有用的東西，**但也表示**只有『降採樣再升採樣』的架構通常**做不出跟輸入同樣尺寸的精確輸出**——你已經丟掉大量細粒度的空間結構了。」

### 19.4 完全不降採樣的 image-to-image（p.69, 72）

**conv → relu → conv → softmax**，全程保持解析度。

⚠️ 【口頭】「但因為你保留整個維度，**計算會非常貴**，模型會非常大。」

### 19.5 ⭐⭐ U-Net（p.68, 70）

> **Skip connection**：把 encoder 某一層的東西**用 identity 直接接到** decoder 對應的位置。

⭐⭐ 【口頭】直覺：
> **「模型不再被迫把一大堆細粒度空間細節硬塞進一個低秩、沒有空間維度的 embedding（那幾乎是不可能的）。它可以透過那個 information bottleneck 學語意細節，同時透過 skip connection 拿到大量空間細節。」**

⭐ 「U-Net 是個**很老**的架構，但至今**仍然是 semantic segmentation 最廣泛使用的架構之一**。遙測領域（例如 land cover mapping）一直在用。」

### 19.6 ⭐⭐ ResNet（p.71）

**Residual connection**：

$$\mathbf{x}_{\mathrm{out}} = F(\mathbf{x}_{\mathrm{in}}) + \mathbf{x}_{\mathrm{in}}$$

要改變維度時：

$$\mathbf{x}_{\mathrm{out}} = F(\mathbf{x}_{\mathrm{in}}) + \mathbf{W}\mathbf{x}_{\mathrm{in}}$$

⭐ 【口頭】背景：「ResNet 是何愷明（Kaiming He）發明的，他現在也是 MIT 教授。這個架構**太神了，我們現在還一直在用，而它已經十歲了**。」

⭐⭐ **最漂亮的直覺**（本節重點）：

> **「每一層都有這條 skip connection，也就是一條 identity 路徑，讓你可以選擇要不要把資料跑過這一層。」**
>
> ⭐⭐ **「所以這其實是一個能學到自己架構的模型——它可以學到最佳深度。你可以說『我只想跑過其中三層，剩下的我就走 skip connection 出去』。模型自己就隱式地學到了這個任務的最佳深度，不需要顯式地把一堆不同架構並排訓練一遍。」**

【Q&A】「skip 是變成另一個 channel，還是二選一？」→
> 「基本上是二選一——假設層的輸出和 residual connection **同尺寸，然後直接相加**。也有實作會加上學習的權重。」
>
> ⭐ 「同樣的 skip connection 在 transformer 的 self-attention 裡也會冒出來。**重點就是『學會怎麼組合資訊』這件事本身很有價值。**」
>
> 「而且因為線性層／卷積層的特性，**就算只是加起來，下一層也有辦法學會忽略某些維度**。例如那個 $\mathbf{W}$ 可以被學成全 0，模型想丟掉就丟掉。」

---

## 20. 時間維度：1D conv、3D conv、video

（投影片 p.73–75）

| 情境 | 做法 |
|---|---|
| **純量時間序列**（例如溫度感測器） | ⭐ 在**時間維度**上做 1D convolution |
| **Video** | RGB × H × W × T ⟹ **4 維輸入**；用 **3D convolution** 在**空間 + 時間**上卷積 |

⭐ 用途：**action recognition（動作辨識）**

### 20.1 ⭐⭐【Q&A】video 的實務現實（課末，很有料）

> 「**很少**有情況下時間空間上的卷積在計算上是划算的。看看 video 的 SOTA 模型，通常都在**時間上降採樣**，因為大部分動作你用『**bag of images**』就能辨識。」
>
> ⭐ 「Alyosha Efros 有篇好論文：拿任何動作辨識問題，用『**需要幾張 frame 才能正確辨識這個動作**』來代表這個問題的複雜度。**而大部分動作，一張 frame 就夠了。**」
>
> ⭐⭐ **實務做法叫 "fast and slow"**：
> - 一條**低維、空間（甚至時間）上降採樣**的通道，**跑得很頻繁** → 建模動作
> - 一條**高解析度**通道，**只週期性地跑**（預先定好，或由模型決定何時跑）→ 建模細粒度語意
>
> 【口頭】結語：**「Video 很迷人，也很複雜、很貴——存很貴、訓練很貴、推論也很貴。是一團亂。」**

---

## 21. ⭐⭐ 不想要 shift invariant 怎麼辦：Positional Encoding

（投影片 p.76–77）

### 21.1 ⭐ 為什麼會不想要

⭐⭐ 【口頭】Beery 給的例子非常精準：

> **「你不會想讓『把杯子拿起來』和『把杯子放下』被當成同一個動作。」**
>
> 「如果 frame rate 很低，你只有這張跟那張，你會希望『哪個先發生』是有意義的。**Translation invariance 不是你永遠都想要的東西。**」

（另一個例子在 [Lecture 5](lecture05_gnn_note.md) §6.3 被 Isola 引用：**影像的上面比較可能是天空**。）

### 21.2 ⭐⭐ 兩個解法（投影片 p.76）

> **1. Use an architecture that is not shift invariant (e.g., MLP)**
>
> **2. Add location information to the *input* to the convolutional filters — this is called *positional encoding***

【口頭】對選項 1 的評語：「但如我們說過的，**MLP 要學複雜函數需要幾乎無限的資料**，很難。」

⭐ 對選項 2：**「這個是真正被證明是贏家的做法。」**

### 21.3 做法（p.77）

輸入端同時有 **`pos`（位置編碼）** 和 **`signal`（原始訊號）** 兩欄，一起送進 $\mathbf{w}$。

⚠️ ⭐ **關鍵細節**：
> **「這個 positional encoding 不是學出來的，是你『建構』出來然後加到輸入上的。」**
> 然後你學一組額外的權重，把它併進卷積裡。

---

## 22. ⭐ Neural Fields：SIREN 與 NeRF

（投影片 p.78–81）

### 22.1 定義（p.78）

| 詞 | 意思 |
|---|---|
| **Field（物理學）** | 隨**空間與時間座標**變化的物理量 |
| ⭐ **Neural field** | **一個（部分或完全）由神經網路參數化的 field** |

$$\Phi: \mathbb{R}^2 \to \mathbb{R}, \qquad l = \Phi(x, y)$$

### 22.2 ⭐ SIREN（p.79；Sitzmann, Martel et al. 2020）

> **CNN applied *per-pixel* to map from a coordinate grid to a color**
>
> ⭐⭐ **Can take continuous coordinates as input!**

⭐【口頭】重點：
> **「你其實是在『學一張單一影像的模型』——學一個從影像位置到顏色的函數映射。」**
>
> 「輸入就是**位置**（也就是 positional encoding）。因為是函數映射，所以座標**可以是連續的** ⟹ 我們可以學到**輸入空間的函數式表示**，而不只是格點式表示。」

### 22.3 ⭐ NeRF（p.80；Mildenhall, Srinivasan, Tancik et al., ECCV 2020）

> **Conv net applied to map from 5-D coordinate grid to a color + volumetric density**

| | |
|---|---|
| **5D Input** | Position + Direction：$(x, y, z, \theta, \phi)$ |
| **Output** | Color + Density：$(\mathrm{RGB}\,\sigma)$ |

⭐ 用途：**從原始訓練資料沒有的視角，生成同一個物體長什麼樣**（場景 walkthrough）。

### 22.4 ⭐⭐【Q&A】NeRF 的三個誠實限制（很值得記）

> **1. NeRF 的輸出不是場景的 3D 模型**
> 「它是『從不同方向拍會長什麼樣的影像』，**那跟 3D 模型不一樣**。實際應用其實很想從稀疏單目影像得到穩健的 3D 模型，最近才開始有人把這兩件事接起來。」
>
> **2. 資料要很多**
> 「經典 NeRF **如果只有幾張稀疏影像，很難擬合好**。」
>
> **3. 只對靜止的東西好用**
> 「如果是**搖晃的樹**，數學上就不太會收斂，因為你拍的東西本身位置隨時間在變。」

⭐ 【口頭】花絮：「Woods Hole 有人做了一個**水下的 NeRF**，處理光在水裡的折射。很讚。」

---

## 23. ⭐⭐ 結語

（投影片 p.82：**Concluding remarks**）

> **Convolution is a fundamental operation for image processing.**
>
> ⭐⭐ **It just means: chop up the image into patches and apply the same function to each patch.**
>
> ⭐ **This concept appears in almost all modern architectures, such as CNNs, transformers, NeRFs, and more.**

【口頭】補充：
> 「其實是『處理任何**有結構**的東西』的基本運算——時間、或時間+空間+多輸入維度都行。比如你想對**每週從全地球拍的 hyperspectral 衛星影像**做卷積也可以。」
>
> ⭐ 「這個概念在 **transformer** 裡也出現，只是那裡的 patch-wise 運算不一定是卷積，**它用的是 attention**。」（→ 這正好接上 [Lecture 5](lecture05_gnn_note.md) §12：**transformer = attention 當 AGGREGATE 的 GNN**）

---

## 24. 精選 Q&A

| 問題 | 答案 |
|---|---|
| **卷積一定是正方形的嗎？** | ⭐ 「**通常是**，那是完全標準的。但不一定要——卷積可以定義成非正方形。只是通常我們看到的是正方形。」 |
| **降採樣之後就回不去原尺寸了吧？** | ⭐ 「**不一定**，但你**可能做得不好**。」（→ §19.5 U-Net 就是答案） |
| **Receptive field 跟 Gaussian blur 是類似的計算嗎？** | 「高層次的直覺有點像，但**計算上差蠻多的**。」 |
| **輸入不只是圖，還有純量 metadata（例如 GPU 座標）怎麼辦？** | ⭐ 「很多做法。最明顯的：先把影像跑過第一個卷積層，然後對那個位置資訊做另一種 conv 或表示，再**concat 到整張輸入影像**，或**稍晚一點才引入網路**。有很多方式在訓練時把額外資訊接進去。」 |
| **video 可以不加維度，直接把每一 frame 當作額外的 RGB channel 嗎？** | ⭐ 「**你可以，但會很貴。**」（見 §20.1） |
| **NeRF 可以做超解析（upscaling）嗎？** | 「可以。有人用它來提升點雲表示的解析度——不完全是原始 NeRF 架構，但有改編版本在做提升空間解析度。」 |

---

## 25. 一頁速查表

### 25.1 ⭐⭐ 心法

| 主張 | 內容 |
|---|---|
| ⭐⭐ **架構的作用** | **Architectures enable us to generalize *outside the training distribution*** |
| ⭐⭐ **兩條路** | **We can pin down truth *either* by adding more data, or by using a more constrained architecture** |
| ⭐ **好架構的定義** | **能表示真函數，其餘部分盡量最小**；而且好用梯度搜、好平行、GPU 快 |
| ⭐⭐ **卷積的一句話定義** | **把影像切成 patch，對每個 patch 套用同一個函數** |
| ⭐⭐ **卷積的本質** | **一個被限制的線性層（Toeplitz 矩陣）** |
| ⭐ **手工 vs 學習** | **資料少 → 手工／強 inductive bias 贏；資料多 → 學出來的贏** |

### 25.2 ⭐⭐ 核心公式

| 名稱 | 式子 |
|---|---|
| **卷積** | $x_{\mathrm{out}}[n,m] = b + \sum_{k_1,k_2=-K}^{K} w[k_1,k_2] x_{\mathrm{in}}[n+k_1, m+k_2]$ |
| **Conv layer** | $\mathbf{z} = \mathbf{w}\star\mathbf{x} + b$ |
| ⭐⭐ **Multi-in Multi-out** | $\mathbf{x}_{\mathrm{out}}[c_2,:,:] = \sum_{c_1} \mathbf{w}[c_1,c_2,:,:] \star \mathbf{x}_{\mathrm{in}}[c_1,:,:] + b[c_2]$ |
| **Translation equivariance** | $f(\mathrm{translate}(x)) = \mathrm{translate}(f(x))$ |
| **Max pooling** | $y_j = \max_{j\in\mathcal{N}(j)} h_j$ |
| **Mean pooling** | $y_j = \frac{1}{\vert\mathcal{N}\vert}\sum_{j\in\mathcal{N}(j)} h_j$ |
| ⭐⭐ **Residual connection** | $\mathbf{x}_{\mathrm{out}} = F(\mathbf{x}_{\mathrm{in}}) + \mathbf{x}_{\mathrm{in}}$（改維度：$+\mathbf{W}\mathbf{x}_{\mathrm{in}}$） |
| **Neural field** | $l = \Phi(x, y)$ |

### 25.3 ⭐⭐ 參數量速查（考試／面試最常錯）

| 量 | 公式 |
|---|---|
| ⭐⭐ **每個 filter 的參數量** | $K_1 \times K_2 \times C_{\mathrm{in}}$ ← **別忘了乘輸入通道數！** |
| ⭐⭐ **filter 個數** | $C_{\mathrm{out}}$ |
| **一層總參數量** | $K_1 K_2 C_{\mathrm{in}} C_{\mathrm{out}}$（+ $C_{\mathrm{out}}$ 個 bias） |
| **範例** | 3×128×128 → 3×3 filters → 96×128×128：**每個 filter 27 個參數，共 96 個 filter** |

### 25.4 ⭐ 卷積層五種看法

1. **Equivariant with translation**
2. **Patch processing**
3. **Image filter**
4. **Parameter sharing**
5. **A way to process variable-sized tensors**

### 25.5 ⭐ 架構元件對照

| 元件 | 解決什麼 |
|---|---|
| **Pooling（空間）** | 對小平移穩定 |
| ⭐ **Pooling（通道）** | 得到別種不變性（例如**對邊緣方向不變**） |
| **Downsampling** | 維度不用一路撐到底 |
| **Strided op** | ⭐ **把 conv/pool 和 downsample 合成一個運算** |
| **Dilated filter** | ⭐ **更少參數、更大 receptive field** |
| **Encoder-Decoder** | 低維表示；兩半可拆開單獨用 |
| ⭐⭐ **U-Net（skip connection）** | **兼得語意瓶頸與空間細節**——segmentation 至今主力 |
| ⭐⭐ **ResNet（residual）** | **模型自己學最佳深度** |
| **3D conv** | 時間維度（action recognition） |
| ⭐⭐ **Positional encoding** | **刻意打破 shift invariance**（建構的，不是學的） |

### 25.6 ⭐ 三種 receptive field 放大法

**加大 kernel** ／ **加深層數** ／ **dilation**

---

## 26. 與其他筆記的連結

| 主題 | 去哪看 |
|---|---|
| **MLP、非線性、Softmax／Cross-Entropy** | [Lecture 1 筆記](lecture01_intro_note.md) §5 |
| **Backprop、computation graph、differentiable programming** | [Lecture 2 筆記](lecture02_backprop_note.md) §7, §14 |
| ⭐ **Universal approximation（本講 §1 的「+ Universal」出自這裡）** | [Lecture 3 筆記](lecture03_approximation_note.md) §5–§6 |
| ⭐⭐ **「不要追求 universality，要加限制」** | [Lecture 5 GNN 筆記](lecture05_gnn_note.md) §0.1 —— 本講 §2 是同一主張的另一種說法 |
| ⭐⭐ **CNN 是格點圖上的 GNN** | [Lecture 5 GNN 筆記](lecture05_gnn_note.md) §4, §11 |
| ⭐ **Positional encoding 在圖上的版本（Laplacian 特徵向量）** | [Lecture 5 GNN 筆記](lecture05_gnn_note.md) §20 |
| **Transformer（＝ attention 當 aggregation 的 patch 處理）** | 課程後續；另見 `transformers.md` |
| **CNN 的另一份筆記（別的教材）** | [`convolutional_neural_nets.md`](convolutional_neural_nets.md) |
| ⭐ **實務上的 CNN 工程取捨（模型壓縮、NAS）** | EfficientML [NAS 筆記](../efficient_ml/lecture07_08_nas_note.md)、[剪枝筆記](../efficient_ml/lecture03_04_pruning_note.md) |
