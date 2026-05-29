1. [Foundations of Learning](https://visionbook.mit.edu/part_foundation_learning.html)
2. [14  Backpropagation](https://visionbook.mit.edu/backpropagation.html)

# 14  反向傳播 (Backpropagation)

## 14.1 簡介

神經網路的核心思想之一是將計算分解為一系列的「層」（layers）。在本章中，我們將把層視為模組化的區塊（modular blocks），它們可以串聯成一個**計算圖**（computation graph）。[圖 14.1](#fig-backpropagation-simple_MLP) 展示了來自[第 12 章](https://visionbook.mit.edu/neural_nets.html) 的雙層多層感知器（MLP）的計算圖。

![圖 14.1：在本章中，我們將神經網路視覺化為一系列層，並稱之為計算圖。](https://visionbook.mit.edu/figures/backpropagation/simple_MLP.png)

每一層接收某些輸入並將其轉換為某些輸出。我們稱之為通過該層的**前向傳播**（$\texttt{forward}$ pass）。如果該層具有參數，我們將把參數視為無參數轉換的*輸入*： $$ \begin{aligned} \mathbf{x}_{\texttt{out}}&= f(\mathbf{x}_{\texttt{in}},\theta)
\end{aligned} $$ 在圖形上，我們將像下方（[圖 14.2](#fig-backpropagation-mod_block_forward)）所示描繪層的前向操作。

![圖 14.2：神經網路層的前向操作。](https://visionbook.mit.edu/figures/backpropagation/mod_block_forward.png)

> [!NOTE]
> 我們將使用該顏色來指示*自由*參數，這些參數是透過學習設定的，而非任何其他處理的結果。

學習問題是尋找能夠實現所需映射的參數 $\theta$ 。通常，我們將透過梯度下降法來解決這個問題。本章的問題是，我們如何計算梯度？

**反向傳播**（Backpropagation）是一種能夠有效計算損失相對於計算圖中每一個參數的梯度的演算法。它依賴於一個特殊的新操作，稱為 `backward`（反向傳播），就像 `forward`（前向傳播）一樣，它可以為每一層獨立定義，並與圖中的其餘部分隔離運作。但在開始定義 `backward` 之前，我們將先建立一些關於反向傳播所利用之關鍵技巧的直覺。

## 14.2 反向傳播的巧妙之處：計算的重複利用

首先，我們考慮一個簡單的計算圖，它是由函數組成的鏈條： $f_L \circ f_{L-1} \circ \cdots f_2 \circ f_1$ ，其中每個函數 $f_l$ 都由參數 $\theta_l$ 參數化。

> [!NOTE]
> 例如，這樣的計算圖可以表示一個多層感知器（MLP），我們將在下一節中看到。

我們的目標是相對於損失函數 $\mathcal{L}$ 來最佳化參數。損失可以被視為計算圖中的另一個節點，它接收 $\mathbf{x}_L$ （ $f_L$ 的輸出）並輸出純量 $J$ ——即損失值。該計算圖如下所示（[圖 14.3](#fig-backpropagation-composed_modules)）。

![圖 14.3：基本的序列式計算圖。](https://visionbook.mit.edu/figures/backpropagation/composed_modulesL.png)

> [!NOTE]
> 這個計算圖是一棵窄樹；參數存在於長度為 1 的分支上。當我們將數據和參數繪製為節點，而將函數繪製為邊時，這會更容易看清： ![將計算圖繪製為樹。](https://visionbook.mit.edu/figures/backpropagation/cgraph_tree.png) 參數以及輸入訓練數據是計算圖的葉節點（leaves）。

我們的目標是更新所有用藍色高亮標記的值： $\theta_1$ ， $\theta_2$ ，依此類推。為此，我們需要計算梯度 $\frac{\partial J}{\partial \theta_1}$ ， $\frac{\partial J}{\partial \theta_2}$ 等。這些梯度中的每一個都可以透過連鎖律（chain rule）進行計算。以下是為 $\theta_1$ 和 $\theta_2$ 的梯度寫出的連鎖律公式： $$ \begin{aligned}
\frac{\partial J}{\partial \theta_1}
&=
 \frac{\partial J}{\partial \mathbf{x}_L}
 \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_{L-1}}
 \cdots
 \frac{\partial \mathbf{x}_3}{\partial \mathbf{x}_2}
\,
\frac{\partial \mathbf{x}_2}{\partial \mathbf{x}_1}
\frac{\partial \mathbf{x}_1}{\partial \theta_1}
\\
\frac{\partial J}{\partial \theta_2}
&=
 \frac{\partial J}{\partial \mathbf{x}_L}
 \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_{L-1}}
 \cdots
 \frac{\partial \mathbf{x}_3}{\partial \mathbf{x}_2}
\,
\frac{\partial \mathbf{x}_2}{\partial \theta_2}
\end{aligned} $$ 我們注意到，大多數項都是共享的，而不是需要分別評估這兩個方程式。我們只需要評估一次這個乘積，然後就可以用它來計算 $\frac{\partial J}{\partial \theta_1}$ 和 $\frac{\partial J}{\partial \theta_2}$ 。現在請注意，這種重複利用的模式也可以以相同的方式應用於 $\theta_3$ ， $\theta_4$ ，依此類推。這就是反向傳播的全部奧妙所在：與其獨立地計算每一層的梯度，不如觀察到它們共享了許多相同的項，因此我們不妨只計算每個共享項一次並重複利用它們。

> [!NOTE]
> 在計算科學中，這種策略通常被稱為**動態規劃**（dynamic programming）。

## 14.3 通用層的反向傳播操作

為了想出一個重新利用所有共享計算的通用演算法，我們先單獨觀察一個通用層，看看我們需要什麼來更新其參數（[圖 14.4](#fig-backpropagation-generic_layer_g_L)）。

![圖 14.4：計算圖中的一個通用層。大括號表示為了評估 $\mathbf{g}_{\texttt{out}}$ 、 $\mathbf{L}$ 和 $\mathbf{g}_{\text{in}}$ ，我們需要考慮的計算圖部分。](https://visionbook.mit.edu/figures/backpropagation/generic_layer_g_L.png)

在這裡，我們引入了兩個新的簡寫： $\mathbf{L}$ 和 $\mathbf{g}$ 。它們代表偏導數（partial derivatives）的陣列（如下所定義），並且是我們在進行反向傳播時需要追踪的關鍵陣列。它們定義為： $$ \begin{aligned}
 \mathbf{g}_l&\triangleq \frac{\partial J}{\partial \mathbf{x}_l} &&\quad\quad \triangleleft \quad \text{代價相對於 } \mathbf{x}_l \text{ 的梯度} \quad [1 \times |\mathbf{x}_l|]\\
 \mathbf{L}&\triangleq \frac{\partial \mathbf{x}_{\texttt{out}}}{\partial [\mathbf{x}_{\texttt{in}}, \theta]} &&\quad\quad \triangleleft \quad \text{該層的梯度}\\
 &\quad\quad \mathbf{L}^{\mathbf{x}} \triangleq \frac{\partial \mathbf{x}_{\texttt{out}}}{\partial \mathbf{x}_{\texttt{in}}} &&\quad\quad \triangleleft \quad \text{相對於層輸入數據的梯度} \quad [|\mathbf{x}_{\texttt{out}}| \times |\mathbf{x}_{\texttt{in}}|]\\
 &\quad\quad \mathbf{L}^{\theta} \triangleq \frac{\partial \mathbf{x}_{\texttt{out}}}{\partial \theta} &&\quad\quad \triangleleft \quad \text{相對於層參數的梯度} \quad [|\mathbf{x}_{\texttt{out}}| \times |\theta|]
\end{aligned} $$ > [!NOTE]
> 所有這些陣列都代表了*在單一工作點*（即數據和參數的當前值）上的梯度。

這些陣列提供了一個簡單的公式來計算我們所需的梯度 $\frac{\partial J}{\partial \theta}$ ，以便「更新」 $\theta$ 以最小化代價： $$ \begin{aligned}
 \frac{\partial J}{\partial \theta} &= \underbrace{\frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}}}_{\mathbf{g}_{\texttt{out}}} \underbrace{\frac{\partial \mathbf{x}_{\texttt{out}}}{\partial \theta}}_{\mathbf{L}^{\theta}} = \mathbf{g}_{\texttt{out}}\mathbf{L}^{\theta}\\
 \theta^{i+1} &\leftarrow \theta^{i} - \eta \Big(\frac{\partial J}{\partial \theta}\Big)^\mathsf{T}&&\quad\quad \triangleleft \quad \texttt{更新}
\end{aligned} \tag{14.1} $$ > [!NOTE]
> 轉置（transpose）是因為按照慣例， $\theta$ 是一個列向量，而 $\frac{\partial J}{\partial \theta}$ 是一個行向量；參見符號標記[章節](https://visionbook.mit.edu/\notations.html) 。

接下來的問題很明確：我們如何為每一層 $l$ 獲得 $\mathbf{g}_l$ 和 $\mathbf{L}^{\theta}_l$ ？

計算 $\mathbf{L}$ 是一個完全局部的過程：對於每一層，我們只需要知道其導數的函數形式 $f^{\prime}$ ，然後在工作點 $[\mathbf{x}_{\texttt{in}}, \theta]$ 處對其進行評估，即可獲得 $\mathbf{L}= f^{\prime}(\mathbf{x}_{\texttt{in}},\theta)$ 。

計算 $\mathbf{g}$ 則稍微複雜一些；它需要評估連鎖律，並且取決於 $\mathbf{x}_{\texttt{out}}$ 和 $J$ 之間的所有層。然而，這可以透過迭代來計算：一旦我們知道了 $\mathbf{g}_l$ , 計算 $\mathbf{g}_{l-1}$ 就只需要再進行一次矩陣相乘！這可以透過以下遞迴關係式來總結： $$ \begin{aligned} \mathbf{g}_{\texttt{in}} &= \mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{x}} &&\quad\quad \triangleleft \quad \text{誤差的反向傳播}
\end{aligned} \tag{14.2} $$ 這個遞迴關係式是反向傳播的本質：它將誤差訊號（梯度）沿著網路反向發送，從最後一層開始，並迭代地應用[等式 14.2](#eq-backpropagation-backward) 來計算每個前一層的 $\mathbf{g}$ 。

> [!NOTE]
> 像 PyTorch 這樣的深度學習庫，每個變量（數據、激活值、參數）都有一個與之關聯的欄位。這個欄位代表每個變量 $v$ 的 $\frac{\partial J}{\partial v}$ 。

我們終於準備好定義在本章開頭承諾的完整 `backward`（反向傳播）函數了！它由以下操作組成（如[圖 14.5](#fig-backpropagation-mod_block_backward) 所示），該操作有三個輸入（ $\mathbf{x}_{\texttt{in}}, \theta, \mathbf{g}_{\texttt{out}}$ ）和兩個輸出（ $\mathbf{g}_{\texttt{in}}$ 和 $\frac{\partial J}{\partial \theta}$ ）。

![圖 14.5：通用層的 `backward`。我們使用該顏色來指示參數梯度。](https://visionbook.mit.edu/figures/backpropagation/mod_block_backward.png)

> [!NOTE]
> 我們使用此顏色來指示沿著網路反向傳遞的數據 / 激活值梯度。

## 14.4 完整演算法：先正向，後反向

我們現在準備好定義完整的反向傳播演算法了。在上一節中，我們看到一旦計算出了 $\mathbf{L}_l$ 和 $\mathbf{g}_l$ ，我們就可以輕鬆計算出 $\theta_l$ 的梯度更新。

> [!NOTE]
> $\mathbf{g}_l$ 和 $\mathbf{L}_l$ 分別是第 $l$ 層的 $\mathbf{g}$ 和 $\mathbf{L}$ 陣列。

因此，我們只需要合理安排我們的操作順序，以便在更新第 $l$ 層時，這兩個陣列已經準備就緒。具體方法是先計算通過整個網路的**前向傳播**（forward pass），這意味著從輸入數據 $\mathbf{x}_0$ 開始，逐層評估以產生序列 $\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_L$ 。[圖 14.6](#fig-backpropagation-forward_pass) 展示了前向傳播的過程。

![圖 14.6：前向傳播。](https://visionbook.mit.edu/figures/backpropagation/forward_pass.png)

> [!NOTE]
> 我們使用該顏色來指示沿著網路前向傳遞的數據 / 激活值。
> (註：我們使用此顏色指示沿著網路前向傳遞的數據/激活值。)

接下來，我們計算**反向傳播**（backward pass），迭代評估 $\mathbf{g}$ 並獲得序列 $\mathbf{g}_L, \mathbf{g}_{L-1}, \ldots$ ，以及每一層的參數梯度（[圖 14.7](#fig-backpropagation-backward_pass)）。

![圖 14.7：反向傳播。](https://visionbook.mit.edu/figures/backpropagation/backward_pass.png)

完整演算法總結在[演算法 14.1](#alg-backpropagation-backprop_for_chains) 中。

![演算法 14.1：反向傳播（針對鏈式計算圖）。反向傳播演算法的簡單版本。這適用於我們迄今為止所見到的計算圖，這些計算圖由一系列的層 $f_1 \circ \ldots \circ f_L$ 組成，沒有合併或分支（關於如何處理具有合併與分支操作之更複雜圖，請參見[第 14.7 節](#sec-backpropagation-branch_and_merge)）。](https://visionbook.mit.edu/figures/backpropagation/backprop_for_chains.png)

## 14.5 在數據批次上的反向傳播

到目前為止，我們只探討了計算單一數據點 $\mathbf{x}$ 的損失梯度。正如您從[第 9 章](https://visionbook.mit.edu/intro_to_learning.html) 和[第 10 章](https://visionbook.mit.edu/gradient_descent.html) 中可能回想起來的，我們希望最小化的總代價函數通常會是訓練集中*所有*數據點損失的*平均值*： $\{\mathbf{x}^{(i)}\}_{i=1}^N$ 。

然而，一旦我們知道如何計算單個數據點的梯度，我們就可以基於以下恆等式輕鬆計算整個數據集的梯度： $$ \begin{aligned}
\frac{\partial \frac{1}{N}\sum_{i=1}^N J_i(\theta)}{\partial \theta} = \frac{1}{N} \sum_{i=1}^N \frac{\partial J_i(\theta)}{\partial \theta}
\end{aligned} \tag{14.3} $$ > [!NOTE]
> 多個項相加之和的梯度，等於每個項的梯度之和。

其中 $J_i(\theta)$ 是單個數據點 $\mathbf{x}^{(i)}$ 的損失。因此，要計算諸如隨機梯度下降 [第 10.7 節](https://visionbook.mit.edu/gradient_descent.html#sec-gradient_descent-SGD) 等演算法的梯度更新，我們以批次模式（batch mode）應用反向傳播，也就是說，我們在批次中的每個數據點上運行它（這可以並行完成），然後對結果取平均值。

在剩下的章節中，我們仍將僅關注單個數據點損失的反向傳播情況。當您繼續閱讀時，請記住，對批次進行相同操作只需應用[等式 14.3](#eq-backpropagation-average_of_gradients) 即可。

> [!NOTE]
> 多個項相加之和的梯度，等於每個項的梯度之和。

## 14.6 範例：多層感知器 (MLP) 的反向傳播

為了完整描述任何給定架構的反向傳播，我們需要網路中每一層的 $\mathbf{L}$ 。一種方法是為加法、乘法等所有原子函數（atomic functions）定義導數 $f^{\prime}$ ，然後將每一層展開為僅包含這些原子操作的計算圖。通過展開後的計算圖進行反向傳播將簡單地利用所有原子 $f^{\prime}$ 來計算所需的 $\mathbf{L}$ 矩陣。然而，對於標準層，通常有更高效的方法來編寫 `backward`。在本節中，我們將為線性層和 ReLU 層——多層感知器（MLP）中的兩個主要層——推導精簡的 `backward` 方法。

### 14.6.1 線性層的反向傳播

線性層在正向傳播方向的定義如下： $$ \begin{aligned} \mathbf{x}_{\texttt{out}}= \mathbf{W} \mathbf{x}_{\texttt{in}}+ \mathbf{b}
\end{aligned} $$ 為了清晰起見，我們將參數分開為 $\mathbf{W}$ 和 $\mathbf{b}$ ，但請記住，我們總是可以將以下內容重寫為 $\theta = \texttt{vec}[\mathbf{W}, \mathbf{b}]$ 的形式。設 $\mathbf{x}_{\texttt{in}}$ 為 $N$ 維， $\mathbf{x}_{\texttt{out}}$ 為 $M$ 維；那麼 $\mathbf{W}$ 是一個 $[M \times N]$ 維的矩陣，而 $\mathbf{b}$ 是一個 $M$ 維的向量。

接下來我們需要該函數相對於其輸入和參數的梯度，即 $\mathbf{L}$ 。矩陣代數通常會隱藏細節，因此我們首先寫出所有單獨的標量梯度： $$ \begin{aligned}
 \mathbf{L}^{\mathbf{x}}[i,j] &= \frac{\partial x_{\texttt{out}}[i]}{\partial x_{\texttt{in}}[j]} =
 \frac{\partial \sum_l \mathbf{W}[i,l] x_{\texttt{in}}[l]}{\partial x_{\texttt{in}}[j]} = \mathbf{W}[i,j]
\end{aligned} \tag{14.4} $$ $$ \begin{aligned}
 \mathbf{L}^{\mathbf{W}}[i,jk] &= \frac{\partial x_{\texttt{out}}[i]}{\partial \mathbf{W}[j,k]} = \frac{\partial \sum_l \mathbf{W}[i,l] x_{\texttt{in}}[l]}{\partial \mathbf{W}[j,k]} =
 \begin{cases}
 x_{\texttt{in}}[k], &\text{如果} \quad i == j\\
 0, & \text{否則}
 \end{cases} \\
 \mathbf{L}^{\mathbf{b}}[i,j] &= \frac{\partial x_{\texttt{out}}[i]}{\partial \mathbf{b}[j]} =
 \begin{cases}
 1, &\text{如果} \quad i == j\\
 0, & \text{否則}
 \end{cases}
\end{aligned} \tag{14.5} $$ [等式 14.4](#eq-lingrad1) 和 [等式 14.5](#eq-lingrad3) 意味著： $$ \begin{aligned}
 \boxed{\mathbf{L}^{\mathbf{x}} = \mathbf{W}} &\quad\quad \triangleleft \quad [M \times N]\\
 \boxed{\mathbf{L}^{\mathbf{b}} = \mathbf{I}} &\quad\quad \triangleleft \quad [M \times M]
\end{aligned} $$ 對於 $\mathbf{L}^\mathbf{W}$ 沒有如此簡單的簡寫，但這沒關係，因為在此時，我們可以將計算出的 $\mathbf{L}^\mathbf{x}$ 代入[等式 14.2](#eq-backpropagation-backward) ，並將 $\mathbf{L}^\mathbf{W}$ 和 $\mathbf{L}^\mathbf{b}$ 代入[等式 14.1](#eq-backpropagation-djdtheta) ，從而實現線性層的 `backward`。 $$ \begin{aligned}
 \mathbf{g}_{\texttt{in}} &= \mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{x}} = \mathbf{g}_{\texttt{out}}\mathbf{W}\\
 \frac{\partial J}{\partial \mathbf{W}} &= \mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{W}}\\
 \frac{\partial J}{\partial \mathbf{b}} &= \mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{b}} = \mathbf{g}_{\texttt{out}}
\end{aligned} \tag{14.6} $$ 為了直觀理解[等式 14.6](#eq-backpropagation-linear_backward_costgrad) ，繪製相乘的矩陣會有所幫助。下方[圖 14.8](#fig-backpropagation-linear_forward_backward_matrices) 中，左側是該層的正向操作（省略了偏置），右側是[等式 14.6](#eq-backpropagation-linear_backward_costgrad) 中的反向操作。

![圖 14.8：線性層的正向與反向矩陣乘法。](https://visionbook.mit.edu/figures/backpropagation/linear_forward_backward_matrices.png)

與其他等式不同，乍看之下 $\frac{\partial J}{\partial \mathbf{W}}$ 似乎沒有簡單的形式。一種幼稚的方法是先構建大型稀疏矩陣 $\mathbf{L}^{\mathbf{W}}$ （大小為 $[M \times MN]$ , 其中當 $\mathbf{L}^\mathbf{W}[i,jk]$ 中 $i \neq k$ 時元素為零），然後執行矩陣乘法 $\mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{W}}$ 。我們可以透過觀察以下簡化來避免所有與零的相乘： $$ \begin{aligned} \frac{\partial J}{\partial \mathbf{W}[i,j]}
 &= \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}} \frac{\partial \mathbf{x}_{\texttt{out}}}{\partial \mathbf{W}[i,j]} \quad\quad\quad\quad\quad\quad \triangleleft \quad[1 \times M][M \times 1] \rightarrow [1 \times 1]\\
 &= \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}} \Big[\frac{\partial x_{\texttt{out}}[0]}{\partial \mathbf{W}[i,j]}, \ldots ,\frac{\partial x_{\texttt{out}}[M-1]}{\partial \mathbf{W}[i,j]} \Big]^\mathsf{T}\\
 &= \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}} \Big[\ldots , 0, \ldots , \frac{\partial x_{\texttt{out}}[i]}{\partial \mathbf{W}[i,j]}, \ldots, 0, \ldots \Big]^\mathsf{T}\\
 &= \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}} \Big[\ldots , 0, \ldots , x_{\texttt{in}}[j], \ldots, 0, \ldots \Big]^\mathsf{T}\\
 &= \frac{\partial J}{\partial x_{\texttt{out}}[i]}x_{\texttt{in}}[j] \quad\quad\quad\quad\quad\quad \triangleleft \quad[1 \times 1][1 \times 1] \rightarrow [1 \times 1]
\end{aligned} $$ > [!NOTE]
> 在矩陣方程式中，檢查維度是否匹配是非常有用的。在本章某些等式的右側，我們標註了乘積中矩陣的維度，其中 $\mathbf{x}_{\texttt{in}}$ 是 M 維， $\mathbf{x}_{\texttt{out}}$ 是 N 維，而損失 $J$ 始終是一個純量。

現在我們只需將所有這些標量導數整理到 $\frac{\partial J}{\partial \mathbf{W}}$ 的矩陣中，即可獲得以下結果： $$ \begin{aligned}
 \frac{\partial J}{\partial \mathbf{W}} & =
 \begin{bmatrix}
 \frac{\partial J}{\partial \mathbf{W}[0,0]} & \ldots & \frac{\partial J}{\partial \mathbf{W}[N-1,0]} \\
 \vdots & \ddots & \vdots \\
 \frac{\partial J}{\partial \mathbf{W}[0,M-1]} & \ldots & \frac{\partial J}{\partial \mathbf{W}[N-1,M-1]} \\
 \end{bmatrix}\\
 & =
 \begin{bmatrix}
 \frac{\partial J}{\partial x_{\texttt{out}}[0]}x_{\texttt{in}}[0] & \ldots & \frac{\partial J}{\partial x_{\texttt{out}}[N-1]}x_{\texttt{in}}[0] \\
 \vdots & \ddots & \vdots \\
 \frac{\partial J}{\partial x_{\texttt{out}}[0]}x_{\texttt{in}}[M-1] & \ldots & \frac{\partial J}{\partial x_{\texttt{out}}[N-1]}x_{\texttt{in}}[M-1] \\
 \end{bmatrix}\\
 & = \mathbf{x}_{\texttt{in}}\frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}}\\
 &= \mathbf{x}_{\texttt{in}}\mathbf{g}_{\texttt{out}}
\end{aligned} $$ > [!NOTE]
> 注意，我們在此使用的是向量和矩陣從零開始索引（zero-indexing）的慣例。

所以我們看到，最終這個梯度具有兩個向量 $\mathbf{x}_{\texttt{in}}$ 和 $\mathbf{g}_{\texttt{out}}$ 之間外積（outer product）的簡單形式（[圖 14.9](#fig-backpropagation-parameter_grad_linear_matrices)）。

![圖 14.9：線性層參數梯度的矩陣相乘。](https://visionbook.mit.edu/figures/backpropagation/parameter_grad_linear_matrices.png)

我們可以將所有這些操作總結在[圖 14.10](#fig-backpropagation-linear_layer_backprop) 中線性層的正向與反向操作圖中。

![圖 14.10：線性層的正向與反向傳播。](https://visionbook.mit.edu/figures/backpropagation/linear_layer_backprop.png)

請注意，所有這些操作都是簡單的表達式，主要涉及矩陣乘法。使用任何提供矩陣乘法（ `matmul` ）作為原語的庫，用代碼編寫線性層的 `forward` 和 `backward` 也非常容易。[圖 14.11](#fig-backpropagation-backprop_code) 給出了該層的 Python 虛擬碼。

 `class linear():
 def __init__(self, W, b, lr):
 self.W = W
 self.b = b
 self.lr = lr # 學習率
 
 def forward(self, x_in):
 self.x_in = x_in
 return matmul(W,x)+b
 
 def backward(self,J_out):
 J_in = matmul(J_out,W)
 dJdW = matmul(self.x_in,J_out)
 dJdb = J_out
 return J_in, dJdW, dJdb
 
 def update(self, dJdW, dJdb):
 self.W -= self.lr*dJdW.transpose()
 self.b -= self.lr*dJdb` **

圖 14.11：具有 `forward` 和 `backward` 的線性層之類 PyTorch 虛擬碼。

### 14.6.2 逐點非線性層的反向傳播

逐點非線性層（pointwise nonlinearities）具有非常簡單的 `backward` 函數。設一個（無參數的）純量非線性函數為 $h: \mathbb{R} \rightarrow \mathbb{R}$ ，其導數函數為 $h^{\prime}: \mathbb{R} \rightarrow \mathbb{R}$ 。定義一個使用 $h$ 的逐點層為 $f(\mathbf{x}_{\texttt{in}}) = [h(x_{\texttt{in}}[0]), \ldots, h(x_{\texttt{in}}[N-1])]^\mathsf{T}$ 。那麼我們有： $$ \begin{aligned}
 \mathbf{L}^{\mathbf{x}} &= f^{\prime}(\mathbf{x}_{\texttt{in}}) = \texttt{diag}([h^\prime(x_{\texttt{in}}[0]), \ldots, h^\prime(x_{\texttt{in}}[N-1])]^\mathsf{T}) \triangleq \mathbf{H}^\prime
\end{aligned} $$ > [!NOTE]
> $\texttt{diag}$ 算子是將一個向量放置在矩陣的對角線上，而該矩陣的其他元素均為零的算子。

由於沒有需要更新的參數，我們只需在 `backward` 操作中計算 $\mathbf{g}_{\texttt{in}}$ ，這需要利用[等式 14.2](#eq-backpropagation-backward)： $$ \begin{aligned}
 \mathbf{g}_{\texttt{in}} = \mathbf{g}_{\texttt{out}}\mathbf{H}^\prime
\end{aligned} $$ 作為一個例子，對於 `relu` 層，我們有： $$ \begin{aligned}
 h^\prime(x) =
 \begin{cases}
 1 &\text{如果} \quad x \geq 0\\
 0 &\text{否則}
 \end{cases}
\end{aligned} $$ 作為矩陣乘法，其 `backward` 操作如[圖 14.12](#fig-backpropagation-pointwise_backward_matices) 所示。

![圖 14.12：逐點層的反向傳播矩陣乘法。](https://visionbook.mit.edu/figures/backpropagation/pointwise_backward_matices.png)

其中 $a = h^\prime(x_{\texttt{in}}[0])$ ， $b = h^\prime(x_{\texttt{in}}[1])$ ，且 $c = h^\prime(x_{\texttt{in}}[2])$ 。我們可以將該方程式簡化如下： $$ \begin{aligned}
 \mathbf{g}_{\texttt{in}}[i] = \mathbf{g}_{\texttt{out}}[i]h^{\prime}(\mathbf{x}_{\texttt{in}}[i]) \quad \forall i
\end{aligned} \tag{14.7} $$ 逐點層的完整操作集接著展示在[圖 14.13](#fig-backpropagation-pointwise_layer_backprop) 中。

![圖 14.13：逐點層的操作示意圖。](https://visionbook.mit.edu/figures/backpropagation/pointwise_layer_backprop.png)

### 14.6.3 損失層的反向傳播

為了定義一個完整的 MLP，我們需要定義的最後一個層是損失層（loss layer）。作為一個簡單的例子，我們將為 $L_2$ 損失函數推導反向傳播： $\left\lVert\hat{\mathbf{y}} - \mathbf{y}\right\rVert^2_2$ ，其中 $\hat{\mathbf{y}}$ 是網路的輸出（預測值），而 $\mathbf{y}$ 是真實標籤（ground truth）。

此層沒有參數，所以我們只需要推導此層的[等式 14.2](#eq-backpropagation-backward)： $$ \begin{aligned}
 \mathbf{L}^{\mathbf{x}} &= \frac{\partial \left\lVert\hat{\mathbf{y}} - \mathbf{y}\right\rVert^2_2}{\partial \hat{\mathbf{y}}} = 2(\hat{\mathbf{y}} - \mathbf{y}) \quad\quad \triangleleft \quad [1 \times |\mathbf{y}|]\\
 \mathbf{g}_{\texttt{in}} &= \mathbf{g}_{\texttt{out}}*2(\hat{\mathbf{y}} - \mathbf{y}) = 2(\hat{\mathbf{y}} - \mathbf{y})
\end{aligned} $$ 在這裡，我們利用了 $\mathbf{g}_{\texttt{out}} = \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}} = \frac{\partial J}{\partial J} = 1$ 的事實，因為損失層的輸出*就是*代價 $J$ 。

因此， $L_2$ 損失層發送的反向訊號是預測與目標之間每個維度誤差的行向量。

這完成了我們對 $L_2$ 損失層的 `forward` 和 `backward` 的推導，得到[圖 14.14](#fig-backpropagation-L2_loss_layer_backprop)。

![圖 14.14： $L_2$ 損失層 `backward` 的矩陣乘法。](https://visionbook.mit.edu/figures/backpropagation/L2_loss_layer_backprop.png)

### 14.6.4 綜合應用：通過 MLP 的反向傳播

讓我們看看當我們將所有這些操作放在一個 MLP 中時會發生什麼。我們將從[圖 14.1](#fig-backpropagation-simple_MLP) 中的 MLP 開始。為簡單起見，我們將省略偏置。設 $\mathbf{x}$ 為四維， $\mathbf{z}$ 和 $\mathbf{h}$ 為三維，而 $\hat{\mathbf{y}}$ 為二維。此網路的前向傳播如底部的[圖 14.15](#fig-backpropagation-forward_pass_MLP) 所示。

![圖 14.15：通過 MLP 的前向傳播。](https://visionbook.mit.edu/figures/backpropagation/forward_pass_MLP.png)

對於反向傳播，我們在這裡對慣例做一些微調，這將揭示前向和反向傳播之間一個有趣的聯繫。我們不再將梯度 $\mathbf{g}$ 表示為行向量，而是將其轉置並視為列向量。轉置向量的反向傳播操作遵循矩陣恆等式 $(\mathbf{A}\mathbf{B})^\mathsf{T}= \mathbf{B}^\mathsf{T}\mathbf{A}^\mathsf{T}$ ： $$ \begin{aligned}
 \mathbf{g}^\mathsf{T}_{\texttt{in}} = (\mathbf{g}_{\texttt{out}}\mathbf{W})^\mathsf{T}= \mathbf{W}^\mathsf{T}\mathbf{g}^\mathsf{T}_{\texttt{out}}
\end{aligned} $$ 現在我們將在[圖 14.16](#fig-backpropagation-backward_pass_MLP) 中使用這些轉置後的 $\mathbf{g}$ 繪製反向傳播。

![圖 14.16：通過 MLP 的反向傳播。](https://visionbook.mit.edu/figures/backpropagation/backward_pass_MLP.png)

這揭示了線性層的 `forward` 和 `backward` 之間一個有趣的聯繫：線性層的反向傳播與前向傳播是完全相同的操作，只是權重矩陣被轉置了！我們在這裡省略了偏置項，但請從[等式 14.6](#eq-backpropagation-linear_backward_costgrad) 中回想，反正傳播到激活值的反向路徑也會忽略偏置。

相反地， `relu` 層在反向路徑上並非執行 ReLU 操作。相反，它變成了一種門控矩陣（gating matrix），由前向傳播中激活值的函數（ $a$ ， $b$ 和 $c$ ）來參數化。該矩陣除了對角線上激活值為非負的元素為 1 之外，其餘均為零。該層的作用是屏蔽掉 $\texttt{relu}$ 負值側變量的梯度。請注意，此操作是矩陣乘法——事實上，無論前向傳播操作是什麼，*所有*的反向操作都是矩陣乘法，正如您在[演算法 14.1](#alg-backpropagation-backprop_for_chains) 中可以觀察到的那樣。

在上面的圖中，我們還沒有包括參數梯度的計算。在反向路徑的每一步中，這些梯度都是作為輸入該層的激活值 $\mathbf{x}_{\texttt{in}}$ 與被送回的梯度 $\mathbf{g}_{\texttt{out}}$ 之間的外積來計算的（參見[圖 14.9](#fig-backpropagation-parameter_grad_linear_matrices)）。

### 14.6.5 前向-反向傳播本身就是一個更大的神經網路

在前一節中，我們看到通過神經網路的反向傳播本身也可以被實現為另一個神經網路，這實際上是一個線性網路，其參數由原始網路的權重矩陣以及原始網路中的激活值決定。因此，完整的反向傳播也是一個神經網路！既然前向傳播也是一個神經網路（原始網路），那麼完整的反向傳播演算法——前向傳播後接反向傳播——就可以被視為只是一個巨大的神經網路。藉由反向網路的每一層進行額外一次矩陣乘法（ `matmul` ），即可從該網路中計算出參數梯度。對於三層 MLP 的完整網路展示在[圖 14.17](#fig-backpropagation-backprop_as_neural_net) 中。

![圖 14.17：通過三層 MLP 進行反向傳播的計算圖。它只是另一個神經網路！實線表示用於計算數據 / 激活值梯度的路徑，虛線表示用於計算參數梯度的路徑。](https://visionbook.mit.edu/figures/backpropagation/backprop_as_neural_net.png)

> [!NOTE]
> `params forward` `params backward` `data forward` `data backward`
> (註：即參數前向、參數反向、數據前向、數據反向)

這個前向-反向網路有一些有趣的地方。其中之一是來自 `relu` 層的**激活值**被轉換成了反向網路中 `linear` 線性層的**參數**（參見[等式 14.7](#eq-backpropagation-pointwise_costgrad)）。對於這種由一個神經網路輸出值來參數化另一個神經網路的設置，有一個通用的術語，稱為**超網路**（hypernetwork）[ [1](https://visionbook.mit.edu/references.html#ref-ha2016hypernetworks) ]。前向網路就是一個用以參數化反向網路的超網路。

另一個我們之前已經指出過的有趣性質是，反向網路僅由線性層組成。無論前向網路由什麼組成（即使它不是傳統的神經網路，而是某種任意的計算圖），這都是成立的。之所以會發生這種情況，是因為反向傳播實現了連鎖律，而連鎖律總是雅可比矩陣（Jacobian matrices）的乘積。既然雅可比矩陣是一個矩陣，顯然它是一個線性函數。但更直觀地說，你可以將每個雅可比矩陣視為損失表面的局部線性逼近；因此每個都可以用一個線性層來表示。

## 14.7 通過有向無環圖 (DAG) 的反向傳播：分支與合併

到目前為止，我們只看到了鏈狀的圖， –[]–[]–[] $\rightarrow$ 。反向傳播可以處理其他結構的圖嗎？答案顯然是**可以的**。目前我們將考慮**有向無環圖**（Directed Acyclic Graphs，簡稱 **DAG**）。在[第 25 章](https://visionbook.mit.edu/recurrent_neural_nets.html) 中，我們還將看到神經網路也可以包含循環，並仍然可以使用反向傳播的變體（例如隨時間反向傳播 BPTT）進行訓練。

在 DAG 中，節點可以有多個輸入和多個輸出。實際上，我們在前面的章節中已經看到了好幾個此類節點的例子。例如，線性層可以被認為具有兩個輸入（ $\mathbf{x}_{\texttt{in}}$ 和 $\theta$ ）和一個輸出 $\mathbf{x}_{\texttt{out}}$ ；或者如果我們累計輸入與輸出向量的每個維度，它可以被認為具有 $N = |\mathbf{x}_{\texttt{in}}| + |\theta|$ 個輸入與 $M = |\mathbf{x}_{\texttt{out}}|$ 個輸出。所以我們已經見過 DAG 計算圖了。

然而，為了處理一般的 DAG，引入兩個新的特殊模組會很有幫助，它們的作用是構建圖的拓撲結構。我們將這些特殊的算子稱為 `merge`（合併）和 `branch`（分支）（[圖 14.18](#fig-backpropagation-branch_and_merge)）。

![圖 14.18： `merge` 和 `branch` 層。](https://visionbook.mit.edu/figures/backpropagation/branch_and_merge.png)

> [!NOTE]
> 我們在這裡僅考慮二元的分支與合併，但 $N$ 路的分支與合併可以類似地進行，或者藉由重複這些算子來實現。

我們在數學上分別將它們定義為變量的拼接（concatenation）與複製（copying）： $$ \begin{aligned}
 \texttt{merge}(\mathbf{x}_{\texttt{in}}^a, \mathbf{x}_{\texttt{in}}^b) &\triangleq [\mathbf{x}_{\texttt{in}}^a, \mathbf{x}_{\texttt{in}}^b] \triangleq \mathbf{x}_{\texttt{out}}\\
 \texttt{branch}(\mathbf{x}_{\texttt{in}}) &\triangleq [\mathbf{x}_{\texttt{in}}, \mathbf{x}_{\texttt{in}}] \triangleq [\mathbf{x}_{\texttt{out}}^a, \mathbf{x}_{\texttt{out}}^b]
\end{aligned} $$ > [!NOTE]
> 如果 $\mathbf{x}_{\texttt{in}}^a$ 和 $\mathbf{x}_{\texttt{in}}^b$ 是形狀不同的張量（tensors）或其他對象怎麼辦？我們還能拼接它們嗎？答案是肯定的。數據張量的形狀對數學沒有影響。我們選擇形狀只是為了符號上的便利；例如，將圖像視為二維陣列是很自然的。

在這裡， $\texttt{merge}$ 接收兩個輸入並將它們拼接。這會產生一個新的多維變量。其反向路徑的方程式非常簡單。為了解釋相對於 $\mathbf{x}_{\texttt{in}}^{a}$ 的代價梯度，即 $\mathbf{g}_{\texttt{in}}^a$ ，我們有： $$ \begin{aligned}
 \mathbf{g}_{\texttt{in}}^{a} &= \mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{x}^a} = \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}} \frac{\partial \mathbf{x}_{\texttt{out}}}{\partial \mathbf{x}_{\texttt{in}}^{a}}\\
 &= \mathbf{g}_{\texttt{out}}\Big[\frac{\partial \mathbf{x}_{\texttt{in}}^{a}}{\partial \mathbf{x}_{\texttt{in}}^{a}}, \frac{\partial \mathbf{x}_{\texttt{in}}^{b}}{\partial \mathbf{x}_{\texttt{in}}^{a}}\Big]^\mathsf{T}\\
 &= \mathbf{g}_{\texttt{out}}[1, 0]^\mathsf{T}
\end{aligned} $$ 對於 $\mathbf{g}_{\texttt{in}}^b$ 也是如此。也就是說，我們只需取出 $\mathbf{g}_{\texttt{out}}$ 梯度向量的前半部分作為 $\mathbf{g}_{\texttt{in}}^a$ ，後半部分作為 $\mathbf{g}_{\texttt{in}}^b$ 。這裡其實沒有任何新東西。我們上面已經為多維變量定義了反向傳播，而 $\texttt{merge}$ 只是構建多維變量的一種顯式方式。

`branch` 算子只稍微複雜一些。在分支中，我們將相同輸出的**副本**發送到多個下游節點。因此，我們有多個來自不同下游路徑的梯度傳回到 `branch` 模組。所以，該模組在反向傳播時的輸入是 $\frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}^{a}}, \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}^{b}}$ ，我們可以將其寫為梯度向量 $\mathbf{g}_{\texttt{out}}= \frac{\partial J}{\partial [\mathbf{x}_{\texttt{out}}^a, \mathbf{x}_{\texttt{out}}^b]} = [\frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}^{a}}, \frac{\partial J}{\partial \mathbf{x}_{\texttt{out}}^{b}}] = [\mathbf{g}_{\texttt{out}}^a, \mathbf{g}_{\texttt{out}}^b]$ 。讓我們計算反向傳播的輸出： $$ \begin{aligned}
 \mathbf{g}_{\texttt{in}}&= \mathbf{g}_{\texttt{out}}\mathbf{L}^{\mathbf{x}}\\
 &= [\mathbf{g}_{\texttt{out}}^a, \mathbf{g}_{\texttt{out}}^b] \frac{\partial [\mathbf{x}_{\texttt{out}}^a, \mathbf{x}_{\texttt{out}}^b]}{\partial \mathbf{x}_{\texttt{in}}}\\
 &= [\mathbf{g}_{\texttt{out}}^a, \mathbf{g}_{\texttt{out}}^b] \frac{\partial [\mathbf{x}_{\texttt{in}}, \mathbf{x}_{\texttt{in}}]}{\partial \mathbf{x}_{\texttt{in}}}\\
 &= [\mathbf{g}_{\texttt{out}}^a, \mathbf{g}_{\texttt{out}}^b][1, 1]^\mathsf{T}\\
 &= \mathbf{g}_{\texttt{out}}^a + \mathbf{g}_{\texttt{out}}^b
\end{aligned} $$ 因此，分支操作只是將沿其反向傳遞的兩個梯度相加。

$\texttt{merge}$ 和 $\texttt{branch}$ 都沒有參數，所以沒有需要定義的參數梯度。因此，我們已經完全指定了這些層的前向與反向行為。接下來的圖總結了其行為（[圖 14.19](#fig-backpropagation-branch_and_merge_layers_backprop_diagram)）。

圖 14.19： `merge` 和 `branch` 層的前向與反向操作。

利用 `merge` 和 `branch` ，我們只需在希望某層具有多個輸入或多個輸出的任何地方插入這些層，即可構建任何 DAG 計算圖。[圖 14.20](#fig-backpropagation-backprop_DAG) 給出了一個範例。

![圖 14.20：一個 DAG 計算圖的示例，我們可以使用前面定義的工具來構建它並進行反向傳播。](https://visionbook.mit.edu/figures/backpropagation/DAG.png)

## 14.8 參數共享

**參數共享**（Parameter sharing）是指單一參數作為輸入被發送到多個不同的層。我們可以將其視為一種分支操作，如[圖 14.21](#fig-backpropagation-parameter_sharing) 所示。

![圖 14.21：參數共享等同於在計算圖中分支一個參數。](https://visionbook.mit.edu/figures/backpropagation/parameter_sharing.png)

那麼，由前一節可知，共享參數的梯度是累加的。設 $\{\theta^i\}^N_{i=1}$ 是一組變量，它們都是同一個自由參數 $\theta$ 的副本。那麼， $$ \begin{aligned}
 \frac{\partial J}{\partial \theta} = \sum_i \frac{\partial J}{\partial \theta^i}
\end{aligned} $$

## 14.9 反向傳播至數據本身

反向傳播並不區分參數和數據——它將兩者都視為無參數模組的通用輸入。因此，我們可以像使用反向傳播最佳化參數輸入一樣，使用反向傳播來最佳化計算圖的數據輸入。

為了看清這一點，僅考慮整個計算圖的輸入與輸出會有所幫助。在前向傳播方向上，輸入是數據與參數設定，而輸出是損失。在反向傳播方向上，輸入是數值 1，而輸出則是損失相對於數據和參數的梯度。對於使用神經網路 $F = f_L \circ \cdots \circ f_1$ 和損失函數 $\mathcal{L}$ 的學習問題，其完整計算圖為 $\mathcal{L}(F(\mathbf{x}_0), \mathbf{y}, \theta) \triangleq J(\mathbf{x}_0, \mathbf{y}, \theta)$ 。這個函數本身可以被視為單個計算區塊，其輸入和輸出如前文所述（[圖 14.22](#fig-backpropagation-J_forward_backward_blocks)）。

> [!NOTE]
> 在 PyTorch 中，你只能將*輸入*變量設置為最佳化目標——這些被稱為計算圖的*葉節點*（leaves），因為在反向路徑上，它們沒有子節點。所有其他變量都完全由輸入變量的值決定——它們不是*自由*變量。

![圖 14.22：針對學習問題 $\min J(\mathbf{x}_0,\mathbf{y},\theta)$ 的完整前向與反向傳播，摺疊為單個計算區塊。](https://visionbook.mit.edu/figures/backpropagation/J_forward_backward_blocks.png)

從這個圖中可以清楚地看出，數據輸入和參數輸入扮演著對稱的角色。正如我們可以藉由沿著由 `backward` 給出的參數梯度下降來最佳化參數以最小化損失一樣，我們也可以藉由沿著數據梯度下降來最佳化輸入數據以最小化損失。

這對於許多不同的應用非常有用。一個例子是視覺化能夠最能激活我們在神經網路中探測的給定神經元的輸入圖像。為此，我們將 $J$ 定義為我們正在探測的神經元值的*負值* [1](#fn1) ，也就是說，如果我們對 $l$ 層上的第 $i$ 個神經元感興趣，則 $J(\mathbf{x}_0,\theta) = -x_{l}[i]$ （請注意，此問題不使用 $\mathbf{y}$ ）。我們在下方的[圖 14.23](#fig-backpropagation-backprop_to_the_data_example) 中展示了這個例子，其中我們使用反向傳播來尋找最能激活計算圖中某個節點的輸入圖像，該節點用於對圖像是否為「貓的照片」進行評分。您在最佳化後的圖像中看到了類似貓的特徵嗎？這告訴了您關於該網路如何運作的什麼資訊？

> [!NOTE]
> 1 它是負值，因此最小化損失即是最大化激活值。

![圖 14.23：根據特定神經網路視覺化貓的最佳圖像。我們使用的網路稱為對比語言-圖像預訓練（CLIP）[ [2](https://visionbook.mit.edu/references.html#ref-radford2021learning) ]，在這裡我們尋找了最大化 CLIP 計算圖中測量圖像與文本「一隻貓的照片」匹配程度之節點的圖像。在[第 51.3 節](https://visionbook.mit.edu/VLMs.html#sec-VLMs-CLIP) 中，我們將更詳細地介紹 CLIP 模型的工作原理。](https://visionbook.mit.edu/figures/backpropagation/backprop_to_the_data_example.png)

> [!NOTE]
> 像這樣的視覺化是找出給定神經元對哪些視覺特徵敏感的有用方法。研究人員通常會將這種視覺化方法與自然圖像先驗（natural image prior）相結合，以便尋找不僅能強烈激活相關神經元，而且看起來像自然照片的圖像（例如，[ [3](https://visionbook.mit.edu/references.html#ref-olah2017feature) ]）。

## 14.10 結論與總結

反向傳播通常被呈現為僅用於訓練神經網路的方法，但它實際上是一個比這更通用的工具。反向傳播是尋找計算圖中偏導數的一種高效方法。它適用於廣泛的計算圖系列，並且不僅可以用於學習參數，還可以用於最佳化數據。

## 參考文獻

[1] D. Ha, A. Dai, Q.V. Le, Hypernetworks., Iclr (2016).

[2] A. Radford, J.W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, others, Learning transferable visual models from natural language supervision., in: Icml, 2021: pp. 8748–8763.

[3] C. Olah, A. Mordvintsev, L. Schubert, Feature visualization., Distill 2 (2017) e7.
