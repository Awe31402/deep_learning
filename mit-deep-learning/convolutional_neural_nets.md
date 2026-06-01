# 24 卷積神經網路

## 24.1 簡介

我們在[第 12 章](neural_nets.html)中看到的神經網路是設計用來處理一般性數據的。但在許多領域中，數據具有特殊的結構，我們可以設計更適合利用該結構的神經網路架構。**卷積神經網路**（Convolutional Neural Networks，也稱為 **convnets** 或 **CNNs**）是一種特別適合處理視覺訊號結構的神經網路架構。

CNN 的核心思想是將輸入圖像切碎成許多小區塊（patches），然後**獨立**且**相同地**處理每個區塊。圖 24.1 呈現了這個概念的精髓：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNNs_as_patch_processing.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNNs_as_patch_processing.png "Figure 24.1: CNNs as patch processing. Photo source: Fredo Durand.")

圖 24.1：將 CNN 視為區塊處理。_照片來源_：Fredo Durand。

CNN 也非常適合處理許多其他空間或時間訊號，例如地理空間數據或聲音。如果有一種自然的方法可以掃描整個訊號，並分別處理每個窗口區域，那麼 CNN 就會是一個合理的選擇。

每個區塊都使用一個分類器模組（即一個神經網路）進行處理。本質上，這個神經網路會掃描輸入中的各個區塊並對其進行分類。輸出是**輸入圖像中每個區塊的標籤**。如果我們將這些預測重新排列回輸入圖像的形狀並進行顏色編碼，我們會得到如下的輸入-輸出映射（圖 24.2）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNN_example_coarse.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNN_example_coarse.png "Figure 24.2: Input-output mapping of a CNN.")

圖 24.2：CNN 的輸入-輸出映射。

請注意，這與我們在[第 12 章](neural_nets.html)中看到的神經網路有很大的不同，後者是對整張圖像輸出單一預測；而 CNN 則是輸出二維（2D）的預測**陣列**。我們也可以將圖像切成**重疊**的區塊。如果我們密集地進行此操作，使得每個區塊與前一個區塊僅偏移一個像素，我們就能得到一個完整解析度的預測圖像（圖 24.3）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNN_example_fine.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNN_example_fine.png "Figure 24.3: Dense input-output mapping.")

圖 24.3：密集輸入-輸出映射。

這看起來非常令人驚嘆！這個 CNN 解決了被稱為**語意分割（semantic segmentation）**的任務，也就是為圖像中的每個像素分配一個類別標籤。CNN 強大的原因之一在於，它們能將輸入圖像映射到**具有相同形狀**的輸出圖像，而不是像我們在前面章節中看到的神經網路那樣只輸出單個標籤。CNN 還可以推廣到輸入和輸出其他類型的結構。關鍵特性在於輸出與輸入的拓撲結構一致：$N$ 維（ND）的輸入張量將被映射到 $N$ 維的輸出張量。

牢記「切碎並預測」就是 CNN 的全部工作後，我們現在將深入探討它們運作的細節。

## 24.2 卷積層

CNN 是由**卷積層**所組成的神經網路。卷積層藉由將輸入 $\mathbf{x}_{\texttt{in}}$ 與一個或多個濾波器 $\mathbf{w}$ 進行卷積，來將輸入 $\mathbf{x}_{\texttt{in}}$ 轉換為輸出 $\mathbf{x}_{\texttt{out}}$。具有單個濾波器的卷積層看起來像這樣：
$$
\begin{aligned}\mathbf{x}_{\texttt{out}}= \mathbf{w} \star \mathbf{x}_{\texttt{in}}+ b & \quad\quad \triangleleft \quad \texttt{conv} \end{aligned} \tag{24.1}
$$

其中 $\mathbf{w}$ 是卷積核，$b$ 是偏置；$\theta = [\mathbf{w}, b]$ 是該層的參數。

在本章中，我們稍微偏離了常用的符號標記方式，無論卷積核是 1D 陣列、2D 陣列還是 ND 陣列，卷積濾波器一律使用小寫的 $\mathbf{w}$ 表示。

回顧[第 15 章](linear_image_filtering.html)中運算子 $\star$ 的定義，我們在此給出一個在 2D 陣列 $\mathbf{x}_{\texttt{in}}$ 上使用大小為 $2K+1 \times 2K+1$ 的方形卷積核進行卷積層運算的範例：

$$
\begin{aligned}x_{\texttt{out}}[n,m] = b + \sum_{k_1,k_2=-K}^K w[k_1,k_2] x_{\texttt{in}}[n+k_1,m+k_2] & \quad\quad \\\ \triangleleft \quad \texttt{conv}\quad \text{(expanded)} \end{aligned} \tag{24.2}
$$

深度網路中的「卷積」層在實際定義上通常是互相關（cross-correlation，$\star$），我們在本書中遵循這一慣例。我們不需要擔心這個名稱上的誤用，因為在學習過程中，不論你是用卷積還是互相關來實現這些層，通常沒有任何差別。這是因為兩者**涵蓋了相同的假設空間**（任何互相關都可以藉由將濾波器進行水平與垂直翻轉來轉換為等效的卷積）。

如[第 15 章](linear_image_filtering.html)所述，卷積只是一種特殊的線性轉換。同樣地，卷積層也只是一種特殊的線性層。它是一個其矩陣 $\mathbf{W}$ 為托普利茲（Toeplitz）矩陣的線性層。我們可以將其視為矩陣，也可以將其視為神經網路，如圖 24.4 所示，該圖展示了在一維（1D）訊號 $\mathbf{x}_{\texttt{in}}$ 上進行無偏置一維卷積的情況。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/conv_matrix_vs_net.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/conv_matrix_vs_net.png "Figure 24.4: Two equivalent visualizations of a convolutional layer.")

圖 24.4：卷積層的兩種等效視覺化呈現。

我們已經在[圖像處理基礎](./part_foundation_image_processing.html)與[線性濾波器](part_linear_filters.html)中看到卷積濾波器在圖像處理中的用處。在那些章節中，我們介紹了多種具有實用特性的手工設計濾波器組。而 CNN 則是去**學習**一個有效的濾波器組。

### 24.2.1 多輸入、多輸出卷積層

在圖像處理中，卷積通常是指對單通道訊號進行濾波並產生單通道輸出，例如對灰階圖像進行濾波並產生純量值的響應圖像。在神經網路中，卷積層更為通用，通常將多通道輸入映射到多通道輸出。在這一小節中，我們先定義如何處理多通道輸入，接著定義如何處理多通道輸出，然後將它們結合起來定義完全通用的卷積層。

##### 多通道輸入

假設我們有一個 RGB 圖像 $\mathbf{x}_{\texttt{in}}\in \mathbb{R}^{3 \times N \times M}$。要將卷積層應用於此多通道圖像，我們只需使用一個多通道濾波器 $\mathbf{w} \in \mathbb{R}^{C \times K \times K}$，並用對應的濾波器通道對每個輸入通道進行濾波，然後將響應相加：
$$
\begin{aligned} \mathbf{x}_{\texttt{out}}= \sum_{c} \mathbf{w}[c,:,:] \star \mathbf{x}_{\texttt{in}}[c,:,:] + b[c] & \quad \triangleleft \quad\texttt{conv}\quad \text{(multichannel in)} \end{aligned}
$$

##### 多通道輸出

上面我們看到了僅具有單個濾波器的卷積層。更常見的是，神經網路中的每個卷積層都會應用一組濾波器，即**濾波器組**。如果我們有一個由 $C$ 個濾波器 $\mathbf{w}_0, \ldots, \mathbf{w}_{C-1}$ 組成的濾波器組，並將它們應用於灰階輸入圖像 $\mathbf{x}_{\texttt{in}}\in \mathbb{R}^{N \times M}$，我們將得到 $C$ 個輸出圖像：
$$
\begin{aligned} \mathbf{x}_{\texttt{out}}[0,:,:] &= \mathbf{w}[0,:,:] \star \mathbf{x}_{\texttt{in}}+ b[0]\\\ &\vdots \nonumber\\\ \mathbf{x}_{\texttt{out}}[C,:,:] &= \mathbf{w}[C-1,:,:] \star \mathbf{x}_{\texttt{in}}+ b[C-1] \end{aligned} \tag{24.3}
$$

現在，$\mathbf{x}_{\texttt{out}}$ 是一個具有 $C$ 個通道的圖像。每個通道都是輸入圖像對其中一個濾波器的響應。

我們使用「圖像」一詞來指代任何二維的測量值或特徵陣列。圖像不一定非得是傳統的照片。

我們將這些通道中的每一個稱為**特徵圖（feature map）**，因為它顯示了輸入的某些特徵，例如垂直邊緣的位置。

##### 多輸入、多輸出

將上述兩者結合起來，我們可以定義一個通用的卷積層，它將具有 $C_{\texttt{in}}$ 個輸入通道的訊號映射到具有 $C_{\texttt{out}}$ 個輸出通道的訊號。對於圖像 $\mathbf{x}_{\texttt{in}}\in \mathbb{R}^{C_{\texttt{in}}\times N \times M}$，其表示如下，其中 $c_2$ 為輸出通道的索引，且 $c_2 \in \{0, \ldots, C_{\texttt{out}}-1\}$：

$$
\begin{aligned}\mathbf{x}_{\texttt{out}}[c_{\texttt{2}},:,:] = \sum_{c_{\texttt{1}}=1}^{C_{\texttt{in}}} \mathbf{w}[c_{\texttt{1}},c_{\texttt{2}},:,:] \star \mathbf{x}_{\texttt{in}}[c_{\texttt{1}},:,:] + b[c_{\texttt{2}}] & \quad \triangleleft \quad\texttt{conv}\quad \text{(multi-in-out)} \end{aligned} \tag{24.4}
$$

多通道卷積的符號表示可能很難追蹤，所以我們在這裡詳細說明其中幾個部分，這些部分也在圖 24.5 中進行了視覺化：

  * $\mathbf{x}_{\texttt{in}}[c_{\texttt{1}},:,:]$ 是輸入訊號的第 $c_{\texttt{1}}$ 個通道。

  * 濾波器組包含 $C_{\texttt{out}}$ 個濾波器，即 $[\mathbf{w}[:,0,:,:], \ldots, \mathbf{w}[:,C_{\texttt{out}}-1,:,:]]$，其中每一個濾波器都會對每個輸入通道應用一個卷積濾波器，然後在所有這些濾波器上對響應求和。

  * 此卷積層將輸入 $\mathbf{x}_{\texttt{in}}\in \mathbb{R}^{C_{\texttt{in}}\times N \times M}$ 映射到輸出 $\mathbf{x}_{\texttt{out}}\in \mathbb{R}^{C_{\texttt{out}}\times N \times M}$。

  * 濾波器組由張量 $\mathbf{w} \in \mathbb{R}^{C_{\texttt{in}}\times C_{\texttt{out}}\times K \times K}$ 表示，其中 $K$ 是（空間、方形）卷積核大小。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/multichannel_conv.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/multichannel_conv.png "Figure 24.5: Multichannel convolution.")

圖 24.5：多通道卷積。

熟悉在不同神經網路架構中處理的數據和參數張量的形狀非常重要。這在設計、建構、分析以及調試這些架構時是不可或缺的。讓我們看一個具體數字的例子。考慮數據 $\mathbf{x}_{\texttt{in}}$，它是一個大小為 $128 \times 128$ 像素的 RGB 圖像。我們將它輸入一個卷積層，該層應用一個由 $3 \times 3$ 濾波器組成的濾波器組（這指的是濾波器的空間維度）。為了簡單起見，我們省略了偏置項。輸出最終會是一個 $96 \times 128 \times 128$ 的張量，如圖 24.6 所示。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/multichannel_conv_diagram.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/multichannel_conv_diagram.png "Figure 24.6: A convolutional layer that applies a bank of 3 \\times 3 filters. How many parameters does each filter have? How many filters are in the filter bank? Source: created by Jonas Wulff.")

圖 24.6：一個應用 $3 \times 3$ 濾波器組的卷積層。每個濾波器有多少個參數？濾波器組中有多少個濾波器？_來源_：由 Jonas Wulff 創建。

為了自我檢測理解程度，你應該要能回答以下問題：

  1. 每個濾波器有多少個參數？(A) 9, (B) 27, (C) 96, (D) 864

  2. 濾波器組中有多少個濾波器？(A) 3, (B) 27, (C) 96, (D) 無法確定

答案寫在腳註中。1

1 答案是 1-B，2-C。

### 24.2.2 步幅卷積

如前所述定義的卷積層會保持其處理訊號的空間解析度。然而，輸出較低的解析度通常就足夠了，甚至更為理想。這可以藉由**步幅卷積（strided convolution）**來實現：

$$
\begin{aligned}x_{\texttt{out}}[n,m] = b + \sum_{k_1,k_2=-K}^K w[k_1,k_2] x_{\texttt{in}}[s_n n-k_1,s_m m-k_2] & \quad\quad \triangleleft \quad \texttt{conv}\quad \text{(strided)} \end{aligned} \tag{24.5}
$$

其中 $s_n$ 和 $s_m$ 分別是垂直和水平方向上的步幅。

在本文及下文中，我們針對最簡單的情況定義了運算，即單個方形濾波器與單通道 2D 訊號的卷積。所有這些運算都可以直接擴展到多通道輸入、多通道輸出的情況，以及 ND 訊號 and 非方形卷積核。我們將這些變體留作讀者的練習，以便在需要時寫出。

通常我們使用相同的步幅 $s_n = s_m = s$。具有這些步幅的卷積層會執行一個映射 $\mathbb{R}^{M \times N} \rightarrow \mathbb{R}^{N/s_n \times M/s_m}$。為了使此映射有明確的定義，我們要求 $N$ 或 $M$ 必須分別能被 $s_n$ 和 $s_m$ 整除；如果不能，我們可以對輸入進行填充（padding，或裁剪 crop）直到可以整除為止。

步幅卷積看起來像這樣（圖 24.7）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/strided_conv_diagram.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/strided_conv_diagram.png "Figure 24.7: Strided convolution.")

圖 24.7：步幅卷積。

當神經網路龐大時，步幅卷積可以顯著減少計算成本與記憶體需求。然而，步幅卷積可能會降低卷積的品質。讓我們看一個具體例子，其中卷積核是 2D 拉普拉斯（Laplacian）卷積核：
$$
\mathbf{w} =\begin{bmatrix} 0 ~& -1 ~& 0 \\\ -1 ~& 4 ~& -1\\\ 0~& -1 ~& 0 \end{bmatrix}
$$

正如我們在[第 18 章](derivatives.html)中看到的，這個濾波器用來檢測圖像上的邊界。圖 24.8 顯示了輸入圖像，以及使用步幅為 1、2 和 4 的拉普拉斯卷積核進行步幅卷積的結果。第二行顯示了離散傅立葉轉換（DFT）的振幅。

輸入 | 步幅 1 | 步幅 2 | 步幅 4
---|---|---|---
[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_0.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_0.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_1.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_1.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_2.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_2.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_4.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_4.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.")
[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_0_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_0_DFT.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_1_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_1_DFT.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_2_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_2_DFT.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_4_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_stride_4_DFT.jpg "Figure 24.8: Figure: Strided convolution results and their Fourier transforms.")

圖 24.8：**圖：**步幅卷積結果及其傅立葉轉換。

步幅為 1 的結果看起來很好，也是我們預期的輸出。然而，步幅為 2 開始在邊界上顯示出一些偽影（artifacts），而步幅為 4 則顯示出非常嚴重的偽影，甚至有些邊界消失了。DFT 讓這些偽影更加明顯。在步幅為 2 的結果中，我們可以看到嚴重的混疊（aliasing）偽影，這在傅立葉域中引入了輸入圖像 DFT 中不存在的新線條。

有人可能會認為，當卷積核是在學習過程中被學習出來時，這些偽影可能並不重要。確實，學習過程可以尋找能最小化混疊所引起之偽影的卷積核，因為這些偽影可能會增加損失（loss）。此外，由於每層由許多通道組成，一組被學習出來的卷積核可以學習補償其他通道產生的混疊。然而，這縮小了有用卷積核的空間，且學習可能無法成功消除所有偽影。

### 24.2.3 空洞卷積

空洞卷積（Dilated convolution，也稱膨脹卷積）與步幅卷積類似，但它是拉開**濾波器本身**的間距，而不是拉開濾波器應用於圖像時的間距：
$$
\begin{aligned} x_{\texttt{out}}[n,m] = b + \sum_{k_1,k_2=-K}^K w[k_1,k_2] x_{\texttt{in}}[n-d_kk_1,m-d_kk_2] & \quad\quad \triangleleft \quad \texttt{conv}\quad \text{(dilated)} \end{aligned} \tag{24.6}
$$

在這裡，我們在兩個空間維度上都以因子 $d_k$ 進行空洞化，但我們也可以在每個維度上選擇不同的空洞率（dilation rate）。或者，如果我們使用多通道卷積，我們甚至可以在通道維度上進行空洞化，但這並不常見。

圖 24.9 直觀地展示了一個空洞濾波器的範例：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/dilated_conv_diagram.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/dilated_conv_diagram.png "Figure 24.9: Dialated convolution. Dark gray cells have value 0.")

圖 24.9：空洞卷積。深灰色單元的值為 0。

如視覺化所示，空洞化是一種在僅需少量權重的情況下，實現具有大卷積核之濾波器的方法。權重只是被拉開間距，以便少數的權重就能覆蓋圖像中更大的區域。

與步幅卷積的情況一樣，空洞化也可能引入偽影。讓我們詳細看一個具體例子，以說明空洞化對濾波器的影響。讓我們考慮模糊卷積核 $b_{2,2}$：

$$
\mathbf{w} = \frac{1}{16}\begin{bmatrix} 1 ~& 2 ~& 1 \\\ 2 ~& 4 ~& 2\\\ 1~& 2 ~& 1 \end{bmatrix}
$$

這個濾波器藉由計算每個像素位置周圍像素強度的加權平均值來模糊輸入圖像。但是，空洞化改變了這個濾波器，改變了濾波器的行為，使其不再具有模糊濾波器的功能。

我們看到 1D 訊號 $[-1, 1, -1, ...]$ 與 $[1, 2, 1]$ 卷積後輸出為零。然而，看看當我們將輸入與空洞卷積核 $[1, 0, 2, 0, 1]$ 進行卷積時會發生什麼事。

下圖顯示了空洞率為 $d_k=1$、$d_k=2$ 和 $d_k=4$ 的卷積核，以及三個所得卷積核的 DFT 振幅（圖 24.10）。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_kernel_binomial.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_kernel_binomial.png "Figure 24.10: Dialed filters and their Fourier transforms.")

圖 24.10：空洞濾波器及其傅立葉轉換。

當使用原始二項式濾波器（對應於 $d_k=1$）時，DFT 顯示該濾波器是一個低通濾波器。當應用空洞化（$d_k=2$）時，DFT 發生了變化，且不再是單峰的。它現在在空間高頻處多出了八個局部極大值。當 $d_k=4$ 時，DFT 顯現出更複雜的頻率行為。圖 24.11 顯示了一張輸入圖像，以及使用模糊卷積核 $b_{2,2}$ 在空洞率為 $d_k=1$、$d_k=2$ 和 $d_k=4$ 下進行空洞卷積的結果。

當使用原始二項式濾波器（對應於 $d_k=1$）時，DFT 顯示該濾波器是一個低通濾波器。當應用空洞化（$d_k=2$）時，DFT 發生了變化，且不再是單峰的。它現在在空間高頻處多出了八個局部極大值。當 $d_k=4$ 時，DFT 顯現出更複雜的頻率行為。圖 24.11 顯示了一張輸入圖像，以及使用模糊卷積核 $b_{2,2}$ 在空洞率為 $d_k=1$、$d_k=2$ 和 $d_k=4$ 下進行空洞卷積的結果。

輸入 | $d_k$ = 1 | $d_k$ = 2 | $d_k$ = 4
---|---|---|---
[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_1.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_1.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_2.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_2.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_3.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_3.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_4.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_4.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.")
[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_1_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_1_DFT.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_2_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_2_DFT.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_3_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_3_DFT.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.") | [![](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_4_DFT.jpg)](https://visionbook.mit.edu/figures/convolutional_neural_nets/aliasing_dilated_4_DFT.jpg "Figure 24.11: Result of the dilated convolutions with the blur kernel, b_{2,2}, with dilations d_k=1, d_k=2, and d_k=4.")

圖 24.11：使用模糊卷積核 $b_{2,2}$ 在空洞率為 $d_k=1$、$d_k=2$ 和 $d_k=4$ 下進行空洞卷積的結果。

總結來說，使用空洞化增加了卷積核的大小而不需要增加計算量（這是原本期望的特性），但它縮小了有用卷積核的空間（這是不期望的特性）。

有一些方法可以利用空洞化來增加有用濾波器的族群。例如，藉由將三個空洞率分別為 $d_k=1$、$d_k=2$ 和 $d_k=4$ 的卷積組合在一起（圖 24.12），可以創建一個在學習過程中能夠在空間高頻與低頻、以及小型與大型卷積核之間切換的卷積核。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/dilated_conv_cascade.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/dilated_conv_cascade.png "Figure 24.12: Convolving dilated filters creates a new filter that can measure effects at a mixture of frequencies.")

圖 24.12：將多個空洞濾波器進行卷積可創建一個新濾波器，能測量多種頻率混合的效應。

這會產生一個大小為 $9 \times 9$（81 個值）的卷積核，但僅由 27 個值定義。當我們級聯（cascade）更多具有更高空洞率的濾波器時，相對計算效率會提高。圖 24.13 顯示了數個多尺度卷積核，這些卷積核可藉由三個空洞卷積核的卷積獲得。你能猜出使用的是哪些卷積核嗎？

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/dilated_examples.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/dilated_examples.png "Figure 24.13: Example kernels that each result from convolving three filters.")

圖 24.13：各由三個濾波器卷積產生的範例卷積核。

如圖所示，三個空洞卷積的級聯可以產生一大類具有不同尺度、方向、偏移的濾波器，以及其他圖樣，例如角點檢測器、長邊緣檢測器和彎曲邊緣檢測器。最後四個卷積核顯示了三個隨機卷積核卷積的結果，這進一步說明了人們可以建構之卷積核的多樣性。每個卷積核都是一個從高斯分佈中採樣的 $3 \times 3$ 陣列。

### 24.2.4 低秩濾波器

空洞化是創建大型濾波器的一種方法，該濾波器僅由少數權重參數化，也就是說，它是一個低秩濾波器。當我們知道優質濾波器具有低秩結構時，這個技巧在許多情況下都很有用。空洞化利用這個技巧來製作大型卷積核，從而捕捉長距離的依賴關係。

可分離濾波器是另一種在許多應用中都非常有用的低秩濾波器（參見[第 16 章](image_processing_fourier.html)）。我們可以藉由簡單地將兩個卷積層順序堆疊，且中間不夾雜其他層，來創建一個具有可分離濾波器的卷積層。第一層是具有 $K \times 1$ 卷積核的濾波器組，第二層則使用 $1 \times K$ 卷積核。這些層的組合相當於一個具有 $K \times K$ 可分離濾波器的單個卷積層。下面給出兩個此類可分離濾波器的例子（圖 24.14）：

當對一個列向量和一個行向量進行卷積時，$\mathbf{w} = \mathbf{u}^\mathsf{T} \circ \mathbf{v}$，結果是外積：$w \left[n,m \right] = u\left[n \right] v\left[m \right]$。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/kernels_separable_aprox.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/kernels_separable_aprox.png "Figure 24.14: Two examples of separable filters.")

圖 24.14：可分離濾波器的兩個範例。

一些重要的卷積核是不可分離的，但可以用少數可分離濾波器的線性組合來近似。例如，高斯拉普拉斯（Gaussian Laplacian）是不可分離的，但可以用可分離濾波器來近似，如圖 24.15 所示：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/laplacian_separable_aprox.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/laplacian_separable_aprox.png "Figure 24.15: Approximating a Gaussian Laplacian filter as the outer product of two 1D filters.")

圖 24.15：將高斯拉普拉斯濾波器近似為兩個 1D 濾波器的外積。

對角高斯導數（diagonal Gaussian derivative）是另一個不可分離的卷積核。當使用 $3 \times 3$ 卷積核來近似它時，我們有：

$$
\mathbf{w} = \begin{bmatrix} 0 ~& -2 ~& -2 \\\ 2 ~& 0 ~& -2\\\ 2~& 2 ~& 0 \end{bmatrix}
$$

但我們從[第 18 章](derivatives.html)中知道，這個卷積核可以寫成兩個可分離卷積核的線性組合：$\mathbf{w} = \text{Sobel}_x + \text{Sobel}_y$，如公式（[Equation 18.3](derivatives.html#eq-sobel_kernels)）中所定義的。一般來說，任何 $M \times N$ 的濾波器都可以分解為 $\min(N,M)$ 個可分離濾波器的線性求和。可分離濾波器可以藉由對卷積核陣列 $\mathbf{w}$ 進行奇異值分解（SVD）來獲得。SVD 會產生三個矩陣 $\mathbf{U}$、$\mathbf{S}$ 和 $\mathbf{V}$，使得 $\mathbf{w} = \mathbf{U} \mathbf{S} \mathbf{V}^\mathsf{T}$，其中 $\mathbf{U}$ 和 $\mathbf{V}$ 的行（columns）是可分離的 1D 濾波器，對角矩陣 $\mathbf{S}$ 的對角值是線性權重。只有在對大型卷積核使用小型線性組合時，才能獲得計算上的好處。此外，在神經網路中，可以對所有單元僅使用可分離濾波器，且學習過程可以發現組合它們的方法，以便建構更複雜的非分離卷積核。

### 24.2.5 降採樣與上採樣層

在[第 23 章](pyramids_new_notation.html)中，我們看到了圖像金字塔，並展示了它們如何用於分析和合成。CNN 也可以被建構成分析與合成金字塔，這是一個非常強大的工具。要創建金字塔，我們只需要引入一種在分析期間對訊號進行降採樣，並在合成期間進行上採樣的方法。在 CNN 中，這是透過**降採樣與上採樣層**來完成的。

降採樣層將輸入張量轉換為空間維度較小的輸出張量：$\mathbb{R}^{N \times M} \rightarrow \mathbb{R}^{N/s_n \times M/s_m}$。我們已經看到了一種降採樣層，即步幅卷積，它等同於卷積後再進行子採樣（subsampling）。另一種常見的降採樣層是**池化（pooling）**，我們將在第 24.3.1 節中遇到它。

上採樣層執行相反的轉換，輸出的張量在空間維度上大於輸入：$\mathbb{R}^{N \times M} \rightarrow \mathbb{R}^{Ns_n \times Ms_m}$。

一種上採樣層可以製作為步幅卷積的對應物。步幅卷積先進行卷積再子採樣；而這個上採樣層則是先對訊號進行空洞化（dilate，或插零）再進行卷積。從一個全零的空白圖像 $\mathbf{h} = \mathbf{0}$ 開始，我們設定：
$$
\begin{aligned} h[ns_n, ms_m] &= x_{\texttt{in}}[n, m] & \quad\quad \triangleleft \quad \texttt{dilation}\\\\ \mathbf{x}_{\texttt{out}}&= \mathbf{w} \star \mathbf{h} + b & \quad\quad \triangleleft \quad \texttt{conv} \end{aligned} \tag{24.7}
$$

此公式適用於所有整數值 $n \in \{1,\ldots,N\}$ 和 $m \in \{1,\ldots,M\}$。

有時，這兩層的組合被稱為 UpConv 層或反卷積層（deconvolution layer，但請注意，反卷積在訊號處理中具有不同的含義）。

## 24.3 非線性濾波層

我們上面涵蓋的所有運算都是線性（或仿射）的。定義非線性的濾波器也是可行的。與卷積濾波器類似，這些濾波器在輸入張量上滑動，並以相同且獨立的方式處理每個窗口，但它們執行的運算是局部窗口的非線性函數。

### 24.3.1 池化層

**池化層**是降採樣層，它們使用某些聚合統計量來總結區塊中的資訊，例如區塊的平均值，稱為**均值池化（mean pooling）**，或者其最大值，稱為**最大值池化（max pooling）**，定義如下：

$$
\begin{aligned} x_{\texttt{out}}[i]= \max_{i \in \mathcal{N}(i)} x_{\texttt{in}}[i]& \quad\quad \triangleleft \quad \texttt{max pooling}\\\\ x_{\texttt{out}}[i]= \frac{1}{|\mathcal{N}|} \sum_{i \in \mathcal{N}(i)} x_{\texttt{in}}[i]& \quad\quad \triangleleft \quad \texttt{mean pooling} \end{aligned} \tag{24.8}
$$

$\mathcal{N}(i)$ 表示與索引 $i$ 在同一個區塊中的索引集合。

與所有降採樣層一樣，池化層可用於降低輸入張量的解析度，從而去除訊號中的高頻資訊。池化在實現**不變性（invariance）**方面也特別有用。卷積層產生的輸出對於其輸入的平移具有**等變性（equivariant）**。池化是將等變性轉化為不變性的一種方法。例如，假設我們運行了一個檢測垂直邊緣的卷積濾波器。其輸出是一個響應圖，在輸入圖像中存在垂直邊緣的任何地方，該響應圖的值都很大。現在，如果我們在這個響應圖上運行一個最大值池化濾波器，它將使該響應圖粗糙化，從而在輸入圖像中靠近垂直邊緣的任何地方產生較大的響應。如果我們使用具有足夠大鄰域 $\mathcal{N}$ 的最大值池化濾波器，輸出將對輸入圖像中邊緣的位置具有不變性。

池化也可以跨通道進行，這可以作為實現額外種類不變性的一種方式。例如，假設我們有一個卷積層，它應用了一個由定向邊緣檢測器濾波器組成的濾波器組，其中每個濾波器尋找不同方向的邊緣。現在，如果我們對這個濾波器組輸出的通道進行最大值池化，那麼不論發現**任何**方向的邊緣，所得的特徵圖都會很大。通常我們不是在尋找邊緣，而是在尋找更複雜的圖樣，但同樣的邏輯依舊適用：首先運行一組尋找 $k$ 個不同方向圖樣的濾波器，然後跨這 $k$ 個通道進行池化，即可檢測該圖樣而不受其方向的影響。這對於 CNN 來說是一個極佳的方法，即使物體在圖像中以各種旋轉角度出現，也能夠識別它們。當然，我們通常不會手工定義這種策略，但如果給定跨通道的池化層，這是 CNN 可以學習去使用的一種策略。

### 24.3.2 全局池化層

池化的一種極限是跨特徵圖的整個空間範圍進行池化。全局池化（Global pooling）是一個將 $C \times M \times N$ 的張量映射到長度為 $C$ 的向量的函數，其中 $C$ 是輸入中的通道數。

全局池化通常用於非常接近輸出的層。如前所述，全局池化可以是**全局平均池化（global average pooling）**，即對特徵圖的所有響應進行平均，或者是**全局最大池化（global max pooling）**，即取特徵圖的最大值。

全局池化會從每個通道中移除空間資訊。然而，如果不同通道學習到對不同空間位置的特徵敏感，那麼關於輸入特徵的空間資訊在輸出向量中可能仍然是可用的。

### 24.3.3 局部歸一化層

另一種非線性濾波器是**局部歸一化層（local normalization layer）**。這些層利用某些鄰域內相鄰激活值的統計數據，來對特徵圖中的每個激活值進行歸一化。對於歸一化類型（$L_1$ 範數、$L_2$ 範數、標準化等）有許多不同的選擇，對於鄰域形狀也有許多不同的選擇，例如空間維度中的方形區塊、一組通道等等。這些選擇中的每一種都會導向具有不同名稱之不同類型的歸一化濾波器。其中一個在歷史上非常重要但現在已不常使用的是 AlexNet 論文 [[1](references.html#ref-krizhevsky2012imagenet)] 中引入的**局部響應歸一化（local response normalization，簡稱 LRN）**濾波器。該濾波器具有以下形式：
$$
\begin{aligned} x_{\texttt{out}}[c,n,m] = x_{\texttt{in}}[c,n,m] / \left( \gamma + \alpha \sum_{i=\max(1,c-l)}^{\max(C,c+l)} x_{\texttt{in}}[i,n,m]^2 \right) ^\beta \quad\quad \triangleleft \quad\texttt{LRN} \end{aligned} \tag{24.9}
$$

其中 $\alpha$、$\beta$、$\gamma$ 和 $l$ 是該層的超參數。這一層藉由相鄰**通道**窗口中激活值的平方和來對每個激活值進行歸一化。

儘管局部歸一化是大腦中常見的結構，但在目前的神經網路中並不常用，現在更常使用的是全局歸一化層，例如 batchnorm 或 layernorm（我們在[第 12 章](neural_nets.html)中看過）。

## 24.4 一個簡單的 CNN 分類器

CNN 是深度網路，它們將卷積層串聯堆疊，並在中間穿插非線性激活函數。CNN 也經常使用降採樣與上採樣層、池化層和歸一化層，如上所述。

CNN 有著各式各樣的架構，每一種都適用於不同類型的問題。我們將在第 24.11 節中看到其中的一些架構。目前，我們將專注於一個適用於圖像分類的簡單架構。這個架構會逐步降採樣圖像，直到最後一層對圖像標籤做出單個全局預測（圖 24.16）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/convnet_motif.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/convnet_motif.png "Figure 24.16: A CNN architecture for image classification.")

圖 24.16：用於圖像分類的 CNN 架構。

我們現在將逐步說明此類分類器的範例。設 $\mathbf{x} \in \mathbb{R}^{M \times N}$ 為一張黑白圖像。為了處理這張圖像，我們可以使用一個具有兩個卷積層的簡單 CNN，其定義如下：

$$
\begin{align} \mathbf{z}_1[c,:,:] &= \mathbf{w}[c,:,:] \star \mathbf{x} + b[c] &\triangleleft \quad \texttt{conv}: [M \times N] \rightarrow [C \times M \times N]\\\ h[c,n,m] &= \max(z_1[c,n,m],0) &\triangleleft \quad \texttt{relu}: [C \times M \times N] \rightarrow [C \times M \times N]\\\ z_2[c] &= \frac{1}{NM} \sum_{n,m} h[c,n,m] &\triangleleft \quad \texttt{gap}: [C \times M \times N] \rightarrow [C]\\\ \mathbf{z}_{3} &= \mathbf{W} \mathbf{z}_{2} + \mathbf{c} &\triangleleft \quad \texttt{fc}: [C] \rightarrow [K]\\\ y[k] &= \frac{e^{-\tau z_3[k]}}{\sum_{l=1}^K e^{-\tau z_3[l]}} &\triangleleft \quad \texttt{softmax}: [K] \rightarrow [K] \end{align}
$$

請注意，這些公式適用於所有 $c \in \{0,\ldots,C-1\}$、$n \in \{0,\ldots,N-1\}$ 和 $m \in \{0,\ldots,M-1\}$。

這個網路具有一個帶有 $C$ 個通道的卷積層，隨後是一個 relu layer。下一層執行空間全局平均池化（`gap`），每個通道都被投影到一個單一數值中，該數值包含 relu 輸出的總和。這產生了一個由長度為 $C$ 的向量所給出的表示。接著，該向量由一個**全連接層**（`fc`）進行處理。全連接層只是滿秩（full rank）線性層的另一個名稱，也就是說，每個輸出神經元都與每個輸入神經元相連，且該映射由一個 $K \times C$ 的矩陣（加上一個偏置）來描述。

這個神經網路可以用來解決一個 $K$ 類別的圖像分類問題（因為對於每個輸入圖像，輸出是一個 $K$ 類別的 softmax）。我們可以使用梯度下降法來訓練它，以尋找優化訓練數據上交叉熵損失的參數 $\theta = [\mathbf{w}_1, \ldots, \mathbf{w}_C, \mathbf{b}_1, \ldots, \mathbf{b}_C, \mathbf{W}, \mathbf{c}]$。

一旦我們有了卷積和 softmax 等基本運算的基元庫（library of primitives），在程式碼中定義這樣一個網路也非常容易：


    # 首先定義參數化層
    conv1 = nn.conv(channels_in=1, channels_out=C, kernel=k, stride=1)
    fc1 = nn.fc(dim_in=C, dim_out=K)

    # 然後將數據輸入網路中
    z1 = conv1(x)
    h = nn.relu(z1)
    z2 = nn.AvgPool2d(h)
    z3 = fc1(z2)
    y = nn.softmax(z3) __

## 24.5 實作範例

在本節中，我們將分析第 24.4 節中描述的簡單網路，該網路經過訓練以區分水平線和垂直線。每個小節將解決分析的一個方面，這應該是訓練任何大型系統的一部分：(1) 訓練與評估、(2) 視覺化並理解網路、(3) 域外泛化（out-of-domain generalization），以及 (4) 識別脆弱性。

### 24.5.1 訓練與評估

讓我們研究一個簡單的分類任務。我們設計了一個包含線條的簡單圖像數據集。線條可以是水平的或垂直的。每張圖像將僅包含一種類型的線條。

我們想設計一個 CNN，根據圖像所包含線條的方向對圖像進行分類。我們將兩個輸出類別定義為：$0$（垂直）和 $1$（水平）。圖 24.17 顯示了訓練集中的一些樣本。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_trainingset.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_trainingset.png "Figure 24.17: A sample of images from the training set. The training set defines the concept we want to learn. In this case we look for lines. But images of lines might not be enough to describe the concept of a line. What is a line? This lack of a precise definition will haunt us later.")

圖 24.17：來自訓練集的圖像樣本。訓練集定義了我們想要學習的概念。在這個例子中，我們尋找線條。但線條的圖像可能不足以描述線條的概念。什麼是線條？這種缺乏精確定義的問題稍後將會困擾我們。

為了解決這個問題，我們使用先前定義 the CNN，在第一層中使用兩個卷積通道 $C=2$。一旦我們訓練了網路，我們可以看到它完美地解決了任務，並且在測試集上的輸出是 100% 正確的（在 10,000 張測試圖像中只有三個錯誤）。測試集中的範例圖像如圖 24.18 所示。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_testset.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_testset.png "Figure 24.18: A sample of images from the test set. The predicted label is shown at the top.")

圖 24.18：來自測試集的圖像樣本。預測標籤顯示在頂部。

### 24.5.2 網路視覺化

網路學到了什麼？它是如何解決問題的？系統開發的一個重要部分是擁有能夠證明、理解和調試它的工具。

為了理解網路，**視覺化**卷積核是有用的。圖 24.19 顯示了兩個學習到的 $9 \times 9$ 卷積核。第一個看起來像高斯濾波器的水平導數（正如我們在[第 18 章](derivatives.html)中看到的），第二個看起來像高斯濾波器的垂直導數（可能更接近二階導數）。事實上，每個卷積核的 DFT 顯示它們對圖像中特定頻帶的頻率內容非常具有選擇性。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_kernels.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_kernels.png "Figure 24.19: Visualization of the learned kernels.")

圖 24.19：學習到的卷積核視覺化。

全連接層學到了權重：
$$
\mathbf{W} = \left[ \begin{array}{cc} 2.83 & -2.36 \\\ -0.60 & 1.14 \end{array} \right]
$$

這對應於兩個通道的對立：第一個特徵是垂直輸出減去水平輸出，第二個特徵計算水平輸出減去垂直輸出。

### 24.5.3 域外泛化

藉由分析訓練後的網路是如何運作的，我們能學到什麼？一個有趣的結果是，我們可以預測先前定義的網路如何推廣到訓練集分佈之外的**域外測試樣本**。

域外（out-of-domain）的另一個術語是**分佈外（out-of-distribution）**。

例如，很自然地會認為，即使圖像中包含的不是線條，該網路在分類圖像包含垂直結構還是水平結構方面仍應表現良好。我們可以藉由產生符合我們方向概念的圖像來測試這個假設。以下測試圖像（圖 24.20）包含不同的定向結構但沒有線條，並且仍然捕捉到了我們對於什麼是正確行為泛化的概念。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_generalization.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_generalization.png "Figure 24.20: Out-of-domain test examples. The predicted label is shown at the top.")

圖 24.20：域外測試範例。預測標籤顯示在頂部。

事實上，即使使用這些來自與訓練集不同分佈的新圖像，網路似乎也能正確執行。

### 24.5.4 識別脆弱性

網路是否解決了我們心目中的任務？我們能否預測哪些輸入會導致輸出失敗？我們能否製作出在我們看來是正確的，但網路卻產生錯誤分類輸出的測試範例？此分析的目的是識別學習到的表示和訓練集中的弱點（缺失的訓練範例、數據中的偏置、架構的局限性等）。

我們看到第一層的輸出實際上並不是在尋找**線條**，而是看能量在傅立葉域中的位置。因此，我們可以藉由創建在我們看來是垂直線，但能量內容卻在傅立葉域中錯誤一側的線條，來愚弄分類器。我們在[第 16 章](image_processing_fourier.html)中看到了一個實現此目的的技巧：調變（modulation）。如果我們將包含水平線的圖像乘以正弦波 $\cos (\pi n / 3)$，我們可以將頻譜內容水平移動，如圖 24.21 所示。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_test_adversarial_creation1.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_test_adversarial_creation1.png "Figure 24.21: Creating a new out-of-domain test image obtained by multiplying an in-domain test image with horizontal lines \(left image its DFT\) with a sinusoidal wave. The resulting image and its DFT are shown on the right\).")

圖 24.21：創建一個新的域外測試圖像，該圖像藉由將具有水平線的域內測試圖像（左圖及其 DFT）乘以正弦波獲得。所得圖像及其 DFT 顯示在右側。

這些線條在我們看來仍然是水平的，但它們現在的頻譜內容在與網路學習到的垂直線檢測器重疊的區域中更高。確實，當網路處理具有這種**正弦紋理**線條的圖像時，它對所有圖像都產生了錯誤的分類結果（圖 24.22）！

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_test_adversarial.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/oriented_bars_cnn_test_adversarial.png "Figure 24.22: Classification results on out-of-domain test images created by modulation.")

圖 24.22：在藉由調變創建的域外測試圖像上的分類結果。

我們剛剛手動設計了一個**對抗樣本**！那麼問題可能是：如果它不是在檢測線條方向，它到底是在檢測什麼？我們對學習到的卷積核進行的分析給出了答案。

對於複雜的架構，**對抗樣本**是藉由解一個優化問題來獲得的：什麼是能使網路產生錯誤輸出的最小輸入擾動？

避免這種情況的一種方法是將這種類型的圖像引入訓練集中並重複整個過程。

## 24.6 CNN 中的特徵圖

在使用 CNN 時，最重要的概念之一是特徵圖（feature map）。特徵圖可以是卷積層輸出的單個通道（如我們上面所定義的），也可以指神經網路某層中整疊的通道。其核心思想是，這些是輸入數據的**特徵**，且這些特徵被排列在一個與輸入數據形狀相匹配的**圖（map）**中。對於圖像，特徵圖是 2D 空間陣列；對於影片，它們是 3D 時空陣列，依此類推。

圖 24.23 顯示了 CNN 中特徵圖與濾波器組之間的相互作用：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/feature_maps_schematic.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/feature_maps_schematic.png "Figure 24.23: The interplay between feature maps and filters banks in a CNN. You can think of the input image itself as an R, G, B feature map and the output logits as a 1x1 resolution feature map with class logits as the channels.")

圖 24.23：CNN 中特徵圖與濾波器組之間的相互作用。你可以將輸入圖像本身視為一個 R、G、B 特徵圖，而將輸出對數機率（logits）視為一個以類別對數機率為通道的 1x1 解析度特徵圖。

網路的輸入是一張圖像，輸出是一個對數機率向量。我們實際上也可以將這些輸入和輸出視為特徵圖：輸入只是一個具有紅、綠、藍通道的特徵圖，而輸出是一個以類別對數機率為通道的 1x1 解析度特徵圖。

現在讓我們看看一個真實網路 AlexNet [[1](references.html#ref-krizhevsky2012imagenet)] 中的特徵圖。圖 24.24 顯示了在網路的第一和第二個卷積層之後，這些特徵圖看起來是什麼樣子：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/alexnet_feature_maps.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/alexnet_feature_maps.png "Figure 24.24: A selection of feature maps and filters in AlexNet. The layer 1 filter with the orange dashed border slides over the input image and produces the layer 1 feature map with the orange dashed border. The layer 2 filter with the green dashed border slides over the layer 1 feature maps and produces the layer 2 feature map with the green dashed border.")

圖 24.24：AlexNet 中精選的特徵圖和濾波器。帶有橙色虛線邊框的第 1 層濾波器在輸入圖像上滑動，產生帶有橙色虛線邊框的第 1 層特徵圖。帶有綠色虛線邊框的第 2 層濾波器在第 1 層特徵圖上滑動，產生帶有綠色虛線邊框的第 2 層特徵圖。

在此圖中有幾點需要注意。首先，隨著我們深入網路，特徵圖的空間解析度會變低，且通道數會增加。這在 CNN 中很常見：每一層進行降採樣並增加通道，以部分補償解析度的降低。其次，雖然在第一層特徵圖對輸入圖像中的基本圖樣（邊緣、線條等）很敏感，但隨著我們深入，特徵圖變得更加抽象。這是圖像分類器網路的典型特徵：淺層中的通道捕捉基本圖像特徵，而深層中的通道則越來越對應於類別語意（例如，一個通道可能是「鳥」像素所在位置的熱圖）。

圖 24.25 顯示了視覺化特徵圖的另一種方法。我們不將通道繪製為一列灰階圖像，而是運行 PCA 將通道維度降低到 3。然後我們可以將每個特徵圖直接渲染為彩色圖像，其中紅色顯示每層特徵圖的第一主成分，綠色顯示第二主成分，藍色顯示第三主成分。我們針對三種常見網路 AlexNet、VGG16 [[2](references.html#ref-vgg16)] 和 ResNet18 [[3](references.html#ref-he2016deep)] 的 5 個層展示此結果。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/feature_maps_pca_viz.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/feature_maps_pca_viz.png "Figure 24.25: PCA visualization of feature maps in three convolutional networks. Because each of these networks has a different number of layers, we select 5 that are evenly spaced from the first to last convolutional feature map.")

圖 24.25：三種卷積網路中特徵圖的 PCA 視覺化。因為這些網路中的每一個都具有不同數量的層，所以我們選擇了從第一個到最後一個卷積特徵圖均勻分佈的 5 個層。

## 24.7 感受野

**感受野（Receptive fields）**是使用 CNN 時的另一個重要概念。在[第 1 章](taxonomy.html)中，我們了解了感受野在神經科學中的歷史。提醒一下，神經元的感受野是該神經元所敏感的輸入訊號區域，即其支持集（support）。在多層感知器（MLPs，[第 12 章](neural_nets.html)）中，每個神經元的感受野是整個輸入向量，因為 MLP 使用的是**全連接層**。另一方面，在 CNN 中，每個神經元只看到輸入的一部分，因為卷積層上的每個輸出神經元僅連接到該卷積層輸入的一個子集，該子集由產生該輸出的濾波器的卷積核大小決定。

CNN 中兩個範例神經元的感受野如下圖所示（圖 24.26）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/RFs.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/RFs.png "Figure 24.26: Receptive fields in a CNN. The black filled neurons are within the receptive fields of each labeled neuron \(left: x_2\[3\], right: x_1\[5\]\).")

圖 24.26：CNN 中的感受野。黑色填充的神經元位於每個標記神經元的感受野內（左：$x_2[3]$，右：$x_1[5]$）。

請注意，我們越深入網路，感受野就越大。要理解其中的原因，請考慮一個沒有非線性激活函數的 CNN。那麼第 $l+1$ 層就是 $l$ 個卷積濾波器的組合。正如我們在[第 15.4.1 節](linear_image_filtering.html#sec-linear_image_filtering-properties_of_the_convolution)中看到的，組合多個濾波器會產生一個具有更大支持度（卷積核大小）的新濾波器。在具有逐點（pointwise）非線性激活函數的 CNN 中也是如此，因為逐點運算不會影響感受野（輸出與輸入具有相同的感受野）。此外，每當我們有一個降採樣率為 $s$ 的降採樣層時，輸出的感受野比輸入的感受野大 $s$ 倍。由於這些特性，隨著我們在 CNN 中走得更深，感受野大小可以迅速增長。通常，我們希望 CNN 的最後一層具有足夠大的感受野以看見整個輸入圖像，以便輸出神經元對輸入中的**所有**像素都敏感。這可以藉由一個 `gap`（全局平均池化）層來實現，其輸出將始終具有覆蓋整個輸入的感受野大小。

## 24.8 空間輸出

在第 24.4 節中，我們看到了一個對圖像輸出單個類別機率向量的 CNN。如果我們想像本章引言中討論的那樣，輸出一個空間上變化的預測圖，該怎麼辦？為了實現這一點，我們只需減少降採樣，使得 CNN 的最後一層是保持較高空間解析度的特徵圖。

移除任何全局池化層也至關重要。

一個範例如下所示：

$$
\begin{aligned} \mathbf{z}_1[c_1,:,:] &= \sum_{c=0}^2 \mathbf{w}_1[c,c_1,:,:] \star \mathbf{x} + b_1[c_1] &\triangleleft \quad \texttt{conv}\\\\ &&[3 \times N \times M] \rightarrow [C_1 \times N \times M]\nonumber\\\\ h[c_1,n,m] &= \max(z_1[c_1,n,m],0) &\triangleleft \quad \texttt{relu}\\\\ &&[C_1 \times N \times M] \rightarrow [C_1 \times N \times M]\nonumber\\\\ \mathbf{z}_2[c_2,:,:] &= \sum_{c_1=0}^{C_1-1} \mathbf{w}_2[c_1,c_2,:,:] \star \mathbf{x} + b_2[c_2] &\triangleleft \quad \texttt{conv}\\\\ &&[C_1 \times N \times M] \rightarrow [C_2 \times N \times M]\nonumber\\\\ y[k,n,m] &= \frac{e^{-\tau z_2[k,n,m]}}{\sum_{l=1}^K e^{-\tau z_2[l,n,m]}} &\triangleleft \quad \texttt{softmax}\\\\ &&[K \times N \times M] \rightarrow [K \times N \times M]\nonumber \end{aligned}
$$

在圖 24.27 中，我們視覺化了這個 CNN（僅展示了這個 2D CNN 的 1D 切片）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/image_to_image_arch.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/image_to_image_arch.png "Figure 24.27: A 1D slice of a CNN that maps an image to an image. The input is the photo of size 3 \\times N \\times M and the output is a class probability map of size K \\times N \\times M; we visualize the corresponding label map on the right \(per-pixel argmax over output probabilities\). The blue arrows are the learnable parameters. The gray arrows share the weights of the blue arrows.")

圖 24.27：將圖像映射到圖像之 CNN 的 1D 切片。輸入是大小為 $3 \times N \times M$ 的照片，輸出是大小為 $K \times N \times M$ 的類別機率圖；我們在右側視覺化了對應的標籤圖（對輸出機率進行逐像素 argmax 運算）。藍色箭頭是可學習的參數。灰色箭頭共享藍色箭頭的權重。

符號提醒：方形節點表示它們代表多個通道（每個節點都是一個神經元向量）。

儘管在歷史上，CNN 最初是作為圖像分類器流行起來的，但這種用法隱藏了它們真正的威力。與其將它們視為「圖像到標籤（image-to-label）」的架構，不如將它們視為**圖像到圖像（image-to-image）**的架構。

更廣泛地說，CNN 是適用於任何可以定義平移（translation）之領域 $\mathcal{X}$ 的 $\mathcal{X}$ 到 $\mathcal{X}$（$\mathcal{X}$-to-$\mathcal{X}$）架構。

## 24.9 作為滑動濾波器的 CNN

CNN 的核心是卷積層，在本節中，我們將考慮一個僅包含 `conv` 層且其間穿插逐點非線性激活函數的 CNN。這樣的 CNN 有時被稱為**全卷積網路**或 **FCN** [[4](references.html#ref-FCNs)]。我們將在下面展示，整個 FCN 只是另一個滑動圖像濾波器。

要理解其原因，請考慮一個處理 1D 訊號並輸出特徵圖 $\mathbf{x}_L$ 的 CNN。取出輸出圖中的兩個特徵向量 $\mathbf{x}_L[:,i]$ 和 $\mathbf{x}_L[:,j]$。位置 $i$ 處的特徵向量是其感受野中輸入區塊的某個函數 $F$，即 $\mathbf{x}_L[:,i] = F(\mathbf{x}_{\texttt{in}}[:,\texttt{RF}(i)])$，其中 $\texttt{RF}$ 返回感受野在輸入圖像中的座標。事實證明，像素 $j$ 處的特徵向量也是由**同一個**函數產生的，只是應用於輸入的不同區塊：$\mathbf{x}_L[:,j] = F(\mathbf{x}_{\texttt{in}}[:,\texttt{RF}(j)])$。

這最容易藉由視覺化證明來理解，我們在圖 24.28 中給出（為了清晰起見，省略了逐點非線性激活函數）：

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNN_as_filter.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/CNN_as_filter.png "Figure 24.28: A CNN is a non-linear filter. Edge colors indicate shared weights; two edges with the same color have the same weight. The colors demonstrate that the same function F is applied to each patch of input nodes.")

圖 24.28：CNN 是一個非線性濾波器。邊緣顏色表示共享的權重；具有相同顏色的兩個邊緣具有相同的權重。這些顏色證明了同一個函數 $F$ 被應用於輸入節點的每個區塊。

為了理解這個特性，首先想像 CNN 沒有逐點非線性激活函數。那麼整個 CNN 就只是序列卷積的組合，其本身就是一個卷積（根據公式（[Equation 15.5](linear_image_filtering.html#eq-linear_image_filtering-conv_associative_property)），連續用多個濾波器對訊號進行卷積，等同於用單個等效濾波器對訊號進行卷積）。因此，一個沒有非線性的 CNN 本身只是一個單個大型卷積濾波器。這種系統的關鍵特性在於它獨立且相同地處理每個輸入區塊。現在請注意，當我們加入逐點非線性激活函數時，這個關鍵特性並未改變，因為它們並未引入神經元或像素之間的相互作用（畢竟它們是逐點的）。因此可以得出，僅由卷積層和逐點非線性激活函數組成的完整 CNN 本身就是一個非線性運算子，它獨立且相同地將相同的轉換應用於輸入訊號的每個區塊，即一個非線性濾波器！

這就是為什麼在本章引言中，我們將 CNN 視覺化為將圖像切碎成區塊，並將相同的「分類器」函數應用於每個區塊。

## 24.10 為什麼要逐區塊處理圖像？

正如我們在上面看到的，一個全卷積的 CNN 可以被認為是一個獨立且相同地處理輸入中每個區塊的函數。

CNN 是一個非線性濾波器。邊緣顏色表示共享的權重；具有相同顏色的兩個邊緣具有相同的權重。這些顏色證明了同一個函數 $F$ 被應用於輸入節點的每個區塊。

在本節中，我們將討論為什麼這兩個特性對圖像處理非常有用。

##### 特性 #1：將區塊視為獨立的

這是一種分而治之（divide-and-conquer）的策略。如果你試圖理解一個複雜的問題，你可能會將其拆解成多個小部分並分別解決每一個部分。這就是 CNN 所做的一切。我們將一個大問題（即「解讀整張照片」）拆解為一堆較小的問題（即「解讀圖像中的每個小區塊」）。

為什麼這是一個好策略？

  1. 小問題比原問題更容易解決。

  2. 小問題可以全部並行解決。

  3. 這種方法**與訊號長度無關（agnostic to signal length）**，也就是說，你只需將一個任意大的問題分解為一口大小的碎片，並「一隻鳥一隻鳥地（bird by bird）」解決它們即可 [[5](references.html#ref-lamott1980)]。

像這樣切碎成小區塊對於許多視覺問題來說是足夠的，因為世界展現出**局部性（locality）**：相關的事物會聚集在一起，即在單個區塊內；相距甚遠的事物通常可以安全地被假設為相互獨立的。

##### 特性 #2：相同地處理每個區塊

對於圖像，卷積是一種特別合適的策略，因為視覺內容往往是**平移不變的（translation invariant）**，且正如我們在前面章節中所學到的，卷積運算子也是平移不變的。

通常，物體可以出現在圖像中的任何位置並且看起來相同，就像圖 24.1 照片中的鳥類一樣。這是因為當鳥類飛過畫面時，它們的位置發生了變化，但它們的身份和外觀並未改變。更廣泛地說，當相機橫移過一個場景時，內容在位置上發生了偏移，但在其他方面保持不變。

因為視覺世界大致上是平移不變的，所以無論每個區塊的位置如何（即它相對於某個規範中心區塊的平移），以相同的方式處理每個區塊都是合理的。

**平移不變（Translation invariant）**只是意味著我們使用相同的函數 $f$ 相同地處理每個區塊。一些文獻則使用該術語來描述卷積。這強調了這樣一個事實：如果我們將輸入訊號平移某個量，那麼輸出訊號也將平移相同的量。也就是說，如果 $f$ 是一個卷積，我們具有以下特性：

$$
\begin{aligned} f(\texttt{translate}(\mathbf{x})) = \texttt{translate}(f(\mathbf{x})) \end{aligned}
$$

## 24.11 熱門 CNN 架構

我們現在已經看到了 CNN 的所有基本建構區塊。在本節中，我們將像對待樂高積木一樣對待這些建構區塊，並展示如何將它們拼接在一起以製作各種實用的架構。

### 24.11.1 編碼器與解碼器

在[第 23 章](pyramids_new_notation.html)中，我們遇到了圖像金字塔，它包括分析管線（將圖像轉換為濾波器響應的多尺度表示）與合成管線（從濾波器響應中重建圖像）。深度網路也可以在分析方向或合成方向上運行。在深度網路的背景下，我們將分析網路稱為**編碼器（encoder）**，將合成網路稱為**解碼器（decoder）**。編碼器將數據映射到該數據的表示形式（通常是較低維度的），而解碼器執行相反的運算，將表示映射回數據。

編碼器和解碼器網路將在本書中多次出現，並且可以由許多不同的架構製成，包括 those 那些不是神經網路的架構。在 CNN 的背景下，編碼器通常是接收圖像作為輸入並逐層降採樣它，直到輸出一個維度低得多的特徵圖的網路。解碼器則相反，接收一組低維度特徵作為輸入，然後逐層上採樣它們，直到輸出圖像作為最終輸出。編碼器的例子是圖像分類器，解碼器的例子是圖像生成器（在[第 32 章](generative_models.html)中介紹）。這兩種架構模式如圖 24.29 所示。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/encoders_and_decoders.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/encoders_and_decoders.png "Figure 24.29: A convolutional encoder \(top\) and a convolutional decoder \(bottom\). The exact ordering of the operations \(e.g., downsample before or after the non-linearity\) is just an example and may vary in different encoder and decoder models. The \\mathbf{z} is a feature map or vector of neural activations, and is sometimes called an embedding \(see sec-representation_learning\)")

圖 24.29：卷積編碼器（上）與卷積解碼器（下）。運算的確切順序（例如，在非線性激活之前或之後進行降採樣）僅為示例，在不同的編碼器和解碼器模型中可能有所不同。$\mathbf{z}$ 是一個特徵圖或神經激活向量，有時也被稱為**嵌入（embedding）**（參見[第 30 章](representation_learning.html)）。

你可以對編碼器和解碼器做的強大事情之一是將它們放在一起，形成一個**編碼器-解碼器（encoder-decoder）**。此類架構首先將輸入圖像編碼為低維度表示，然後將該表示解碼回圖像輸出。因此，這是一個適用於圖像到圖像問題的架構，並且與我們在第 24.8 節中看到的圖像到圖像 CNN 相比，它具有一些優勢：1) 藉由降採樣，內部特徵圖更小，使用更少的記憶體和計算量，2) 編碼器-解碼器引入了**資訊瓶頸（information bottleneck）**——即編碼器和解碼器之間的表示是低維度的，且只能傳輸有限的資訊——這迫使表示進行抽象。我們將在[第 30 章](representation_learning.html)中更詳細地研究後者這一概念，在那裡我們將看到壓縮訊號的若干好處。編碼器-解碼器架構的示意圖如圖 24.30 所示。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/encoder_decoder_arch.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/encoder_decoder_arch.png "Figure 24.30: Encoder-decoder architecture. In this example, the input is an image and the output is a segmentation map.")

圖 24.30：編碼器-解碼器架構。在此範例中，輸入是一幅圖像，輸出是一幅分割圖（segmentation map）。

### 24.11.2 U-Nets

編碼器-解碼器迫使訊號通過一個瓶頸，儘管這可能是一件好事（如上所述），但也使得解碼器的任務變得相當困難。特別是，解碼器可能無法輸出高頻細節；對於語意分割網路，其後果可能是預測的標籤圖非常粗糙。

為了避開這個問題，我們可以加入直接穿梭於網路中各層區塊之間的通道。跳躍連接（skip connection） $f$ 是一個簡單的恆等式路徑，連接網路上兩個不同的層，$f(\mathbf{x}) = \mathbf{x}$。

將跳躍連接加入編碼器-解碼器會產生一種被稱為 **U-Net** 的架構 [[6](references.html#ref-ronneberger2015u)]。在這種架構中，跳躍連接以鏡像圖樣排列，其中第 $l$ 層直接連接到第 $(L-l)$ 層。跳躍連接的輸出必須以某種方式重新整合到網路中。U-Net 藉由沿著通道維度，將來自前一層的激活值與後一層的激活值拼接（concatenate）來實現這一點。這種架構可以保持編碼器-解碼器的資訊瓶頸，及其在記憶體和計算效率以及強制抽象方面的固有好處，同時也允許殘差資訊流過跳躍連接，從而不犧牲輸出高頻空間預測的能力。U-Net 看起來非常像[第 23 章](pyramids_new_notation.html)中的可操縱金字塔（steerable pyramids），後者也由降採樣**分析**路徑和隨後的上採樣**合成**路徑組成，且在路徑中的鏡像階段之間設有跳躍連接。主要區別在於 U-Net 具有**學習到的濾波器** and **非線性激活函數**。U-Net 的示意圖如圖 24.31 所示。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/unet.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/unet.png "Figure 24.31: U-net architecture. Each block contains a series of layers. The skip connections concatenate activations.")

圖 24.31：U-Net 架構。每個區塊包含一系列層。跳躍連接會拼接激活值。

### 24.11.3 殘差網路 (ResNets)

另一個使用跳躍連接的熱門架構被稱為**殘差網路（Residual Networks，或 ResNets）** [[3](references.html#ref-he2016deep)]。

另請參見 Highway Networks [[7](references.html#ref-srivastava2015highway)]，這是一個相關的架構，它也使用了一種跳躍連接，但由乘性閘（multiplicative gates）控制。

在 ResNets 的背景下，跳躍連接被稱為**殘差連接（residual connections）**。這種類型的跳躍連接被**加**到一個層區塊 $F$ 的輸出中：
$$
\begin{aligned} \mathbf{x}_{\texttt{out}}= F(\mathbf{x}_{\texttt{in}}) + \mathbf{x}_{\texttt{in}}\quad\quad \triangleleft \quad \texttt{residual block} \end{aligned}
$$

在這樣一個**殘差區塊（residual block）**中，你可以將 $F$ 視為一個**殘差**，它以加性方式微擾 $\mathbf{x}_{\texttt{in}}$，以將其轉換為改進後的 $\mathbf{x}_{\texttt{out}}$。如果 $\mathbf{x}_{\texttt{out}}$ 與 $\mathbf{x}_{\texttt{in}}$ 的維度不相同，那麼我們可以加入一個線性映射來轉換維度：$\mathbf{x}_{\texttt{out}}= F(\mathbf{x}_{\texttt{in}}) + \mathbf{W}\mathbf{x}_{\texttt{in}}$。

對於殘差區塊來說，要簡單地執行恆等映射是非常容易的，它只需學習將 $F$ 設置為零即可。正因為如此，如果我們連續堆疊許多殘差區塊，網路最終可能會只學習使用其中的一個子集。如果我們將堆疊 of 殘差區塊數量設置得非常大，那麼網路本質上可以學習自身應該具有多深，根據解決任務的需要使用盡可能多的區塊。ResNets 經常利用這個事實來做到非常深；例如，它們可能深達數百個區塊。圖 24.32 描繪了一個 5 個區塊深的 ResNet。

[![](https://visionbook.mit.edu/figures/convolutional_neural_nets/resnet.png)](https://visionbook.mit.edu/figures/convolutional_neural_nets/resnet.png "Figure 24.32: ResNet architecture. Each block contains a series of layers. The skip connections add activations from one block to the next.")

圖 24.32：ResNet 架構。每個區塊包含一系列層。跳躍連接將一個區塊的激活值加到下一個區塊。

## 24.12 結語

我們將在後面的章節中看到，幾種新型模型最近正在取代 CNN，成為視覺問題中最成功的架構。其中一種架構是 **Transformer**（[第 26 章](transformers.html)）。可能有人會想：「如果 Transformer 更好，那我們為什麼還要費心學習 CNN 呢！」我們介紹 CNN 的原因，並非因為此處展示的具體架構會永遠存在，而是因為它所體現的底層原理在感官處理中無處不在。第 24.10 節中提到的兩個關鍵特性實際上也存在於 Transformer 以及 CNN 之外的許多其他架構中：Transformer 也包含獨立且相同地處理每個區塊的階段，但它們將這些階段與跨區塊全局化資訊的其他階段穿插在一起。不論你是否想稱這些較新的架構為卷積，完全取決於個人偏好，目前在社群中對此也存在一些爭論。對我們來說這並不重要，因為如果我們學會了這些原理，我們就可以在我們遇到的所有系統中識別出它們，而不需要糾結於名稱。

[1] A. Krizhevsky, I. Sutskever, G.E. Hinton, Imagenet classification with deep convolutional neural networks., in: Nips, 2012: pp. 1097–1105.

[2] K. Simonyan, A. Zisserman, Very deep convolutional networks for large-scale image recognition., in: Iclr, 2015.

[3] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition., in: Cvpr, 2016: pp. 770–778.

[4] J. Long, E. Shelhamer, T. Darrell, Fully convolutional networks for semantic segmentation., in: Cvpr, 2015: pp. 3431–3440.

[5] A. Lamott, Bird by bird., Bantam Doubleday Dell Publishing Group, 1980.

[6] O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation., in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2015: pp. 234–241.

[7] R.K. Srivastava, K. Greff, J. Schmidhuber, Highway networks., (2015). <https://arxiv.org/abs/1505.00387>.
