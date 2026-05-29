# 12 神經網路

## 12.1 前言

神經網路（Neural networks）是模擬大腦運作方式所建立的函數模型。在大腦中，我們有數百億個相互連接的神經元（neurons）。每個神經元都可以被視為圖（graph）中的一個節點（node），而邊（edges）則是連接神經元之間的管道（[圖 12.1](#fig-neural_nets-fig1_net)）。這些邊是有方向性的，電訊號在大腦的導線中僅朝單一方向傳播。

<a id="fig-neural_nets-fig1_net"></a>

![圖 12.1：神經網路可以繪製為一個有向圖。](https://visionbook.mit.edu/figures/neural_nets/fig1_net.png)

輸出的邊稱為軸突（axons），輸入的邊稱為樹突（dendrites）。當來自樹突的輸入訊號總和超過某個門檻時，神經元就會「激發」（fires），沿著其軸突發送脈衝。

## 12.2 感知器：單一神經元的簡單模型

讓我們考慮一個灰色陰影標示的神經元，它具有四個輸入和一個輸出（[圖 12.2](#fig-neural_nets-perceptron_fig2)）。

<a id="fig-neural_nets-perceptron_fig2"></a>

![圖 12.2：單一神經元的一個簡單模型是感知器。](https://visionbook.mit.edu/figures/neural_nets/perceptron_fig2.png)

此神經元的一個簡單模型是感知器（perceptron）。感知器是一個具有 $n$ 個輸入 $\{x_i\}_{i=1}^n$ 和一個輸出 $y$ 的神經元，它根據以下方程將輸入映射到輸出：<a id="eq-neural_nets-perceptron_activation"></a>

$$
\begin{aligned}
z = f(\mathbf{x}) &= \sum_{i=1}^n w_i x_i + b = \mathbf{w}^\mathsf{T}\mathbf{x} + b &\triangleleft \quad \text{線性層}\\
    g(z) &= \begin{cases}
    1, &\text{若} \quad z > 0\\
    0,              & \text{否則}
\end{cases} &\triangleleft \quad \text{啟用函數}\\
    y &= g(f(\mathbf{x})) &\triangleleft \quad \text{感知器}
\end{aligned} \tag{12.1}
$$

簡而言之，我們對輸入進行加權求和，如果該總和超過門檻（此處為 0），神經元就會激發（輸出 1）。函數 $f$ 被稱為線性層（linear layer），因為它計算輸入的線性函數 $\mathbf{w}^\mathsf{T}\mathbf{x}$ 加上一個**偏差**（bias）$b$。函數 $g$ 被稱為啟用函數（activation function），因為它決定神經元是否啟用（激發）。

數學上，$f$ 是一個仿射函數（affine function），但按照慣例我們稱之為「線性層」。一種思考方式是，$f$ 是 $\begin{bmatrix}\mathbf{x}\ 1\end{bmatrix}$ 的線性函數。

### 12.2.1 作為分類器的感知器

感知器在 1950 年代後期引起了人們的極大關注，因為研究表明它們可以學習對數據進行分類 [[1](#ref-rosenblatt1958perceptron)]。讓我們來看看它是如何運作的。我們將考慮一個具有兩個輸入 $x_1$ 和 $x_2$ 以及一個輸出 $y$ 的感知器。假設輸入連接權重為 $w_1 = 2$、$w_2 = 1$，且 $b=0$。$z$ 和 $y$ 作為 $x_1$ 和 $x_2$ 的函數值如 [圖 12.3](#fig-neural_nets-perceptron_as_classifier) 所示。

<a id="fig-neural_nets-perceptron_as_classifier"></a>

![圖 12.3：感知器中隱藏單元 (\(z\)) 和輸出單元 (\(y\)) 的值，作為輸入數據的函數。](https://visionbook.mit.edu/figures/neural_nets/perceptron_as_classifier.png)

請注意，$y$ 的值為 0 或 1，因此您可以將其視為一個分類器，為繪製區域的右上半部分分配類別標籤 1。

### 12.2.2 使用感知器進行學習

感知器就像一個分類器，但我們該如何用它來學習呢？這個概念是，給定數據 $\{\mathbf{x}^{(i)}, y^{(i)}\}_{i=1}^N$，我們將調整權重 $\mathbf{w}$ 和偏差 $b$，以最小化分類損失（classification loss）$\mathcal{L}$，該損失對錯誤分類的數量進行評分：

<a id="eq-neural_nets-perceptron_learning_problem"></a>

$$
\begin{aligned}
z = f(\mathbf{x}) &= \sum_{i=1}^n w_i x_i + b = \mathbf{w}^\mathsf{T}\mathbf{x} + b &\triangleleft \quad \text{linear layer}\\
    g(z) &= \begin{cases}
    1, &\text{if} \quad z > 0\\
    0,              & \text{otherwise}
\end{cases} &\triangleleft \quad \text{activation function}\\
    y &= g(f(\mathbf{x})) &\triangleleft \quad \text{perceptron}
\end{aligned} \tag{12.1}
$$

在 [圖 12.4](#fig-neural_nets-fitting_a_perceptron) 中，此優化過程對應於移動和旋轉**決策邊界**（decision boundary），直到找到一條線將標記為 $y = 0$ 的數據與標記為 $y = 1$ 的數據分開。

<a id="fig-neural_nets-fitting_a_perceptron"></a>

![圖 12.4：感知器不同可能的決策表面。](https://visionbook.mit.edu/figures/neural_nets/fitting_a_perceptron.png)

您可能會好奇，尋找分開類別之最佳直線的具體優化演算法是什麼？原始的感知器論文提出了一種特殊的演算法，即「感知器學習演算法」。這是一種針對感知器特定結構客製化的優化器。關於神經網路的早期論文充滿了針對特定架構的具體學習規則：Delta 規則、Rescorla-Wagner 模型等等 [[2](#ref-rescorla1972theory)]。如今，我們很少使用這些特殊用途的演算法。相反，我們使用*通用*優化器，例如梯度下降（用於可微分目標）或零階方法（用於不可微分目標）。下一章將介紹反向傳播（backpropagation）演算法，這是一種通用的基於梯度的優化器，幾乎適用於我們在本書中將看到的所有神經網路（但請注意，對於感知器目標，由於它具有不可微分的步階閾值函數，我們會選擇零階優化器）。

## 12.3 多層感知器

感知器可以解決線性可分的二元分類問題，但在其他方面相當受限。首先，它們僅產生單個輸出。如果我們想要多個輸出呢？我們可以藉由在感知器之後增加向外發散的邊來實現（[圖 12.5](#fig-neural_nets-fan_out)）。

<a id="fig-neural_nets-fan_out"></a>

![圖 12.5：多個輸出從一個神經元發散出來。](https://visionbook.mit.edu/figures/neural_nets/fan_out.png)

這個網絡將輸入**層**數據 $\mathbf{x}$ 映射到輸出**層** **y**。輸入和輸出之間的神經元被稱為**隱藏單元**（hidden units），以灰色陰影表示。此處，$z$ 是一個隱藏單元，而 $h$ 也是一個隱藏單元，即 $h = g(z)$，其中 $g(\cdot)$ 是如 [公式 12.1](#eq-neural_nets-perceptron_activation) 中所示的啟用函數。

更常見的情況是，我們可能會有許多隱藏單元堆疊在一起，我們稱之為多層感知器（[圖 12.6](#fig-neural_nets-MLP1)）。

<a id="fig-neural_nets-MLP1"></a>

![圖 12.6：多層感知器。](https://visionbook.mit.edu/figures/neural_nets/MLP1.png)

這個網路有幾層？有些教科書會說是兩層 [$\mathbf{W}_1$, $\mathbf{W}_2$]，有些說是三層 [$\mathbf{x}$, $\{\mathbf{z}, \mathbf{h}\}$, $\mathbf{y}$]，有些則說是四層 [$\mathbf{x}$, $\mathbf{z}$, $\mathbf{h}$, $\mathbf{y}$]。我們必須習慣這種模糊性。

因為這個網絡具有多層神經元，且網絡中的每個神經元都扮演著感知器的角色，我們稱之為**多層感知器**（**multilayer perceptron**，簡稱 **MLP**）。此 MLP 的方程式為：

$$
\begin{aligned}    \mathbf{z} &= \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 &\triangleleft \quad \text{線性層}\\
    \mathbf{h} &= g(\mathbf{z}) &\triangleleft \quad \text{啟用函數}\\
    \mathbf{y} &= \mathbf{W}_2\mathbf{h} + \mathbf{b}_2 &\triangleleft \quad \text{線性層}
\end{aligned}
$$

一般而言，MLP 可以按照此模式構建任意數量的圖層：線性層、啟用函數、線性層、啟用函數，依此類推。

啟用函數 $g$ 可以是如 [公式 12.1](#eq-neural_nets-perceptron_activation) 的門檻函數，但更一般地，它可以是任何逐點非線性（pointwise nonlinearity），即 $g(\mathbf{h}) = [\tilde{g}(h_1), \ldots, \tilde{g}(h_N)]$，其中 $\tilde{g}$ 是任何將 $\mathbb{R} \rightarrow \mathbb{R}$ 的非線性函數。

除了 MLP 之外，這種序列（線性層、逐點非線性、線性層、逐點非線性等）是幾乎所有神經網路中的典型基本單元（motif），包括我們稍後將在本書中看到的絕大多數網絡。

## 12.4 啟用值與參數

在處理深度網路時，區分「啟用值」（activations）和「參數」（parameters）非常有用。啟用值是神經元所取得的值 $[\mathbf{x}, \mathbf{z}_1, \mathbf{h}_1, \ldots, \mathbf{z}_{L-1}, \mathbf{h}_{L-1}, \mathbf{y}]$；稍微濫用一下符號，我們將此術語同時用於啟用函數前（preactivation）的神經元與啟用函數後（postactivation）的神經元。啟用值是被處理數據的神經表徵（neural representations）。通常，我們不會糾結於區分網路的輸入、隱藏單元和輸出，而是簡單地將網路中逐層的所有數據和神經啟用值稱為序列 $[\mathbf{x}_0, \ldots, \mathbf{x}_L]$，在這種情況下，$\mathbf{x}_0$ 是原始輸入數據。

![多層網絡是由一系列轉換 \(f_1, \ldots, f_L\) 組成，這些轉換會產生一系列啟用值 \(\mathbf{x}_1, \ldots, \mathbf{x}_L\)。](https://visionbook.mit.edu/figures/neural_nets/transformations.png)

相反地，參數是網絡的權重和偏差。這些是正在被學習的變數。啟用值和參數都是變數的張量（tensors）。

通常我們將一層視為一個函數 $\mathbf{x}_{l+1} = f_{l+1}(\mathbf{x}_l)$，但我們也可以將參數顯式化，並將每一層視為一個函數：

$$
\begin{aligned}
    \mathbf{x}_{l+1} = f_{l+1}(\mathbf{x}_{l}, \theta_{l+1})
\end{aligned}
$$

也就是說，每一層都將前一層的啟用值以及當前層的參數作為輸入，並產生下一層的啟用值。改變輸入啟用值或輸入參數都會影響該層的輸出。從這個角度來看，我們能用參數做的任何事情，也都可以用啟用值來代替，反之亦然，這正是許多應用和技巧的基礎。例如，雖然我們通常是學習參數的值，但我們也可以選擇保持參數固定，轉而學習能夠實現某些目標的啟用值。事實上，許多方法（如網絡提示（prompting）、對抗性攻擊和網絡視覺化）正是這麼做的，我們將在後續章節中更詳細地看到這些內容。

### 12.4.1 快速啟用值與慢速參數

那麼，啟用值與參數有什麼不同呢？一種思考方式是，啟用值是針對個別「數據點」的「快速」函數：它們是處理該數據點的少數幾層網路運算的結果。參數「也是」數據的函數（它們是從數據中學習而來的），但它們是針對「數據集」的「慢速」函數：參數是透過在整個數據集上進行優化程序而得到的。因此，啟用值和參數都是數據的統計量，也就是從數據中提取出來、用以組織或總結數據的資訊。參數是一種「後設總結」（metasummary），因為它們指定了從數據產生啟用值的函數式轉換，而啟用值本身則是數據的總結。[圖 12.7](#fig-neural_nets-params_vs_activations) 顯示了這個過程。

<a id="fig-neural_nets-params_vs_activations"></a>

![圖 12.7：學習是一個將數據集映射到參數的函數。透過神經網路進行推論（inference）則是一個將數據點映射到啟用值的函數。](https://visionbook.mit.edu/figures/neural_nets/params_vs_activations.png)

## 12.5 深度網路

深度網路（Deep nets）是將線性-非線性單元堆疊多次的神經網路（[圖 12.8](#fig-deep_nets)）：

<a id="fig-deep_nets"></a>

![圖 12.8：深度網絡由線性層與非線性函數交織而成。](https://visionbook.mit.edu/figures/neural_nets/deep_nets.png)

每一層都是一個函數。因此，深度網路是許多函數的複合（composition）：

$$
\begin{aligned}
    f(\mathbf{x}) = f_{L}(f_{L-1}(\ldots f_2(f_1(\mathbf{x}))))
\end{aligned}
$$

其中 $L$ 是網路中的層數。

這些函數由權重 $[\mathbf{W}_1, \ldots, \mathbf{W}_L]$ 和偏差 $[\mathbf{b}_1, \ldots, \mathbf{b}_L]$ 參數化。我們稍後會看到某些層還具有其他參數。總體而言，我們將深度網路中所有參數的拼接稱為 $\theta$。

深度網路之所以強大，是因為它們可以執行非線性映射。事實上，一個擁有足夠多神經元的深度網路可以極其精確地擬合幾乎任何所需的函數，我們將在 [第 12.5.2 節](#sec-neural_nets-universal_approximation) 進一步探討此性質。

### 12.5.1 深度網路可執行非線性分類

讓我們回到前面顯示的二元分類問題，但現在讓這兩個類別變得非線性可分。我們的新數據集如 [圖 12.9](#fig-neural_nets-nonseparable_dataset) 所示。

<a id="fig-neural_nets-nonseparable_dataset"></a>

![圖 12.9：非線性可分的數據集。](https://visionbook.mit.edu/figures/neural_nets/nonseparable_dataset.png)

在這裡，沒有任何一條直線能將 0 和 1 分開。儘管如此，我們將展示一個可以解決此問題的多層網路。秘訣就是增加更多層！我們將使用如 [圖 12.10](#fig-neural_nets-simple_MLP_network) 所示的雙層 MLP。

<a id="fig-neural_nets-simple_MLP_network"></a>

![圖 12.10：一個簡單的 MLP 網絡。](https://visionbook.mit.edu/figures/neural_nets/simple_MLP_network.png)

考慮對 $\mathbf{W}_1$ 和 $\mathbf{W}_2$ 使用以下設定：

$$
\begin{aligned}
    \mathbf{W}_1 =
        \begin{bmatrix}
            1 & -1 \\
            2 & 1
        \end{bmatrix}
    ,\quad\quad
    \mathbf{W}_2 =
        \begin{bmatrix}
            1 & -1
        \end{bmatrix}
\end{aligned}
$$

整個網路隨即執行以下運算：

$$
\begin{aligned}
    z_1 &= x_1 - x_2, \quad z_2 = 2x_1 + x_2 &\triangleleft \quad \texttt{線性}\\
    h_1 &= \max(z_1,0), \quad h_2 = \max(z_2,0) &\triangleleft \quad \texttt{ReLU}\\
    z_3 &= h_1-h_2 &\triangleleft \quad \texttt{線性}\\
    y &= \mathbb{1}(z_3 > 0) &\triangleleft \quad \text{\texttt{門檻值}}
\end{aligned}
$$

其中 $\mathbb{1}$ 是指示函數（indicator function），我們定義為：

$$
\begin{equation*}
    \mathbb{1}(x) = \begin{cases}
                1 & \text{若 $x$ 為真} \\
                0 & \text{否則}
             \end{cases}
\end{equation*}
$$

在這裡，我們引入了一種新的逐點非線性函數：**修正線性單元**（**Rectified linear unit**，簡稱 **ReLU**），它就像是門檻函數的漸進版本，其優點是在其一半的定義域內產生非零梯度，從而便於進行基於梯度的學習。

我們在 [圖 12.11](#fig-neural_nets-simple_MLP_network_values) 中將神經元的值視覺化為 $x_1$ 和 $x_2$ 的函數：

<a id="fig-neural_nets-simple_MLP_network_values"></a>

![圖 12.11：圖 12.10 中所示 MLP 的隱藏單元和輸出單元的值。](https://visionbook.mit.edu/figures/neural_nets/simple_MLP_network_values.png)

如最右側的圖表所示，在輸出 $y$ 處，該神經網路成功地將值 1 分配給數據空間中標記為 1 的數據點所在的區域。這個例子表明，使用深度網路可以解決非線性分類問題。在實務上，我們會希望「學習」能夠實現此分類的參數設定。一種方法是列舉所有可能的參數設定，並選擇一個成功將 0 和 1 分開的設定。這種窮舉列舉是一個緩慢的過程，但別擔心，在 [第 14 章](https://visionbook.mit.edu/backpropagation.html) 中，我們將看到如何使用梯度下降來加速。但值得指出的是，列舉始終是一個足夠的解決方案，至少在可能的參數值形成有限集合時是如此。

<a id="sec-neural_nets-universal_approximation"></a>

### 12.5.2 深度網路是通用逼近器

深度網路不僅能執行非線性分類，原則上還能執行「任何」連續的輸入-輸出映射。**通用逼近定理**（**universal approximation theorem**）[[3](#ref-Cybenko1989)] 表明，即使是僅具有單個隱藏層的網路也是如此。但限制是，隱藏層中的神經元數量必須非常大，才能擬合複雜的函數。

嚴格來說，該定理僅適用於 $\mathbb{R}^N$ 緊緻子集上的連續函數——例如，神經網路無法擬合不可計算的函數。我們在本節中不進行嚴格證明。有興趣的讀者可以參考 [[4](#ref-telgarsky2016benefits)] 以獲得關於通用逼近的正式探討。

為了直觀理解這為什麼是真的，我們將考慮使用 ReLU-MLP（具有 ReLU 非線性的 MLP）來逼近 $\mathbb{R} 
\rightarrow \mathbb{R}$ 的任意函數。首先觀察到，任何函數都可以藉由指示函數（即放置在不同位置的脈衝/階梯函數）之和來任意逼近：<a id="eq-neural_nets-sum_of_indicators"></a>

$$
\begin{aligned}
    f(x) \approx \sum_i w_i \mathbb{1}(\alpha_i < x < \beta_i)
\end{aligned} \tag{12.3}
$$

例如，在 [圖 12.12](#fig-neural_nets-curve_as_bump) 中，我們顯示了一條以此方式逼近的曲線（藍線）。隨著脈衝（黑線）的寬度 $\beta-\alpha$ 趨近於零，擬合誤差也會趨近於零。

<a id="fig-neural_nets-curve_as_bump"></a>

![圖 12.12：任何 \(\mathbb{R} \rightarrow \mathbb{R}\) 的函數都可以藉由基本脈衝函數的和來任意逼近。](https://visionbook.mit.edu/figures/neural_nets/curve_as_bump.png)

雖然我們這裡只考慮純量函數 $\mathbb{R} 
\rightarrow \mathbb{R}$，但類似的構造也可以用來逼近 $\mathbb{R}^n 
\rightarrow \mathbb{R}^m$ 形式的一般函數。

接下來，我們將展示一個 ReLU-MLP 可以表示 [公式 12.3](#eq-neural_nets-sum_of_indicators)。加權和 $\sum_i w_i \ldots$ 是簡單的部分：那只是一個線性層。因此，我們只需要展示我們也可以使用線性層和 ReLU 層來寫出 $\mathbb{1}(\alpha < x < \beta)$。

結果表明，其構造相當簡單：<a id="eq-neural_nets_bump_as_relu_net"></a>

$$
\begin{aligned}
&\mathbb{1}(\alpha < x < \beta)\\
&\approx \\
&\texttt{relu}\bigg(\frac{x-(\alpha-\gamma)}{\gamma}\bigg) - \texttt{relu}\bigg(\frac{x-\alpha}{\gamma}\bigg) - \texttt{relu}\bigg(\frac{x-(\beta-\gamma)}{\gamma}\bigg) + \texttt{relu}\bigg(\frac{x-\beta}{\gamma}\bigg)
\end{aligned}
\tag{12.4}
$$

在這裡，我們展示了神經網路如何將函數表示為基底函數（basis functions）之和。這個概念也是訊號處理的基石，在訊號處理中，訊號通常表示為正弦波之和（[第 16 章](https://visionbook.mit.edu/image_processing_fourier.html)）、方塊（[圖 21.18](https://visionbook.mit.edu/upsamplig_downsampling_2.html#fig-upsampling_and_downsampling-nn_interp)）或梯形（[圖 21.19](https://visionbook.mit.edu/upsamplig_downsampling_2.html#fig-bilinear_interp)）。

當 $\gamma 
\rightarrow 0$ 時，此逼近會變得精確。[公式 12.4](#eq-neural_nets_bump_as_relu_net) 中四個 ReLU 的每一個輸入都是輸入 $x$ 的仿射函數，因此這四個值可以由一個具有四個輸出的線性層來表示。然後我們對這四個值應用一個 ReLU 層，最後我們應用一個線性層來計算這些 ReLU 的和（權重為 [1, -1, -1, 1] 的加權和）。因此，[公式 12.4](#eq-neural_nets_bump_as_relu_net) 可以實現為一個 `線性`-`ReLU`-`線性` 的網路。在 [圖 12.13](#fig-neural_nets-bump_as_relus) 中，我們展示了以此方式構建脈衝的示例：

<a id="fig-neural_nets-bump_as_relus"></a>

![圖 12.13：一個脈衝可以表示為平移和縮放之 ReLU 函數的加權和。](https://visionbook.mit.edu/figures/neural_nets/bump_as_relus.png)

綜合以上，我們為每個脈衝設計了一個 `線性`-`ReLU`-`線性` 網路，隨後使用一個線性層來將所有脈衝相加。序列中的兩個線性層可以合併為單個線性層，因此整個函數可以藉由一個 `線性`-`ReLU`-`線性` 網路逼近至任意精度。

大多數文獻將此類網路稱為具有單個隱藏層的網路，這採用了我們不將啟用前與啟用後神經元算作獨立層的慣例。

請注意，在此逼近中，我們建模的每個脈衝都需要四個 ReLU 神經元。因此，如果我們要逼近一個非常多起伏的函數，例如具有 $N$ 個脈衝，我們將需要 $4N$ 個 ReLU 神經元。一般而言，要在曲線擬合上達到任意好的逼近，我們的網路可能需要無限數量的神經元。

### 12.5.3 深度與寬度

上面我們看到，如果您有一個包含 $N$ 個神經元的隱藏層，您可以擬合一個具有 $\mathcal{O}(N)$ 個脈衝的純量函數。單個隱藏層上的神經元數量稱為其**寬度**（width）。

如果不同的層具有不同數量的神經元，則我們可以指定每層的寬度。在這裡，我們假設所有層都具有相同的寬度，並簡單地稱之為網路的寬度。

因此，隨著我們增加網路的寬度，我們可以擬合愈加複雜的函數。如果我們轉而增加網路的**深度**（depth，即其層數）呢？事實證明，這也是提高網路容量的有效方法，但其效果與增加寬度有些許不同。

有趣的是，有時*深度*網路擬合數據所需的參數比*寬度*網路少得多。這一結論主要來自經驗主義，研究人員發現，在許多熱門問題上，較深的網路在實務上的效果就是更好。然而，這也揭示了關於何時以及為什麼會發生這種情況之數學理論的開端。該理論的基本概念是確立某些類別的函數在深度為 $d$ 的網路中可以使用多項式數量的神經元來表示，但在深度為 $d^\prime$ 的網路中則需要指數數量的神經元（對於某些 $d^\prime < d$）。沿著這些思路進行的論證被稱為**深度分離**（**depth separations**），有興趣的讀者可以參考 [[4](#ref-telgarsky2016benefits)] 以深入了解這一持續進行的研究方向。

## 12.6 深度學習：使用神經網路進行學習

使用我們在 [第 9 章](https://visionbook.mit.edu/intro_to_learning.html) 中定義的公式，學習包含使用一個*優化器*在一個*假設空間*（hypothesis space）中尋找一個函數，以最大化一個*目標*（objective）。從這個角度來看，神經網路僅僅是一種特殊的假設空間（以及該假設空間的特定參數化）。**深度學習**（**Deep learning**）就是指使用此參數化假設空間的學習演算法。

深度學習通常還涉及使用基於梯度的優化來搜尋假設空間，以尋找與數據的最佳擬合。我們將在 [第 14 章](https://visionbook.mit.edu/backpropagation.html) 中詳細探討此方法，在那裡我們將學習用於神經網路基於梯度學習的**反向傳播**（**backpropagation**）演算法。然而，使用其他方法優化神經網路當然也是可能的，包括零階優化器，如演化策略（Evolution Strategies）[[5](#ref-salimans2017evolution)]（見 [第 10.6.1 節](https://visionbook.mit.edu/gradient_descent.html#sec-gradient_descent-zeroth_order)）。

反向傳播的一個迷人替代方案被稱為**赫布學習**（**Hebbian learning**）[[6](#ref-hebb2005organization)]。反向傳播是一種*由上而下*（top-down）的學習演算法，在網路輸出端（頂部）產生的誤差會向後傳播，以通知較早的層如何更新其權重和偏差以最小化損失，這是一種基於*反饋*（feedback）的學習形式。相比之下，赫布學習是一種*由下而上*（bottom-up）的方法，神經元僅基於網路中活動的*前饋*（feedforward）模式來建立連接。赫布方法中的經典學習規則是**赫布法則**（**Hebb's rule**）：「一起激發，連結在一起」（fire together, wire together）。也就是說，每當兩個神經元同時處於活躍狀態時，我們就會增加它們之間連接的權重。儘管此學習規則並未顯式地最小化損失函數，但研究表明它能產生有效的神經表徵。例如，類似赫布的規則可以學習**資訊最大化**（**infomax**）表徵，這些表徵在神經啟用值中捕獲盡可能多關於輸入訊號的資訊 [[7](#ref-linsker1988self)]。類似的規則可以構建出類似記憶庫（memory banks）的網路 [[8](#ref-hopfield1982neural)]。赫布學習也引起了人們的興趣，因為它被認為比反向傳播更具生物學上的合理性（biologically plausible）。這是因為赫布法則可以進行*局部*（locally）計算——每個神經元僅根據相鄰神經元的活動來加強或減弱其權重——而反向傳播則需要整個神經網路的全局協調。目前尚不清楚這種全局協調在生物大腦中是如何實現的。

### 12.6.1 深度學習的數據結構：張量與批次

我們在深度學習中會遇到的主要數據結構是**張量**（**tensor**），它就是一個多維陣列。這看似簡單，但熟悉張量處理的慣例非常重要。

一般而言，深度學習中的一切都表示為張量——輸入是一個張量，啟用值是張量，權重是張量，輸出也是張量。如果您擁有的數據本身並非以張量表示，那麼在將其輸入深度網路之前的首要步驟就是將其轉換為張量格式。我們最常使用實數張量，也就是張量的元素屬於 $\mathbb{R}$。

假設我們有一個由圖像 $\mathbf{x}$ 和標籤 $\mathbf{y}$ 組成的數據集 $\{\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\}_{i=1}^N$。以張量方式思考此數據集，將其視為兩個張量：$\mathbf{X} \in \mathbb{R}^{N \times C_0 \times H \times W}$ 和 $\mathbf{Y} \in \mathbb{R}^{N \times K}$。張量的第一個維度是我們數據集中的元素數量。其餘維度是圖像的維度（$C_0$ 個顏色通道，高度為 $H$，寬度為 $W$）和標籤的維度（$K$ 類分類）。

網路中的啟用值也是張量。對於我們目前看到的 MLP 網路，啟用值張量的形狀為 $N \times C_{\ell}$，其中 $C_{\ell}$ 是第 $\ell$ 層上的神經元數量，有時也比照輸入圖像的顏色通道稱為**通道**（channels）。在後續章節中，我們將遇到其他啟用層具有額外維度的架構，例如在卷積網路中，我們將看到形狀為 $N \times C_{\ell} \times H_{\ell} \times W_{\ell}$ 的啟用層。

另一個重要的概念是**批次**（batches）。通常，我們不會逐張圖像地透過神經網路進行處理。相反地，我們是一次運行一個*批次*的圖像，並且它們是並行處理的。從訓練數據中取樣的一個批次可以表示為 $\{\mathbf{x}_{\texttt{batch}}^{(i)}, \mathbf{y}_{\texttt{batch}}^{(i)}\}_{i=1}^{N_{\texttt{batch}}}$，以張量表示的批次形狀為 $\mathbf{X} \in \mathbb{R}^{N_{\texttt{batch}} \times C_0 \times H \times W}$ 和 $\mathbf{Y} \in \mathbb{R}^{N_{\texttt{batch}} \times K}$。

網路的權重和偏差通常也表示為張量。線性層的權重和偏差將是形狀為 $\mathbf{W}_{\ell} \in \mathbb{R}^{C_{\ell+1} \times C_{\ell}}$ 和 $\mathbf{b}_{\ell} \in \mathbb{R}^{C_{\ell+1}}$ 的張量。

例如，在下方的 [圖 12.14](#fig-neural_nets-simple_MLP_network_tensors_and_batches) 中，我們將與來自 [圖 12.10](#fig-neural_nets-simple_MLP_network) 的 MLP 處理三個數據點的批次相關之所有張量視覺化，其中大寫字母代表數據點和啟用值的批次，對應於 [圖 12.10](#fig-neural_nets-simple_MLP_network) 中數據點和隱藏單元的小寫名稱。對於這個網路，輸入不是一組圖像，而是一組向量 $\mathbf{X} \in \mathbb{R}^{N_{\texttt{batch}} \times C_0}$。輸出是每個輸入向量對應的一個值，因此我們有 $\mathbf{Y} \in \mathbb{R}^{N_{\texttt{batch}} \times 1}$。

<a id="fig-neural_nets-simple_MLP_network_tensors_and_batches"></a>

![圖 12.14：代表通過圖 12.10 中 MLP 一次前向傳播的張量](https://visionbook.mit.edu/figures/neural_nets/simple_MLP_network_tensors_and_batches-2.png)

這個例子展示了處理一維數據的張量和批次的基本概念，但在視覺領域，大多數時候我們將處理更高維度的張量。對於圖像數據，我們通常使用四維張量：批次 $\times$ 通道 $\times$ 高度 $\times$ 寬度；對於影片，我們可能會使用五維張量：批次 $\times$ 通道 $\times$ 高度 $\times$ 寬度 $\times$ 時間。三維（3D）掃描具有額外的*深度*空間維度；因此，3D 數據的影片可以由六維張量來表示。如您所見，僅以二維矩陣的方式進行思考是不太夠的。相反地，您應該將數據處理想像為在 $N$ 維張量上運作，以不同的方式對其進行切片和切割。朝著這個方向邁出一步，您可能會發現將張量在 3D 中視覺化非常有用，如 [圖 12.15](#fig-neural_nets-3D_tensor_example) 所示。

<a id="fig-neural_nets-3D_tensor_example"></a>

![圖 12.15：可代表 \(C \times H \times W\) 彩色圖像的 3D 張量。](https://visionbook.mit.edu/figures/neural_nets/3D_tensor_example.png)

這更接近視覺系統實際運作的 $N$ 維張量，而且許多概念僅藉由 3D 思考就可以充分掌握。我們將在後續章節中看到一些例子。

## 12.7 圖層型錄

在下方，我們使用藍色表示**參數**（parameters），使用紅色表示**數據/啟用值**（data/activations，即輸入到每一層與從每一層輸出的變數）。

我們僅在本章中以此方式為方程式著色，以明確不同變數的角色。但是，請留意本書後面圖表中的這些顏色。我們通常會用紅色繪製啟用值，用藍色繪製參數。

### 12.7.1 線性層

線性層是深度網路的主力。網絡幾乎所有的參數都包含在這些圖層中；我們將這些參數稱為權重和偏差。我們之前已經介紹過線性層。它們看起來像這樣：

$$
\begin{aligned}
    {\mathbf{x}_{\texttt{out}}} = {\mathbf{W}} {\mathbf{x}_{\texttt{in}}} + {\mathbf{b}} & \quad\quad \triangleleft \quad\texttt{linear}
\end{aligned}
$$

### 12.7.2 啟用層

<a id="fig-neural_nets-pointwise_nonlinearities"></a>

![圖 12.16：常見的逐點非線性。](https://visionbook.mit.edu/figures/neural_nets/pointwise_nonlinearities.png)

如果一個網路僅包含線性層，那麼它就只能計算線性函數。這是因為 $N$ 個線性函數的複合仍是一個線性函數。**啟用層**（**Activation layers**）引入了非線性。啟用層通常是逐點函數（pointwise functions），在輸入向量的每個維度上應用純量到純量的映射。通常這些層的參數（若有）是不用被學習的（但也可以被學習）。一些常見的啟用層定義如下，並繪製在 [圖 12.16](#fig-neural_nets-pointwise_nonlinearities) 中：

$$
\begin{aligned}
    {x_{\texttt{out}}}[i] =
    {\gamma}
    \frac{{x_{\texttt{in}}}[i] - \mathbb{E}[{x_{\texttt{in}}}[i]]}
         {\sqrt{\texttt{Var}[{x_{\texttt{in}}}[i]]}}
    + {\beta} & \quad\quad \triangleleft \quad \texttt{batchnorm}
\end{aligned}
$$

<a id="sec-neural_nets-normalization_layers"></a>

### 12.7.3 正規化層

正規化層（Normalization layers）引入了另一種非線性。它們不像啟用層那樣是逐點非線性，而是根據一組神經元的集體行為來擾動每個神經元的非線性。讓我們從**批次正規化**（**batchnorm**，簡稱**批次規一化**）[[9](#ref-ioffe2015batch)] 的例子開始。

批次正規化對每個神經啟用值進行**標準化**，使其相對於一個批次數據點的平均值和變異數（variance）進行歸一。在數學上，

$$
\begin{aligned}
    x_{\texttt{out}}[i] =
    \gamma
    \frac{x_{\texttt{in}}[i] - \mathbb{E}[x_{\texttt{in}}[i]]}
         {\sqrt{\texttt{Var}[x_{\texttt{in}}[i]]}}
    + \beta & \quad\quad \triangleleft \quad \texttt{batchnorm}
\end{aligned}
$$

其中 $\gamma$ 和 $\beta$ 是該層的學習參數，用於維持表達能力，以便該層可以輸出具有非零均值和非單位變異數的值。最常見的是，批次正規化是使用訓練批次統計數據來計算均值和變異數，這些統計數據在不同批次之間是會改變的。在測試時，則使用來自訓練數據的聚合統計數據。然而，使用測試批次統計數據對於實現對訓練數據到測試數據統計數據變化的不變性非常有用 [[10](#ref-wang2020fully)]。

從統計學中回想一下，隨機變數抽樣的標準分數（z-score）代表它與平均值相差多少個標準差：$z = \frac{x-\mu}{\sigma}$。

多年來還定義了許多其他正規化層。我們將重點介紹的另外兩個是 **$L_2$ 正規化**和**圖層正規化**（**layer normalization**，簡稱 **layernorm**）[[11](#ref-ba2016layer)]。$L_2$ 正規化將輸入投影到單位超球體上，這對於將啟用值限制為單位向量非常有用：

$$
\begin{aligned}
    x_{\texttt{out}}[i] &=
    \frac{x_{\texttt{in}}[i]}{\left\lVert\mathbf{x}_{\texttt{in}}\right\rVert_2}
    & \quad\quad \triangleleft \quad \texttt{L2-norm}
\end{aligned}
$$

圖層正規化（Layernorm）非常相似，不同之處在於它標準化的是輸入啟用值的向量：

$$
\begin{aligned}
    \mu &= \frac{1}{n} \sum_{i=1}^n x_{\texttt{in}}[i]\\
    \sigma^2 &= \frac{1}{n} \sum_{i=1}^n \left(x_{\texttt{in}}[i] - \mu\right)^2\\
    x_{\texttt{out}}[i] &=
    \gamma
    \frac{x_{\texttt{in}}[i] - \mu}{\sigma} +
    \beta
    & \quad\quad \triangleleft \quad \texttt{layernorm}
\end{aligned}
$$

請注意，圖層正規化（layernorm）就像 $L_2$-正規化一樣，將啟用值向量映射到超球體的表面，但它也將啟用值居中以具有零均值，然後透過 $\gamma$ 和 $\beta$ 對啟用值進行縮放和移動。作為一個練習，看看您是否可以使用 $L_2$-正規化作為其中一個步驟來寫出圖層正規化。

請注意，圖層正規化看起來也與批次正規化相當相似。兩者都標準化了啟用值，但針對不同的統計數據進行操作。圖層正規化計算單個數據點 $\mathbf{x}_{\texttt{in}}$ 的元素之均值和變異數，並且對批次中的每個此類數據點分別進行計算。批次正規化則計算批次中跨數據點的每個通道的均值和變異數。如果我們有一個儲存在張量 $\mathbf{X} \in \mathbb{R}^{N_{\texttt{batch}} \times C}$ 中的批次，那麼圖層正規化所做的看起來就像是批次正規化所做的「轉置」。批次正規化根據其列（column）的均值和變異數來標準化張量的每個元素。圖層正規化則根據其行（row）的均值 and 變異數來標準化每個元素：

<a id="fig-neural_nets-batchnorm_vs_layernorm_diagram"></a>

![圖 12.17：批次正規化 vs 圖層正規化。灰色表示計算均值和變異數的區域。另請參閱 的圖 2 以獲得更多此類視覺化。](https://visionbook.mit.edu/figures/neural_nets/batchnorm_vs_layernorm_diagram.png)

批次正規化的一個問題是，它需要一次處理整個數據點批次，並在批次中的每個數據點之間引入依賴性。這違反了數據點應該被獨立同分布（iid）處理的原則，如果您的方法依賴於 iid 假設，這可能會導致錯誤。圖層正規化沒有這個問題，並且確實是以 iid 的方式處理每個數據點。

### 12.7.4 輸出層

我們需要的最後一個部分是一個**輸出層**（**output layer**），它將神經表徵——一個高維的浮點數陣列——映射到所需的輸出表徵。在分類問題中，所需的輸出是類別標籤，最常見的輸出操作是 softmax 函數，我們在前面的章節中已經遇到過。在圖像合成問題中，所需的輸出通常是維度為 $N \times M \times 3$ 且值在 $[0, 255]$ 範圍內的 3D 陣列。乘以 255 的 sigmoid 是此設定的典型輸出轉換。這兩個圖層的方程式為：

$$
\begin{aligned}
    x_{\texttt{out}}[i] &=
    \frac{e^{-\tau x_{\texttt{in}}[i]}}
         {\sum_{k=1}^K e^{-\tau x_{\texttt{in}}[k]}}
    & \quad\quad \triangleleft \quad \texttt{softmax}\\
    x_{\texttt{out}}[i] &=
    255*\texttt{sigmoid}(x_{\texttt{in}}[i])
    & \quad\quad \triangleleft \quad \text{圖像輸出問題的常見圖層}
\end{aligned}
$$

在 softmax 定義中，我們增加了一個**溫度**（**temperature**）參數 $\tau$，用於縮放預測的陡峭度（或信心度）。

輸出層是損失函數的輸入，從而完成了我們對深度學習問題的規約。然而，在實務中使用這些輸出需要將它們翻譯成實際的圖片、行動或決策。對於分類問題，這可能意味著取 softmax 分布的 argmax，以便我們報告單個類別。對於圖像預測問題，這可能意味著將每個輸出四捨五入為整數值，因為常見的圖像格式將 RGB 值表示為整數。

當然，您還可以嘗試許多其他輸出轉換。通常，它們會非常針對具體問題，因為它們取決於您所針對的輸出空間之結構。

## 12.8 為什麼神經網路是一個好架構？

正如您很快就會學到的，幾乎所有現代電腦視覺演算法都以某種方式涉及深度網路。因此您可能會想知道：為什麼深度網路是一個如此出色的架構？我們在這裡強調五個原因：

1. 它們具有高容量（足夠大的網路是通用逼近器）。
2. 它們是可微分的（參數可以透過梯度下降進行優化）。
3. 它們具有良好的歸一化偏置（神經架構反映了現實世界中的真實結構）。
4. 它們在並行硬體上運行效率高。
5. 它們構建了抽象表徵。

讓我們結合第 11 章關於尋找真理的討論（見 [第 11 章](https://visionbook.mit.edu/problem_of_generalization.html) 以及 [圖 11.4](https://visionbook.mit.edu/problem_of_generalization.html#fig-problem_of_generalization-search_space_tools)）來看看原因 1-3。原因 1 與假設空間的大小有關。如果我們使用具有許多參數的大型神經網路，假設空間可以變得非常大。因此，我們通常可以確定我們的真實解（或其近似解）確實存在於神經網路架構所跨越的空間中。原因 2 指出，在此空間中尋找解相對容易，因為我們可以使用梯度來引導我們朝向愈來愈好的數據擬合。原因 3 是我們在本書後面研究更先進的神經網路架構時才會體會到的。事實證明，這些架構施加了約束和正則化器，使我們的搜尋偏向於能夠捕獲關於視覺世界的真實結構之解，從而使學到的解能夠泛化。

原因 4 與前三個同樣重要：它指出我們可以*高效地*進行所有這些運算，因為大多數計算可以在現代硬體上並行化；特別是矩陣乘法（線性層）和逐點運算（例如 ReLU 層）在圖形處理單元（GPU）上都很容易並行化。此外，大多數操作應用於圖像批次，其中批次中的每個項目都可以發送到不同的並行計算節點。

原因 5 也許是最細微的。它與神經網路的層次結構有關。逐層地，神經網路建立起輸入數據之日益抽象的表徵，且這些抽象傾向於愈來愈有用。這個論點乍看之下不易體會，但它將是本書後續章節（特別是關於表徵學習的章節）的一大主題。目前，請記住，在深度網路中逐層建立起來的*內部表徵*（internal representations），其用處和重要性遠遠超出了網路整體的輸入-輸出行為。

## 12.9 結語

神經網路是一個非常簡單且有用的參數化假設空間。它們是通用逼近器，可以透過梯度下降進行訓練並在並行硬體上運行。*深度*網路在電腦視覺中特別有效；正如我們很快就會看到的，可以構建專門反映視覺世界結構的深度架構，從而使視覺處理極內有且性能優異。人工神經網路也與我們大腦中的真實神經網路有著聯繫。這種聯繫比僅僅共享一個名稱更為深厚：我們稍後在本書中將看到之深度網路架構（例如卷積網絡、Transformer）是我們目前對動物大腦中計算的最佳模型，因為它們比任何競爭模型都能更好地解釋大腦數據 [[12](#ref-Schrimpf2020integrative)]。這是一類真正值得關注的模型。

## 參考文獻

<a id="ref-rosenblatt1958perceptron"></a>[1] F. Rosenblatt, The perceptron: A probabilistic model for information storage and organization in the brain., Psychological Review 65 (1958) 386.

<a id="ref-rescorla1972theory"></a>[2] R.A. Rescorla, A theory of pavlovian conditioning: Variations in the effectiveness of reinforcement and non-reinforcement., Classical Conditioning, Current Research and Theory 2 (1972) 64–69.

<a id="ref-Cybenko1989"></a>[3] G. Cybenko, Approximation by superpositions of a sigmoidal function., Mathematics of Control, Signals and Systems 2 (1989) 303–314.

<a id="ref-telgarsky2016benefits"></a>[4] M. Telgarsky, Benefits of depth in neural networks., in: Conference on Learning Theory, PMLR, 2016: pp. 1517–1539.

<a id="ref-salimans2017evolution"></a>[5] T. Salimans, J. Ho, X. Chen, S. Sidor, I. Sutskever, Evolution strategies as a scalable alternative to reinforcement learning., (2017). https://arxiv.org/abs/1703.03864.

<a id="ref-hebb2005organization"></a>[6] D.O. Hebb, The organization of behavior: A neuropsychological theory., Psychology Press, 2005.

<a id="ref-linsker1988self"></a>[7] R. Linsker, Self-organization in a perceptual network., Computer 21 (1988) 105–117.

<a id="ref-hopfield1982neural"></a>[8] J.J. Hopfield, Neural networks and physical systems with emergent collective computational abilities., Proceedings of the National Academy of Sciences 79 (1982) 2554–2558.

<a id="ref-ioffe2015batch"></a>[9] S. Ioffe, C. Szegedy, Batch normalization: Accelerating deep network training by reducing internal covariate shift., in: Icml, 2015: pp. 448–456.

<a id="ref-wang2020fully"></a>[10] D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell, Fully test-time adaptation by entropy minimization., (2020). https://arxiv.org/abs/2006.10726.

<a id="ref-ba2016layer"></a>[11] J.L. Ba, J.R. Kiros, G.E. Hinton, Layer normalization., (2016). https://arxiv.org/abs/1607.06450.

<a id="ref-Schrimpf2020integrative"></a>[12] M. Schrimpf, J. Kubilius, M.J. Lee, N.A.R. Murty, R. Ajemian, J.J. DiCarlo, Integrative benchmarking to advance neurally mechanistic models of human intelligence., Neuron (2020).\n
