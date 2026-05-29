1. [Foundations of Learning](https://visionbook.mit.edu/part_foundation_learning.html)
2. [10  Gradient-Based Learning Algorithms](https://visionbook.mit.edu/gradient_descent.html)

# 10  基於梯度的學習演算法

## 10.1 簡介

一旦你指定了一個學習問題（損失函數、假設空間、參數化），下一步就是尋找能夠最小化損失的參數。這是一個最佳化問題，而我們將使用最常見的最佳化演算法是**梯度下降法**。梯度下降就像滑雪者在白雪皚皚的山上往下行進，而山的形狀就是損失函數。

梯度下降有很多變體，我們將這整個系列稱為**基於梯度的學習演算法**。所有這些演算法都共享相同的基本思想：在某個工作點，計算最陡峭下降的方向，然後利用這個方向來尋找一個具有更低損失的新工作點。

> [!NOTE]
> 我們使用**工作點**（operating point）這個術語來指代我們目前正在評估損失的特定點（參數設定）。

## 10.2 技術背景設定

在本章中，我們考慮最小化代價函數 $J: \cdot \rightarrow \mathbb{R}$ 的任務，該函數將某個任意輸入映射到一個純量代價。

在學習問題中， $J$ 的定義域是訓練數據和參數 $\theta$ 。我們通常會將訓練數據視為固定的，並僅將目標函數表示為參數的函數，即 $J(\theta)$ 。我們的目標是求解： $$ \theta^* = \arg\min_{\theta} J(\theta) $$ 幾乎所有的最佳化器都是透過某種迭代過程來運作，在此過程中，它們會不斷更新參數使其變得越來越好。不同的最佳化器在參數更新函數的運作方式上有所不同。如[圖 10.1](#fig-gradient_descent-optimization_schematic)所示，更新函數可以查看有關損失地貌（loss landscape）的一些資訊，然後利用該資訊來更新參數。

![圖 10.1：通用最佳化迴路。](https://visionbook.mit.edu/figures/gradient_descent/optimization_schematic.png)

在最簡單的設定中，稱為**零階最佳化**（zeroth-order optimization），更新函數只能觀察到值 $J(\theta)$ 。因此，尋找能夠最小化損失的 $\theta$ 的唯一方法，就是對不同的 $\theta$ 值進行採樣，並朝向值較低的方向移動。

對於**基於梯度的最佳化**（gradient-based optimization，也稱為**一階最佳化**），更新函數將目前工作點上代價對參數的梯度 $\nabla_{\theta}J(\theta)$ 作為輸入。這揭示了關於損失的極其有用的資訊，直接告訴我們如何最小化它：只需朝向最陡峭下降的方向移動，也就是梯度的反方向（或稱梯度方向）。

高階最佳化方法會觀察損失的高階導數，例如海森矩陣（Hessian） $H$ ，它能告訴你地貌在局部的彎曲情況。海森矩陣的計算成本很高，但許多方法會使用海森矩陣的近似值，或者與損失曲率相關的其他屬性，這些方法也越來越受歡迎~。

## 10.3 基本梯度下降法

最簡單版本的梯度下降法僅沿著梯度方向邁出一步，其步長與梯度大小成正比。此演算法在[演算法 10.1](#alg-gradient_descent_basic_gradient_descent)中進行了說明。

![演算法 10.1：梯度下降法 `GD`。藉由沿著梯度 $\nabla_{\theta} J$ 下降來最佳化代價函數 $J: \theta \rightarrow \mathbb{R}$ 。](https://visionbook.mit.edu/figures/gradient_descent/alg1.png)

該演算法有兩個超參數：**學習率** $\eta$ ，用以控制步長（學習率乘以梯度大小），以及步數 $K$ 。如果學習率足夠小且初始參數向量 $\theta^0$ 是隨機的，那麼當 $K \rightarrow \infty$ 時，該演算法幾乎肯定會收斂到 $J$ 的局部極小值~。然而，為了更快速地下降，將學習率設定為較高的值會非常有用。

## 10.4 學習率調整排程

一個普遍有用的策略是從較高的 $\eta$ 值開始，然後根據**學習率調整排程**將其衰減直至收斂。研究人員提出了無數種排程，它們通常藉由呼叫某個函數 $\texttt{lr}(\eta^0,k)$ 來獲取每次下降迭代中的學習率： $$ \eta^{k} = \texttt{lr}(\eta^0,k) $$ 一般來說，我們希望更新規則滿足 $\eta^{k+1} < \eta^k$ ，以便在接近極小值點時採取較小的步長。以下給出幾種簡單且受歡迎的方法： $$ \begin{aligned}
 \texttt{lr}(\eta^0,k) &= \beta^{-k} \eta^0 &\quad\quad \triangleleft\quad \text{指數衰減 (exponential decay)}\\
 \texttt{lr}(\eta^0,k) &= \beta^{-\lfloor k/M \rfloor} \eta^0 &\quad\quad \triangleleft\quad \text{階梯式指數衰減 (stepwise exponential decay)}\\
 \texttt{lr}(\eta^0,k) &= \frac{(K - k)}{K} \eta^0 &\quad\quad \triangleleft\quad \text{線性衰減 (linear decay)}
\end{aligned} $$ > [!NOTE]
> 線性衰減的一個缺點是它取決於總步數 $K$ 。這使得我們難以比較不同長度的最佳化運行過程。在更進階的學習率調整排程（例如餘弦衰減 cosine decay~）中也需要注意這一點，因為它們對於不同的 $K$ 設定也會有不同的行為。

其中 $\beta$ 和 $M$ 是這些方法的額外超參數。學習率衰減的一般方法總結在[演算法 10.2](#alg-gradient_descent_gradient_descent_with_lr_decay)中。

![演算法 10.2：具有學習率衰減的梯度下降演算法。](https://visionbook.mit.edu/figures/gradient_descent/alg2.png)

此演算法的變體包括僅在達到高原期（即損失連續多次迭代沒有下降）時才衰減學習率，或者根據更複雜的非線性排程（例如形狀像餘弦函數的排程~）來衰減學習率。

## 10.5 動量

我們能否進行比單純沿著梯度方向邁出一步更聰明的更新？在無數被提出的想法中，少數被保留下來的其中之一就是**動量**~。動量讓滑雪的比喻變得更加精確：動量就像是滑雪者的慣性，帶著他們越過滑雪坡上的小顛簸和不完美之處，並在他們沿著直線路徑下降時增加速度。在數學上，動量意味著我們將參數更新設定為方向 $\mathbf{v}^{k+1}$ ，該方向由前一次更新方向 $\mathbf{v}^{k}$ 與目前負梯度方向的加權組合給出： $$ \mathbf{v}^{k+1} = \mu \mathbf{v}^{k} - \eta\nabla_{\theta} J(\theta^k) $$ 此組合中的權重 $\mu$ 是一個新的超參數，有時簡稱為動量。完整演算法參見[演算法 10.3](#alg-gradient_descent_gradient_descent_with_momentum) 。

![演算法 10.3：具有動量的梯度下降演算法。](https://visionbook.mit.edu/figures/gradient_descent/alg3.png)

[圖 10.2](#fig-gradient_descent-momentum_out1) 顯示了動量如何影響簡單目標函數 $J = \texttt{abs}(\theta)$ （ $\theta$ 的絕對值）的梯度下降。如圖所示，適度的動量可以幫助提高收斂速度（ $\mu = 0.5$ ），但過多的動量會導致軌跡越過最佳值，甚至在達到最佳損失時，軌跡也可能不會停止（[圖 10.2](#fig-gradient_descent-momentum_out1) 中的 $\mu = 0.95$ ）。

![圖 10.2：（左）一個簡單的損失函數 $J = \texttt{abs}(\theta)$ 。（右）三種不同動量 $\mu$ 設定下的最佳化軌跡。白線表示每次最佳化迭代時參數的值，從頂部開始並向底部前進。顏色表示損失的值。紅點是損失首次達到最佳值 $0.01$ 範圍內的位置。](https://visionbook.mit.edu/figures/gradient_descent/momentum_out1.png)

我們也可以設計其他類型的動量，這些動量根據先前更新中累積的一些資訊來調整更新方向的偏差。你可以從其他地方閱讀到另外兩個受歡迎的替代方案，即 Nesterov 加速梯度~ 和 Adam~。

## 10.6 哪些函數可以用梯度下降法最小化？

如果一個函數不可微怎麼辦？我們還能使用梯度下降嗎？有時是可以的！我們需要的屬性是，我們能夠獲得一個有意義的訊號，以了解如何微調函數的參數以減少損失。這個屬性與數學教科書中定義的可微性*並不*相同。一個函數可能是可微的，但卻無法提供有用的梯度（例如，如果梯度處處為零）；而一個函數在某些點上可能是不可微的，但仍然允許進行有意義的基於梯度的更新（例如：）。

[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) 給出了使用梯度下降最小化不同類型函數的例子。[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (b) 和 [圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (d) 是函數不連續且在不連續點處解析導數未定義的情況。令人驚訝的是，在[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (b) 中，這對梯度下降來說並不是問題。這是因為我們這裡使用的梯度下降演算法（PyTorch~ 所使用的演算法）在不連續點處使用了**單側導數**，也就是說，我們將不連續點處的梯度設定為在固定任意方向上距離不連續點極小步長處的梯度值。在底層，對於每個原子不連續操作，PyTorch 要求我們定義其在不連續點處的梯度，而單側梯度是一個標準選擇。這就是為什麼在深度學習中使用像修正線性單元（ReLU）這樣在深層網路中很常見且具有不連續梯度的函數是沒問題的。

[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (c) 和 (e) 給出了函數連續但梯度表現不佳的情況。在[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (c) 中，我們遇到了一個幾乎**消失**的梯度，即它處處接近於零，因此使用固定學習率的梯度下降將會非常緩慢。[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (e) 顯示了相反的情況：在極小值點處的梯度趨近於無窮大；我們稱之為**梯度爆炸**，這會導致收斂失敗。

最後，[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (f) 顯示了另一個棘手的情況：當存在多個極小值時，梯度下降可能會卡在次優的局部極小值中。我們最終到達哪一個極小值將取決於我們初始化 $x$ 的位置。

![圖 10.3：梯度下降在各種函數上的表現。** 在每個子圖中，左側顯示函數 $J$ ，紅點表示使用 $\eta=0.01$ 和 $\mu=0.9$ 的梯度下降（GD）所找到的解。右側顯示了在每次 GD 迭代中， $x$ 值在 $J$ 之上繪製的軌跡。(a) 當 $\eta$ 趨近於零時，GD 對凸函數會收斂。(b) 只要不連續點的任一側有定義梯度，不連續性就不會構成實質問題。(c) 幾乎平坦的函數會表現出非常緩慢的下降。(d) 分段常數函數是有問題的，因為梯度會完全消失。(e) 對於函數 $J=\texttt{sqrt}(\texttt{abs}(\theta))-0.25$ , 梯度在極小值點處趨近於無窮大，從而導致不穩定。(f) 當 $J$ 具有多個局部極小值時，我們可能找不到全域極小值。](https://visionbook.mit.edu/figures/gradient_descent/grad_descent.png)

### 10.6.1 針對缺乏良好梯度的函數之類梯度最佳化

對於最小化像[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (d) 這樣梯度處處為零的函數該怎麼辦？這是梯度下降法真正面臨困境的情況。然而，通常可以將此類問題轉化為可以使用梯度下降處理的問題。請記住，從最佳化的角度來看，梯度的關鍵屬性是它是參數空間中局部損失最小化的方向。大多數基於梯度的最佳化器實際上並不需要真正的梯度；相反地，它們的更新函數與更廣泛的局部損失最小化方向系列 $\mathbf{v}$ 相容。

除了真正的梯度之外， $\mathbf{v}$ 還有哪些其他好的選擇？一個常見的想法是將 $\mathbf{v}$ 設定為**代理損失**函數的梯度，這是一個具有有意義（非零）梯度且逼近 $J$ 的函數 $J_{\texttt{surr}}$ 。例如，可以是 $J$ 的平滑版本。獲取 $\mathbf{v}$ 的另一種方法是透過對 $\theta$ 的擾動進行採樣，並查看哪種擾動會導致更低的損失來計算它。在這種策略中，我們評估一組擾動 $\epsilon$ 的 $J(\theta+\epsilon)$ ，然後朝向能夠降低損失的 $\epsilon$ 移動。此類方法有時被稱為**演化策略**~，此演算法的基本版本在[演算法 10.4](#alg-gradient_descent_ES)中給出。

![演算法 10.4：演化策略演算法。](https://visionbook.mit.edu/figures/gradient_descent/alg4.png)

如[圖 10.4](#fig-gradient_descent-sampling_out1) 所示，該演算法可以成功地最小化[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (c) 中的函數。

![圖 10.4：使用 [演算法 10.4](#alg-gradient_descent_ES) 最小化不可微（零梯度）損失，其中使用 $\sigma=1$ ， $M=10$ ，且 $\eta=0.02$ 。](https://visionbook.mit.edu/figures/gradient_descent/sampling_out1.png)

### 10.6.2 梯度裁剪

對於[圖 10.3](#fig-gradient_descent-grad_descent_simple_examples) (e) 中梯度在極值點附近爆炸的情況該怎麼辦？我們有什麼方法可以改善這個函數的最佳化嗎？為了解決梯度爆炸問題，一個有用的技巧是**梯度裁剪**，它指的是將梯度的幅度限制在某個最大值之內。[演算法 10.5](#alg-gradient_descent_grad_clipping) 描述了這種方法。

![演算法 10.5：梯度裁剪演算法。](https://visionbook.mit.edu/figures/gradient_descent/alg5.png)

> [!NOTE]
> `clip` 是「裁剪」函數： $\texttt{clip}(v, -m, m) = \max(\min(v,m),-m)$。如[圖 10.5](#fig-gradient_descent-clipped_out1) 所示，該演算法確實成功地最小化了我們梯度爆炸的範例。

![圖 10.5：使用帶有裁剪的 `GD` 來最小化具有爆炸梯度的損失，其中使用 $m=0.1$ 。](https://visionbook.mit.edu/figures/gradient_descent/clipped_out1.png)

## 10.7 隨機梯度下降法

我們目前所看到的基於梯度的方法有一個問題，那就是梯度的計算實際上可能非常昂貴，而在學習問題中通常就是這種情況。這是因為學習問題通常具有 $J$ 是每個訓練數據點所產生的損失之平均值的形式。計算 $\nabla_{\theta}J(\theta)$ 需要計算平均值中每個元素的梯度，也就是在訓練集中每個數據點的位置上評估被學習函數的梯度。如果我們在一個大型數據集上進行訓練，比如有 100 萬個訓練點，那麼僅執行一步梯度下降端賴於計算 100 萬個梯度！為了解釋清楚，我們將 $J$ 寫為訓練數據 $\{\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\}_{i=1}^N$ 的顯式函數。對於典型的學習問題， $\nabla_{\theta} J(\theta, \{\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\}_{i=1}^N)$ 可以分解如下： $$ \begin{align}
 \nabla_{\theta} J(\theta, \{\mathbf{x}^{(i)}, \mathbf{y}^{(i)}\}_{i=1}^N) &=
 \nabla_{\theta} \frac{1}{N}\sum_{i=1}^N \mathcal{L}(f_{\theta}(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})\\
 &= \frac{1}{N}\sum_{i=1}^N \nabla_{\theta} \mathcal{L}(f_{\theta}(\mathbf{x}^{(i)}), \mathbf{y}^{(i)})
\end{align} $$ 對於大型的 $N$ ，計算這個總和非常昂貴。相反地，假設我們從該總和中隨機抽樣（無放回抽樣）一部分項， $\{\mathbf{x}^{(b)}, \mathbf{y}^{(b)}\}_{b=1}^B$ ，其中 $B$ 是**批次大小**。然後，我們透過計算該批次的平均梯度來估計總梯度，如下所示： $$ \begin{align}
 \tilde{\mathbf{g}} = \frac{1}{N}\sum_{b=1}^B \nabla_{\theta} \mathcal{L}(f_{\theta}(\mathbf{x}^{(b)}), \mathbf{y}^{(b)})
\end{align} $$ 如果我們取樣一個大批次，其中 $B$ 幾乎與 $N$ 一樣大，那麼對這 $B$ 個項的平均值應該與對所有 $N$ 個項的平均值大致相同。如果我們取樣一個較小的批次，那我們對梯度的估計精確度會降低，但計算速度會更快。因此，我們在精確度和速度之間需要進行權衡，我們可以透過超參數 $B$ 來調整這個權衡。

使用這個想法的梯度下降變體稱為**隨機梯度下降法**（SGD），因為每次下降迭代都使用不同隨機（stochastic）取樣的訓練數據批次來估計梯度。

SGD 的完整描述參見[演算法 10.6](#alg-gradient_descent_SGD) 。

![圖 10.6：隨機梯度下降演算法。隨機梯度下降從完整訓練數據的隨機子集（批次）中估計梯度，並在此基礎上進行更新。](https://visionbook.mit.edu/figures/gradient_descent/alg6.png)

除了比 `GD` 計算速度更快之外， `SGD` 還具有許多有用的特性。由於每次下降步驟都帶有某種隨機性，只要在某些隨機取樣的批次中這些小顛簸消失， $\texttt{SGD}$ 就可以跳過損失地貌中的小顛簸。另一個重要的特性是， $\texttt{SGD}$ 可以隱式地為學習問題提供正規化。例如，對於線性問題（即 $f_\theta$ 是線性的），如果存在多個最小化損失的參數設定， $\texttt{SGD}$ 通常會收斂到具有最小參數範數的解~。

## 10.8 結論與總結

最佳化的研究可以寫滿數十本教科書和數千篇學術論文。但對我們來說幸運的是，現代機器學習已經收斂到實務中僅使用的幾種非常簡單的最佳化方法。我們很快就會接觸到深度學習，這是用於電腦視覺的主要機器學習類型。在深度學習中，基於梯度的最佳化是主力核心。信不信由你，上面描述的少數幾種演算法就足以訓練大多數最先進的深度學習模型。每年都會有對這些想法的新闡述，二階方法也指日可待，然而基本概念仍然非常簡單：計算損失地貌形狀的局部估計，然後根據這個形狀，朝向較低的損失邁出一小步。
