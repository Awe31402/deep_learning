1. [Foundations of Learning](https://visionbook.mit.edu/part_foundation_learning.html)
2. [13  Neural Networks as Distribution Transformers](https://visionbook.mit.edu/neural_nets_as_distribution_transformers.html)

# 13  作為分佈轉換器的神經網路

## 13.1 簡介

到目前為止，我們已經看到深層網路是簡單函數的堆疊，它們組合起來可以實現從輸入到輸出的有趣映射。本節將介紹一種稍微不同的深層網路思考方式。這個想法是將每個層視為*數據分佈的幾何轉換*。

## 13.2 一種不同的函數繪圖方式

深層網路中的每個層都是從數據的一種表示到另一種表示的映射： $f: \mathbf{x}_{\text{in}} \rightarrow \mathbf{x}_{\text{out}}$ 。如果 $\mathbf{x}_{\text{in}}$ 和 $\mathbf{x}_{\text{out}}$ 都是一維（1D）的，那麼我們可以將此映射繪製為一個函數，其中 $\mathbf{x}_{\text{in}}$ 在 $x$ 軸上，而 $\mathbf{x}_{\text{out}}$ 在 $y$ 軸上。

![圖 13.1：繪製函數 $\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}}$ 的傳統方式。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/trad_plot.png)

現在，我們將改為考慮一種不同的映射繪圖方式，在此方式中，我們只需將 $y$ 軸旋轉為水平方向，而不是垂直方向。

![圖 13.2：另一種函數繪圖方式。函數是重新排列輸入空間的映射。這裡顯示的恆等函數 $\mathbf{x}_{\text{out}}=\mathbf{x}_{\text{in}}$ 意味著「不進行重新排列」，因此其映射是直線。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/new_way_plot.png)

這種描繪方式顯而易見地表明了 $\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}}$ 的圖是恆等映射：數據點被映射到未改變的位置。

![圖 13.3：幾種可能是神經網路層的簡單函數的映射圖。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/new_way_plot_examples.png)

上述每一個都是可以在深層網路中找到的層。如上圖第一行所示的線性層會拉伸和擠壓數據分佈。ReLU 非線性激活函數將所有負數據映射到 0，並對所有非負數據應用恆等映射。Sigmoid 函數則將負數據拉向 0，正數據拉向 1。

## 13.3 深層網路如何重新映射數據分佈

藉由這種方式，輸入的數據分佈可以被逐層重塑為所需的配置。例如，二元 Softmax 分類器的目標是移動數據點，直到所有類別 0 的點最終都被移動到輸出層上的 $(1,0)$ ，而所有類別 1 的點最終都被移動到 $(0,1)$ 。

> [!NOTE]
> 這是因為整數 0 的獨熱編碼是 $[1,0]$ ，而整數 1 的獨熱編碼是 $[0,1]$ 。

深層網路將這些操作進行堆疊； [圖 13.4](#fig-neural_nets_as_data_transformations-new_way_plot_two_layer) 給出了一個範例。

![圖 13.4： `linear` - `relu` 堆疊的映射圖。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/new_way_plot_two_layer.png)

上圖顯示了均勻網格的數據點如何在深層網路中逐層映射。我們還可以使用這種繪圖風格來顯示非均勻分佈的輸入數據點是如何被轉換的。這才是深層網路實際運作的場景，有時從這個角度來看，網路的實際行為會顯得非常不同。我們可以將深層網路視為將輸入數據分佈 $p_{\text {data }}$ 轉換為輸出數據分佈 $p_{\text {out}}$ 的過程。網路中的每一層激活值都是數據的不同表示或**嵌入**，我們可以將某層 $\ell$ 上的激活值分佈視為 $p_{\ell}$ 。接著，深層網路逐層將 $p_{\text {data }}$ 轉換為 $p_{\texttt{1}}$ ，再轉換為 $p_{\texttt{2}}$ ，以此類推，直到最終將數據轉換為分佈 $p_{\text {out}}$ 。大多數損失函數也可以從這個角度來解釋：它們以某種形式懲罰輸出分佈 $p_{\text {out}}$ 與目標分佈 $p_{\texttt{target}}$ 之間的散度。

這種繪圖方式的一個極佳屬性是，它還可以擴展到視覺化二維（2D）到二維（2D）的映射（這是傳統 $x$ 軸 / $y$ 軸繪圖難以做到的）。實際的深層網路執行的是 $N$ 維（ND）到 $N$ 維（ND）的映射，但光是 2D 到 2D 的視覺化就能為一般情況提供許多見解。在[圖 13.5](#fig-neural_nets_as_data_transformations-2D_mapping_diagrams) 中，我們展示了三個常見的神經網路層如何作用於轉換以原點為中心的二維高斯數據群。

![圖 13.5：幾種神經網路層的 2D 映射圖。 `linear` 線性層映射的位移、拉伸和旋轉取決於其權重與偏置。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/2D_mapping_diagrams.png)

這裡需要注意的一個有趣現象是，ReLU 層將許多點映射到正象限的軸上。一般來說，在 ReLU 網路中，大量的數據密度會沿著這些軸積聚，因為正象限之外的任何點都會被折射到軸上。這種效應在具有高維度嵌入的實際網路中會變得更加誇張。特別是，對於寬度為 $N$ 的層，嚴格為正的區域僅佔嵌入空間的 $\frac{1}{2^N}$ 比例，因此在經過 ReLU 層之後，幾乎整個空間都會被映射到軸上。高維神經表示的幾何結構可能會因此變得非常稀疏，即大部分的表示空間體積都沒有被任何數據點所佔據。

## 13.4 二元分類器範例

考慮一個多層感知器（MLP），它執行被公式化為兩路 Softmax 回歸的二元分類。輸入數據點均在 $\mathbb{R}^2$ 中，目標輸出在 $\vartriangle^1$ 中，層結構為 $\texttt{linear}-\texttt{relu}-\texttt{linear}-\texttt{relu}-\texttt{linear}$ 。該網路繪製於下方的[圖 13.6](#fig-neural_nets-simple_MLP_network2) 中。

![圖 13.6：一個具有三個線性層與兩個輸出的 MLP，適用於執行二元 Softmax 回歸。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/simple_MLP_network2.png)

或者用數學公式表示如下： $$ \begin{align}
 \mathbf{z}_1 &= \mathbf{W}_1\mathbf{x} + \mathbf{b}_1 &\triangleleft \quad \texttt{linear}\\
 \mathbf{h}_1 &= \texttt{relu}(\mathbf{z}_1) &\triangleleft \quad \texttt{relu}\\
 \mathbf{z}_2 &= \mathbf{W}_2\mathbf{h}_1 + \mathbf{b}_2 &\triangleleft \quad \texttt{linear}\\
 \mathbf{h}_2 &= \texttt{relu}(\mathbf{z}_2) &\triangleleft \quad \texttt{relu}\\
 \mathbf{z}_3 &= \mathbf{W}_3\mathbf{h}_2 + \mathbf{b}_3 &\triangleleft \quad \texttt{linear}\\
 \mathbf{y} &= \texttt{softmax}(\mathbf{z}_3) &\triangleleft \quad \text{\texttt{softmax}}
\end{align} $$ 現在我們希望訓練這個網路來對兩個數據分佈進行分類。目標是將輸入分佈轉換為區分不同類別的目標分佈，如下方的[圖 13.7](#fig-neural_nets_as_data_transformations-goal_of_classifier) 所示。

![圖 13.7：神經網路分類器的目標是重新排列輸入數據分佈，以匹配目標標籤分佈。（左）包含紅色和藍色兩個類別的輸入數據集。（右）目標輸出（獨熱編碼）。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/goal_of_classifier.png)

在此示例中，目標輸出將所有紅點置於 $(0,1)$ ，並將所有藍點置於 $(1,0)$ 。這些是我們兩個類別的獨熱編碼坐標。訓練網路的過程包含尋找一系列幾何轉換，以將輸入分佈重新排列為此目標輸出分佈。

我們將在訓練過程中的四個**檢查點**，逐層視覺化網路如何轉換訓練數據集。

> [!NOTE]
> 檢查點是訓練中某次迭代的參數記錄，也就是說，如果參數向量的迭代值為 $\theta^0, \theta^1, \ldots, \theta^T$ ，那麼任何 $\theta^k$ 都可以被記錄為一個檢查點。

在[圖 13.8](#fig-neural_nets-nn_training_viz) 中，我們將其繪製為 $\mathbb{R}^2 \rightarrow \mathbb{R}^2$ 映射的 3D 視覺化圖。每條虛線將數據點在一層的表示連接到它在下一層的表示。灰色線條展示了在每一層中，原點周圍的一個正方形區域如何被拉伸、旋轉或擠壓，從而映射到下一層轉換後的區域。

![圖 13.8：深層網路如何逐層重新映射輸入數據。目標輸出是將所有紅點移動到 $(0,1)$ ，將所有藍點移動到 $(1,0)$ （兩個類別的獨熱編碼）。隨著訓練的進行，網路逐漸實現了這種分離。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/nn_training_viz.png)

當然，「拉伸、旋轉或擠壓」只是一種直觀的思考方式，我們可以更精確地描述。線性層執行空間的仿射轉換，而 ReLU 層則將所有負值投影到維度嚴格為正的錐體邊界上。在訓練的過程中，網路學會了逐層將這兩個類別解耦，並將點拉向一維單體的頂點，從而實現了對點的正確分類！

## 13.5 高維度數據點如何被深層網路重新映射

如果我們的數據和激活值是高維度的呢？上述的圖只能視覺化一維和二維的數據分佈。深層表示通常比這高得多維，為了視覺化它們，我們需要應用**降維**工具。這些工具將高維數據投影到較低的維度，例如二維，從而進行視覺化。一個常見的目標是執行投影，使得二維投影中兩個數據點之間的距離大致與它們在高維空間中的實際距離成正比。在下一個圖[圖 13.9](#fig-neural_nets-vit_mapping_plot) 中，我們使用了一種稱為 **t-分佈隨機鄰近嵌入**（t-Distributed Stochastic Neighbor Embedding，簡稱 **t-SNE**）~ 的降維技術，來視覺化現代深層網路不同層如何表示不同語義類別的圖像數據集，其中每種顏色代表不同的語義類別。我們正在視覺化的網路是 Transformer 架構，我們將在[第 26 章](https://visionbook.mit.edu/transformers.html) 中學習其所有細節。目前，需要注意的重要事項是：(1) 我們只展示了精選的幾層（該網路實際上具有數十層），(2) 其嵌入是高維的（特別是 38,400 維），但透過 t-SNE 被映射到了 2D。因此，與前一節展示了確切嵌入以及逐層轉換的每一步之視覺化不同，此視覺化僅是對網路內部發生情況的粗略展示。

![圖 13.9：強大的深層網路如何將輸入圖像重新映射到將語義類別（以不同顏色顯示）分開的解耦表示中。此深層網路是視覺 Transformer（ViT~），我們將在後續章節中學習它。它是使用對比語言-圖像預訓練（CLIP~，參見）進行訓練的。每個 `ViT 區塊`都包含多層神經處理（參見；我們在區塊中第一個 `token norm` 之後立即視覺化嵌入）。我們在所有顯示的層上聯合應用 t-SNE。](https://visionbook.mit.edu/figures/neural_nets_as_data_transformations/vit_mapping_plot.png)

請注意，在第一層上，語義類別並沒有很好地被分開，但到了最後一層，該表示已經將語義類別**解耦**，從而使每個類別佔據表示空間的不同部分。這是預期之中的，因為此視覺化中的最終層接近網路的輸出，並且該網路已被訓練為輸出語義的直接表示（特別是，此網路是使用對比語言-圖像預訓練 [CLIP [1](https://visionbook.mit.edu/references.html#ref-radford2021learning) ] 進行訓練的，這是一種學習語義表示的方法，我們將在[第 51.3 節](https://visionbook.mit.edu/VLMs.html#sec-VLMs-CLIP) 中學習它）。

## 13.6 結論與總結

逐層地，深層網路將數據從其原始格式轉換為更加抽象且有用的表示。將這個過程視為對數據分佈的一系列幾何轉換，或者視為一種將最初混亂的數據進行重新組織以使不同數據類別清晰分離的解耦過程，會非常有幫助。

## 參考文獻

[1] A. Radford, J.W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, others, Learning transferable visual models from natural language supervision., in: Icml, 2021: pp. 8748–8763.
