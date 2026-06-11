1. [視覺神經網路架構](<https://visionbook.mit.edu/part_neural_architectures.html>)
2. [26 Transformer](<https://visionbook.mit.edu/transformers.html>)

# 26 Transformer

## 26.1 前言

**Transformer** 是近年來發展的一系列架構，它們推廣並擴展了卷積神經網路（CNN）背後的概念。此系列架構的名稱是由 [[1](<https://visionbook.mit.edu/references.html#ref-vaswani2017attention>)] 首次提出，當時被應用於語言建模。我們在本章中的討論更接近 [[2](<https://visionbook.mit.edu/references.html#ref-dosovitskiy2020vit>)] 中引入的**視覺 Transformer**（**ViT**）。

與 CNN 類似，Transformer 將訊號處理問題分解為多個階段，其中包含獨立且相同處理的區塊（chunks）。然而，它們也包含了跨區塊混合資訊的層，稱為**注意力層**（**attention layers**），使得整個管道（pipeline）能夠對區塊之間的依賴關係進行建模。

Transformer 最初是引入於自然語言處理領域，用於語言建模，即字元與單字的序列。因此，某些文獻將 Transformer 視為循環神經網路（RNN）在序列建模上的替代方案，但事實上，Transformer 層是像卷積層一樣的*平行*處理機器，而不是像循環層那樣的序列式機器。

## 26.2 CNN 的局限性：遠距離區塊之間的獨立性

CNN 是圍繞著*局部性*（locality）的概念建構的：影像中不同的局部區域可以安全地獨立進行處理。這正是我們能夠使用具有小核心（kernels）之濾波器（filters）的原因。然而，通常影像中所有感受野（receptive fields）之間都需要共享全局資訊。卷積層並不適合用來將資訊*全局化*，因為它們唯一的做法是要麼增加濾波器的核心大小，要麼藉由堆疊多層以增加深層神經元的感受野。圖 26.1 展示了淺層 CNN 無法比較空間上距離太遠的兩個輸入節點（$x_1$ 和 $x_7$）：

[![](https://visionbook.mit.edu/figures/transformers/fig-transformers-CNN_limitations.png)](<https://visionbook.mit.edu/figures/transformers/fig-transformers-CNN_limitations.png> "圖 26.1：考慮一個核心大小為 3 的 2 層 CNN，其任務是比較 x_1 和 x_7。它無法做到這一點：沒有任何神經元同時與 x_1 和 x_7 相連。斜線標記表示分別與 x_1 和 x_7 相連的神經元。")

圖 26.1：考慮一個核心大小為 3 的 2 層 CNN，其任務是比較 $x_1$ 和 $x_7$。它無法做到這一點：沒有任何神經元同時與 $x_1$ 和 $x_7$ 相連。斜線標記（Hatch marks）表示分別與 $x_1$ 和 $x_7$ 連接的神經元。

我們該如何高效地在巨大的空間距離上傳遞訊息？我們已經看過一種選擇：直接使用全連接層（fully connected layer），這樣一來，該層之後的每個輸出神經元都會接收前一層中每個神經元的輸入。然而，全連接層擁有極多的參數（如果其輸入和輸出是 $N$ 維向量，則有 $N^2$ 個參數），且擬合（fit）所有這些參數需要耗費大量的時間與數據。我們能否想出一種更有效率的策略？

## 26.3 注意力的概念

注意力（Attention）是一種高效處理全局資訊的策略，它僅專注於對當前任務最顯著的訊號部分。這個想法可以從人類知覺中的注意力得到啟發。當我們觀看一個場景時，我們的眼睛會四處掃視，並注意某些突出的元素，而不是一次性接收整個場景 [[3](<https://visionbook.mit.edu/references.html#ref-wolfe2000visual>)]。如果我們被問到關於場景中汽車顏色的問題，我們會移動眼睛去看那輛車，而不是只是消極地凝視。我們能給予神經網路同樣的能力嗎？

在神經網路中，注意力遵循相同的直覺想法。層 $l+1$ 的一組神經元可以*關注*（attend to）層 $l$ 的一組神經元，以決定其回應應該是什麼。如果我們「要求」該組神經元報告輸入影像中任何汽車的顏色，那麼它們應該將注意力投向上一層中代表汽車顏色的神經元。我們很快就會詳細看到這是如何實現的，但首先我們需要引入一種新的數據結構以及一種思考神經處理的新方法。

## 26.4 新的數據類型：Token

我們曾討論過，深度學習中的主要數據結構是不同種類的神經元分組：通道（channels）、張量（tensors）、批次（batches）等等。現在我們將引入另一種基本數據結構：**Token**。Token 是另一種神經元分組，但我們對 Token 進行操作的特定方式，與我們之前對通道、批次及其他分組的操作方式不同。具體來說，我們會將 Token 視為*封裝*（encapsulated）的資訊群組；我們將在 Token 上定義運算子，而這些運算子將是我們存取與修改 Token 內部內容的唯一介面。從程式語言的角度來看，你可以將 Token 視為一種新的數據*類型*（type）。

在本章中，我們僅考慮內部內容為神經元向量的 Token。因此，單個 Token 將由一個列（column）向量 $\mathbf{t} \in \mathbb{R}^{d \times 1}$ 表示，有時也被稱為該 Token 的**編碼向量**（**code vector**）。

### 26.4.1 數據的 Token 化

使用 Token 的第一步是將原始輸入數據進行 *Token 化*（tokenize）。完成此步驟後，所有後續層都將在 Token 上運作，直到輸出層根據最終的 Token 集做出某些決策或預測。我們該如何將輸入影像進行 Token 化？那麼，在一般的神經網路中，我們是如何將影像「神經元化」以進行處理的？我們只需用一個神經元來表示影像中的每個*像素*（如果是彩色影像則為三個神經元）。要將影像 Token 化，我們只需用一個 Token 來表示影像中的每個*像素區塊*（patch of pixels）。Token 向量是向量化後的區塊（將三個顏色通道相繼堆疊），或者是向量化區塊的低維度投影。藉由每個區塊都用一個 Token 來表示，整張影像就對應到一個 Token 陣列。圖 26.2展示了以這種方式將野生動物園（safari）影像進行 Token 化的外觀。

[![](https://visionbook.mit.edu/figures/transformers/fig-transformers-tokenization.png)](<https://visionbook.mit.edu/figures/transformers/fig-transformers-tokenization.png> "圖 26.2：Token 化：將影像轉換為一組向量。\mathbf{W}_{\texttt{tokenize}} 是從向量化裁剪區域的維度到 d 維的可學習線性投影。這只是將影像 Token 化的眾多可能方法之一。")

圖 26.2：Token 化：將影像轉換為一組向量。$\mathbf{W}_{\texttt{tokenize}}$ 是從向量化裁剪區域（crops）的維度到 $d$ 維的可學習線性投影。這只是將影像 Token 化的眾多可能方法之一。

### 26.4.2 處理 Token 的數據結構與表示法

Token 序列將由矩陣 $\mathbf{T} \in \mathbb{R}^{N \times d}$ 表示，其中序列中的每個 Token $\mathbf{t}_1, \ldots, \mathbf{t}_N$ 經轉置後成為該矩陣的行（row）：

$$
\mathbf{T} =
  \begin{bmatrix}
    \mathbf{t}_1^\mathsf{T}\\
    \vdots \\
    \mathbf{t}_N^\mathsf{T}\\
\end{bmatrix}
$$

在圖形上，$\mathbf{T}$ 是由 $\mathbf{t}_1, \ldots, \mathbf{t}_N$ 建構而成的，如下所示（圖 26.3）：

[![](https://visionbook.mit.edu/figures/transformers/T_notation.png)](<https://visionbook.mit.edu/figures/transformers/T_notation.png> "圖 26.3：在本章中，我們將一組 Token 表示為一個矩陣，其每一行（rows）即為這些 Token 向量。")

圖 26.3：在本章中，我們將一組 Token 表示為一個矩陣，其每一行（rows）即為這些 Token 向量。

正如我們將看到的，Transformer 對輸入序列的排列（permutations）具有不變性，因此就 Transformer 而言，Token 組應該被視為*集合*，而不是有順序的序列。

這種表示法的概念是：*Token 之於 Transformer，就像神經元之於神經網路*。神經網路層在神經元陣列上進行操作；例如，MLP 接收一個列（column）向量 $\mathbf{x}$ 作為輸入，其每一行（rows，在此處即為各元素）是純量神經元。Transformer 則在 Token 陣列上進行操作。矩陣 $\mathbf{T}$ 只是向量值 Token 的一維陣列的便利表示形式。

雖然在本章中我們僅考慮向量值的 Token，但很容易想像 Token 可以是任何形式的結構化群組。我們只需要定義基本運算子（如加法）如何在此類群組上進行運作（且理想情況下是以可微分的方式）。

Transformer 對 Token 的主要操作包含兩個：（1）透過加權和來*混合*（mixing）Token，以及（2）透過非線性轉換來*修改*（modifying）每個獨立的 Token。這些操作類似於常規神經網路的兩個主力：線性層（linear layer）和逐點非線性（pointwise nonlinearity）。

### 26.4.3 混合 Token

將數據轉換為 Token 後，我們現在需要定義轉換這些 Token 並最終以此做出決策的操作。我們定義的第一個操作是：如何對 Token 進行*線性組合*（linear combination）。

[![](https://visionbook.mit.edu/figures/transformers/lin_comb_neurons_vs_tokens.png)](<https://visionbook.mit.edu/figures/transformers/lin_comb_neurons_vs_tokens.png> "圖 26.4：神經元的線性組合與 Token 的線性組合對比。")

圖 26.4：神經元的線性組合與 Token 的線性組合對比。

Token 的線性組合與神經網路中的全連接層不同。它不是對純量神經元進行加權求和，而是對向量值的 Token 進行加權求和（圖 26.4）。多個輸入和輸出神經元/Token 的這些等式的一般形式為：

$$
\begin{aligned}
    x_{\texttt{out}}[i]&= \sum_{j=1}^N w_{ij}x_{\texttt{in}}[j]\\
    \mathbf{x}_{\texttt{out}}&= \mathbf{W}\mathbf{x}_{\texttt{in}}&\quad\quad \triangleleft \quad \text{神經元的線性組合}\\
    \mathbf{T}_{\texttt{out}}[i,:]&= \sum_{j=1}^N w_{ij} \mathbf{T}_{\texttt{in}}[j,:]\\
     \mathbf{T}_{\texttt{out}}&= \mathbf{W}\mathbf{T}_{\texttt{in}}&\quad\quad \triangleleft \quad \text{Token 的線性組合}
\end{aligned} \tag{26.1}
$$

如上所示，對 Token 的操作定義與對神經元的操作類似，不同之處在於 Token 是向量值，而神經元是純量值。我們在前面章節中遇到的大多數網路層，都可以用類似於定義神經元的方式來為 Token 進行定義。

例如，我們可以在 Token 上定義一個全連接層（fc layer），作為從 $N_1$ 個輸入 Token 到 $N_2$ 個輸出 Token 的映射，並由矩陣 $\mathbf{W} \in \mathbb{R}^{N_2 \times N_1}$ 參數化（以及選擇性地由一組 Token 偏差 $\mathbf{b} \in \mathbb{R}^{N_2 \times d}$ 參數化）：

$$
\begin{aligned}
    \mathbf{T}_{\texttt{out}}&= \mathbf{W}\mathbf{T}_{\texttt{in}}+ \mathbf{b} \quad\quad &\triangleleft \quad \text{Token 上的全連接層}
\end{aligned} \tag{26.2}
$$

### 26.4.4 修改 Token

線性組合僅允許我們線性地混合與重組 Token，而堆疊線性函數只能得到另一個線性函數。在標準神經網路中，全連接層和卷積層本身無法對非線性函數進行建模，我們也遇到了同樣的問題。為了解決這個限制，我們在神經網路中加入了*逐點非線性*（pointwise nonlinearities）。這些是*個別*對每個神經元應用非線性轉換的函數，獨立於所有其他神經元。類似地，對於 Token 網路，我們將引入*逐 Token*（tokenwise）運算子；這些是*個別*對每個 Token 應用非線性轉換的函數，獨立於所有其他 Token。給定一個非線性函數 $F_{\theta}: \mathbb{R}^N \rightarrow \mathbb{R}^N$，以 $\mathbf{T}_{\texttt{in}}$ 作為輸入的逐 Token 非線性層可以表示為：

$$
\begin{aligned}
    \mathbf{T}_{\texttt{out}}=
        \begin{bmatrix}
        F_{\theta}(\mathbf{T}_{\texttt{in}}[0,:]) \\
        \vdots \\
        F_{\theta}(\mathbf{T}_{\texttt{in}}[N-1,:]) \\
        \end{bmatrix}\quad\quad \triangleleft \quad\text{單個 Token 非線性運算}
\end{aligned}
$$

請注意，此操作是常規神經網路中逐點非線性的推廣；ReLU 層是 $F_{\theta} = \texttt{relu}$ 且該層在神經元輸入（純量）而非 Token 輸入（向量）上運行的特例：

$$
\begin{aligned}
    \mathbf{x}_{\texttt{out}}=
        \begin{bmatrix}
        \texttt{relu}(x_{\texttt{in}}[0]) \\
        \vdots \\
        \texttt{relu}(x_{\texttt{in}}[N-1]) \\
        \end{bmatrix}\quad\quad \triangleleft \quad\text{單個神經元非線性運算 (\texttt{relu})}
\end{aligned}
$$

$F_{\theta}$ 可以是任何非線性函數，但某些選擇的效果會比其他選擇更好。一個常見的選擇是讓 $F_{\theta}$ 成為多層感知器（MLP）；參見「神經網路」一章。在這種情況下，$F_{\theta}$ 具有可學習的參數 $\theta$，即該 MLP 的權重與偏差。這揭示了常規神經網路中的逐點運作與 Token 網路中的 Token 運作之間的一個重要區別：ReLU 以及大多數其他逐神經元非線性沒有可學習的參數，而 $F_{\theta}$ 通常有。這是使用 Token 的有趣之處之一，逐 Token 的操作變得極具表達力且參數豐富。

## 26.5 Token 網路

我們將使用 **Token 網路**（**token nets**）一詞來指稱以 Token 作為主要節點，而非以神經元作為主要節點的計算圖。

請注意，本章中的術語並非標準術語。*Token 網路*一詞以及我們給出的一些定義，是我們自己的發明。

Token 網路就像神經網路一樣，在以線性組合混合節點的層（例如全連接線性層、卷積層等）與對每個節點應用逐點非線性的層（例如 ReLU、逐 Token MLP）之間交替進行。當然，由於 Token 只是神經元的群組，因此每個 Token 網路本身也是一個神經網路，只是從不同的角度來看——它是子網路的網路。在圖 26.5 中，我們並排展示了一個標準神經網路和一個 Token 網路，以強調它們在操作上的相似性。

[![](https://visionbook.mit.edu/figures/transformers/neural_nets_vs_token_nets.png)](<https://visionbook.mit.edu/figures/transformers/neural_nets_vs_token_nets.png> "圖 26.5：神經網路與 Token 網路對比。這裡的箭頭代表節點之間的任何函數依賴關係（請注意，不同的箭頭代表不同類型的函數）。")

圖 26.5：神經網路與 Token 網路對比。這裡的箭頭代表節點之間的任何函數依賴關係（請注意，不同的箭頭代表不同類型的函數）。

## 26.6 注意力層

**注意力層**（**Attention layers**）定義了一種特殊的 Token 線性組合。注意力層不使用自由參數矩陣 $\mathbf{W}$ 來參數化線性組合，而是使用另一個矩陣，我們稱之為注意力矩陣 $\mathbf{A}$。$\mathbf{A}$ 與 $\mathbf{W}$ 之間的重要區別在於 $\mathbf{A}$ 是*數據依賴型*（data-dependent）的，也就是說，$\mathbf{A}$ 的值是輸入到該網路之數據的函數。此外，$\mathbf{A}$ 通常僅包含非負值，這與將其視為分配我們對每個輸入 Token 支付多少（非負）注意力的矩陣的想法一致。在下面的圖表（圖 26.6）中，我們用標記為 $f$ 的函數來表示數據依賴性，並將注意力矩陣塗成紅色，以表示它是從*轉換後的數據*中建構的，而不是自由參數（對於自由參數，我們使用藍色）：

[![](https://visionbook.mit.edu/figures/transformers/fc_vs_attn.png)](<https://visionbook.mit.edu/figures/transformers/fc_vs_attn.png> "圖 26.6：全連接層與注意力層對比。")

圖 26.6：全連接層與注意力層對比。

在這裡，我們將注意力描述為具有數據依賴權重的全連接層。我們也可以將注意力描述為一種**動態池化**（**dynamic pooling**），即均值池化（mean pooling），但使用根據輸入數據動態決定的加權平均值。

除了權重是其他數據的函數外，注意力層的方程式與線性層相同（目前未具體說明，但我們隨後會看到具體示例）：

$$
\begin{aligned}
    \mathbf{A} &= f(\ldots) \quad\quad \triangleleft \text{ 注意力}\\
    \mathbf{T}_{\texttt{out}}&= \mathbf{A}\mathbf{T}_{\texttt{in}}
\end{aligned}
$$

當然，關鍵問題是 $f$ 到底是什麼？$f$ 依賴於哪些輸入，且 $f$ 的數學形式是什麼？在寫出確切的方程式之前，我們將從直覺開始：$f$ 是一個決定對 $\mathbf{T}_{\texttt{in}}$ 中的每個 Token 應用多少注意力的函數；因為該層只是 Token 的加權組合，所以 $f$ 只是在決定該組合中的權重。$f$ 可以依賴於任何數量的輸入訊號，這些訊號告訴網路該注意什麼。

作為一個具體的例子，考慮我們希望能夠對野生動物園示例影像中的不同對象提出問題，例如照片中有多少隻動物。那麼一種策略是關注代表動物頭部的每個 Token，然後將它們相加計算。$f$ 將接收文字查詢（text query）作為輸入，並產生輸出權重 $\mathbf{A}$，對於對應於任何動物頭部的 $\mathbf{T}_{\texttt{in}}$ Token，其權重較高，而對於所有其他 $\mathbf{T}_{\texttt{in}}$ Token，其權重較低。如果我們訓練這樣一個系統來回答有關計算動物數量的問題，那麼 Token 的編碼向量可能會自然而然地編碼一個代表其感受野中動物頭部數量的特徵；畢竟，這將是一個解決我們問題的方案（它將最小化損失並正確回答問題）。其他解決方案也是可能的，但我們將專注於這種直覺的解決方案，並在圖 26.7 中進行說明。

[![](https://visionbook.mit.edu/figures/transformers/attention_layer_safari_query_cartoon.png)](<https://visionbook.mit.edu/figures/transformers/attention_layer_safari_query_cartoon.png> "圖 26.7：注意力如何在影像中的不同區域（Token）之間進行分配。Token 的編碼向量包含多個維度，每個維度可以編碼 Token 的不同屬性。左側顯示了編碼動物頭部數量的維度。右側顯示了編碼顏色的另一個維度（或者這可以是三個維度，編碼 RGB）。輸出 Token 是對所有受關注 Token 的加權和。")

圖 26.7：注意力如何在影像中的不同區域（Token）之間進行分配。Token 的編碼向量包含多個維度，每個維度可以編碼 Token 的不同屬性。左側顯示了編碼動物頭部數量的維度。右側顯示了編碼顏色的另一個維度（或者這可以是三個維度，編碼 RGB）。輸出 Token 是對所有受關注 Token 的加權和。

這裡巧妙的地方在於，注意力為我們提供了一種方法，使該網路層能夠根據不同的輸入問題動態改變其行為；提出不同的問題會產生不同的答案，如下圖 26.7 所示。

讓我們逐步梳理圖 26.7 的邏輯。在這裡，我們假設有一種 Token 表徵可以回答兩種不同的問題，一個是關於數量，另一個是關於顏色。我們設計出的表徵（這可以透過學習得到）是在 Token 向量的一個維度中編碼一個常數值 $1$，用於加總受關注 Token 的數量。在另一組維度中，我們有該 Token 代表的區塊（patch）的平均 RGB 顏色。請注意，Token 僅在網路的輸入端（即 Token 化步驟之後）直接代表影像區塊；在網路的更深層，Token 所代表的內容可能會更加抽象。每個文字查詢都會引導出不同的注意力分配，我們稍後會詳細介紹該過程是如何運作的。目前只需考慮文字查詢會根據該 Token 的內容與查詢內容的匹配程度，為每個 Token 分配一個純量權重。輸出 Token $\mathbf{t}_{\texttt{out}}$ 是所有以注意力純量加權的 Token 的總和。如果文字查詢「這張照片中有多少隻動物」僅對代表動物頭部的 Token 給予注意力權重 $1$，而文字查詢「黑斑羚（impala）的顏色是什麼」僅對黑斑羚的 Token 給予權重 $\frac{1}{3}$，那麼這個方案將會對這些問題得出合理的答案。如此一來，前一種情況下的輸出向量在代表受關注 Token 數量的維度中包含正確答案 $4$，並在代表平均區塊顏色的維度中包含偏棕色的 RGB 值。

記住這個直覺的畫面，我們現在轉向定義注意力分配函數 $f$ 的方程式。我們將專注於 Transformer 中出現的特定版本 $f$，稱為**查詢-鍵-值注意力**（**query-key-value attention**）。

### 26.6.1 查詢-鍵-值注意力

Transformer 使用一種基於查詢（queries）、鍵（keys）和值（values）概念的特定注意力。在查詢-鍵-值注意力中，每個 Token 都與一個**查詢**（**query**）向量、一個**鍵**（**key**）向量和一個**值**（**value**）向量相關聯。

查詢、鍵和值的概念來自於資料庫，其中資料庫單元格儲存一個*值*（value），當*查詢*（query）與該單元格的*鍵*（key）相匹配時，該值就會被檢索出來。Token 就像資料庫單元格，而注意力就像是從 Token 的資料庫中檢索資訊。

我們將這些向量定義為 Token 編碼向量的線性轉換，投影到長度為 $m$ 的查詢/鍵/值向量。對於一個 Token $\mathbf{t}$，我們有：

$$
\begin{aligned}
    \mathbf{q} &= \mathbf{W}_q \mathbf{t} \quad\quad \triangleleft \text{ 查詢 (query)}\\
    \mathbf{k} &= \mathbf{W}_k \mathbf{t} \quad\quad \triangleleft \text{ 鍵 (key)}\\
    \mathbf{v} &= \mathbf{W}_v \mathbf{t} \quad\quad \triangleleft \text{ 值 (value)}
\end{aligned}
$$

這是一個值得思考的問題：您能否使用其他可微分的函數來計算查詢、值 and 鍵？那會有用嗎？

在 Transformer 中，網路的所有輸入都會被 Token 化，因此文字問題「照片中有多少隻動物？」也將被表示為一個 Token。

我們在本書中不會涵蓋這些內容，但可以使用自然語言處理的方法將文字轉換為一個 Token，或轉換為一個 Token 序列。

此 Token 將提交其查詢向量 $\mathbf{q}_{\texttt{question}}$，與代表影像中不同區塊的 Token 的鍵（keys）進行匹配；查詢與鍵之間的相似度決定了該查詢將應用於具有該鍵之 Token 的注意力權重大小。衡量查詢 $\mathbf{q}$ 與鍵 $\mathbf{k}$ 之間相似度最常用的度量是內積（dot product） $\mathbf{q}^\mathsf{T}\mathbf{k}$。

以此方式查詢 $\mathbf{T}_{\texttt{in}}$ 中的每個 Token，會給我們一個相似度向量：

$$
\begin{aligned}
    \mathbf{s} = [s_1, \ldots, s_N]^\mathsf{T}&= [\mathbf{q}_{\texttt{question}}^\mathsf{T}\mathbf{k}_1, \ldots, \mathbf{q}_{\texttt{question}}^\mathsf{T}\mathbf{k}_N]^\mathsf{T}
\end{aligned} \tag{26.3}
$$

然後，我們使用 softmax 函數對向量 $\mathbf{s}$ 進行正規化，以得到我們的注意力權重 $\mathbf{a} \in \mathbb{R}^{N \times 1}$。最後，我們不是直接在 Token 編碼上應用 $\mathbf{a}$（即對 Token 進行加權求和），而是對 Token 的值（value）向量進行加權求和，以得到 $\mathbf{T}_{\texttt{out}}$：

$$
\begin{aligned}
    \mathbf{a} &= \texttt{softmax}(\mathbf{s})\\
    \mathbf{T}_{\texttt{out}}&= \begin{bmatrix}
        a_1\mathbf{v}_1^\mathsf{T}\\
        \vdots \\
        a_N\mathbf{v}_N^\mathsf{T}\\
    \end{bmatrix}
\end{aligned} \tag{26.4}
$$

其中 $\mathbf{v}_1$ 是 $\mathbf{t}_1=\mathbf{T}_{\text {in }}[0,:]$ 的值向量，依此類推。

我們在此處以及本章稍後使用以下配色方案：

[![](https://visionbook.mit.edu/figures/transformers/fig-transformers-color_scheme.png)](<https://visionbook.mit.edu/figures/transformers/fig-transformers-color_scheme.png>)

圖 26.8 將這些步驟進行了視覺化。

[![](https://visionbook.mit.edu/figures/transformers/attn_arch1.png)](<https://visionbook.mit.edu/figures/transformers/attn_arch1.png> "圖 26.8：注意力層的機制。來自問題的查詢與代表黑斑羚的 Token 鍵相匹配；黑斑羚 Token 的值向量隨後對求和的貢獻最大，從而產生 \mathbf{t}_{\texttt{out}} 的編碼向量。（在此示例中省略了 Softmax。）")

圖 26.8：注意力層的機制。來自問題的查詢與代表黑斑羚（impala）的 Token 鍵相匹配；黑斑羚 Token 的值向量隨後對求和的貢獻最大，從而產生 $\mathbf{t}_{\texttt{out}}$ 的編碼向量。（在此示例中省略了 Softmax。）

### 26.6.2 自注意力

正如我們現在所看到的，注意力是一種通用方法，用於根據來自另一組 Token 的查詢，動態地池化一組 Token 中的資訊。我們將考慮的下一個問題是，哪些 Token 應該進行查詢，而我們又應該與哪些 Token 進行匹配？在上一節的示例中，答案很直觀，因為我們有一個文字問題在詢問視覺影像中的內容，因此自然是由文字給出查詢，而我們與代表影像的 Token 進行匹配。但是，我們能否設計出一個更通用的架構，而不需要手動設計哪些 Token 以哪些方式進行交互？

**自注意力**（**Self-attention**）就是這樣一種架構。其想法是在自注意力層上，*所有* Token 都提交查詢，且對於每個查詢，我們對該層中的*所有* Token 進行加權求和。如果 $\mathbf{T}_{\texttt{in}}$ 是一組 $N$ 個輸入 Token，那麼我們就有 $N$ 個查詢、$N$ 個加權和以及 $N$ 個輸出 Token，以形成 $\mathbf{T}_{\texttt{out}}$。這在下面的圖 26.9 中進行了視覺化。

[![](https://visionbook.mit.edu/figures/transformers/self_attn_layer.png)](<https://visionbook.mit.edu/figures/transformers/self_attn_layer.png> "圖 26.9：自注意力層。")

圖 26.9：自注意力層。

為了計算一組輸入 Token $\mathbf{T}_{\texttt{in}}$ 的查詢、鍵和值，我們對該組中的每個 Token 應用相同的線性轉換，從而得到矩陣 $\mathbf{Q}_{\texttt{in}}, \mathbf{K}_{\texttt{in}} \in \mathbb{R}^{N \times m}$ 和 $\mathbf{V}_{\texttt{in}} \in \mathbb{R}^{N \times d}$，其中每一行分別為每個 Token 的查詢/鍵/值：

請注意，查詢向量和鍵向量必須具有相同的維度 $m$，因為我們要在它們之間進行內積。相反地，值向量必須與 Token 編碼向量的維度 $d$ 相匹配，因為這些向量會被加總以產生新的 Token 編碼向量。

$$
\begin{aligned}
    \mathbf{Q}_{\texttt{in}}&=
     \begin{bmatrix}
        \mathbf{q}_1^\mathsf{T}\\
        \vdots \\
        \mathbf{q}_N^\mathsf{T}\\
     \end{bmatrix}
    =
    \begin{bmatrix}
        (\mathbf{W}_q \mathbf{t}_1)^\mathsf{T}\\
        \vdots \\
        (\mathbf{W}_q \mathbf{t}_N)^\mathsf{T}\\
    \end{bmatrix}
    = \mathbf{T}_{\texttt{in}}\mathbf{W}_q^\mathsf{T}&\triangleleft \quad\quad \text{查詢 (query) 矩陣} \\
    \mathbf{K}_{\texttt{in}}&=
     \begin{bmatrix}
        \mathbf{k}_1^\mathsf{T}\\
        \vdots \\
        \mathbf{k}_N^\mathsf{T}\\
     \end{bmatrix}
    =
    \begin{bmatrix}
        (\mathbf{W}_k \mathbf{t}_1)^\mathsf{T}\\
        \vdots \\
        (\mathbf{W}_k \mathbf{t}_N)^\mathsf{T}\\
    \end{bmatrix}
    = \mathbf{T}_{\texttt{in}}\mathbf{W}_k^\mathsf{T}&\triangleleft \quad\quad \text{鍵 (key) 矩陣}\\
    \mathbf{V}_{\texttt{in}}&=
     \begin{bmatrix}
        \mathbf{v}_1^\mathsf{T}\\
        \vdots \\
        \mathbf{v}_N^\mathsf{T}\\
     \end{bmatrix}
    =
    \begin{bmatrix}
        (\mathbf{W}_v \mathbf{t}_1)^\mathsf{T}\\
        \vdots \\
        (\mathbf{W}_v \mathbf{t}_N)^\mathsf{T}\\
    \end{bmatrix}
    = \mathbf{T}_{\texttt{in}}\mathbf{W}_v^\mathsf{T}&\triangleleft \quad\quad \text{值 (value) 矩陣}
\end{aligned} \tag{26.5}
$$

最後，我們得到注意力方程式：

$$
\begin{aligned}
    \mathbf{A} &= f(\mathbf{T}_{\texttt{in}}) = \texttt{softmax}\Big(\frac{\mathbf{Q}_{\texttt{in}}\mathbf{K}_{\texttt{in}}^\mathsf{T}}{\sqrt{m}}\Big) &\triangleleft \quad\quad \text{注意力矩陣}\\
    \mathbf{T}_{\texttt{out}}&= \mathbf{A}\mathbf{V}_{\texttt{in}}
\end{aligned} \tag{26.6}
$$

其中 softmax 是在每一行內進行的（即對每個單獨查詢向量的匹配向量進行計算，如等式 26.3 所示）。以下展示了自注意力層的完整詳細機制（圖 26.10）：

[![](https://visionbook.mit.edu/figures/transformers/attn_arch2.png)](<https://visionbook.mit.edu/figures/transformers/attn_arch2.png> "圖 26.10：展開的自注意力層。帶虛線外框的節點互相比對；它們代表一個查詢與一個鍵相匹配，從而在灰色框中產生一個純量相似度值，該值作為由 \mathbf{A} 計算的加權和中的權重。")

圖 26.10：展開的自注意力層。帶虛線外框的節點互相比對；它們代表一個查詢與一個鍵相匹配，從而在灰色框中產生一個純量相似度值，該值作為由 $\mathbf{A}$ 計算的加權和中的權重。

這完整定義了自注意力層，也就是 Transformer 中使用的注意力層類型。不過，在我們繼續之前，讓我們思考一下自注意力可能在做些什麼的直覺。

考慮我們正在處理野生動物園影像，而我們的任務是語意分割（semantic segmentation，即用對象類別標記每個區塊）。圖 26.11 說明了這種情況。我們首先對影像進行 Token 化，使每個區塊都由一個 Token 表示。現在我們有一個 Token $\mathbf{t}_2$，它代表黑斑羚軀幹周圍的像素區塊。我們希望透過一層自注意力來更新這個 Token。由於該網路的目標是分類區塊，因此更新 $\mathbf{t}_2$ 以獲得該區塊中正在發生事情的更好語意表徵是有意義的。一種方法是關注代表黑斑羚其他區塊的 Token，並使用它們將 $\mathbf{t}_2$ 精煉為一個更抽象的 Token 向量，捕捉「黑斑羚」這個標籤。直覺是，給定周圍其他相關區塊的背景資訊，會更容易識別該區塊。精煉操作只是對 Token 的編碼向量求和，其效果是減少了三個受關注的黑斑羚區塊之間不共享的噪聲，從而放大了它們之間的共同點——「黑斑羚」標籤。更複雜的精煉可以透過多層自注意力來實現。此外，黑斑羚區塊的查詢也可以從長頸鹿和斑馬區塊中檢索資訊，因為這些區塊提供了可能有用的額外背景資訊（如果查詢中的動物是在長頸鹿和斑馬附近被發現，它就更有可能是黑斑羚，因為所有這些動物都傾向於聚集在同一個生態棲地中）。

[![](https://visionbook.mit.edu/figures/transformers/attention_layer_cartoon.png)](<https://visionbook.mit.edu/figures/transformers/attention_layer_cartoon.png> "圖 26.11：一種可使用自注意力來聚合包含相同對象的所有區塊之資訊的方法，從而為查詢區塊 \mathbf{t}_2 取得更好的對象表徵。")

圖 26.11：一種可使用自注意力來聚合包含相同對象的所有區塊之資訊的方法，從而為查詢區塊 $\mathbf{t}_2$ 取得更好的對象表徵。

這只是網路使用自注意力的一種方式。它具體如何被使用將由訓練數據和任務決定。實際發生的情況可能與我們的直覺故事有所偏差：隱藏層上的 Token 不一定代表空間上局部的像素區塊。雖然初始的 Token 化層會從局部影像區塊中創建 Token，但在這之後，注意力層可以混合空間上相距甚遠的 Token 之間的資訊；請注意， $\mathbf{T}_{\texttt{out}}[0,:]$ 不一定代表影像中與 $\mathbf{T}_{\texttt{in}}[0,:]$ 相同的空間區域。

圖 26.12 給出了一個自注意力地圖（attention maps）在野生動物園影像上的示例。在此示例中，我們僅使用區塊顏色作為查詢和鍵的特徵。每張注意力地圖顯示了將 $\mathbf{A}$ 的一行重新調整為輸入影像大小後的結果。

[![](https://visionbook.mit.edu/figures/transformers/transformers_attn_ex.png)](<https://visionbook.mit.edu/figures/transformers/transformers_attn_ex.png> "圖 26.12：自注意力地圖的示例，其中每個 Token 是一個影像區塊，且查詢和鍵向量都被設置為該區塊的平均顏色，並正規化為單位向量。")

圖 26.12：自注意力地圖的示例，其中每個 Token 是一個影像區塊，且查詢和鍵向量都被設置為該區塊的平均顏色，並正規化為單位向量。

### 26.6.3 多頭自注意力

儘管自注意力層功能強大，但它們仍然受到限制，因為它們只有一組查詢/鍵/值投影矩陣（即 $\mathbf{W}_q$, $\mathbf{W}_k$, $\mathbf{W}_v$）。這些矩陣定義了用於將查詢與鍵進行匹配的相似度概念。特別是，兩個 Token $i$ 和 $j$ 之間的相似度測量為：

$$
\begin{aligned}
     s_{ij} &= \mathbf{q}_i^\mathsf{T}\mathbf{k}_j\\
     &= (\mathbf{W}_q \mathbf{t}_i)^\mathsf{T}\mathbf{W}_k \mathbf{t}_j\\
     &= \mathbf{t}_i^\mathsf{T}\mathbf{W}_q^\mathsf{T}\mathbf{W}_k \mathbf{t}_j\\
     &= \mathbf{t}_i^\mathsf{T}\mathbf{S}\mathbf{t}_j
\end{aligned}
$$

這表明 $\mathbf{W}_q$ and $\mathbf{W}_k$ 定義了某個矩陣 $\mathbf{S} = \mathbf{W}_q^\mathsf{T}\mathbf{W}_k$，該矩陣調節了我們如何測量 $\mathbf{t}_i$ 和 $\mathbf{t}_j$ 之間的相似度（內積）。因此，單個自注意力層僅以一種方式測量相似度。

如果我們想以不止一種方式測量相似度呢？例如，也許我們希望我們的網路基於顏色相似性執行一組計算，基於紋理相似性執行另一組計算，並基於形狀相似性執行又一組計算？Transformer 做到這一點的方法是使用**多頭自注意力**（**multihead self-attention**，**MSA**）。此方法僅包含平行運行 $k$ 個注意力層。所有這些層都應用於同一個輸入 $\mathbf{T}_{\texttt{in}}$。這會產生 $k$ 個輸出 Token 集：$\mathbf{T}_{\texttt{out}}^1, \ldots, \mathbf{T}_{\texttt{out}}^k$。為了合併這些輸出，我們將它們全部串接（concatenate），並投影回 $\mathbf{T}_{\texttt{in}}$ 的原始維度。這些步驟如以下數學公式所示：

$$
\begin{aligned}
    \mathbf{T}_{\texttt{out}}^i &= \texttt{attn}^i(\mathbf{T}_{\texttt{in}}) \quad \text{for } i \in \{1,\ldots,k\}\\
    \bar{\mathbf{T}}_{\texttt{out}} &= \begin{bmatrix}
        \mathbf{T}_{\texttt{out}}^1[0,:] & \ldots & \mathbf{T}_{\texttt{out}}^k[0,:]\\
        \vdots & \vdots & \vdots \\
        \mathbf{T}_{\texttt{out}}^1[N-1,:] & \ldots & \mathbf{T}_{\texttt{out}}^k[N-1,:]\\
    \end{bmatrix} &\quad\quad \triangleleft \quad \bar{\mathbf{T}}_{\texttt{out}} \in \mathbb{R}^{N \times kv}\\
    \mathbf{T}_{\texttt{out}}&= \bar{\mathbf{T}}_{\texttt{out}}\mathbf{W}_{\texttt{MSA}} &\quad\quad \triangleleft \quad \mathbf{W}_{\texttt{MSA}} \in \mathbb{R}^{kv \times d}
\end{aligned} \tag{26.7}
$$

其中 $v$ 是值向量的維度，而 $d$ 是輸出編碼向量的維度（[[2](<https://visionbook.mit.edu/references.html#ref-dosovitskiy2020vit>)] 建議設定 $kv = d$）。矩陣 $\mathbf{W}_{\texttt{MSA}}$ *合併*了所有的頭；它的值是可學習的參數。MSA 的其他可學習參數是 $k$ 個注意力頭中每一個的查詢、鍵和值投影。

請注意，與前面介紹的單頭自注意力層不同，這裡的值向量不需要與 Token 編碼向量具有相同的維度，因為我們應用了投影等式 26.7。

這裡的基本道理非常簡單：如果自注意力層是件好事，為什麼不直接多加一些呢？我們可以透過構建更深的 Transformer 來增加*順序*自注意力層，或者我們可以透過使用 MSA 來增加*平行*自注意力層。

## 26.7 完整的 Transformer 架構

完整的 Transformer 架構是交替堆疊自注意力層與逐 Token 非線性層而成的。這兩個步驟類似於 MLP 中交替堆疊線性層與逐神經元非線性層，如下圖 26.13 所示：

[![](https://visionbook.mit.edu/figures/transformers/transformer_vs_MLP.png)](<https://visionbook.mit.edu/figures/transformers/transformer_vs_MLP.png> "圖 26.13：基本 Transformer 架構與 MLP 對比。")

圖 26.13：基本 Transformer 架構與 MLP 對比。

除了這個基本模板之外，還可以添加許多變體，從而在 Transformer 系列中產生不同的特定架構。一些常見的添加是正規化層（normalization layers）和殘差連接（residual connections）。在圖 26.14 中，我們繪製了來自 [[2](<https://visionbook.mit.edu/references.html#ref-dosovitskiy2020vit>)] 的 ViT 架構，顯示了這些額外組件在何處加入。

[![](https://visionbook.mit.edu/figures/transformers/ViT_arch.png)](<https://visionbook.mit.edu/figures/transformers/ViT_arch.png> "圖 26.14：ViT Transformer 架構。這組層形成了一個計算區塊（以灰色陰影表示），對於深度為 L 的 ViT，該區塊可以重複 L 次。為了澄清參數在此架構中的位置，我們將所有具有可學習參數的邊塗成藍色（請注意，MSA 合併在此圖中並未明確顯示，但它也是可學習的）。")

圖 26.14：ViT Transformer 架構。這組層形成了一個計算區塊（以灰色陰影表示），對於深度為 $L$ 的 ViT，該區塊可以重複 $L$ 次。為了澄清參數在此架構中的位置，我們將所有具有可學習參數的邊塗成藍色（請注意，MSA 合併在此圖中並未明確顯示，但它也是可學習的）。

此架構在每個注意力層之前以及每個逐 Token MLP 層之前使用圖層正規化（layer normalization，[第 12.7.3 節](https://visionbook.mit.edu/neural_nets.html#sec-neural_nets-normalization_layers)）。該正規化是在每個 Token *內部*進行的（Token 編碼向量被視為類似於一個圖層；此向量的每個維度都根據該向量所有維度的平均值與變異數進行標準化），因此我們將此層稱為 `token norm`。請注意，`token norm` 是一個逐 Token 的操作，就像我們的逐 Token MLP 一樣，但它執行不同類型的轉換，且沒有可學習的參數。殘差連接被添加在每組層的周圍。

此 ViT（具有單頭注意力）的虛擬碼如下所示：

```python
# x : 輸入數據 (RGB 影像)
# K : Token 化的區塊 (patch) 大小
# d : Token/查詢/鍵/值維度 (將這些均設為相同)
# L : 網路層數
# W_q_T, W_k_T, W_v_T : 轉置後的查詢/鍵/值投影矩陣
# mlp: 逐 Token MLP

# 將輸入影像進行 Token 化
T = tokenize(x,K) # 3 x H x W 影像 --> N x d Token 編碼向量陣列

# 讓 Token 運行通過所有 L 層
for l in range(L):

    # 注意力層
    Q, K, V = nn.matmul(nn.layernorm(T),[W_q_T[l], W_k_T[l], W_v_T[l]]) 
    # nn.matmul 執行矩陣乘法
    A = nn.softmax(nn.matmul(Q,K.transpose())/sqrt(d), dim=0)
    T = nn.matmul(A,V) + T # 注意殘差連接

    # 逐 Token MLP
    T = mlp[l](nn.layernorm(T)) + T # 注意殘差連接

# T 現在包含由 Transformer 計算出的輸出 Token 表徵
```

我們目前定義的 Transformer 輸出是一組 Token $\mathbf{T}_{\texttt{out}}$。通常我們希望輸出具有不同的格式，例如用於影像分類的單個對數幾率（logits）向量（[第 9.7.3 節](https://visionbook.mit.edu/intro_to_learning.html#sec-intro_to_learning-image_classification)），或者用於影像到影像（image-to-image）任務的影像格式（[第 34.6 節](https://visionbook.mit.edu/conditional_generative_models.html#sec-conditional_generative_models-im2im)）。為了處理這些情況，我們通常會定義一個特定任務的輸出層，該層將 $\mathbf{T}_{\texttt{out}}$ jako 作為輸入，並產生所需的格式作為輸出。例如，要產生一個對數幾率預測向量，我們可以先對 $\mathbf{T}_{\texttt{out}}$ 中的所有 Token 編碼向量求和，然後使用單個線性層將得到的 $d$ 維向量投影到 $K$ 維向量中（用於 $K$ 類分類）。

## 26.8 排列等變性

Transformer 的一個重要性質是，它們對於輸入 Token 序列的排列具有等變性（equivariance）。這是因為逐 Token 層 $F_{\theta}$ 與注意力層 $\texttt{attn}$ 都是**排列等變**的：

$$
\begin{aligned}
    F_{\theta}(\texttt{permute}(\mathbf{T}_{\texttt{in}})) &= \texttt{permute}(F_{\theta}(\mathbf{T}_{\texttt{in}}))\\
    \texttt{attn}(\texttt{permute}(\mathbf{T}_{\texttt{in}})) &= \texttt{permute}(\texttt{attn}(\mathbf{T}_{\texttt{in}}))
\end{aligned}
\tag{26.8}
$$

其中 $\texttt{permute}$ 是對 $\mathbf{T}_{\texttt{in}}$ 中 Token 順序的排列（即對矩陣的列（rows）進行排列）。這意味著，如果你打亂（即排列）輸入影像中的區塊，然後應用注意力，則輸出將保持不變，除了原始輸出的排列也會相應改變之外。由於完整的 Transformer 架構僅是這兩類層的組合（加上可能存在的殘差連接和 Token 正規化，它們也是排列等變的），並且由於組合兩個排列等變函數會產生排列等變運算，因此我們有：

$$
\begin{aligned}
    \texttt{transformer}(\texttt{permute}(\mathbf{T}_{\texttt{in}})) &= \texttt{permute}(\texttt{transformer}(\mathbf{T}_{\texttt{in}}))
\end{aligned}
$$

此性質在圖 26.15 中進行了視覺化。

[![](https://visionbook.mit.edu/figures/transformers/permutation_equivariance.png)](<https://visionbook.mit.edu/figures/transformers/permutation_equivariance.png> "圖 26.15：Transformer 是排列等變的。為了簡化表示法，我們在此省略了 Token 變數上的層索引。")

圖 26.15：Transformer 是排列等變的。為了簡化表示法，我們在此省略了 Token 變數上的層索引。

從不變性（invariances）和等變性（equivariances）的角度來理解網絡層通常很有用。卷積層是平移等變的，但不必是排列等變的，而注意力層則是平移等變*和*排列等變的（由於平移是排列的一種特殊情況，任何排列等變層也必然是平移等變層）。其他層也可以類似地進行歸類：全局平均池化層是排列*不變*的，ReLU 層是排列等變的，逐 Token MLP 層也是排列等變的（但相對於 Token 集合而非神經元集合），依此類推。

一個通常很好的策略是選擇能夠反映你的數據或任務對稱性的網路層：在目標檢測中，平移等變性是合理的，因為大致上，一隻鳥無論出現在影像的哪個位置都依然是一隻鳥。出於同樣的原因，排列等變性也可能合理，但也僅在一定程度上合理：如果你把一張影像分成小區塊並將它們打亂，這可能會破壞對於識別非常重要的空間佈局。我們將在第 26.11 節中看到 Transformer 如何使用所謂的位置編碼來重新插入有關空間佈局的有用資訊。

## 26.9 偽裝的 CNN

Transformer 提供了一種思考數據處理的新方法，看起來它們與以往的架構非常不同。然而，正如我們所提到的，它們實際上與 CNN 有許多共同點。事實上，Transformer 架構的大部分（但不是全部）都可以被視為偽裝的 CNN。在本節中，我們將梳理上面學到的幾個網路層，並看看它們實際上是如何執行卷積的。

### 26.9.1 Token 化

使用 Transformer 的第一步是將輸入進行 Token 化。最基本的方法是將輸入影像切成大小為 $K \times K$ 的非重疊區塊，然後透過線性投影將這些區塊轉換為向量。你可能已經注意到，此操作可以寫成卷積；畢竟我們說過 CNN 的核心想法就是將訊號切成區塊。特別是，這種形式的 Token 化可以寫成核心大小和步長（stride）都等於 $K$ 的卷積層：

$$
\begin{aligned}
&\mathbf{T}[n(N/K)+m,c_{\texttt{2}}] = \nonumber\\
&b[c_{\texttt{2}}] +  \sum_{c_{\texttt{1}}=1}^{C_{\texttt{in}}} \sum_{k_1,k_2=-K}^K w[c_{\texttt{1}},c_{\texttt{2}},k_1,k_2] x_{\texttt{in}}[c_{\texttt{1}},K n-k_1,K m-k_2] \quad \triangleleft \quad \text{(Token 化)}
\end{aligned} \tag{26.9}
$$

其中，對於 RGB 影像， $\mathbf{x}_{\texttt{in}}\in \mathbb{R}^{3 \times N \times M}$，$C_{\texttt{in}}=3$，且 $C_{\texttt{out}}=d$（Token 的維度）。此數學公式假設 $N$ 和 $M$ 可以被 $K$ 整除；如果不能，則可以對輸入進行縮放或填充（pad），直到可以整除為止。

雖然這個等式看起來開始變得複雜，但它其實就是一個具有以下參數的 `conv` 運算子：

```python
T = conv(x_in, channels_in=3, channels_out=d, kernel=K, stride=K) # Token 化
```

### 26.9.2 查詢-鍵-值投影

接下來我們看看作為注意力層一部分的查詢、鍵和值投影。為簡單起見，我們將僅考慮查詢投影，因為鍵和值遵循完全相同的模式。

我們將此操作寫為矩陣乘法 $\mathbf{T}_{\texttt{in}}\mathbf{W}_q^\mathsf{T}$（等式 26.5）。這個乘法所做的是對每個 Token 向量（$\mathbf{T}_{\texttt{in}}$ 的每一行）應用相同的線性轉換（$\mathbf{W}_q$）。對序列中的每個元素應用相同的線性運算正是卷積所做的事情。具體來說，查詢操作可以寫成將包含 $N$ 個 $d$ 通道 Token 的集合與包含 $m$ 個濾波器、核心大小為 1 的濾波器組（filter bank）進行卷積，從而產生一組新的 $N$ 個 $m$ 通道 Token。這種等價性在下圖 26.16 中進行了視覺化：

[![](https://visionbook.mit.edu/figures/transformers/conv_matmul_equivalence-2.png)](<https://visionbook.mit.edu/figures/transformers/conv_matmul_equivalence-2.png> "圖 26.16：Transformer 中的查詢、鍵和值投影既可以寫成卷積，也可以寫成矩陣乘法。")

圖 26.16：Transformer 中的查詢、鍵 and 值投影既可以寫成卷積，也可以寫成矩陣乘法。

因此，查詢、鍵和值投影都是核心大小為 1 的多通道卷積。

卷積實際上在線性代數中無處不在！每當你看到乘積 $\mathbf{A}\mathbf{B}$ 時，你都可以將其視為多通道濾波器組 $\mathbf{B}$（每一行一個濾波器；核心大小為 1）與訊號 $\mathbf{A}$（時間索引行，通道在列中）的卷積。

### 26.9.3 逐 Token MLP

接下來我們將考慮逐 Token MLP 層。逐 Token MLP 對序列中的每個 Token 應用相同的 MLP $F_{\theta}$。$F_{\theta}$ 由線性層和逐點非線性組成。為簡單起見，我們假設沒有偏差（bias）（作為練習，可以將此條件放寬）。 $F_{\theta}$ 中的線性層都具有以下形式：

$$
\begin{aligned}
    \mathbf{t}_{\texttt{out}} &= \mathbf{W}\mathbf{t}_{\texttt{in}}
\end{aligned} \tag{26.10}
$$

當我們對序列中的每個 Token 應用此層時，我們有：

$$
\begin{aligned}
    \mathbf{T}_{\texttt{out}}&= \mathbf{T}_{\texttt{in}}\mathbf{W}^{\mathsf{T}}
\end{aligned}
$$

請注意，這看起來與我們在上一節介紹的查詢操作（等式 26.5）完全一樣。因此，相同的結果成立：逐 Token MLP 的線性層都可以寫成核心大小為 1 的卷積。

現在，MLP 中的逐點非線性是逐神經元應用的，因此這些層的功能與 CNN 中的逐點非線性完全相同。這就是 MLP 中的完整層集，因此我們得到：逐 Token MLP 可以寫成一系列卷積與逐神經元非線性交替進行，即一個 CNN。

### 26.9.4 CNN 與 Transformer 的相似之處

正如我們所看到的，Transformer 中的大多數層都是卷積層。這些層將訊號處理問題分解為多個區塊，然後獨立且相同地處理每個區塊。Transformer 中的其他一些操作——正規化層、殘差連接等——在 CNN 中也很常見。那麼，Transformer 和 CNN 之間有什麼*不同*呢？

將問題分解為區塊是如此具有根本性用處的想法，以至於它以不同的名稱出現在許多不同的領域。它的一個通用名稱是將問題*因子化*（factorizing）為更小的部分。

### 26.9.5 CNN 與 Transformer 的相異之處

#### 26.9.5.1 CNN 可以具有非單位空間範圍的核心

當我們將它們寫成卷積時，查詢-鍵-值投影和逐 Token MLP *僅使用 1x1 濾波器*。

我們使用「1x1 濾波器」一詞來指稱核心大小在所有維度上均為 1 的任何濾波器，無論該訊號是一維、二維、三維等。

事實上，情況不可能以其他方式呈現。如果你使用更大的核心，它將破壞 Transformer 的排列等變性質，因為濾波器的輸出將取決於哪個 Token 挨著哪個 Token。這是 CNN 和 Transformer 之間的核心區別之一。CNN 使用 $K \times K$ 濾波器，這使得相鄰的影像區域可以一起被處理。Transformer 使用 1x1 濾波器，這意味著該網路在架構上無法得知空間結構（哪個 Token 挨著哪個 Token）。對於空間結構通常極為重要的視覺問題，Transformer 可以藉由網路的*輸入*（而非架構結構）來賦予位置知識。我們將在「位置編碼」一節中討論這個想法。

#### 26.9.5.2 Transformer 具有注意力層

注意力層*不是*卷積層。它們不將處理過程分解為獨立的區塊，而是執行全局運算，在該運算中所有輸入 Token 都可以相互作用。由此產生的 Token 線性組合並非 CNN 中所見的混合操作，並且解決了 CNN 視野狹窄（myopic）的局限性，即每個濾波器僅能看到其感受野中的資訊。

## 26.10 遮罩注意力

有時我們想要限制哪些 Token 可以關注哪些 Token。這可以透過*遮罩*（masking）注意力矩陣來實現，也就是將某些權重固定為零。這在許多設定中都很有用，包括透過遮罩自動編碼（masked autoencoding）學習特徵 [[4](<https://visionbook.mit.edu/references.html#ref-he2022masked>)]、不同數據模態之間的交叉注意力（cross-attention） [[5](<https://visionbook.mit.edu/references.html#ref-wei2020multi>)]，以及序列預測（sequential prediction） [[6](<https://visionbook.mit.edu/references.html#ref-chen2020generative>)]。為了解釋這一點，我們將描述序列預測的使用場景。

為簡單起見，本節中我們用一維編碼向量來描述 Token，但請記住，對於 $d$ 維編碼向量，$\mathbf{T}$ 會有 $d$ 列（columns，在此指矩陣的直行）。

一個常見的問題是，給定前 $n$ 個 Token，預測序列中的第 $(n+1)$ 個 Token。例如，我們可能正試圖預測代表影片中下一幀、句子中下一個單字，或下一天天氣的 Token。

$\mathbf{T}_{1:n}$ 是 Token 序列 $\begin{bmatrix} \mathbf{t}_1^\mathsf{T}\\ \vdots\\ \mathbf{t}_{n}^\mathsf{T} \end{bmatrix}$ 的簡寫。

對此預測問題進行建模的一種簡單方法是使用線性層：$\mathbf{y}_{n+1} = \mathbf{A}\mathbf{T}_{1:n}$。以下是其圖示，右側是以矩陣乘法表示的網路層：

[![](https://visionbook.mit.edu/figures/transformers/masked_prediction1.png)](<https://visionbook.mit.edu/figures/transformers/masked_prediction1.png> "圖 26.17：根據時間索引 1-3 遮罩預測時間索引 4。")

圖 26.17：根據時間索引 1-3 遮罩預測時間索引 4。

在訓練期間，我們會給出如下的範例：

$$
\begin{aligned}
    \{\mathbf{t}_1, \ldots, \mathbf{t}_n\} &\rightarrow \mathbf{t}_{n+1}\\
    \{\mathbf{t}_1, \ldots, \mathbf{t}_{n-1}\} &\rightarrow \mathbf{t}_n\\
    \{\mathbf{t}_1, \ldots, \mathbf{t}_{n-2}\} &\rightarrow \mathbf{t}_{n-1}
\end{aligned} \tag{26.11}
$$

依此類推。我們可以使用單個矩陣乘法一次性做出所有這些預測（圖 26.18）：

[![](https://visionbook.mit.edu/figures/transformers/masked_attn_one_matmul.png)](<https://visionbook.mit.edu/figures/transformers/masked_attn_one_matmul.png> "圖 26.18：使用遮罩注意力一次性做出多個因果預測。黑色單元格是被遮罩的，它們被填入零。")

圖 26.18：使用遮罩注意力一次性做出多個因果預測。黑色單元格是被遮罩的，它們被填入零。

透過這種方式，一次前向傳播（forward pass）可以做出 $N$ 個預測，而不是一個預測。這相當於將「預測下一個 Token」執行了 $N$ 次，但這一切都發生在單個矩陣乘法中，使用的是右側所示的矩陣。

這種矩陣被稱為*因果*（causal）矩陣，因為每個輸出索引 $i$ 僅依賴於滿足 $j < i$ 的輸入索引 $j$。如果 $\mathbf{A}$ 是一個注意力矩陣，那麼這種策略就被稱為**因果注意力**（**causal attention**）。這是一種遮罩策略，其中每個 Token 只能關注序列中的*前述*（previous）Token。這種方法可以顯著加快訓練速度，因為所有的子序列預測問題（給定 $\mathbf{T}_{1:n-2}$ 預測 $\mathbf{t}_{n-1}$、給定 $\mathbf{T}_{1:n-1}$ 預測 $\mathbf{t}_{n}$、給定 $\mathbf{T}_{1:n}$ 預測 $\mathbf{t}_{n+1}$）都是同時受到監督（supervised）學習的。

這也適用於多於一層的 Transformer，其遮罩策略如圖 26.19 所示。

[![](https://visionbook.mit.edu/figures/transformers/multilayer_masked_attention.png)](<https://visionbook.mit.edu/figures/transformers/multilayer_masked_attention.png> "圖 26.19：多層遮罩注意力利用深層網路實現因果預測。")

圖 26.19：多層遮罩注意力利用深層網路實現因果預測。

請注意，每一層 $l$ 上的輸出 Token 都具有 $\mathbf{t}^l_n$ 僅依賴於 $\mathbf{T}^0_{1:n-1}$ 的性質，其中 $\mathbf{T}^0$ 是輸入至 Transformer 的初始 Token 集合。還請注意，在第一層之後，所有後續層都可以使用時間上沒有偏移的因果注意力，且上述性質仍然得以保持。最後，請注意預測每個後續輸出 Token 的子網路與預測每個前述 Token 的子網路有著實質上的重疊。也就是說，在所有預測問題之間存在著計算共享。當我們在[第 32.7 節](https://visionbook.mit.edu/generative_models.html#sec-generative_models-autoregressive)討論自迴歸（autoregressive）模型時，將會看到這種策略的更具體應用。

## 26.11 位置編碼

另一個與 Transformer 相關的想法是**位置編碼**（**Positional Encodings**）。Transformer 中對 Token 的操作是排列等變的，這意味著我們可以打亂 Token 的位置而不會發生實質性的變化（唯一的變化是輸出會被排列打亂）。其後果是，Token 無法得知它們在訊號表徵中的位置。然而，有時我們可能希望保留位置知識。例如，知道一個 Token 是影像頂部區域的表徵，可以幫助我們識別該 Token 很可能代表天空。位置編碼將一個代表*在訊號中位置*的編碼拼接到每個 Token 上。如果訊號是影像，那麼位置編碼應該代表 x 和 y 座標。然而，它不需要將這些座標表示為純量；更常用的是位置的週期性表徵，其中座標被編碼為一組正弦波（sinusoidal waves）在每個位置上取得的值向量：

$$
\begin{aligned}
    \mathbf{p}_x &= [\sin(x), \sin(x/B), \sin(x/B^2), \ldots, \sin(x/B^P)]^\mathsf{T}\\
    \mathbf{p}_y &= [\sin(y), \sin(y/B), \sin(y/B^2), \ldots, \sin(y/B^P)]^\mathsf{T}\\
    \mathbf{p} &= \begin{bmatrix}
        \mathbf{p}_x\\
        \mathbf{p}_y
    \end{bmatrix}
\end{aligned} \tag{26.12}
$$

其中 $x$ 和 $y$ 是該 Token 的座標。此表徵在圖 26.20 中進行了視覺化：

[![](https://visionbook.mit.edu/figures/transformers/positional_codes.png)](<https://visionbook.mit.edu/figures/transformers/positional_codes.png> "圖 26.20：位置編碼。")

圖 26.20：位置編碼。

另一種策略是簡單地讓位置編碼由模型*學習*（learned）得到，這空間表徵可能會比正弦波編碼更好 [[2](https://visionbook.mit.edu/references.html#ref-dosovitskiy2020vit)]。

雖然位置編碼在 Transformer 中很有用且很常見，但它並非此架構所獨有。同樣的位置編碼對於 CNN 也有用，可以作為一種使卷積濾波器以位置為條件（conditioned on position）的方法，從而在影像的每個位置應用不同的加權和 [[7](https://visionbook.mit.edu/references.html#ref-liu2018intriguing)]。位置編碼也出現在輻射場（radiance fields）中，我們將在[第 45 章](https://visionbook.mit.edu/nerf.html)中進行介紹。

[![](https://visionbook.mit.edu/figures/transformers/affine_layer_comparison.png)](<https://visionbook.mit.edu/figures/transformers/affine_layer_comparison.png> "圖 26.21：")

圖 26.21

## 26.12 比較全連接層、卷積層與自注意力層

簡寫「fc」通常用於表示全連接線性層（fully-connected linear layer）。

深度網路中的許多層都是特殊類型的仿射轉換（affine transformations）。我們目前看到的有三種：全連接層（`fc`）、卷積層（`conv`）和自注意力層（`attn`）。所有這些層的相似之處在於，它們的前向傳播都可以寫成 $\mathbf{X}_{\texttt{out}}= \mathbf{W}\mathbf{X}_{\texttt{in}}+ \mathbf{b}$ ，對應於某個矩陣 $\mathbf{W}$ 和某個向量 $\mathbf{b}$。在 `conv` 和 `attn` 層中，$\mathbf{W}$ 和 $\mathbf{b}$ 是作為輸入 $\mathbf{X}_{\texttt{in}}$ 的某個函數而確定的。在 `conv` 層中，這個函數非常簡單：只需建構一個 Toeplitz 矩陣，該矩陣重複卷積核以匹配 $\mathbf{X}_{\texttt{in}}$ 的維度。在 `attn` 層中，確定 $\mathbf{W}$ 的函數稍微複雜一些，正如我們在上面看到的，而且通常我們不使用偏差 $\mathbf{b}$。

這些層中的每一層都可以表示為一個矩陣，檢查這些矩陣中的結構是實踐理解它們相似之處與相異之處的一種有用方法。`fc` 層的矩陣是滿秩（full rank）的，而 `conv` 和 `attn` 層的矩陣具有低秩（low-rank）結構，但低秩結構的類型不同。在圖 26.21 中，我們展示了這些矩陣的樣子，並歸納了這三種層的其他一些重要屬性。

## 26.13 結語

截至撰寫本文時，Transformer 是電腦視覺以及事實上人工智慧的大多數領域中的主導架構。它們結合了早期架構的許多最佳概念——卷積逐區塊處理、殘差連接、ReLU 非線性和正規化層——以及幾項較新的創新，特別是向量值 Token、注意力層和位置編碼。Transformer 也可以被視為**圖神經網路**（**Graph Neural Networks**，**GNN**）的一個特例。我們在本書中沒有關於 GNN 的獨立章節，因為除 Transformer 之外的 GNN 在電腦視覺中尚未普及。GNN 是一類非常通用的架構，用於透過在集合上形成操作圖（graph of operations）來處理*集合*。Transformer 正是在做這件事：它接收一個輸入的 Token 集合，並逐層在該集合上應用轉換網絡，直到經過足夠的層之後，讀出最終的表徵或預測。

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need., in: Nips, 2017: pp. 5998–6008.

[2] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby, An image is worth 16x16 words: Transformers for image recognition at scale., Iclr (2021).

[3] J.M. Wolfe, Visual attention., Seeing (2000) 335–386.

[4] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, Masked autoencoders are scalable vision learners., in: Cvpr, 2022: pp. 15979–15988.

[5] X. Wei, T. Zhang, Y. Li, Y. Zhang, F. Wu, Multi-modality cross attention network for image and sentence matching., in: Cvpr, 2020: pp. 10941–10950.

[6] M. Chen, A. Radford, R. Child, J. Wu, H. Jun, D. Luan, I. Sutskever, Generative pretraining from pixels., in: Icml, 2020: pp. 1691–1703.

[7] R. Liu, J. Lehman, P. Molino, F.P. Such, E. Frank, A. Sergeev, J. Yosinski, An intriguing failing of convolutional neural networks and the coordconv solution., (2018). <https://arxiv.org/abs/1807.03247>.


