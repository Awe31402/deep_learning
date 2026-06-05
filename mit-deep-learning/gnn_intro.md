# 圖神經網路入門 (A Gentle Introduction to Graph Neural Networks)

> **描述：** 建立利用圖 (Graph) 的結構和屬性的學習演算法需要哪些組件？

**作者：** Benjamin Sanchez-Lengeling, Emily Reif, Adam Pearce, Alexander B. Wiltschko
**發表於：** 2021 年 9 月 2 日，*Distill*
**DOI：** [10.23915/distill.00033](https://doi.org/10.23915/distill.00033)
**原始文章：** [https://distill.pub/2021/gnn-intro/](https://distill.pub/2021/gnn-intro/)

---

*這篇文章是關於圖神經網路的兩篇 Distill 出版物之一。請閱讀[理解圖上的卷積](https://distill.pub/2021/understanding-gnns/)[^1]，以了解影像上的卷積如何自然地推廣到圖上的卷積。*

圖無處不在；現實世界中的物體通常是根據它們與其他事物的連接來定義的。一組物體以及它們之間的連接，自然地可以表示為一個**圖**。研究人員開發在圖資料上運行的神經網路（稱為圖神經網路，或 GNN）已經有十多年了[^2]。最近的發展提高了它們的能力和表達力。我們開始在抗菌藥物發現[^3]、物理模擬[^4]、假新聞檢測[^5]、交通預測[^6] 和推薦系統[^7] 等領域看到實際應用。

本文將探索並解釋現代圖神經網路。我們將這項工作分為四個部分。首先，我們看看哪種資料最自然地被表述為圖，以及一些常見的例子。其次，我們探索是什麼讓圖與其他類型的資料不同，以及我們在使用圖時必須做出的一些專門選擇。第三，我們構建一個現代的 GNN，逐步介紹模型的每個部分，從該領域歷史上的建模創新開始。我們逐漸從一個最基礎的實現過渡到最先進 GNN 模型。第四，也是最後，我們提供了一個 GNN 遊樂場，您可以在其中體驗真實世界的任務和資料集，以建立更強的直覺，了解 GNN 模型的每個組件如何對其做出的預測做出貢獻。

首先，讓我們建立什麼是圖。圖表示實體集合（**節點**，nodes）之間的關係（**邊**，edges）。

*[交互式可視化/組件：'graph-description' - 在原始文章中查看]*

我們在圖中可能找到的三種類型的屬性，懸停以高亮顯示每個屬性。其他類型的圖和屬性將在「其他類型的圖」部分中探討。

為了進一步描述每個節點、邊或整個圖，我們可以在圖的每個部分中存儲資訊。

*[交互式可視化/組件：'graph-description-embeddings' - 在原始文章中查看]*

標量或嵌入 (embeddings) 形式的資訊可以存儲在每個圖節點（左）或邊（右）上。

我們還可以通過將方向性與邊關聯來使圖專門化（**有向圖、無向圖**）。

![](https://distill.pub/2021/gnn-intro/directed_undirected.e4b1689d.png) 邊可以是有向的，其中邊 $e$ 具有源節點 $v_{src}$ 和目標節點 $v_{dst}$。在這種情況下，資訊從 $v_{src}$ 流向 $v_{dst}$。它們也可以是無向的，其中沒有源節點或目標節點的概念，資訊雙向流動。請注意，具有單個無向邊相當於具有從 $v_{src}$ 到 $v_{dst}$ 的一個有向邊，以及從 $v_{dst}$ 到 $v_{src}$ 的另一個有向邊。

圖是非常靈活的資料結構，如果現在這看起來很抽象，我們將在下一節中用具體例子使其具體化。

## 圖以及在哪裡可以找到它們

您可能已經熟悉某些類型的圖資料，例如社交網絡。然而，圖是一種非常強大且通用的資料表示方式，我們將展示兩種您可能認為無法建模為圖的資料類型：影像和文字。雖然這很違反直覺，但通過將影像和文字視為圖，可以了解更多關於它們的對稱性和結構，並建立直覺，這將有助於理解我們稍後將討論的其他較不規則的圖資料。

### 作為圖的影像

我們通常將影像視為具有影像通道的矩形網格，將它們表示為數組（例如，244x244x3 個浮點數）。另一種思考影像的方法是將其視為具有規則結構的圖，其中每個像素代表一個節點，並通過一條邊連接到相鄰像素。每個非邊界像素正好有 8 個鄰居，存儲在每個節點上的資訊是一個 3 維向量，代表像素的 RGB 值。

可視化圖連接性的一種方法是通過其**鄰接矩陣** (adjacency matrix)。我們對節點進行排序（在這種情況下，一個簡單的 5x5 笑容影像中的 25 個像素中的每一個），如果兩個節點共享一條邊，則在 $n_{nodes} \times n_{nodes}$ 的矩陣中填入一個項目。請注意，以下這三種表示形式中的每一種都是同一資料片段的不同視角。

單擊影像像素以切換其值，並查看圖表示如何變化。

### 作為圖的文字

我們可以通過將索引與每個字元、單詞或標記 (token) 關聯來數字化文字，並將文字表示為這些索引的序列。這創建了一個簡單的有向圖，其中每個字元或索引是一個節點，並通過一條邊連接到緊隨其後的節點。

編輯上面的文字以查看圖表示如何變化。

當然，在實踐中，這通常不是文字和影像的編碼方式：這些圖表示是多餘的，因為所有影像和所有文字都將具有非常規則的結構。例如，影像在其鄰接矩陣中具有帶狀結構，因為所有節點（像素）都連接在一個網格中。文字的鄰接矩陣只是一條對角線，因為每個單詞只連接到前一個單詞和下一個單詞。

此表示（字元標記序列）指的是文字在 RNN 中通常的表示方式；其他模型（例如 Transformer）可以被認為是將文字視為一個完全連接的圖，我們在其中學習標記之間的關係。在「圖注意力網絡」中查看更多資訊。

### 現實世界中的圖值資料

圖是描述您可能已經熟悉的資料的有用工具。讓我們轉向結構更異質的資料。在這些例子中，每個節點的鄰居數量是可變的（與影像和文字的固定鄰域大小相反）。除了圖之外，很難用任何其他方式來表述這些資料。

**作為圖的分子。** 分子是物質的基石，由 3D 空間中的原子和電子組成。所有粒子都在相互作用，但當一對原子固定在彼此穩定的距離時，我們說它們共享一個共價鍵。不同的原子對和鍵具有不同的距離（例如單鍵、雙鍵）。將這個 3D 對象描述為一個圖是非常方便且常見的抽象，其中節點是原子，邊是共價鍵。[^8] 這是兩個常見的分子及其關聯的圖。

（左）香茅醛 (Citronellal) 分子的 3D 表示（中）分子中化學鍵的鄰接矩陣（右）分子的圖表示。

（左）咖啡因 (Caffeine) 分子的 3D 表示（中）分子中化學鍵的鄰接矩陣（右）分子的圖表示。

**作為圖的社交網絡。** 社交網絡是研究人群、機構和組織集體行為模式的工具。我們可以通過將個人建模為節點，將他們的關係建模為邊，來構建一個代表人群的圖。

（左）戲劇《奧賽羅》中的場景影像。（中）劇中人物之間互動的鄰接矩陣。（右）這些互動的圖表示。

與影像和文字資料不同，社交網絡沒有相同的鄰接矩陣。

（左）空手道錦標賽的影像。（中）空手道俱樂部成員之間互動的鄰接矩陣。（右）這些互動的圖表示。

**作為圖的引用網絡。** 科學家在發表論文時通常會引用其他科學家的工作。我們可以將這些引用網絡可視化為一個圖，其中每篇論文是一個節點，每個**有向**邊是兩篇論文之間的引用關係。此外，我們可以將每篇論文的資訊添加到每個節點中，例如摘要的字詞嵌入。 （參見 [^9], [^10] , [^11]）。

**其他例子。** 在電腦視覺中，我們有時想要標記視覺場景中的物體。然後，我們可以通過將這些物體視為節點，將它們的關係視為邊來構建圖。[機器學習模型](https://www.tensorflow.org/tensorboard/graphs)、[程式碼](https://openreview.net/pdf?id=BJOFETxR-) [^12] 和 [數學方程式](https://openreview.net/forum?id=S1eZYeHFDS)[^13] 也可以表述為圖，其中變數是節點，邊是以這些變數作為輸入和輸出的運算。您可能會在其中一些語境中看到使用「資料流圖」(dataflow graph) 一詞。

現實世界圖的結構在不同類型的資料之間可能會有很大差異——有些圖有許多節點，但它們之間的連接很少，反之亦然。就節點、邊的數量以及節點的連接性而言，圖資料集可能存在很大差異（不論是在給定資料集內，還是在資料集之間）。

現實世界中發現的圖的摘要統計資訊。數字取決於特徵化決策。更多有用的統計資訊 and 圖可以在 KONECT[^14] 中找到。

## 哪些類型的問題具有圖結構資料？

我們已經描述了現實世界中圖的一些例子，但我們想在這些資料上執行什麼任務？圖上有三種通用的預測任務：圖級別、節點級別和邊級別。

在圖級別任務中，我們預測整個圖的單個屬性。對於節點級別任務，我們預測圖中每個節點的某些屬性。對於邊級別任務，我們想要預測圖中邊的屬性或存在與否。

對於上述三個級別的預測問題（圖級別、節點級別和邊級別），我們將展示所有這些問題都可以用單個模型類別（GNN）來解決。但首先，讓我們更詳細地參觀這三類圖預測問題，並提供每個類別的具體例子。

還有其他相關的任務是目前活躍的研究領域。例如，我們可能想要生成圖，或者解釋圖上的預測。更多主題可以在「深入探討」部分找到。

### 圖級別任務

在圖級別任務中，我們的目標是預測整個圖的屬性。例如，對於表示為圖的分子，我們可能想要預測該分子的氣味，或者它是否會與涉及某種疾病的受體結合。

這類似於使用 MNIST 和 CIFAR 的影像分類問題，我們希望將標籤與整個影像關聯。對於文字，一個類似的問題是情感分析，我們希望一次識別整個句子的情緒或情感。

### 節點級別任務

節點級別任務關注預測圖中每個節點的身份或角色。

節點級別預測問題的一個經典例子是 Zach 的空手道俱樂部。[^15] 該資料集是一個單一的社交網絡圖，由在政治分歧後宣誓效忠兩個空手道俱樂部之一的個人組成。故事是這樣的，Hi 先生（教練）和 John H（管理員）之間的恩怨導致空手道俱樂部分裂。節點代表個人空手道練習者，邊代表這些成員在空手道之外的互動。預測問題是分類在恩怨之後，給定成員會忠於 Hi 先生還是 John H。在這種情況下，節點到教練或管理員的距離與該標籤高度相關。

在左側，我們有問題的初始條件，在右側，我們有一個可能的解決方案，其中每個節點都根據聯盟進行了分類。該資料集可用於其他圖問題，如無監督學習。

沿用影像的類比，節點級別的預測問題類似於**影像分割**，我們試圖標記影像中每個像素的角色。對於文字，類似的任務是預測句子中每個單詞的詞性（例如名詞、動詞、副詞等）。

### 邊級別任務

圖中剩餘的預測問題是**邊預測**。

邊級別推理的一個例子是影像場景理解。除了識別影像中的物體之外，深度學習模型還可以用於預測它們之間的關係。我們可以將其表述為邊級別的分類：給定代表影像中物體的節點，我們希望預測這些節點中哪些共享一條邊，或者該邊的值是多少。如果我們希望發現實體之間的連接，我們可以考慮將圖完全連接，並根據它們的預測值修剪邊以得到一個稀疏圖。

![](https://distill.pub/2021/gnn-intro/merged.0084f617.png) 在上面的 (b) 中，原始影像 (a) 已被分割為五個實體：每個戰鬥人員、裁判、觀眾和墊子。(c) 顯示了這些實體之間的關係。![](https://distill.pub/2021/gnn-intro/edges_level_diagram.c40677db.png) 在左側，我們有一個根據先前的視覺場景構建的初始圖。右側是當根據模型的輸出修剪某些連接時，該圖可能的邊標記結果。

## 在機器學習中使用圖的挑戰

那麼，我們該如何使用神經網路來解決這些不同的圖任務呢？第一步是思考我們將如何表示圖，使其與神經網路兼容。

機器學習模型通常接受矩形或網格狀數組作為輸入。因此，如何將它們表示為與深度學習兼容的格式並不是很直觀。圖有多達四種類型的資訊，我們可能希望使用這些資訊來做出預測：節點、邊、全局上下文和連接性。前三者相對簡單：例如，對於節點，我們可以通過為每個節點分配一個索引 $i$ 並將 $node_i$ 的特徵存儲在 $N$ 中來形成一個節點特徵矩陣 $N$。雖然這些矩陣具有可變數量的樣本，但它們可以在不需要任何特殊技術的情況下進行處理。

然而，表示圖的連接性更加複雜。也許最顯著的選擇是使用鄰接矩陣，因為這很容易張量化。但是，這種表示有一些缺點。從示例資料集表中，我們看到圖中的節點數量可能在百萬量級，並且每個節點的邊數可能高度可變。通常，這會導致非常稀疏的鄰接矩陣，這在空間上是低效的。

另一個問題是，有許多鄰接矩陣可以編碼相同的連接性，並且不能保證這些不同的矩陣在深度神經網路中會產生相同的結果（也就是說，它們不是置換不變的，permutation invariant）。

學習置換不變的操作是最近研究的一個領域。[^16][^17]

例如，之前的奧賽羅圖可以用這兩個鄰接矩陣等價地描述。它也可以用節點的每個其他可能置換來描述。

![](https://distill.pub/2021/gnn-intro/othello1.246371ea.png) ![](https://distill.pub/2021/gnn-intro/othello2.6897c848.png)

兩個代表相同圖的鄰接矩陣。

下面的示例顯示了可以描述這個只有 4 個節點的小圖的每個鄰接矩陣。這已經是相當數量的鄰接矩陣了——對於像奧賽羅這樣更大的例子，這個數量是無法承受的。

所有這些鄰接矩陣都代表同一個圖。單擊一條邊以在「虛擬邊」上將其刪除，單擊以將其添加，矩陣將相應更新。

一種優雅且記憶體高效的表示稀疏矩陣的方法是作為**鄰接表** (adjacency lists)。這些將節點 $n_i$ 和 $n_j$ 之間邊 $e_k$ 的連接性描述為鄰接表第 k 個條目中的元組 (i,j)。由於我們預期邊的數量遠低於鄰接矩陣的條目數 ($n_{nodes}^2$)，我們避免了在圖的未連接部分上進行計算和存儲。

另一種用大 O 符號表示的說法是，相較於 $O(n_{nodes}^2)$，更傾向於使用 $O(n_{edges})$。

為了使這個概念更具體，我們可以看到在該規範下如何表示不同圖中的資訊：

懸停並單擊邊、節點和全局圖標記以查看和更改屬性表示。一側是小圖，另一側是張量表示中的圖資訊。

應該注意的是，該圖對每個節點/邊/全局使用標量值，但大多數實際的張量表示對每個圖屬性都有向量。我們將處理大小為 $[n_{nodes}, node_{dim}]$ 的節點張量，而不是大小為 $[n_{nodes}]$ 的節點張量。其他圖屬性也是如此。


## 圖神經網路

現在圖的描述已經是置換不變的矩陣格式了，我們將介紹使用圖神經網路 (GNN) 來解決圖預測任務。**GNN 是對圖的所有屬性（節點、邊、全局上下文）進行的可優化轉換，該轉換保留了圖的對稱性（置換不變性）。** 我們將使用 Gilmer 等人[^18] 提出的「訊息傳遞神經網路」 (message passing neural network) 框架來構建 GNN，並採用 Battaglia 等人[^19] 引入的 Graph Nets 架構圖解。GNN 採用「圖輸入，圖輸出」 (graph-in, graph-out) 的架構，這意味著這些模型類型接受一個圖作為輸入，將資訊加載到其節點、邊和全局上下文中，並逐步轉換這些嵌入，而不改變輸入圖的連接性。

### 最簡單的 GNN

利用我們在上面構建的圖的數值表示（使用向量而不是標量），我們現在準備構建一個 GNN。我們將從最簡單的 GNN 架構開始，在該架構中，我們為所有圖屬性（節點、邊、全局）學習新的嵌入，但我們尚未利用圖的連接性。

為簡單起見，先前的圖表使用標量來表示圖屬性；在實踐中，特徵向量或嵌入要有用得多。

這個 GNN 在圖的每個組件上使用一個獨立的多層感知器 (MLP)（或您喜歡的任何可微模型）；我們將其稱為 GNN 層 (GNN layer)。對於每個節點向量，我們應用 MLP 並得到一個學習到的節點向量。我們對每條邊執行相同的操作，學習每個邊的嵌入，並且也對全局上下文向量執行相同的操作，為整個圖學習單個嵌入。

您也可以將其稱為 GNN 塊 (GNN block)。因為它包含多個操作/層（類似於 ResNet 塊）。![](https://distill.pub/2021/gnn-intro/arch_independent.0efb8ae7.png) 簡單 GNN 的單個層。輸入是一個圖，每個組件 (V,E,U) 都被 MLP 更新以產生一個新圖。每個函數下標表示在 GNN 模型的第 n 層中，針對不同圖屬性的獨立函數。

正如神經網路模組或層中常見的那樣，我們可以將這些 GNN 層堆疊在一起。

因為 GNN 不會更新輸入圖的連接性，所以我們可以用與輸入圖相同的鄰接表和特徵向量數量來描述 GNN 的輸出圖。但是，輸出圖具有更新後的嵌入，因為 GNN 已經更新了每個節點、邊 and 全局上下文的表示。

### 通過池化資訊進行 GNN 預測

我們已經構建了一個簡單的 GNN，但我們該如何在上述任何任務中做出預測呢？

我們將以二元分類為例，但這個框架可以很容易地擴展到多分類或迴歸的情況。如果任務是對節點進行二元預測，且圖中已包含節點資訊，則方法非常簡單——對每個節點嵌入應用一個線性分類器。

![](https://distill.pub/2021/gnn-intro/prediction_nodes_nodes.c2c8b4d0.png) 我們可以想像一個社交網絡，我們希望通過不使用用戶資料（節點），而僅使用關係資料（邊）來匿名化用戶資料。這種情況的一個例子是我們在「節點級別任務」子部分中指定的節點任務。在空手道俱樂部的例子中，這將僅使用人與人之間的會議次數來確定與 Hi 先生或 John H 的聯盟關係。

然而，事情並不總是那麼簡單。例如，您可能在邊中存儲了圖中的資訊，但節點中沒有資訊，但仍需要對節點進行預測。我們需要一種方法來從邊收集資訊並將其提供給節點進行預測。我們可以通過**池化** (pooling) 來做到這一點。池化分為兩個步驟：

  1. 對於每個要池化的項目，**收集** (gather) 它們的每個嵌入並將它們拼接成一個矩陣。

  2. 收集到的嵌入隨後被**聚合** (aggregate)，通常通過求和 (sum) 操作。

有關聚合操作的更深入討論，請轉到「比較聚合操作」部分。

我們用字母 $\rho$ 表示**池化**操作，並將從邊到節點收集資訊的過程表示為 $\rho_{E_n \to V_{n}}$。

懸停在一個節點（黑色節點）上，以可視化收集並聚合了哪些邊，從而為該目標節點產生嵌入。

因此，如果我們只有邊級別的特徵，並試圖預測二元節點資訊，我們可以使用池化將資訊路由（或傳遞）到它需要去的地方。模型看起來像這樣。

![](https://distill.pub/2021/gnn-intro/prediction_edges_nodes.e6796b8e.png)

如果我們只有節點級別的特徵，並試圖預測二元邊級別的資訊，模型看起來像這樣。

![](https://distill.pub/2021/gnn-intro/prediction_nodes_edges.26fadbcc.png) 這種情況的一個例子是我們在「邊級別任務」子部分中指定的邊任務。節點可以被識別為影像實體，我們正試圖預測這些實體是否共享關係（二元邊）。

如果我們只有節點級別的特徵，並且需要預測二元全局屬性，我們需要將所有可用的節點資訊收集在一起並進行聚合。這類似於 CNN 中的**全局平均池化** (Global Average Pooling) 層。對於邊也可以做同樣的事情。

![](https://distill.pub/2021/gnn-intro/prediction_nodes_edges_global.7a535eb8.png) 這是預測分子屬性的常見場景。例如，我們有原子資訊、連接性，並且我們想知道分子的毒性（有毒/無毒），或者它是否具有特定的氣味（玫瑰香/非玫瑰香）。

在我們的例子中，分類模型 $c$ 可以很容易地替換為任何可微模型，或使用廣義線性模型適應多分類任務。

![](https://distill.pub/2021/gnn-intro/Overall.e3af58ab.png) 使用 GNN 模型的端到端預測任務。

現在我們已經證明了我們可以構建一個簡單的 GNN 模型，並通過在圖的不同部分之間路由資訊來做出二元預測。這種池化技術將作為構建更複雜 GNN 模型的基石。如果我們有新的圖屬性，我們只需要定義如何將資訊從一個屬性傳遞到另一個屬性。

請注意，在這個最簡單的 GNN 公式中，我們根本沒有在 GNN 層內部使用圖的連接性。每個節點都是獨立處理的，每條邊以及全局上下文也是如此。我們僅在池化資訊以進行預測時才使用連接性。

### 在圖的不同部分之間傳遞訊息

我們可以通過在 GNN 層內使用池化來做出更複雜的預測，從而使我們學習到的嵌入意識到圖的連接性。我們可以使用**訊息傳遞** (message passing)[^18] 來做到這一點，其中相鄰的節點或邊交換資訊並相互影響其更新後的嵌入。

訊息傳遞分為三個步驟：

  1. 對於圖中的每個節點，**收集** (gather) 所有相鄰節點的嵌入（或訊息），這就是上面描述的 $g$ 函數。

  2. 通過聚合函數（如求和）**聚合** (aggregate) 所有訊息。

  3. 所有池化後的訊息都通過一個**更新函數** (update function)，通常是一個學習到的神經網路。

您也可以 1) 收集訊息，3) 更新它們，然後 2) 聚合它們，這仍然是一個置換不變的操作。[^20]

正如池化可以應用於節點或邊一樣，訊息傳遞也可以在節點或邊之間發生。

這些步驟是利用圖連接性的關鍵。我們將在 GNN 層中構建更複雜的訊息傳遞變體，從而產生具有更高表達力和能力強大的 GNN 模型。

懸停在一個節點上，以高亮顯示相鄰節點，並可視化將被池化、更新和存儲的相鄰嵌入。

當應用一次時，這個操作序列就是最簡單類型的訊息傳遞 GNN 層。

這讓人聯想到標準卷積：在本質上，訊息傳遞和卷積都是聚合和處理元素鄰居的資訊以更新該元素值的操作。在圖中，元素是節點，在影像中，元素是像素。然而，圖中相鄰節點的數量是可變的，這與影像中每個像素具有固定數量的相鄰元素不同。

通過將訊息傳遞 GNN 層堆疊在一起，節點最終可以整合來自整個圖的資訊：在三層之後，一個節點就擁有了關於與其相距三步的節點的資訊。

我們可以更新我們的架構圖，以包含節點的這一新資訊源：

![](https://distill.pub/2021/gnn-intro/arch_gcn.40871750.png) GCN 架構的示意圖，該架構通過池化距離為一度的相鄰節點來更新圖的節點表示。

### 學習邊表示

我們的資料集並不總是包含所有類型的資訊（節點、邊和全局上下文）。當我們想要對節點進行預測，但我們的資料集只有邊資訊時，我們在上面展示了如何在模型的最終預測步驟中使用池化將資訊從邊路由到節點。我們可以使用訊息傳遞在 GNN 層內部的節點和邊之間共享資訊。

我們可以使用與之前使用相鄰節點資訊相同的方式，將來自相鄰邊的資訊整合進來：首先池化邊資訊，使用更新函數對其進行轉換，然後將其存儲。

然而，圖中存儲的節點和邊資訊不一定具有相同的大小或形狀，因此如何將它們結合起來並不是很直觀。一種方法是學習從邊空間到節點空間的線性映射，反之亦然。或者，也可以在更新函數之前將它們拼接在一起。

![](https://distill.pub/2021/gnn-intro/arch_mpnn.a13c2294.png) 訊息傳遞層 (Message Passing layer) 的架構示意圖。第一步「準備」一條由邊及其相連節點的資訊組成的訊息，然後將該訊息「傳遞」給節點。

在構建 GNN 時，我們更新哪些圖屬性以及以何種順序更新它們是一個設計決策。我們可以選擇是在邊嵌入之前更新節點嵌入，還是相反。這是一個開放的研究領域，有各種各樣的解決方案——例如，我們可以以「編織」 (weave) 方式進行更新[^21]，在這種方式中，我們有四個更新後的表示，它們被結合到新的節點 and 邊表示中：節點到節點（線性）、邊到邊（線性）、節點到邊（邊層）、邊到節點（節點層）。

![](https://distill.pub/2021/gnn-intro/arch_weave.352befc0.png) 我們可能在 GNN 層中結合邊和節點表示的一些不同方式。

### 加入全局表示

我們到目前為止所描述的網絡存在一個缺陷：圖中相距很遠的節點可能永遠無法高效地相互傳輸資訊，即使我們應用了多次訊息傳遞。對於一個節點，如果我們有 k 層，資訊最多只能傳播到相距 k 步的地方。對於預測任務取決於相距很遠的節點或節點群組的情況，這可能會是一個問題。一種解決方案是讓所有節點都能相互傳遞資訊。不幸的是，對於大型圖，這很快就會在計算上變得非常昂貴（儘管這種稱為「虛擬邊」 (virtual edges) 的方法已被用於小分子等小型圖）。[^18]

此問題的一種解決方案是使用圖的全局表示 (U)，有時也稱為**主節點** (master node)[^19][^18] 或上下文向量。這個全局上下文向量連接到網絡中的所有其他節點和邊，並可以作為它們之間的橋樑來傳遞資訊，從而為整個圖建立表示。這創造了比原本可以學習到的更豐富、更複雜的圖表示。

![](https://distill.pub/2021/gnn-intro/arch_graphnet.b229be6d.png) 利用全局表示的 Graph Nets 架構示意圖。

在這種視角下，所有圖屬性都具有學習到的表示，因此我們可以在池化過程中利用它們，方法是將我們感興趣的屬性資訊與其餘屬性進行條件化 (conditioning)。例如，對於一個節點，我們可以考慮來自相鄰節點、相連邊和全局資訊的資訊。為了使新的節點嵌入基於所有這些可能的資訊源，我們可以直接將它們拼接在一起。此外，我們也可以通過線性映射將它們映射到相同的空間並將它們相加，或者應用一個特徵級別的調製層 (feature-wise modulation layer)[^22]，這可以被認為是一種特徵級別的注意力機制。

![](https://distill.pub/2021/gnn-intro/graph_conditioning.3017e214.png) 根據其他三個嵌入（相鄰節點、相鄰邊、全局）對一個節點的資訊進行條件化的示意圖。此步驟對應於 Graph Nets 層中的節點運算。


## GNN 遊樂場

我們在此處描述了廣泛的 GNN 組件，但它們在實踐中到底有何不同？這個 GNN 遊樂場允許您查看這些不同的組件和架構如何促成 GNN 學習真實任務的能力。

我們的遊樂場展示了使用小分子圖的圖級別預測任務。我們使用 Leffingwell 氣味資料集[^23][^24]，該資料集由分子和相關的氣味感知（標籤）組成。預測分子結構（圖）與其氣味的關係是一個跨越化學、物理、神經科學和機器學習的百年難題。

為了簡化問題，我們對每個分子僅考慮單個二元標籤，分類分子圖是否聞起來有「刺激性」 (pungent)，這由專業調香師標記。如果一個分子具有強烈、醒目的味道，我們就說它具有「刺激性」氣味。例如，可能含有「烯丙醇」 (allyl alcohol) 分子的蒜和芥末就具有這種特質。常用於薄荷味糖果的「胡椒酮」 (piperitone) 分子也被描述為具有刺激性氣味。

我們將每個分子表示為一個圖，其中原子是節點，包含其原子身份（碳、氮、氧、氟）的單熱編碼 (one-hot encoding)；化學鍵是邊，包含其鍵類型（單鍵、雙鍵、三鍵或芳香鍵）的單熱編碼。

我們針對此問題的通用建模模板將使用順序 GNN 層構建，隨後是用於分類的帶有 Sigmoid 激活函數的線性模型。我們的 GNN 設計空間有許多可以自定義模型的控制桿：

  1. GNN 層的數量，也稱為**深度** (depth)。

  2. 每個屬性在更新時的維度。更新函數是一個 1 層 MLP，帶有 ReLU 激活函數和用於激活值歸一化的 Layer Norm。

  3. 池化中使用的聚合函數：最大值 (max)、平均值 (mean) 或求和 (sum)。

  4. 更新的圖屬性或訊息傳遞的樣式：節點、邊和全局表示。我們通過布林切換（開啟或關閉）來控制這些。基準模型將是獨立於圖的 GNN（所有訊息傳遞關閉），它在最後將所有資料聚合到單個全局屬性中。開啟所有訊息傳遞函數將產生 GraphNets 架構。

為了更好地理解 GNN 如何學習任務優化的圖表示，我們還查看了 GNN 的倒數第二層激活值。這些「圖嵌入」 (graph embeddings) 是 GNN 模型在預測之前的輸出。由於我們使用廣義線性模型進行預測，因此線性映射足以讓我們看到我們是如何在決策邊界周圍學習表示的。

由於這些是高維向量，我們通過主成分分析 (PCA) 將它們降至 2D。一個完美的模型將能肉眼分離標記資料，但由於我們降低了維度且模型也不完美，這個邊界可能較難看出。

嘗試不同的模型架構以建立您的直覺。例如，看看您是否可以編輯左側的分子以使模型預測值增加。相同的編輯對不同的模型架構有相同的效果嗎？

這個遊樂場在瀏覽器中通過 [tfjs](https://www.tensorflow.org/js/) 即時運行。

編輯分子以查看預測如何變化，或更改模型參數以加載不同的模型。在散點圖中選擇不同的分子。

### 一些經驗性的 GNN 設計教訓

在探索上述架構選擇時，您可能已經發現某些模型的性能優於其他模型。是否存在一些明顯的 GNN 設計選擇可以給我們帶來更好的性能？例如，更深的 GNN 模型是否比更淺的模型表現更好？或者聚合函數之間是否有明確的選擇？答案將取決於資料[^25][^26]，甚至特徵化和構建圖的不同方式都可能給出不同的答案。

通過以下交互式圖表，我們探索了 GNN 架構的空間，以及該任務在幾個主要設計選擇上的性能：訊息傳遞樣式、嵌入維度、層數和聚合運算類型。

散點圖中的每個點代表一個模型：x 軸是可訓練變數的數量，y 軸是性能。懸停在一個點上以查看 GNN 架構參數。

*[交互式可視化/組件：'BasicArchitectures' - 在原始文章中查看]*

每個模型性能與其可訓練變數數量的散點圖。懸停在一個點上以查看 GNN 架構參數。

首先要注意的是，令人驚訝的是，較高的參數數量確實與較高的性能相關。GNN 是一種非常參數高效的模型類型：即使參數數量很少（3k），我們也可以找到具有高性能的模型。

接下來，我們可以查看基於不同圖屬性的學習表示維度聚合的性能分佈。

*[交互式可視化/組件：'ArchitectureNDim' - 在原始文章中查看]*

不同節點、邊和全局維度下模型的聚合性能。

我們可以注意到，具有更高維度的模型往往具有更好的平均和下限性能，但在最大值方面卻沒有發現相同的趨勢。一些性能最佳的模型可以在較小維度中找到。由於較高維度也將涉及較多的參數，這些觀察結果與前一張圖相呼應。

接下來我們可以看到基於 GNN 層數的性能細分。

*[交互式可視化/組件：'ArchitectureNLayers' - 在原始文章中查看]*

層數與模型性能的圖表，以及模型性能與參數數量的散點圖。每個點按層數著色。懸停在一個點上以查看 GNN 架構參數。

箱形圖顯示了類似的趨勢，雖然平均性能往往隨層數增加而增加，但表現最好的模型不是具有三層或四層，而是兩層。此外，四層的性能下限有所下降。這種效應之前已被觀察到，具有更多層數的 GNN 將在更遠的距離傳播資訊，並可能面臨其節點表示因連續多次迭代而被「稀釋」 (diluted) 的風險[^27]。

我們的資料集是否有偏好的聚合運算？我們的下一張圖表按聚合類型細分了性能。

*[交互式可視化/組件：'ArchitectureAggregation' - 在原始文章中查看]*

聚合類型與模型性能的圖表，以及模型性能與參數數量的散點圖。每個點按聚合類型著色。懸停在一個點上以查看 GNN 架構參數。

總體而言，求和 (sum) 對平均性能似乎有非常微小的提升，但最大值 (max) 或平均值 (mean) 可以給出同樣好的模型。這在研究聚合運算的辨別/表達能力時有助於提供背景資訊。

先前的探索給出了好壞參半的資訊。我們可以找到複雜度越高、性能越好的平均趨勢，但我們也可以找到清晰的反例，即參數較少、層數較少或維度較低的模型表現更好。一個更清晰的趨勢是關於相互傳遞資訊的屬性數量。

在這裡，我們根據訊息傳遞的樣式來細分性能。在這兩個極端上，我們考慮圖實體之間不進行通信的模型（「無」）以及在節點、邊和全局之間傳遞訊息的模型。

*[交互式可視化/組件：'ArchitectureMessagePassing' - 在原始文章中查看]*

訊息傳遞與模型性能的圖表，以及模型性能與參數數量的散點圖。每個點按訊息傳遞著色。懸停在一個點上以查看 GNN 架構參數。

總體而言，我們看到通信的圖屬性越多，平均模型的性能就越好。我們的任務以全局表示為中心，因此顯式學習該屬性也往往會提高性能。我們的節點表示似乎也比邊表示更有用，這是有道理的，因為這些屬性中加載了更多資訊。

從這裡開始，您可以朝許多方向前進以獲得更好的性能。我們希望強調兩個大方向，一個與更複雜的圖演算法有關，另一個則指向圖本身。

到目前為止，我們的 GNN 是基於鄰域的池化操作。有一些圖概念很難以這種方式表達，例如線性圖路徑（連接的節點鏈）。在 GNN 中設計可以提取、執行和傳播圖資訊的新機制是當前的一個研究領域[^28], [^29], [^30], [^31]。

GNN 研究的前沿之一不是製造新的模型和架構，而是「如何構建圖」——更準確地說，是為圖賦予可以利用的額外結構或關係。正如我們粗略看到的，通信的圖屬性越多，我們往往能獲得更好的模型。在這個特定情況下，我們可以考慮通過增加節點之間額外的空間關係、增加非化學鍵的邊，或者子圖之間顯式可學習的關係，來使分子圖的特徵更加豐富。

在「其他類型的圖」中查看更多資訊。


## 深入探討

接下來，我們有幾個章節討論與 GNN 相關的無數圖相關主題。

### 其他類型的圖（多重圖、超圖、超節點、分層圖）

雖然我們只描述了每個屬性具有向量化資訊的圖，但圖結構更加靈活，可以容納其他類型的資訊。幸運的是，訊息傳遞框架足夠靈活，通常使 GNN 適應更複雜的圖結構就是定義資訊如何被新的圖屬性傳遞和更新。

例如，我們可以考慮多邊圖或**多重圖** (multigraphs)[^32]，其中一對節點可以共享多種類型的邊，這在我們想要根據節點的類型不同地建模節點之間的相互作用時發生。例如，在社交網絡中，我們可以根據關係類型（熟人、朋友、家人）指定邊類型。GNN 可以通過為每種邊類型設置不同類型的訊息傳遞步驟來進行適應。我們還可以考慮嵌套圖 (nested graphs)，例如，一個節點代表一個圖，也稱為超節點圖 (hypernode graph)。[^33] 嵌套圖對於表示分層資訊非常有用。例如，我們可以考慮一個分子網絡，其中一個節點代表一個分子，如果我們有方法（反應）將一個分子轉化為另一個分子，則在兩個分子之間共享一條邊[^34][^35]。在這種情況下，我們可以通過讓一個 GNN 學習分子級別的表示，另一個學習反應網絡級別的表示，並在訓練期間交替使用它們，來在嵌套圖上進行學習。

另一種圖是**超圖** (hypergraph)[^36]，其中一條邊可以連接到多個節點，而不僅僅是兩個。對於給定的圖，我們可以通過識別節點社群並分配連接到該社群中所有節點的超邊 (hyper-edge) 來構建超圖。

![](https://distill.pub/2021/gnn-intro/multigraphs.1bb84306.png) 更複雜圖的示意圖。左側是具有三種邊類型（包括一條有向邊）的多重圖示例。右側是三級分層圖，中間級別的節點是超節點。

如何訓練和設計具有多種類型圖屬性的 GNN 是當前的一個研究領域[^37], [^38]。

### GNN 中的圖採樣與批次處理

訓練神經網路的常見做法是使用在隨機、恆定大小（批次大小）的訓練資料子集（迷你批次，mini-batches）上計算的梯度來更新網路參數。由於相鄰節點和邊數量的多樣性，這對圖提出了挑戰，意味著我們無法擁有恆定的批次大小。圖的批次處理的核心思想是創建保留較大圖之基本屬性的子圖。這種圖採樣操作高度依賴於上下文，涉及從圖中子選擇節點和邊。這些操作在某些語境中（引用網絡）可能是有意義的，而在其他語境中，這些操作可能太過強烈（例如分子，子圖僅代表一個新的、更小的分子）。如何對圖進行採樣是一個開放的研究問題。[^39] 如果我們關心保留鄰域級別的結構，一種方法是隨機採樣均勻數量的節點，即我們的**節點集**。然後添加與該節點集相鄰且距離為 k 的相鄰節點，包括它們的邊。[^40] 每個鄰域都可以被認為是一個單獨的圖，並且可以在這些子圖的批次上訓練 GNN。損失可以被屏蔽以僅考慮節點集，因為所有相鄰節點都將具有不完整的鄰域。一種更有效的策略可能是首先隨機採樣單個節點，將其鄰域擴展到距離 k，然後在擴展集中選擇另一個節點。一旦構建了特定數量的節點、邊或子圖，就可以終止這些操作。如果上下文允許，我們可以通過挑選初始節點集，然後子採樣恆定數量的節點（例如隨機，或通過隨機遊走或 Metropolis 演算法[^41]）來構建恆定大小的鄰域。

![](https://distill.pub/2021/gnn-intro/sampling.968003b3.png) 對同一個圖進行採樣的四種不同方式。採樣策略的選擇高度依賴於上下文，因為它們將產生不同的圖統計分佈（節點數、邊數等）。對於高度連接的圖，邊也可以被子採樣。

當圖足夠大以至於無法放入記憶體中時，對圖進行採樣就顯得尤為重要。這啟發了新的架構和訓練策略，例如 Cluster-GCN [^42] 和 GraphSaint [^43]。我們預期圖資料集在未來會持續增長。

### 歸納偏置

當構建模型來解決特定類型資料上的問題時，我們希望使我們的模型專門化，以利用該資料的特徵。當成功做到這一點時，我們通常會看到更好的預測性能、更短的訓練時間、更少的參數以及更好的泛化能力。

例如，在影像上進行標記時，我們希望利用以下事實：無論狗是在影像的左上角還是右下角，它仍然是一隻狗。因此，大多數影像模型使用具有平移不變性的卷積。對於文字，標記 (tokens) 的順序非常重要，因此循環神經網路會順序處理資料。此外，一個標記的存在（例如單詞「not」）會影響句子其餘部分的含意，因此我們需要能夠「注意」文字其他部分的組件，像 BERT 和 GPT-3 這樣的 Transformer 模型就可以做到這一點。這些是歸納偏置 (inductive biases) 的一些例子，我們在其中識別資料中的對稱性或規律性，並添加利用這些屬性的建模組件。

在圖的情況下，我們關心每個圖組件（邊、節點、全局）如何相互關聯，因此我們尋求具有關係歸納偏置 (relational inductive bias) 的模型。[^19] 模型應該保留實體之間的顯式關係（鄰接矩陣）並保留圖對稱性（置換不變性）。我們預期實體之間交互作用很重要的問題將受益於圖結構。具體來說，這意味著設計針對集合的轉換：在節點或邊上的操作順序應該無關緊要，且操作應該適用於可變數量的輸入。

### 比較聚合操作

在任何相當強大的 GNN 架構中，池化來自相鄰節點和邊的資訊都是關鍵的一步。因為每個節點都有可變數量的鄰居，並且因為我們想要一種可微的聚合此資訊的方法，所以我們希望使用一種對節點順序和提供的節點數量不變的平滑聚合操作。

選擇和設計最佳聚合操作是一個開放的研究課題。[^44] 聚合操作的理想屬性是相似的輸入提供相似的聚合輸出，反之亦然。一些非常簡單的置換不變操作候選者是求和 (sum)、平均值 (mean) 和最大值 (max)。像方差這樣的統計摘要也起作用。所有這些都接受可變數量的輸入，並提供相同的輸出，無論輸入順序如何。讓我們探索這些操作之間的差異。

沒有任何池化類型可以始終區分所有圖對，例如左側的最大池化與右側的求和/平均池化。

沒有任何一種操作是始終最佳的選擇。當節點具有高度可變數量的鄰居，或者您需要局部鄰域特徵的歸一化視角時，平均值 (mean) 操作可能很有用。當您想要突出局部鄰域中單個顯著特徵時，最大值 (max) 操作可能很有用。求和 (sum) 在這兩者之間提供了平衡，它提供了局部特徵分佈的快照，但因為它沒有經過歸一化，所以也可以突出離群值。在實踐中，求和是常用的。

設計聚合運算是一個與集合機器學習相交的開放研究問題。[^45] 像主鄰域聚合 (Principal Neighborhood aggregation)[^27] 這樣的新方法通過將多個聚合操作拼接在一起並添加一個取決於要聚合的實體的連接度（度數）的縮放函數，來將幾個聚合操作考慮在內。同時，也可以設計特定領域的聚合運算。一個例子是「四面體手性」 (Tetrahedral Chirality) 聚合算子[^46]。

### GCN 作為子圖函數逼近器

另一種看待具有 1 度鄰居查找的 k 層 GCN（和 MPNN）的方法是，將其視為在大小為 k 的子圖的已學習嵌入上運行的神經網路。[^47][^44]

當關注一個節點時，在 k 層之後，更新後的節點表示對所有相距 k 距離的鄰居具有有限的視野，本質上是一個子圖表示。邊表示也是如此。

So 一個 GCN 正在收集所有可能的大小為 k 的子圖，並從一個節點或邊的視角學習向量表示。可能子圖的數量會以組合方式增長，因此從一開始就列舉這些子圖與在 GCN 中動態構建它們相比，可能是令人望而卻步的。

![](https://distill.pub/2021/gnn-intro/arch_subgraphs.197f9b0e.png)

### 邊與對偶圖

需要注意的一點是，邊預測和節點預測雖然看起來不同，但通常會歸結為同一個問題：圖 $G$ 上的邊預測任務可以被表述為 $G$ 的對偶圖 (dual graph) 上的節點級別預測。

為了獲得 $G$ 的對偶圖，我們可以將節點轉換為邊（並將邊轉換為節點）。圖及其對偶圖包含相同資訊，只是以不同的方式表達。有時，這種性質使在一個表示中解決問題比在另一個表示中更容易，就像傅立葉空間中的頻率一樣。簡而言之，為了在 $G$ 上解決邊分類問題，我們可以考慮在 $G$ 的對偶圖上進行圖卷積（這與在 $G$ 上學習邊表示相同），這個想法是隨對偶-對稱圖卷積網絡 (Dual-Primal Graph Convolutional Networks)[^48] 一起發展起來的。

### 圖卷積作為矩陣乘法，以及矩陣乘法作為圖上的遊走

我們談了很多關於圖卷積和訊息傳遞，當然，這提出了一個問題：我們在實踐中如何實現這些操作？在本節中，我們將探索矩陣乘法、訊息傳遞的一些性質，以及它與遍歷圖的聯繫。

我們想要說明的第一點是，鄰接矩陣 $A$（大小為 $n_{nodes} \times n_{nodes}$）與大小為 $n_{nodes} \times node_{dim}$ 的節點特徵矩陣 $X$ 的矩陣乘法，實現了具有求和聚合的簡單訊息傳遞。設矩陣為 $B=AX$，我們可以觀察到任何項目 $B_{ij}$ 都可以表示為 $<A_{row_i}, X_{column_j}>= A_{i,1}X_{1,j}+A_{i,2}X_{2, j}+\dots+A_{i,n}X_{n, j}=\sum_{A_{i,k}>0} X_{k,j}$。因為 $A_{i,k}$ 僅在 $node_i$ 和 $node_k$ 之間存在邊時才是二元項目（為 1），所以內積本質上是在「收集」與 $node_i$ 共享一條邊的維度為 $j$ 的所有節點特徵值。應該注意的是，這種訊息傳遞並未更新節點特徵的表示，只是池化了相鄰的節點特徵。但是，這可以通過在矩陣相乘之前或之後將 $X$ 傳遞給您最喜歡的可微轉換（例如 MLP）來輕鬆適應。

從這個角度，我們可以體會使用鄰接表的好處。由於 $A$ 預期的稀疏性，我們不必在 $A_{i,j}$ 為零的地方求和。只要我們有基於索引收集值的操作，我們就應該能夠只檢索正項目。此外，這種免矩陣乘法的方法使我們免於使用求和作為聚合操作。

我們可以想像，多次應用此操作使我們能夠在更大的距離傳播資訊。在這個意義上，矩陣乘法是遍歷圖的一種形式。當我們看鄰接矩陣的冪 $A^K$ 時，這種關係也是顯而易見的。如果我們考慮矩陣 $A^2$，項目 $A^2_{ij}$ 計算從 $node_{i}$ 到 $node_{j}$ 長度為 2 的所有遊走 (walks)，並且可以表示為內積 $<A_{row_i}, A_{column_j}> = A_{i,1}A_{1, j}+A_{i,2}A_{2, j}+\dots+A_{i,n}A_{n, j}$。直覺是，第一項 $A_{i,1}A_{1, j}$ 僅在兩個條件下為正：存在連接 $node_i$ 到 $node_1$ 的邊，以及另一條連接 $node_{1}$ 到 $node_{j}$ 的邊。換句話說，兩條邊形成了一條長度為 2、從 $node_i$ 出發經過 $node_1$ 到達 $node_j$ 的路徑。由於求和，我們正在對所有可能的中間節點進行計數。當我們考慮 $A^3=A \times A^2$ 等等直到 $A^k$ 時，這種直覺也適用。

關於如何將矩陣視為圖，還有更深層次的聯繫可以探索[^49][^50][^51]。

### 圖注意力網絡

在圖屬性之間傳遞資訊的另一種方法是通過注意力機制。[^52] 例如，當我們考慮一個節點及其一度相鄰節點的求和聚合時，我們也可以考慮使用加權和。接下來的挑戰是以置換不變的方式關聯權重。一種方法是考慮一個根據節點對分配權重的純量評分函數（$f(node_i, node_j)$）。在這種情況下，評分函數可以被解釋為衡量相鄰節點與中心節點關聯程度的函數。權重可以被歸一化，例如使用 Softmax 函數，以將大部分權重集中在與任務最相關的鄰居上。這個概念是圖注意力網絡 (Graph Attention Networks, GAT) [^53] 和 Set Transformers[^54] 的基礎。因為評分是基於節點對進行的，所以保留了置換不變性。常見的評分函數是內積，且節點通常在評分之前通過線性映射被轉換為查詢 (query) 和鍵 (key) 向量，以增加評分機制的表達能力。此外，為了可解釋性，評分權重可以用作衡量邊與任務相關之重要性的指標。

![](https://distill.pub/2021/gnn-intro/attention.3c55769d.png) 單個節點對其相鄰節點注意力的示意圖。對於每條邊，計算一個相互作用評分，進行歸一化，並用於加權節點嵌入。

此外，Transformer 可以被視為具有注意力機制的 GNN [^55]。在這種視角下，Transformer 模型將多個元素（例如字元標記）建模為完全連接圖中的節點，而注意力機制則為每個節點對分配邊嵌入，這些嵌入用於計算注意力權重。不同之處在於實體之間假設的連接模式，GNN 假設稀疏模式，而 Transformer 則對所有連接進行建模。

### 圖解釋與歸因

當在實際中部署 GNN 時，我們可能會關心模型的可解釋性，以建立可信度、進行調試或科學發現。我們關心解釋的圖概念因語境而異。例如，對於分子，我們可能關心特定子圖的存在或不存在[^56]；而在引用網絡中，我們可能關心文章的連接程度。由於圖概念的多樣性，有許多方法可以建立解釋。GNNExplainer[^57] 將此問題表述為提取對任務重要的最相關子圖。歸因技術 (Attribution techniques)[^58] 將排名的重要性值分配給與任務相關的圖屬性部分。因為可以合成生成具有挑戰性的真實圖問題，所以 GNN 可以作為評估歸因技術的嚴格且可重複的測試平台[^59]。

![](https://distill.pub/2021/gnn-intro/graph_xai.bce4532f.png) 圖上一些可解釋性技術的示意圖。歸因將排名的值分配給圖屬性。排名可以用作提取可能與任務相關之連接子圖的基礎。

### 生成建模

除了在圖上學習預測模型之外，我們還可能關心學習圖的生成模型。通過生成模型，我們可以通過從學習到的分佈中採樣或在給定起點的情況下完成圖來生成新圖。一個相關的應用是在新藥設計中，其中需要具有特定屬性的新型分子圖作為治療疾病的候選藥物。

圖生成模型的一個主要挑戰在於對圖的拓撲結構進行建模，該拓撲結構在大小上可能會有巨大差異，且具有 $N_{nodes}^2$ 個項目。一種解決方案是使用自動編碼器框架將鄰接矩陣直接建模為影像。[^60] 預測邊存在與否被視為二元分類任務。通過僅預測已知的邊和不存在的邊子集，可以避免 $N_{nodes}^2$ 項。graphVAE 學習在鄰接矩陣中建模正連接模式和一些非連接模式。

另一種方法是順序構建圖，從一個圖開始，迭代地應用離散動作，例如節點和邊的添加或刪除。為了避免估計離散動作的梯度，我們可以使用策略梯度。這已通過自迴歸模型（例如 RNN[^61]）或在強化學習場景[^62]中實現。此外，有時圖可以被建模為僅具有語法元素的序列。[^63][^64]


## 結語

圖是一種功能強大且結構豐富的資料類型，其優勢和挑戰與影像和文字非常不同。在本文中，我們概述了研究人員在構建基於神經網路處理圖的模型時取得的一些里程碑。我們逐步介紹了使用這些架構時必須做出的一些重要設計選擇，希望 GNN 遊樂場能夠為這些設計選擇的經驗性結果提供直覺。近年來 GNN 的成功為解決廣泛的新問題提供了絕佳的機會，我們很高興看到該領域未來會帶來什麼。

---
## 附錄

### 致謝

我們深深感謝 Andy Coenen, Brian Lee, Chaitanya K. Joshi, Ed Chi, Humza Iqbal, Fernanda Viegas, Jasper Snoek, Jennifer Wei, Martin Wattenberg, Patricia Robinson, Wesley Qian 和 Yiliu Wang 提供的有益反饋和建議，並感謝 Michael Terry 進行程式碼審查。

我們的許多 GNN 架構圖都是基於 Graph Nets 的圖表。

### 作者貢獻

所有作者都參與了寫作。

Adam Pearce 和 Emily Reif 製作了交互式圖表並設定了圖形美學。Benjamin Sanchez-Lengeling 和 Emily Reif 製作了部分初始影像草圖。Alexander B. Wiltschko 提供了編輯和寫作指導。

### 討論與評審

[Review #1 - Chaitanya K. Joshi](https://github.com/distillpub/post--gnn-intro/issues/1)   
[Review #2 - Patricia Robinson](https://github.com/distillpub/post--gnn-intro/issues/2)   
[Review #3 - Humza Iqbal](https://github.com/distillpub/post--gnn-intro/issues/3)

### 更新與修正

如果您發現錯誤或想提出更改建議，請[在 GitHub 上創建 issue](https://github.com/distillpub/post--gnn-intro/issues/new)。

### 重複使用

除非另有說明，否則圖表和文字均根據 Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) 許可進行授權，[原始碼可在 GitHub 上獲得](https://github.com/distillpub/post--gnn-intro)。從其他來源重複使用的圖表不屬於此許可範圍，可以通過其說明文字中的註釋「Figure from ...」來識別。

### 引用

對於學術背景下的歸因，請將此工作引用為
    
    Sanchez-Lengeling, et al., "A Gentle Introduction to Graph Neural Networks", Distill, 2021.

BibTeX 引用
    
    @article{sanchez-lengeling2021a,
      author = {Sanchez-Lengeling, Benjamin and Reif, Emily and Pearce, Adam and Wiltschko, Alexander B.},
      title = {A Gentle Introduction to Graph Neural Networks},
      journal = {Distill},
      year = {2021},
      note = {https://distill.pub/2021/gnn-intro},
      doi = {10.23915/distill.00033}
    }

## 參考文獻與腳註

[^1]: Understanding Convolutions on Graphs    Daigavane, A., Ravindran, B. and Aggarwal, G., 2021. Distill. [DOI: 10.23915/distill.00032](https://doi.org/10.23915/distill.00032)
[^2]: The Graph Neural Network Model    Scarselli, F., Gori, M., Tsoi, A.C., Hagenbuchner, M. and Monfardini, G., 2009. IEEE Transactions on Neural Networks, Vol 20(1), pp. 61--80.
[^3]: A Deep Learning Approach to Antibiotic Discovery    Stokes, J.M., Yang, K., Swanson, K., Jin, W., Cubillos-Ruiz, A., Donghia, N.M., MacNair, C.R., French, S., Carfrae, L.A., Bloom-Ackermann, Z., Tran, V.M., Chiappino-Pepe, A., Badran, A.H., Andrews, I.W., Chory, E.J., Church, G.M., Brown, E.D., Jaakkola, T.S., Barzilay, R. and Collins, J.J., 2020. Cell, Vol 181(2), pp. 475--483.
[^4]: Learning to simulate complex physics with graph networks    Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J. and Battaglia, P.W., 2020.
[^5]: Fake News Detection on Social Media using Geometric Deep Learning    Monti, F., Frasca, F., Eynard, D., Mannion, D. and Bronstein, M.M., 2019.
[^6]: Traffic prediction with advanced Graph Neural Networks    *, O.L. and Perez, L..
[^7]: Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in {Real-Time}    Eksombatchai, C., Jindal, P., Liu, J.Z., Liu, Y., Sharma, R., Sugnet, C., Ulrich, M. and Leskovec, J., 2017.
[^8]: Convolutional Networks on Graphs for Learning Molecular Fingerprints    Duvenaud, D., Maclaurin, D., Aguilera-Iparraguirre, J., Gomez-Bombarelli, R., Hirzel, T., Aspuru-Guzik, A. and Adams, R.P., 2015.
[^9]: Distributed Representations of Words and Phrases and their Compositionality    Mikolov, T., Sutskever, I., Chen, K., Corrado, G. and Dean, J., 2013.
[^10]: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding    Devlin, J., Chang, M., Lee, K. and Toutanova, K., 2018.
[^11]: Glove: Global Vectors for Word Representation    Pennington, J., Socher, R. and Manning, C., 2014. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP).
[^12]: Learning to Represent Programs with Graphs    Allamanis, M., Brockschmidt, M. and Khademi, M., 2017.
[^13]: Deep Learning for Symbolic Mathematics    Lample, G. and Charton, F., 2019.
[^14]: KONECT    Kunegis, J., 2013. Proceedings of the 22nd International Conference on World Wide Web - WWW '13 Companion.
[^15]: An Information Flow Model for Conflict and Fission in Small Groups    Zachary, W.W., 1977. J. Anthropol. Res., Vol 33(4), pp. 452--473. The University of Chicago Press.
[^16]: Learning Latent Permutations with Gumbel-Sinkhorn Networks    Mena, G., Belanger, D., Linderman, S. and Snoek, J., 2018.
[^17]: Janossy Pooling: Learning Deep Permutation-Invariant Functions for Variable-Size Inputs    Murphy, R.L., Srinivasan, B., Rao, V. and Ribeiro, B., 2018.
[^18]: Neural Message Passing for Quantum Chemistry    Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl, G.E., 2017. Proceedings of the 34th International Conference on Machine Learning, Vol 70, pp. 1263--1272. PMLR.
[^19]: Relational inductive biases, deep learning, and graph networks    Battaglia, P.W., Hamrick, J.B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C., Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C., Langston, V., Dyer, C., Heess, N., Wierstra, D., Kohli, P., Botvinick, M., Vinyals, O., Li, Y. and Pascanu, R., 2018.
[^20]: Deep Sets    Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R. and Smola, A., 2017.
[^21]: Molecular graph convolutions: moving beyond fingerprints    Kearnes, S., McCloskey, K., Berndl, M., Pande, V. and Riley, P., 2016. J. Comput. Aided Mol. Des., Vol 30(8), pp. 595--608.
[^22]: Feature-wise transformations    Dumoulin, V., Perez, E., Schucher, N., Strub, F., Vries, H.d., Courville, A. and Bengio, Y., 2018. Distill, Vol 3(7), pp. e11.
[^23]: Leffingwell Odor Dataset    Sanchez-Lengeling, B., Wei, J.N., Lee, B.K., Gerkin, R.C., Aspuru-Guzik, A. and Wiltschko, A.B., 2020.
[^24]: Machine Learning for Scent: Learning Generalizable Perceptual Representations of Small Molecules    Sanchez-Lengeling, B., Wei, J.N., Lee, B.K., Gerkin, R.C., Aspuru-Guzik, A. and Wiltschko, A.B., 2019.
[^25]: Benchmarking Graph Neural Networks    Dwivedi, V.P., Joshi, C.K., Laurent, T., Bengio, Y. and Bresson, X., 2020.
[^26]: Design Space for Graph Neural Networks    You, J., Ying, R. and Leskovec, J., 2020.
[^27]: Principal Neighbourhood Aggregation for Graph Nets    Corso, G., Cavalleri, L., Beaini, D., Lio, P. and Velickovic, P., 2020.
[^28]: Graph Traversal with Tensor Functionals: A Meta-Algorithm for Scalable Learning    Markowitz, E., Balasubramanian, K., Mirtaheri, M., Abu-El-Haija, S., Perozzi, B., Ver Steeg, G. and Galstyan, A., 2021.
[^29]: Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels    Du, S.S., Hou, K., Poczos, B., Salakhutdinov, R., Wang, R. and Xu, K., 2019.
[^30]: Representation Learning on Graphs with Jumping Knowledge Networks    Xu, K., Li, C., Tian, Y., Sonobe, T., Kawarabayashi, K. and Jegelka, S., 2018.
[^31]: Neural Execution of Graph Algorithms    Velickovic, P., Ying, R., Padovano, M., Hadsell, R. and Blundell, C., 2019.
[^32]: Graph Theory    Harary, F., 1969.
[^33]: A nested-graph model for the representation and manipulation of complex objects    Poulovassilis, A. and Levene, M., 1994. ACM Transactions on Information Systems, Vol 12(1), pp. 35--68.
[^34]: Modeling polypharmacy side effects with graph convolutional networks    Zitnik, M., Agrawal, M. and Leskovec, J., 2018. Bioinformatics, Vol 34(13), pp. i457--i466.
[^35]: Machine learning in chemical reaction space    Stocker, S., Csanyi, G., Reuter, K. and Margraf, J.T., 2020. Nat. Commun., Vol 11(1), pp. 5505.
[^36]: Graphs and Hypergraphs    Berge, C., 1976. Elsevier.
[^37]: HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs    Yadati, N., Nimishakavi, M., Yadav, P., Nitin, V., Louis, A. and Talukdar, P., 2018.
[^38]: Hierarchical Message-Passing Graph Neural Networks    Zhong, Z., Li, C. and Pang, J., 2020.
[^39]: Little Ball of Fur    Rozemberczki, B., Kiss, O. and Sarkar, R., 2020. Proceedings of the 29th ACM International Conference on Information & Knowledge Management.
[^40]: Sampling from large graphs    Leskovec, J. and Faloutsos, C., 2006. Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining - KDD '06.
[^41]: Metropolis Algorithms for Representative Subgraph Sampling    Hubler, C., Kriegel, H., Borgwardt, K. and Ghahramani, Z., 2008. 2008 Eighth IEEE International Conference on Data Mining.
[^42]: Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks    Chiang, W., Liu, X., Si, S., Li, Y., Bengio, S. and Hsieh, C., 2019.
[^43]: GraphSAINT: Graph Sampling Based Inductive Learning Method    Zeng, H., Zhou, H., Srivastava, A., Kannan, R. and Prasanna, V., 2019.
[^44]: How Powerful are Graph Neural Networks?    Xu, K., Hu, W., Leskovec, J. and Jegelka, S., 2018.
[^45]: Rep the Set: Neural Networks for Learning Set Representations    Skianis, K., Nikolentzos, G., Limnios, S. and Vazirgiannis, M., 2019.
[^46]: Message Passing Networks for Molecules with Tetrahedral Chirality    Pattanaik, L., Ganea, O., Coley, I., Jensen, K.F., Green, W.H. and Coley, C.W., 2020.
[^47]: N-Gram Graph: Simple Unsupervised Representation for Graphs, with Applications to Molecules    Liu, S., Demirel, M.F. and Liang, Y., 2018.
[^48]: Dual-Primal Graph Convolutional Networks    Monti, F., Shchur, O., Bojchevski, A., Litany, O., Gunnemann, S. and Bronstein, M.M., 2018.
[^49]: Viewing matrices & probability as graphs    Bradley, T..
[^50]: Graphs and Matrices    Bapat, R.B., 2014. Springer.
[^51]: Modern Graph Theory    Bollobas, B., 2013. Springer Science & Business Media.
[^52]: Attention Is All You Need    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I., 2017.
[^53]: Graph Attention Networks    Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y., 2017.
[^54]: Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks    Lee, J., Lee, Y., Kim, J., Kosiorek, A.R., Choi, S. and Teh, Y.W., 2018.
[^55]: Transformers are Graph Neural Networks    Joshi, C., 2020. NTU Graph Deep Learning Lab.
[^56]: Using Attribution to Decode Dataset Bias in Neural Network Models for Chemistry    McCloskey, K., Taly, A., Monti, F., Brenner, M.P. and Colwell, L., 2018.
[^57]: GNNExplainer: Generating Explanations for Graph Neural Networks    Ying, Z., Bourgeois, D., You, J., Zitnik, M. and Leskovec, J., 2019. Advances in Neural Information Processing Systems, Vol 32, pp. 9244--9255. Curran Associates, Inc.
[^58]: Explainability Methods for Graph Convolutional Neural Networks    Pope, P.E., Kolouri, S., Rostami, M., Martin, C.E. and Hoffmann, H., 2019. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
[^59]: Evaluating Attribution for Graph Neural Networks [[HTML]](https://papers.nips.cc/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html)   Sanchez-Lengeling, B., Wei, J., Lee, B., Reif, E., Qian, W., Wang, Y., McCloskey, K.J., Colwell, L. and Wiltschko, A.B., 2020. Advances in Neural Information Processing Systems 33.
[^60]: Variational Graph Auto-Encoders    Kipf, T.N. and Welling, M., 2016.
[^61]: GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models    You, J., Ying, R., Ren, X., Hamilton, W.L. and Leskovec, J., 2018.
[^62]: Optimization of Molecules via Deep Reinforcement Learning    Zhou, Z., Kearnes, S., Li, L., Zare, R.N. and Riley, P., 2019. Sci. Rep., Vol 9(1), pp. 1--10. Nature Publishing Group.
[^63]: Self-Referencing Embedded Strings (SELFIES): A 100% robust molecular string representation    Krenn, M., Hase, F., Nigam, A., Friederich, P. and Aspuru-Guzik, A., 2019.
[^64]: GraphGen: A Scalable Approach to Domain-agnostic Labeled Graph Generation    Goyal, N., Jain, H.V. and Ranu, S., 2020.
