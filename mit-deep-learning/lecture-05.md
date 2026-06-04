# Session Transcript: MIT Deep Learning Lec 05 (Graphs) - 2026-06-02

## Context
Verbatim transcript of a deep dive session into MIT Deep Learning Lecture 05: Architectures: Graphs, discussing GNN structures, CNN limitations on graphs, neighborhood encoding, and aggregation functions.

---

### Round 1: Read Lecture Transcript and Summarize
**User:**
讀取 .books/mit-deep-learning/transcripts/05_Lec_05._Architectures_Graphs.md

**Gemini:**
我已為您讀取並整理了 `05_Lec_05._Architectures_Graphs.md` 的逐字稿內容。以下是本堂課程（由 MIT 的 Phillip Isola 教授主講）的核心內容摘要：

### 1. 適用問題與圖的表達 (Graph-Structured Data)
* **適用領域**：社交網路、分子結構、藥物與蛋白質交互作用、物理粒子/流體模擬、地圖最短路徑搜尋（如 Google Maps 尋找啟發式加速演算法）以及離散組合優化問題。
* **資料結構**：輸入包含代表節點特徵的屬性向量（Node Attributes）以及代表節點連接關係的鄰接矩陣（Adjacency Matrix）。任務通常分為：
  * **Node-level** 預測（例如：社交網路中的用戶興趣分類）
  * **Graph-level** 預測（例如：預測分子是否具備毒性）

### 2. 圖的神經網路設計原則 (Inductive Biases & Constraints)
* **排列不變性 (Permutation Invariance)**：重新排列圖中節點的索引順序（Index）不應改變整個圖的預測結果（對應 Graph-level 預測）。
* **排列同變性 (Permutation Equivariance)**：若打亂輸入節點的順序，輸出對應各節點的預測也會隨之打亂（對應 Node-level 預測）。
* **與 CNN 的關聯**：CNN 可以看作是運作在「規則網格圖 (Grid Graphs)」上的特殊 GNN。GNN 則是將這種局部運算、參數共享（Parameter Sharing）與隨輸入尺寸變動（Scale Invariance）的特性推廣到任意拓撲結構上。

### 3. 訊息傳遞機制 (Message-Passing Framework)
GNN 的每一層運算可拆解為兩個步驟：
1. **聚合 (Aggregate)**：一個具排列不變性的多重集函數（Multiset Function），將鄰近節點的特徵向量聚合成一個代表訊息的向量（如 Sum, Mean, Max, 或 Transformer 中所使用的 Attention 機制）。
2. **更新 (Update)**：將聚合後的訊息與節點自身的特徵結合，通過非線性變換（如 MLP + ReLU）得到該節點的新嵌入向量（Node Embedding）。
* **感受野 (Receptive Field)**：隨着層數 $k$ 的增加，每個節點能接收到更遠距離（$k$-hop）鄰近節點的資訊。

### 4. GNN 的表達能力與理論限制 (Theory of Approximation Power)
* **萬能逼近性 (Universal Approximation)**：MLP-Sum-MLP 的架構已被證明是多重集（Permutation-Invariant）函數的萬能逼近器。而使用 Mean 替代 Sum 會損失「節點鄰居數量」的資訊，使表達能力顯著降低。
* **表達力上限與 1-WL 測試**：GNN 區分非同構圖的能力，在理論上限與經典圖論中的 **1-WL (Weisfeiler-Leman) 演算法**等價。
* **無法區分的圖**：如果兩個結構完全不同的圖，其所有節點從局部看出去的「鄰域樹狀結構 (Tree Structures)」完全一致，GNN 將無法區分它們（例如具備不同環結構或直徑的特定圖）。
* **對策：位置編碼 (Positional Encodings)**：可透過加入代表圖中絕對/相對位置的編碼（例如 **圖拉普拉斯矩陣的特徵向量 Graph Laplacian Eigenvectors**，類似 Transformer 中的正弦位置編碼），以此打破對稱性並提升區分力，但這也需要在排列不變性與區分度之間做折衷（Trade-off）。

---

### Round 2: CNN Challenges on Graphs
**User:**
CNN 在處理圖的時候有什麼難點

**Gemini:**
根據逐字稿內容與圖神經網路（GNN）的理論，傳統卷積神經網路（CNN）在直接處理圖結構（Graphs）時，主要面臨以下幾個核心難點：

### 1. 鄰居節點的數量不固定 (Variable Number of Neighbors)
* **CNN 的做法**：影像中的像素在規則的二維網格上，除了邊界之外，每個像素點都有固定數量的鄰居（例如 $3 \times 3$ 卷積核的鄰域永遠是 8 個像素）。這使得卷積核可以設計成固定大小，非常適合 GPU 進行大規模並行加速。
* **圖的難點**：在圖結構中，每個節點的度（Degree，即鄰居數量）是高度可變的。一個節點可能只有 1 個鄰居，另一個節點卻可能有成千上萬個鄰居（例如社交網路中的熱門帳號）。**傳統 CNN 的固定尺寸濾波器（Filter）無法直接適應這種動態變化的局部結構**。

### 2. 局部鄰域缺乏空間順序 (Lack of Ordering/Structure in Neighborhoods)
* **CNN 的做法**：在影像中，鄰域像素有明確的空間關係（如「左上」、「右下」）。卷積核中的每個權重參數與特定的空間位置是一一對應的。
* **圖的難點**：在圖中，一個節點的鄰居集合本質上是**無序的（Set/Multiset）**。我們無法定義誰是「左邊的鄰居」或「右邊的鄰居」。如果硬是用 CNN 的權重矩陣去乘以鄰居節點，就會強行引入任意的順序，這會導致模型的預測結果隨着節點編號的順序而改變，違反了**排列不變性（Permutation Invariance）**。

### 3. 非歐幾里得空間的幾何挑戰 (Non-Euclidean Domain)
* **CNN 的做法**：CNN 預設輸入資料存在於規則的「歐幾里得空間」（如 1D 音訊、2D 影像、3D 體素）。這類空間有統一的坐標系與距離定義。
* **圖的難點**：圖是非歐幾里得（Non-Euclidean）結構，節點之間僅由拓撲關係（連接與否）決定。在圖中，我們很難定義什麼是「平移（Translation）」，這使得 CNN 最核心的**平移不變性（Translation Invariance）與權重共享機制**無法直接在圖上實行。

### 4. 硬體加速與計算效率的瓶頸 (Hardware Implications)
* **CNN 的做法**：CNN 的規則網格計算可以完美轉化為高度優化的密集矩陣乘法（Dense Matrix Multiplication），這是 GPU 硬體最擅長的工作。
* **圖的難點**：圖的表示通常需要使用**稀疏矩陣（Sparse Matrix）**來節省記憶體，這會帶來不規則的記憶體存取模式（Irregular Memory Access Patterns）。在硬體層面上，這種不規則的稀疏運算比 CNN 的密集網格計算更難進行並行優化。

---

### GNN 是如何解決這些難點的？
為了解決上述 CNN 處理圖的限制，GNN 引入了以下機制：
1. **使用聚合函數（Aggregate Function）**：採用如 `Sum`、`Mean` 或 `Max` 等無關輸入順序與數量的運算，將任意數量鄰居的資訊壓縮成單個固定長度的向量。
2. **排列不變性設計**：確保不論如何打亂節點的索引（Permutation），模型輸出的物理特徵依然保持一致。

---

### Round 3: Neighborhood Encoding
**User:**
encoding neiborhood 說明

**Gemini:**
在圖神經網路（GNN）的脈絡中，**「鄰域編碼」（Neighborhood Encoding）** 指的是**如何將一個節點周圍的拓撲結構與鄰居的特徵資訊，壓縮（編碼）成一個低維度的稠密向量（Embedding）**。

這是 GNN 能夠理解圖形結構的關鍵機制。我們可以從以下幾個核心維度來詳細說明：

---

### 1. 鄰域編碼的核心機制：消息傳遞（Message Passing）
GNN 對鄰域的編碼通常是透過「聚合（Aggregate）」與「更新（Update）」兩步遞迴進行的：

$$\mathbf{m}_v^{(k)} = \text{Aggregate}^{(k)} \left( \left\{ \mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v) \right\} \right)$$
$$\mathbf{h}_v^{(k)} = \text{Update}^{(k)} \left( \mathbf{h}_v^{(k-1)}, \mathbf{m}_v^{(k)} \right)$$

* **訊息收集 (Aggregation)**：節點 $v$ 收集它所有鄰居 $\mathcal{N}(v)$ 的狀態特徵。為了應對鄰居數量不固定且無序的問題，聚合函數（如 `Sum`、`Mean` 或 `Max`）必須具備**排列不變性（Permutation Invariance）**。
* **狀態融合 (Update)**：將鄰域聚合出來的訊息 $\mathbf{m}_v$ 與節點自身的特徵融合，產生節點在下一層的新編碼 $\mathbf{h}_v$。

---

### 2. 樹狀結構視角 (The Tree Perspective / Computation Tree)
當我們堆疊多層 GNN 時，鄰域編碼的範圍會呈指數型擴展。從單個節點的角度來看，**鄰域編碼本質上是在對一棵「計算樹」（Computation Tree）進行編碼**：

* **1 層 GNN**：編碼節點 $v$ 的直接鄰居（1-hop 鄰域）。
* **2 層 GNN**：直接鄰居會先聚合它們各自的鄰居（2-hop），再傳遞給節點 $v$。此時，節點 $v$ 的編碼包含了高度為 2 的局部樹狀結構。
* **$K$ 層 GNN**：節點 $v$ 最後得到的向量，就是**高度為 $K$、以 $v$ 為根節點的計算樹**的編碼表徵。

> [!NOTE]
> GNN 並非直接處理整張圖的全局結構，而是將圖中的局部關係「展開（Unroll）」成以各個節點為根的樹狀結構，然後對這些樹進行編碼。

---

### 3. 鄰域編碼的理論極限：1-WL 測試
因為 GNN 的鄰域編碼本質上是基於**計算樹的展開**，這帶來了一個致命的限制：
* **樹狀同構限制**：如果兩個結構上不同的子圖，展開後的鄰域樹狀結構（Neighborhood Tree Structures）完全相同，GNN 對這兩個鄰域編碼出來的向量將會**完全一樣**。
* 這是為什麼傳統 GNN 無法分辨某些特定環狀結構（如三角形 vs 五角星）或計算圖的直徑。其編碼能力的上限等同於經典圖論中的 **1-WL (Weisfeiler-Leman) 算法**。

---

### 4. 如何強化鄰域編碼？(Structural & Positional Encodings)
為了解決樹狀編碼的局限性，現代 GNN 會在鄰域編碼中加入額外的結構或位置資訊：

1. **節點度數編碼 (Node Degree Encoding)**：在初始特徵中加入節點的度數（Degree），讓聚合函數能區分鄰居的多寡。
2. **位置編碼 (Positional Encoding)**：加入如**圖拉普拉斯特徵向量 (Graph Laplacian Eigenvectors)**，為圖中的每個節點提供一個廣義的「空間坐標」，打破局部樹狀結構的對稱性。
3. **子圖/環編碼 (Subgraph/Cycle Encoding)**：顯式地將局部環結構（如「此節點是否在一個三角形中」）作為特徵輸入，直接增強鄰域編碼對拓撲特徵的感知能力。

---

### Round 4: Aggregation Functions
**User:**
Aggregation functions 是什麼

**Gemini:**
在圖神經網路（GNN）中，**聚合函數（Aggregation Functions）** 是最核心的運算元件。它的任務是：**將一個節點周圍所有鄰居的特徵向量（數量不固定且無序），合併（聚合）成一個單一的、固定維度的訊息向量。**

---

### 1. 為什麼需要聚合函數？（面臨的挑戰）
在處理圖形資料時，有兩個傳統神經網路（如 MLP）無法解決的挑戰：
1. **鄰居數量不固定**：每個節點連接的鄰居數（度數）都不同。
2. **無序性**：鄰居節點之間沒有「第一、第二」的順序關係。

因此，聚合函數在數學上必須是一個 **多重集函數（Multiset Function）**，且必須具備以下兩個關鍵性質：
* **排列不變性（Permutation Invariance）**：改變鄰居輸入的順序，輸出的結果不能改變。例如：$f(\{A, B\}) = f(\{B, A\})$。
* **輸入長度可變性**：不論輸入是 1 個向量還是 10,000 個向量，輸出的維度都必須固定。

---

### 2. 常見的聚合函數及其特性

不同的聚合函數會保留不同維度的結構資訊，在實務上有各自的優缺點：

#### ① Sum Aggregation (求和)
直接將所有鄰居的特徵向量相加。
* **公式**：$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u$
* **特點**：
  * **保留「數量（Cardinality）」資訊**：鄰居越多，相加的值通常越大。因此模型能隱式地得知節點的度數（鄰居數）。
  * **理論能力最強**：配合前後的 MLP（即 `MLP(Sum(MLP(h)))`），在理論上被證明是**多重集函數的萬能逼近器**，能區分最多的非同構圖。
  * **缺點**：如果某些節點的鄰居極多（如社交網路中的網紅），數值可能會非常大，導致梯度不穩定。

#### ② Mean Aggregation (求平均)
將鄰居的特徵向量相加後，除以鄰居的數量。
* **公式**：$\mathbf{m}_v = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u$
* **特點**：
  * **數值穩定**：將特徵限制在相同的尺度，不易發生數值爆炸。
  * **著重「比例/分佈」**：適合用在只關心鄰居特徵的相對分佈，而不關心絕對數量的任務。
  * **缺點（會損失資訊）**：無法區分「鄰居數少但特徵一致」與「鄰居數多且特徵一致」的情況。例如：1 個特徵為 $[1, 1]$ 的鄰居，跟 100 個特徵都是 $[1, 1]$ 的鄰居，平均出來的結果完全一樣。

#### ③ Max / Min Aggregation (最大值/最小值)
在鄰居特徵的每一個維度上，只取最大值或最小值（類似 CNN 的 Max Pooling）。
* **公式**：$\mathbf{m}_v = \max_{u \in \mathcal{N}(v)} (\mathbf{h}_u)$
* **特點**：
  * **著重「最顯著特徵」**：只保留鄰域中最突出或最具代表性的特徵。
  * **演算法表徵能力**：例如在求解圖的最短路徑（Bellman-Ford 演算法）時，`Min` 聚合函數是實現該演算法的天然基礎。

#### ④ Attention Aggregation (注意力機制)
不採用均等的權重，而是動態計算目標節點與各個鄰居之間的「關聯度」，進行加權求和（這也是 **Transformer** 和 **GAT (Graph Attention Networks)** 的核心）。
* **公式**：$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{h}_u$ （其中 $\alpha_{vu}$ 是動態計算的注意力權重）。
* **特點**：
  * **表達力極強**：模型可以自主學習「哪些鄰居更重要」，並在不同上下文下動態調整關注點。
  * **計算成本較高**：需要額外計算注意力權重矩陣。

---

### 3. 總結：理論與實務的權衡

| 聚合函數 | 排列不變性 | 是否保留鄰居數量資訊 | 計算複雜度 | 適用場景 |
| :--- | :---: | :---: | :---: | :--- |
| **Sum** | 是 | **是** | 低 | 理論區分力要求高、圖規模中等、節點度數差異不大時。 |
| **Mean** | 是 | 否 | 低 | 節點度數差異極大（如無尺度網路）、數值易不穩定時。 |
| **Max** | 是 | 否 | 低 | 尋找局部最值、特徵極稀疏或特定演算法模擬。 |
| **Attention** | 是 | 否 (通常) | 高 | 需要精細特徵交互、計算資源充足時（如 Transformer / GAT）。 |

---

### Round 5: Trigger Record Session
**User:**
用 @[skill/verbatim-recorder.skill] 記錄本輪會話
