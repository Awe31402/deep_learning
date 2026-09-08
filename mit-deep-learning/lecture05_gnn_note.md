# MIT 6.7960 第五講完整筆記：Graph Neural Networks（圖神經網路）

> **課程**：MIT 6.7960 Deep Learning, Fall 2024 — Lecture 5
> **講者**：**Phillip Isola**（課程第三位講者，EECS 教授）
> **來源**：
>   - 官方投影片：`mit6_7960_f24_lec5.pdf`（47 頁）
>   - 課堂錄影逐字稿：[Lecture 5 錄影](https://youtu.be/0niIwb37nF0)（1:21:00）
>   - 指定讀物：Hamilton, *Graph Representation Learning Book*（課堂只指定一章，但講者說「後面還有很多，那幾乎是一本書，寫得非常好」）
>
> **標記說明**（同 [Lecture 1 筆記](lecture01_intro_note.md)）
> - `【口頭】` = 課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 課堂提問／教授反問
> - 未標記者 = 投影片本身的內容
> - `⭐` = 重點；`⭐⭐` = 核心
>
> **公告**：【口頭】Pset 1 **下週二截止**，是**兩週份**的作業，「如果你還沒開始，現在就開始」。

---

## 目錄

- [0. ⭐⭐ 本講定位：架構設計的重點是「加限制」](#0--本講定位架構設計的重點是加限制)
- [1. 什麼問題適合用 Graph Net](#1-什麼問題適合用-graph-net)
- [2. ⭐ 資料表示與兩個目標](#2--資料表示與兩個目標)
- [3. ⭐⭐ 為什麼不能直接丟 MLP：Permutation Invariance](#3--為什麼不能直接丟-mlppermutation-invariance)
- [4. ⭐⭐ CNN 是格點上的 GNN](#4--cnn-是格點上的-gnn)
- [5. ⭐⭐ Message Passing：AGGREGATE 與 UPDATE](#5--message-passingaggregate-與-update)
- [6. ⭐ AGGREGATE 有哪些選擇](#6--aggregate-有哪些選擇)
- [7. ⭐⭐ 用 min 聚合實作 Bellman-Ford 最短路徑](#7--用-min-聚合實作-bellman-ford-最短路徑)
- [8. ⭐⭐ 通用的聚合函數：MLP-sum-MLP](#8--通用的聚合函數mlp-sum-mlp)
- [9. UPDATE 與 READOUT](#9-update-與-readout)
- [10. ⭐⭐ GNN 攤開來看就是一個 MLP](#10--gnn-攤開來看就是一個-mlp)
- [11. ⭐ MLP 是單一節點的 GNN](#11--mlp-是單一節點的-gnn)
- [12. 各種推廣：邊屬性、多關係、Attention](#12-各種推廣邊屬性多關係attention)
- [13. ⭐ 樹狀視角與權重共享](#13--樹狀視角與權重共享)
- [14. 怎麼訓練一個 GNN](#14-怎麼訓練一個-gnn)
- [15. 兩個真實架構：Polypharmacy 與 Google Maps](#15-兩個真實架構polypharmacy-與-google-maps)
- [16. ⭐⭐ 逼近能力：GNN 分不出哪些圖](#16--逼近能力gnn-分不出哪些圖)
- [17. ⭐⭐ Weisfeiler-Leman 演算法](#17--weisfeiler-leman-演算法)
- [18. ⭐⭐ 理論真的有用嗎：sum vs mean 的實驗](#18--理論真的有用嗎sum-vs-mean-的實驗)
- [19. ⭐ GNN 學不到的結構性質](#19--gnn-學不到的結構性質)
- [20. ⭐⭐ 破解方法：Positional Encoding](#20--破解方法positional-encoding)
- [21. 一頁速查表](#21-一頁速查表)
- [22. 與其他筆記的連結](#22-與其他筆記的連結)

---

## 0. ⭐⭐ 本講定位：架構設計的重點是「加限制」

【口頭】Isola 一開場先把這一講放進「架構系列」的地圖裡：

| 講次 | 架構 | 關係 |
|---|---|---|
| [Lecture 4](lecture04_cnn_note.md)（Sara Beery） | **ConvNet** | GNN 的特例 |
| **Lecture 5（本講）** | **Graph Net** | ConvNet 的推廣 |
| 之後 | **Transformer** | **GNN 的一種特例**（aggregation ＝ attention） |
| 之後 | RNN / memory-based | |

### 0.1 ⭐⭐ 這一講最重要的一句話

【口頭】講者在第 2 分鐘就丟出整堂課的核心主張：

> **「Graph nets 並不像 MLP 那樣是 universal 的。而事實上，力量正是從這裡來的。」**
>
> **「架構設計的重點，是做出一個『刻意不能擬合某些函數』的函數逼近器，因為我們就是想把那些函數排除掉。」**
>
> **「Graph 是一種語言，用來引入限制（constraints），讓你的系統做不到你不想要它做的事。」**
>
> **「所以在架構設計裡，universality 其實不是我們要追求的東西。」**

⭐⭐ 這一段直接把 [Lecture 3 逼近理論](lecture03_approximation_note.md) 的結論翻了個面：Lecture 3 花整堂課證明「MLP 什麼都能逼近」，Lecture 5 說「能逼近所有東西反而是缺點」。

**架構設計的兩個動機**（承接 Lecture 4）：
1. **硬體友善**：好平行化、GPU 跑得快
2. ⭐⭐ **加限制**：把不該學的函數排除掉

### 0.2 本講三段路線圖（投影片 p.2）

1. **Learning tasks with graphs** —— 什麼問題適合用圖
2. **Message passing GNNs** —— 機制（就是一個有 forward / backward 的神經網路）
3. **Approximation Power** —— 接續 Lecture 3，問「GNN 能逼近哪些函數」

---

## 1. 什麼問題適合用 Graph Net

### 1.1 【Q&A】講者先問學生：你們領域有什麼圖？

學生答案 + 講者補充：

| 學生的答案 | 講者的補充 |
|---|---|
| **Social networks** | 人 + 朋友關係 |
| **Hidden Markov models** | 一種 probabilistic **graphical model**，用圖表示機率分布是經典想法 |
| **Gene interactions** | 基因與轉錄出的蛋白質是節點，交互作用是邊 |
| **網頁交叉引用** | ⭐ **PageRank**——「Google 基本上是一家 graph search 公司起家的」 |

> 【口頭】「不管你是唸物理、化學還是資工，你一定看過一百萬張圖。圖是我們表示世界上有趣資料的最基本資料結構之一。」

### 1.2 投影片上的例子（p.3–9）

| 例子 | 輸入 → 輸出 | 任務型態 |
|---|---|---|
| **Node classification** | 一個節點 → 標籤 | 「這個用戶喜不喜歡爵士樂？」 |
| **Link prediction**（Pinterest，Ying et al. 2018） | 一對節點 → 有沒有邊 | 推薦系統（user × 興趣的**二分圖**） |
| **分子性質預測**（Duvenaud 2015, Stokes 2020） | **整張圖** → 性質 | 溶解度、毒性、藥效。Stokes 2020 上了 *Cell* 封面（抗生素抗藥性） |
| **Polypharmacy 副作用**（Zitnik et al. 2018） | **一對節點** → 邊的型態 | 「兩種藥各自沒事，合吃就出事」——藥–藥、藥–蛋白、蛋白–蛋白**多種邊** |
| **Google Maps ETA**（Derrow-Pinion et al. 2021） | 路網圖 → 到達時間 | 已上線；投影片顯示各城市負面 ETA 減少 16–51% |
| **物理模擬**（Sanchez-Gonzalez et al. 2020） | **一堆節點**（粒子）→ 新狀態 | 流體模擬。邊 = 粒子間的斥力 |
| **組合最佳化**（Velickovic 2020, Gasse 2019, Selsam 2018） | 問題實例圖 → 解／決策 | 「Neural Algorithmic Reasoning」；LP 的 constraint × variable 二分圖、SAT 的 clause × variable 二分圖 |

### 1.3 ⭐⭐【口頭】明明有古典演算法，為什麼還要用神經網路？

這是講者自己提出並回答的問題（最短路徑、線性規劃都有最佳解法）：

| | **古典演算法** | **Graph Net** |
|---|---|---|
| 正確性 | **保證**收斂到真解 | 只是近似 |
| 速度 | **可能很慢**（Dijkstra 跑整個世界的路網） | **快**（學到的 heuristic） |
| 適用範圍 | 任何輸入 | ⭐ **只在訓練的那個資料分布上好**，換分布就沒保證 |

> 【口頭】「你學到的是**針對那個資料分布特化**的東西。」

而且 Google Maps 的真實問題還有一堆古典演算法難處理的額外限制：路多安全、多風景、避開高速公路……

---

## 2. ⭐ 資料表示與兩個目標

### 2.1 輸入（投影片 p.10）

一張圖用兩個東西表示：

| 符號 | 是什麼 | 形狀 |
|---|---|---|
| $\mathbf{A}$ | **鄰接矩陣**（adjacency matrix） | $\mathbb{R}^{n\times n}$ |
| $\mathbf{X}$ | **節點屬性矩陣**（feature matrix） | $\mathbb{R}^{n\times d}$ |

【口頭】屬性向量是什麼？分子的話就是「這顆原子是碳還是氫」；社群網路的話就是「這個人的個人檔案」；藥物的話就是「藥名」。投影片上那些**灰色小長條**就是它。

### 2.2 ⭐ 兩個目標

| 目標 | 輸出 | 別名 |
|---|---|---|
| **1. Node embeddings** | 每個節點一個向量 | node2vec 式 |
| **2. Graph embedding** | 整張圖一個向量 | graph2vec 式 |

> **GNN 的定義（投影片紅字）**：學一個**函數**，從「圖／鄰域結構 + 節點／邊屬性」映到「向量」。

### 2.3 【Q&A】三個開場問題

| 問題 | 回答 |
|---|---|
| node embedding 跟 graph embedding 是同一個模型嗎？ | 都可以。⭐ **常見做法：先學一組 node embeddings，最後把它們加總起來得到 graph embedding。** |
| 圖很大、很密的話鄰接矩陣不會爆掉嗎？ | ⭐ 「這**不是我們建模者的選擇**，是資料本身的性質。我們通常不會去簡化輸入的圖，就把它當成給定的。」 |
| 兩者要 joint learning 嗎？ | 可以任選其一，也可以兩個都做 |

---

## 3. ⭐⭐ 為什麼不能直接丟 MLP：Permutation Invariance

### 3.1 【Q&A】暖身小考

投影片 p.11 給了一個 4 節點的圖和它的鄰接矩陣，講者問：**「鄰接矩陣的第四列對應圖上哪個節點？」**
學生答**藍色**——因為那一列有**三個 1**，而圖上只有藍色節點有三條邊。

### 3.2 天真做法（Idea 1）

把 $\mathbf{A}$ 和 $\mathbf{X}$ **攤平串成一根很長的向量**，丟進 MLP。

### 3.3 ⭐⭐【Q&A】為什麼這樣不好？

| 學生答案 | 講者評語 |
|---|---|
| 「鄰接矩陣只有 0 和 1，浪費了 MLP 吃實數的能力」 | 「不錯的想法」 |
| ⭐⭐ 「**這取決於你怎麼給節點編號。你一 permute 就壞了。**」 | **「這就是我自己心裡的答案。」** |

> 【口頭】「我們決定鄰接矩陣的第 4 列是藍色節點。**但為什麼我不說第 3 列才是藍色節點？** 這個決定是任意的。」

**分子的結構不會因為你怎麼排原子的編號而改變。**

### 3.4 ⭐⭐ 形式化（投影片 p.11）

令 $\mathbf{P}$ 是置換矩陣（$\mathbf{P}$ 換列，$\mathbf{P}^\top$ 換行）：

| 性質 | 式子 | 用在哪 |
|---|---|---|
| ⭐ **Permutation invariance** | $f(\mathbf{PAP}^\top, \mathbf{PX}) = f(\mathbf{A}, \mathbf{X})$ | **graph embedding**（輸出單一向量） |
| ⭐ **Permutation equivariance** | $f(\mathbf{PAP}^\top, \mathbf{PX}) = \mathbf{P}f(\mathbf{A}, \mathbf{X})$ | **node embeddings**（每節點一個向量） |

⭐⭐ **和 CNN 的完美對應**：

| | ConvNet | Graph Net |
|---|---|---|
| 對稱性 | **translation**（平移） | **permutation**（重新編號） |
| invariance | 平移輸入 → 輸出不變（整圖分類） | 換編號 → 輸出不變（整圖預測） |
| equivariance | 平移輸入 → feature map 跟著平移 | 換編號 → 每節點的預測跟著換位置 |

### 3.5 【Q&A】有沒有「不該」permutation invariant 的情況？

講者說**有，最後一節會講**（→ §20 positional encoding）。他當場給的例子：
> 「如果分子在你的世界裡總是以某個固定方向出現，藍色那顆比較重的原子總是在某個**標準位置**上，那你當然會想用那個資訊。」

另一個【Q&A】：invariance 為什麼輸出是單一向量？→ **不一定，可以是任意維度**；invariance 只是說「permute 輸入後輸出完全不變」。

---

## 4. ⭐⭐ CNN 是格點上的 GNN

（投影片 p.12–13：「Images are like graphs…」／「A CNN is a GNN over a grid graph」）

### 4.1 ⭐ 圖跟影像**不一樣**的地方（【Q&A】學生答）

| 學生答案 | 講者補充 |
|---|---|
| ⭐ **「頂點的鄰域形狀不固定」** | 「所以你需要能處理**可變鄰居數**的架構。這**有硬體上的意涵**——我不能把『我要看 3×3』焊死在硬體上。」 |
| ⭐⭐ **「鄰域裡沒有結構，所有鄰居都該被平等對待」** | 「CNN 有空間維度，左右和上下意義不同；GNN 的更新方式**不能依賴鄰居空間上落在哪裡**。結構比 ConvNet 的空間結構更抽象。」 |

### 4.2 ⭐ 圖跟影像**相同**的地方（投影片藍字）

| 共通點 | 說明 |
|---|---|
| **local operations** | 每個節點只從鄰居拿資訊 |
| ⭐⭐ **globalize through depth** | **靠堆疊局部操作來擴大感受野**。CNN 反覆濾波 → 感受野變大；GNN 每跑一輪 message passing → 多跳一個鄰居，最後涵蓋整張圖 |
| **weight sharing** | CNN 把影像切 patch、每塊用同一組參數；GNN 把圖切成局部區域、每塊用同一組參數 |
| ⭐ **input can have varying size** | CNN 可以吃任意大小影像；GNN 可以吃任意大小的圖。**這是和 MLP 最大的差別**（MLP 需要固定維度輸入） |

### 4.3 ⭐【Q&A】CNN 複習小考（投影片 p.13）

投影片把 CNN 畫成 GNN 的樣子，然後問：

| 問題 | 答案 |
|---|---|
| 那些**灰色長條**在 CNN 裡叫什麼？ | ⭐ **feature map 上每個位置的那一整條 channel**（＝ GNN 的節點屬性向量） |
| kernel size 多少？ | **3×3** |
| stride 多少？ | **3**（因為圖上沒有重疊，每次整個移 3 格） |

---

## 5. ⭐⭐ Message Passing：AGGREGATE 與 UPDATE

### 5.1 兩步驟的想法（投影片 p.15–16）

> **Idea:**
> 1. **Encode each node**（基於節點間的 message passing）
> 2. **Aggregate** 這**一組** node embeddings 成為 graph embedding

### 5.2 ⭐⭐ 一般式（投影片 p.16）

在第 $k$ 輪：

**Aggregate** —— 從鄰居收集資訊：

$$\mathbf{m}^{(k)}_{\mathcal{N}(v)} = \mathrm{AGGREGATE}^{(k)}\big(\{\mathbf{h}^{(k-1)}_u : u \in \mathcal{N}(v)\}\big)$$

**Update** —— 把鄰居的訊息併進自己：

$$\mathbf{h}^{(k)}_v = \mathrm{UPDATE}^{(k)}\big(\mathbf{h}^{(k-1)}_v,\ \mathbf{m}^{(k)}_{\mathcal{N}(v)}\big)$$

其中 $\mathcal{N}(v) = \{u \mid \exists (u,v) \in E(\mathcal{G})\}$ 是節點的鄰域。

投影片 p.17 進一步把 AGGREGATE 拆成兩塊：

$$\mathbf{m}^{(k)}_{\mathcal{N}(v)} = \underbrace{\bigoplus_{u\in\mathcal{N}(v)}}_{\text{排列不變的池化}} \underbrace{\psi^{(k)}\big(\mathbf{h}^{(k-1)}_u, \mathbf{h}^{(k-1)}_v\big)}_{\text{每條邊算一個訊息}}$$

### 5.3 ⭐⭐ 感受野每輪加一

> 【口頭】「k 越大就像神經網路越深，我不斷把表示改得越來越好，最後做預測。**每套用一次這兩步，感受野就增加 1。**」
>
> 「所以那顆紅色節點，如果我跑兩輪，就會拿到綠色節點的資訊——訊息從綠傳到藍，第二輪再從藍傳到紅。」

引用文獻密密麻麻一整排（Merkwirth & Lengauer 2005、Scarselli 2009、Bruna 2014、Dai 2016、Battaglia 2016、Defferrard 2016、Duvenaud 2015、Hamilton 2017、Kearnes 2016、Kipf & Welling 2017、Gilmer 2017、Li 2016、Velickovic 2018、Verma & Zhang 2018、Ying 2018、Zhang 2018…）。

---

## 6. ⭐ AGGREGATE 有哪些選擇

### 6.1 ⭐⭐ AGGREGATE 的必要性質

> **Aggregate: permutation invariant, multi-set function**

【口頭】把三個詞拆開：
- **吃一個集合**（順序無所謂）→ 輸出一個向量
- ⭐ **multi-set**（多重集）：因為節點向量 $\mathbf{h}$ **可能重複**。「multiset 只是集合論裡『允許重複元素的集合』的術語。除非你很深入集合論，不然它就是『一個可以有重複的集合 → 向量』的函數。」
- ⭐⭐ **AGGREGATE 不是 universal function，它被限制住了**——這正是 §0.1 說的「力量來自限制」

### 6.2 常見選擇（投影片 p.18）

| 名稱 | 式子 | 出處／備註 |
|---|---|---|
| **Sum** | $\sum_{u\in\mathcal{N}(v)}\mathbf{h}_u$ | |
| **Average** | $\frac{1}{\vert\mathcal{N}(v)\vert}\sum_{u\in\mathcal{N}(v)}\mathbf{h}_u$ | Merkwirth & Lengauer 2005, Scarselli 2009 |
| **對稱歸一化** | $\sum_{u\in\mathcal{N}(v)}\dfrac{\mathbf{h}_u}{\sqrt{\vert\mathcal{N}(v)\vert\,\vert\mathcal{N}(u)\vert}}$ | ⭐ **Kipf & Welling 2016（GCN）**, Hamilton 2017。「用你**和**訊息來源**雙方**的鄰居數來歸一化」 |
| **Min / Max**（逐座標） | $\max\{\mathbf{h}^{(k-1)}_u : u\in\mathcal{N}(v)\}$ | ⭐ 「很像 CNN 裡的 pooling」；輸出**固定長度**，與集合大小無關 |

> ⭐⭐ 【口頭】劇透：sum 和 mean 只差一個除法，但**後面 §18 會看到這個小差別讓實驗完全垮掉**。

### 6.3 【Q&A】怎麼知道我的問題該不該 permutation invariant？

> 「這**很看領域**。任何『輸入順序改變不會改變你想做的最佳預測』的問題，就該套這個限制。」

反例（講者當場給的，也是 §20 的伏筆）：
> 「影像處理。我知道影像**上面**比較可能是天空、下面不是。如果我在處理局部 patch，我**想知道我在影像的哪裡**。Sara 講過用 positional encoding 告訴你在影像的哪裡。⭐ **translation invariance 是 permutation invariance 的一種，而 permutation invariance 是更一般的版本。**」

【Q&A】AGGREGATE 輸出的維度？→ **一個 $D$ 維的行向量**。

---

## 7. ⭐⭐ 用 min 聚合實作 Bellman-Ford 最短路徑

（投影片 p.18 底部 + p.19）

### 7.1 為什麼要講這個

⭐ 目的是**證明**：古典圖演算法可以直接用 AGGREGATE / UPDATE 表達。

### 7.2 Bellman-Ford 回顧

- 每個節點的「屬性向量」就是一個**純量** $d_v$：**我到目標節點的最短距離估計**
- 初始化：目標節點 $d = 0$，其他全部 $d = \infty$
- 反覆跑遞迴式直到收斂：

$$d^{(k)}_v = \min_{u\in\mathcal{N}(v)}\ d^{(k-1)}_u + \mathrm{cost}(u,v)$$

【口頭】「距離會從目標往外傳播。目標的鄰居先變成 1（假設每跳成本為 1），然後往外擴散。」

### 7.3 ⭐⭐ 與 GNN 的逐行對照（投影片 p.19）

| Bellman-Ford | GNN |
|---|---|
| `for k = 1 … |S| - 1:` | `for k = 1 … GNN iter:` |
| `  for u in S:` | `  for u in S:` |
| `    d[k][u] = min_v d[k-1][v] + cost(v,u)` | `    h_u^(k) = Σ_v MLP(h_v^(k-1), h_u^(k-1))` |
| | ↑ **sum or max pooling** |

⭐ 【口頭】「Bellman-Ford 可以實作成一個**只做 AGGREGATE 的 GNN，UPDATE 在這裡是 trivial 的**。AGGREGATE 用 min 而不是 sum，而『鄰居的 embedding + cost』可以用一個 MLP 表示——如果每條邊長度都是 1，那個 MLP 就只是加 1。」

> 【口頭】**這題會出現在 Pset 上。**

---

## 8. ⭐⭐ 通用的聚合函數：MLP-sum-MLP

（投影片 p.20 / p.40）

### 8.1 問題

「sum 好像不錯，min 好像也不錯。**有沒有一族聚合函數，是所有 multiset 排列不變映射的 universal approximator？**」

### 8.2 ⭐⭐ 答案：有

$$\mathbf{m}_{\mathcal{N}(v)} = \mathrm{MLP}_2\Big(\sum_{u\in\mathcal{N}(v)}\mathrm{MLP}_1(\mathbf{h}_u, \mathbf{h}_v)\Big)$$

一般寫法（投影片 p.40）：$f_{\mathrm{Agg}}(S) = g_1\Big(\sum_{\mathbf{h}\in S} g_2(\mathbf{h})\Big)$

出處：**Zaheer et al. 2017（Deep Sets）、Qi et al. 2017（PointNet）、Xu et al. 2019（GIN）**

拆解成三步：
1. 把「鄰居的向量 + 自己的向量」丟進 $\mathrm{MLP}_1$
2. **對所有鄰居 sum**
3. 輸出再過 $\mathrm{MLP}_2$

### 8.3 ⭐⭐【口頭】最重要的一段話：這裡的 universal 跟 Lecture 3 的 universal 不一樣

學生心裡的疑問（講者自己代問）：「等等，我們不是說 MLP 本來就是 universal approximator 嗎？」

> **「是啊，只要寬度趨近無限。但我們這裡**不是**要逼近所有函數。我們要一個**沒辦法**擬合非 multiset 函數的逼近器。」**
>
> **「中間那個 sum 就是讓它變成 multiset 函數的關鍵——鄰居的順序不重要，因為我們就只是把它們加起來。」**
>
> **「所以它是 multiset 函數族的 universal approximator，但它表示不了其他函數。That's good. 這是一個受限的函數族。」**

⭐⭐ 這是整堂課「限制即力量」主張最精準的一次落地。

### 8.4 §17 會用到的兩段式論證

1. 某種非線性函數是任意 multiset 函數的 universal approximator
2. MLP 是任意非線性函數的 universal approximator（← Lecture 3）
3. ⟹ **某個 MLP 是 multiset 函數的 universal approximator**

---

## 9. UPDATE 與 READOUT

### 9.1 ⭐ 典型的 UPDATE（投影片 p.20）

$$\mathbf{h}^{(k)}_v = \sigma\big(\mathbf{W}_{\text{self}}\mathbf{h}^{(k-1)}_v + \mathbf{W}_{\text{neigh}}\mathbf{m}^{(k)}_{\mathcal{N}(v)} + b\big)$$

【口頭】「就是一個很小的、單層的 MLP：自己的線性變換 + 鄰居訊息的線性變換 + bias，過一個 pointwise 非線性（可以是 ReLU）。」

### 9.2 ⭐【Q&A】為什麼 self 和 neigh 用**兩個不同**的 W？

> 「你其實**可以**把 $\mathbf{h}$ 和 $\mathbf{m}$ **concat 起來，用一個大的 $\mathbf{W}$**，那它就長得像一個 linear + ReLU。**這裡拆開只是為了符號上清楚。**」

### 9.3 READOUT（投影片 p.21）

$$\mathbf{h}_{\mathcal{G}} = \mathrm{READOUT}\big(\{\mathbf{h}^{(K)}_v : v \in \mathcal{G}\}\big)$$

- 【口頭】「**形式跟 AGGREGATE 完全一樣，只是範圍是整張圖的所有節點，而不是某個節點的鄰居。**」
- 一樣是 multiset 函數（可以有重複），輸出一個向量
- 講者強調：「拆成這幾塊只是**其中一種約定**，我選這個是因為後面的分析比較乾淨。」

---

## 10. ⭐⭐ GNN 攤開來看就是一個 MLP

（投影片 p.22：「GNNs unrolled」）

⭐⭐ 講者說：「message passing 的觀點連到經典的圖論訊息傳遞理論，**但我覺得有一個更好、更簡單的理解方式**，而且它會和之後要講的 transformer 密切相關。」

### 10.1 攤開（unroll）之後長什麼樣

把 $k$ 步 message passing 攤開，每一步就是「拿一組節點向量 → 更新它們」：

```
        ⋮
   UPDATE⁽²⁾      ← node-wise 非線性
  AGGREGATE⁽²⁾    ← 對鄰居的「線性層」
   UPDATE⁽¹⁾
  AGGREGATE⁽¹⁾
   輸入節點向量
```

### 10.2 ⭐⭐ 精確的對應（投影片 p.22 原文）

> - **像 MLP，但節點是向量而不是純量**，邊可能是複雜的函數（例如一條邊可以是一個 MLP）
> - **每一輪 GNN message passing 就是一層**
>   - ⭐ **AGGREGATE ≈ linear layer**
>   - ⭐ **UPDATE ≈ pointwise layer**

【口頭】「拓樸上看起來就跟一個普通的老式神經網路一模一樣。唯一的差別是這個 MLP 裡的『神經元』是**向量**。」

### 10.3 ⭐【Q&A】那 backprop 在哪裡？

> **「message passing 不是最佳化演算法，它就是 forward pass。」**
>
> 「怎麼訓練？**就 backprop 過這個 computation graph。沒有任何不同。**跟訓練任何其他神經網路是完全一樣的演算法。」

### 10.4 ⭐【Q&A】k 要設多少？權重要不要跨層共享？

| 選擇 | 後果 |
|---|---|
| **每層不同參數**（$\mathrm{AGGREGATE}^{(k)}$ 帶上標 $k$） | 必須**固定深度 $k$** 展開並訓練 |
| ⭐ **跨層共享參數**（不依賴 $k$） | **想跑多久就跑多久**；此時會想確保**漸近性質好**（跑到無窮會收斂到好的值） |

### 10.5 【Q&A】這跟 attention head 有什麼不同？

> ⭐⭐ **「Transformer 是一種特別的 graph net。Attention 就是一種特別的 aggregation operator。」**

【Q&A】UPDATE 一定要非線性嗎？→ 「你需要有非線性。GNN 裡 AGGREGATE 和／或 UPDATE 都可以有非線性，通常兩個都有。理論上大概只要一個非線性就能證 universal approximation，但我要再想想。」

---

## 11. ⭐ MLP 是單一節點的 GNN

（投影片 p.23：「An MLP is a GNN over single node」）

⭐⭐【Q&A】講者問「**MLP 的圖長什麼樣？**」，學生答「一條鏈」。講者說更精確地講：

> **「其實就只是一個節點。」**

推導：
- 一個 GNN 節點自帶一個**向量** → 剛好就是 MLP 的輸入向量
- 只有一個節點 ⟹ **沒有鄰居** ⟹ AGGREGATE 是 **identity**（trivial）
- UPDATE 只剩下 $\mathbf{h}^{(k)} = \sigma(\mathbf{W}_{\text{self}}\mathbf{h}^{(k-1)} + b)$ ⟹ **這就是 MLP**

### 11.1 ⭐⭐ 三個架構的統一表

| 架構 | 是什麼樣的圖 |
|---|---|
| **MLP** | **單一節點**的 GNN |
| **CNN** | **格點圖（grid graph）**上的 GNN |
| **Transformer** | **attention 當 AGGREGATE** 的 GNN |

> 【口頭】「我整堂課都會不斷指出不同東西之間的連結，因為初學深度學習時什麼都像是不同的東西，**但其實什麼都是同一個東西**，只要小小改一下就能把一個概念變成另一個概念。我覺得這真的很有助於理解這些東西怎麼運作。」

---

## 12. 各種推廣：邊屬性、多關係、Attention

（投影片 p.24）

| 推廣 | 式子／說明 |
|---|---|
| ⭐ **Edge attributes** | $\mathbf{m}_{\mathcal{N}(v)} = \sum_{u\in\mathcal{N}(v)}\mathrm{MLP}^{(k)}(\mathbf{h}^{(k-1)}_u, \mathbf{h}^{(k-1)}_v, \mathbf{w}_{uv})$。【Q&A】有人問「邊上有距離資訊怎麼辦」，答案就是這個 |
| **Multi-relational** | 多個「channel」——**不同邊型用不同的 aggregation**（→ §15.1 的 polypharmacy） |
| ⭐⭐ **Attention**（Velickovic 2018） | $\mathbf{m}_{\mathcal{N}(v)} = \sum_{u\in\mathcal{N}(v)}\alpha_{v,u}\mathbf{h}_u$。「**加總的權重取決於節點 embedding 的值**——這個系統的名字叫 transformer」 |
| **Janossy pooling**（Murphy 2018） | 用**排列敏感**的函數，然後**對所有排列取平均** |

---

## 13. ⭐ 樹狀視角與權重共享

### 13.1 樹狀視角（投影片 p.25–27）

⭐⭐ 這是為 §16 的理論鋪路的關鍵視角。

站在節點 A 上，「**誰在對你說話**」形成一棵樹：

| 層 | 內容 |
|---|---|
| layer 0（輸入） | 各節點的原始屬性向量 |
| layer 1 | B 從 {A, C} 收訊息；C 從 {A, B, E, F} 收；D 從 {A} 收 |
| layer 2 | A 從 {B, C, D} 收（而它們已帶著 layer 1 的資訊） |

> **灰色方塊就是我們要學的 aggregation 函數。**
>
> 【口頭】「每一輪 message passing 就把這棵樹的深度加 1。」

### 13.2 ⭐ 權重共享（投影片 p.28）

> **所有節點用同一組 aggregation 函數。**（同一層內的所有灰盒子是同一個函數；不同層可以不同）
>
> ⭐⭐ **所以我們可以為「沒見過的節點與圖」產生 encoding！**（動態圖、不同的分子……）

【口頭】「就像 ConvNet 我可以在小圖上訓練、在大圖上跑濾波器一樣。」

⚠️ 但講者馬上補了一句誠實的警告：
> **「統計上它可能只 generalize 到大小差不多的圖，因為它可能對訓練資料的尺寸過擬合。但功能上，圖再大它都能重複同樣的操作。」**

### 13.3 ⭐【Q&A】圖的連通結構是學出來的嗎？

> **「Vanilla 的答案是：資料告訴你連通結構。」**
>
> 「你要預測的那個分子，它的圖結構是資料給定的。如果是某種碳氫化合物，就是氫接著幾個碳。**這不是你要優化的東西。**」
>
> 「你說的『能不能學出圖結構』是**另一個問題**（meta 層級），今天不會講到。」

---

## 14. 怎麼訓練一個 GNN

（投影片 p.29）

### 14.1 ⭐ 一個 data point 是什麼？（【Q&A】學生回答得很到位）

| 任務 | 一個訓練樣本 |
|---|---|
| **Node-level** | **{節點, 標籤} pairs** ⭐ 外加**鄰接矩陣**（講者事後補充：「應該說是 node, 逗號, adjacency matrix, 逗號, labels」） |
| **Graph-level** | **{整張圖, 標籤} pairs**（例如「這個分子有沒有毒」的二元標籤） |

### 14.2 要指定什麼

- **AGGREGATE、UPDATE、READOUT** 函數
- **loss function**（分類問題就 cross-entropy）

### 14.3 怎麼訓練

**Train with SGD**（backprop 過整個展開的計算圖）。

---

## 15. 兩個真實架構：Polypharmacy 與 Google Maps

### 15.1 Polypharmacy（投影片 p.30；Zitnik, Agrawal & Leskovec）

不同的邊型：**藥–藥、藥–蛋白、蛋白–蛋白**

$$\mathbf{h}^{(k+1)}_v = \mathrm{ReLU}\Big(\sum_r \sum_{u\in\mathcal{N}_r(v)} c^{vu}\mathbf{W}^{(k)}_r \mathbf{h}^{(k)}_u + c^v_r \mathbf{h}^{(k)}_v\Big)$$

> ⭐ **每種邊型 $r$ 各做一次 aggregation，然後全部加起來。**

投影片圖裡把三種關係（$r_1$ 腸胃出血、$r_2$ 心搏過緩、drug target relation）各自算好，再經 $\phi$ 合成一個 $\mathbf{h}^{(k+1)}_c$。

### 15.2 Google Maps ETA（投影片 p.31；Derrow-Pinion et al., CIKM 2021）

| 項目 | 內容 |
|---|---|
| **節點** | segment $v$（50–100 公尺）、supersegment $u$（約 20 個 segment） |
| **輸入特徵** | 歷史通行時間、道路型態、預測當下的交通樣態 |
| ⭐ **三種 aggregation/update** | $\mathbf{h}_v$（用相鄰的 $\mathbf{h}_e, \mathbf{h}_u$）／$\mathbf{h}_e$（用相鄰的 $\mathbf{h}_v, \mathbf{h}_u$）／$\mathbf{h}_u$（用 supersegment 裡的 $\mathbf{h}_v, \mathbf{h}_e$） |
| **pooling** | 每個 aggregation 用**多種 pooling 的組合** |
| **loss** | 各時間尺度 loss 的線性組合（segment 時間、supersegment 時間、累積 segment 時間、self-supervised／generative loss） |

### 15.3 Many connections（投影片 p.32）

| 領域 | 連結 |
|---|---|
| **Graph signal processing and convolutions** | |
| **Inference in graphical models**（Dai 2016） | ⭐ **node embedding ＝ latent variable**；給定節點特徵與圖，推論潛變數 →「Neural message passing」 |
| **Distributed / Local algorithms**（Sato 2019, Loukas 2020） | GNN 做偵測、驗證、計算的界限 |
| **Random walks**（Xu 2018） | ⭐ **oversmoothing**、圖結構與深度；**(adaptive) skip connections** |
| **Graph isomorphism testing**（Morris 2019, Xu 2019） | → §17 |

---

## 16. ⭐⭐ 逼近能力：GNN 分不出哪些圖

（投影片 p.34–36）

### 16.1 ⭐⭐ 核心問題的轉換

> **「Which graphs can GNNs distinguish? $f(G) \neq f(G')$？」**

⭐⭐ **Distinction implies function approximation**（投影片：Symmetric Stone-Weierstrass 定理；Azizian & Lelarge 21, Chen-Villar-Chen-Bruna 19, Keriven & Peyré 19, Maron-Fetaya-Segol-Lipman 19）：

$$(G, G') \in \rho(\mathcal{F}) \iff \forall F \in \mathcal{F},\ F(G) = F(G')$$

**定理**：如果函數 $H$（在緊緻定義域上）**不會**對同一個等價類裡的圖給出不同標籤，那它就能被 message passing GNN 逼近：

$$\forall \epsilon > 0,\ \exists F \in \mathcal{F}^{\mathrm{GNN}} : \sup_{G\in K}\Vert H(G) - F(G)\Vert \leq \epsilon$$

⭐ **邏輯**：反過來說，如果 GNN 分不出等價類裡的兩張圖，那它**必然**在需要區分它們的函數上產生逼近誤差。

### 16.2 ⭐⭐ 等價類是什麼？就是**樹狀結構相同**

> 【口頭】「從 graph net 的角度，**那些樹就是圖的全部特徵**。That's all they can see，因為訊息就是這樣傳的。」
>
> ⭐⭐ **「如果兩張圖的所有節點都有相同的樹狀結構，那對 graph net 來說這兩張圖就是等價的。」**

### 16.3 ⭐⭐ 具體反例（投影片 p.36 左）

兩張**結構明顯不同**的圖：
- 一張是「兩個封閉的四邊形區域」（梯子狀，6 節點）
- 一張是「兩個三角形」（蝴蝶結狀，6 節點）

但左上角紅色節點展開的樹**完全一樣**：
> 紅節點的鄰居是「黃 + 紅」，黃節點的鄰居是「紅、黃、紅」……一直展下去都一樣。

⭐⭐ **後果**：「如果這是分子，一個有毒一個沒毒，**graph net 必須給它們同一個標籤**。」

投影片右邊還有第二組例子（**一個三角柱 vs $K_{3,3}$ 完全二分圖**），一樣是等價類。

### 16.4 ⭐【Q&A】這在實務上多常見？

⭐⭐ 這是全場最誠實的一段：

> 「**去年我說：噢，這是 edge case，不太常見。然後我發現我其實不知道。**」
>
> 「用紙筆很難構造出來。我猜這是**理論上重要**的東西，像是 adversarial／worst-case——它可能存在，然後你整套機制就垮了。」
>
> 「但**我真的很想知道**：現實中你要分類毒性的分子，會不會真的碰到這個問題？會不會有一個良性分子和一個有毒分子，在 graph net 眼中無法區分，但對人體的作用完全不同？」
>
> **「答案是，我不知道。但我想對於健康與安全關鍵的應用，這是很重要的。」**

### 16.5 【Q&A】學生提早講出解法

有學生馬上說：「如果你給每個節點一個 one-hot encoding 加在屬性上，不就可以了？」

講者：「你把我幾張投影片後要問的問題答掉了。⭐ **是的，但 graph net 的力量正是來自對節點排序不變。你打破那個對稱，就打破了 permutation invariance。所以這是 trade-off。**」

---

## 17. ⭐⭐ Weisfeiler-Leman 演算法

（投影片 p.37–39）

### 17.1 ⭐⭐ 主定理

> **Theorem**（Morris-Ritzert-Fey-Hamilton-Lenssen-Rattan-Grohe 19；Xu-Hu-Leskovec-Jegelka 19）
> **任何 GNN 最多只能區分 1-dim WL 演算法能區分的那些圖。**
> **而且對任何 $n$，存在一個 GNN 使得對任何 $t$，$c^{(t)} \equiv h^{(t)}$。**

⭐ 兩個方向都要注意：
1. **上界**：GNN ≤ 1-WL
2. ⭐⭐ **可達成**：**這個上界是能達到的**——「所以我們在這一族圖裡有 universal 的判別能力」

### 17.2 演算法本身（Morgan 65, Weisfeiler & Leman 68）

**Coloring** $c^{(t)}: V(G) \to \Sigma$

$$c^{(t)}(v) = \mathrm{Hash}\big(c^{(t-1)}(v),\ c^{(t-1)}(u) \mid u\in\mathcal{N}(v)\big)$$

**同構測試**：$\{c^{(t_\infty)}(v) \mid v\in V(G)\} \neq \{c^{(t_\infty)}(v') \mid v'\in V(G')\}$？

【口頭】「這回到 1960 年代，是很經典的圖論。大致的做法是給節點上色，做某種傳播，如果兩張圖最後的著色相同就判定為同一張圖。」

### 17.3 ⭐⭐ 與 GNN 的逐項對照（投影片 p.38–39）

| WL | GNN |
|---|---|
| $c^{(t)}(v) = \boxed{\mathrm{Hash}}\big(c^{(t-1)}(v),\ c^{(t-1)}(u)\mid u\in\mathcal{N}(v)\big)$ | $\mathbf{h}^{(t)}_v = \boxed{f_{\mathrm{Update}}}\big(\mathbf{h}^{(t-1)}_v,\ \boxed{f_{\mathrm{Agg}}}(\{\mathbf{h}^{(t-1)}_u \mid u\in\mathcal{N}(v)\})\big)$ |
| **Hash** | ⭐ **UPDATE ∘ AGGREGATE** ——GNN 就是在**學**這個 hash |

### 17.4 ⭐⭐ 怎麼保證 AGGREGATE 是 injective（單射）？

（投影片 p.40）這就是 §8 那條式子的用途：

$$f_{\mathrm{Agg}}(S) = g_1\Big(\sum_{\mathbf{h}\in S}g_2(\mathbf{h})\Big),\qquad \mathbf{m}^{(k)}_{\mathcal{N}(v)} = \mathrm{MLP}_2\sum_{u\in\mathcal{N}(v)}\mathrm{MLP}_1(\mathbf{h}^{(k-1)}_u)$$

⭐ **兩段論證**：
1. 這個形式是**任意 multiset 函數**的通用表示
2. **MLP 能通用逼近 $g_1, g_2$**（Lecture 3）
3. ⟹ 可以逼近一個 injective 的 hash ⟹ 達到 1-WL 的判別力

---

## 18. ⭐⭐ 理論真的有用嗎：sum vs mean 的實驗

（投影片 p.41，PROTEINS 資料集，y 軸是**訓練**準確率）

⭐⭐ 講者特別強調：「這是**訓練**準確率，所以這只關於 **approximation**，不關於 generalization——**我能不能擬合訓練資料**。」

| 曲線 | 聚合函數 | 結果 |
|---|---|---|
| 🔴 **紅** | **Sum — MLP**（injective） | ⭐ **100%**，完全擬合 |
| 🟠 **橘** | **Sum — linear+ReLU** | ~0.9，**卡住** |
| 🔵🟢 **藍／綠** | **Mean/Max — MLP 或 linear+ReLU** | ~0.78–0.8，**更差** |

### 18.1 ⭐ 為什麼 linear+ReLU 不行？

> 「你可能在 PyTorch 裡想：噢我不用 MLP，我就 sum + linear + ReLU 好了，省點計算。**但 linear+ReLU 不是 universal approximator**——要 linear-ReLU-linear 才是。結果就是你根本擬合不了你想擬合的資料。」

### 18.2 ⭐⭐【Q&A】為什麼 mean 比 sum 差這麼多？

講者說這是最有趣的一條曲線，然後問學生。學生答對了：

> **「因為你算不出有幾個鄰居。」**
>
> 講者：「**sum 知道鄰居總數的資訊，因為鄰居越多值越大。mean 除以鄰居數，就把那個資訊弄丟了。**」

⭐⭐ 這裡的教訓（講者自己下的結論）：

> 「你可能就這樣寫程式，心想『取平均比較數值穩定吧』，隨手改了。**結果整個東西完全失效。**」
>
> **「有時候會覺得，為什麼要學這些『什麼網路能逼近什麼函數』的理論？——它在實務上真的會有影響。如果你用了一個沒有能力逼近你在乎的函數族的東西，你就是被根本地卡住了。」**

---

## 19. ⭐ GNN 學不到的結構性質

（投影片 p.42）

> **Lemma**（Garg et al. 2020, Chen et al. 2020）
> **不行！** 本講討論的 Message Passing GNN **一般而言算不出**下面這些：

| 算不出的東西 |
|---|
| **最短／最長 cycle 的長度** |
| **圖的 diameter（直徑）** |
| **某個 motif（模體）出現的次數** |

⭐ 【口頭】「不管你有多少訓練資料，都會有讓你誤差任意大的情況——因為你可以構造出**兩張在 GNN 眼中同構、但直徑不同**的圖。」

---

## 20. ⭐⭐ 破解方法：Positional Encoding

（投影片 p.43–44）

### 20.1 想法

> **加上編碼「在圖中的位置」的節點輸入特徵。**

⭐ 這就是刻意**打破** permutation invariance。

### 20.2 ⭐ 怎麼做：Graph Laplacian 的特徵向量

$$\boldsymbol{L} = \boldsymbol{D} - \boldsymbol{A}$$

- $\boldsymbol{D}$：對角矩陣，對角線是**節點的度數**（$D_{ii} = \deg(v_i)$ 或 $\sum_{(v_i,v_j)\in E} w_{ij}$，$D_{ij}=0$ for $i\neq j$）
- $\boldsymbol{A}$：鄰接矩陣
- **歸一化版本**（附錄 p.46）：$\boldsymbol{I} - \boldsymbol{D}^{-1}\boldsymbol{A}$ 或 $\boldsymbol{I} - \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2}$

> 【口頭】「就是一個廣義的座標系統，告訴你節點在圖中的位置。」

| 優點 | 缺點 |
|---|---|
| **加入全域結構資訊** | ⭐ **ambiguity**：**符號翻轉（sign flips）、特徵值重根（multiplicities）** |
| 打破等價類，區分得出以前分不出的圖 | ⭐ **可能就不再 generalize 到新的 permutation** |

**實驗**（ZINC 分子迴歸，Lim-Robinson-Zhao-Smidt-Sra-Maron-J 22）：test error 從 "standard" 的 ~0.25 → basic Laplacian PE ~0.20 → improved ~0.12。

### 20.3 ⭐ 回到 CNN：不想要 shift invariant 怎麼辦（投影片 p.44）

完全平行的兩個選項：

1. **用一個本來就不 shift invariant 的架構**（例如 MLP）
2. ⭐ **把位置資訊加到卷積濾波器的「輸入」上 —— 這就叫 positional encoding**

> ⭐⭐ 【口頭】「就跟 Sara 上次講的 sinusoidal positional encoding 一樣。**這是一個 trade-off**：你可能不再能 generalize 到新的排列，但你能區分以前區分不了的東西。」

---

## 21. 一頁速查表

### 21.1 ⭐⭐ 心法

| 主張 | 內容 |
|---|---|
| ⭐⭐ **架構設計的本質** | **不是追求 universality，而是加限制**——做一個「刻意擬合不了某些函數」的逼近器 |
| ⭐⭐ **一切都是同一個東西** | **MLP ＝ 單節點 GNN；CNN ＝ 格點 GNN；Transformer ＝ attention 當 AGGREGATE 的 GNN** |
| ⭐ **對稱性對照** | ConvNet ↔ **translation** invariance/equivariance；Graph Net ↔ **permutation** invariance/equivariance。**translation invariance 是 permutation invariance 的特例** |
| ⭐ **為什麼不用古典演算法** | 古典演算法**保證正確但慢**；GNN **快但只在訓練分布上好** |
| ⭐⭐ **理論不是裝飾** | sum→mean 一個除法之差，PROTEINS 訓練準確率就從 100% 掉到 78% |

### 21.2 ⭐⭐ 核心公式

| 名稱 | 式子 |
|---|---|
| **Permutation invariance** | $f(\mathbf{PAP}^\top, \mathbf{PX}) = f(\mathbf{A},\mathbf{X})$ |
| **Permutation equivariance** | $f(\mathbf{PAP}^\top, \mathbf{PX}) = \mathbf{P}f(\mathbf{A},\mathbf{X})$ |
| **AGGREGATE** | $\mathbf{m}^{(k)}_{\mathcal{N}(v)} = \bigoplus_{u\in\mathcal{N}(v)}\psi^{(k)}(\mathbf{h}^{(k-1)}_u,\mathbf{h}^{(k-1)}_v)$ |
| **UPDATE** | $\mathbf{h}^{(k)}_v = \sigma(\mathbf{W}_{\text{self}}\mathbf{h}^{(k-1)}_v + \mathbf{W}_{\text{neigh}}\mathbf{m}^{(k)}_{\mathcal{N}(v)} + b)$ |
| **READOUT** | $\mathbf{h}_{\mathcal{G}} = \mathrm{READOUT}(\{\mathbf{h}^{(K)}_v : v\in\mathcal{G}\})$ |
| ⭐⭐ **通用 multiset 聚合** | $f_{\mathrm{Agg}}(S) = \mathrm{MLP}_2\big(\sum_{\mathbf{h}\in S}\mathrm{MLP}_1(\mathbf{h})\big)$ |
| **WL 上色** | $c^{(t)}(v) = \mathrm{Hash}(c^{(t-1)}(v), c^{(t-1)}(u)\mid u\in\mathcal{N}(v))$ |
| **Graph Laplacian** | $\boldsymbol{L} = \boldsymbol{D} - \boldsymbol{A}$ |

### 21.3 ⭐ AGGREGATE 選擇對照

| 選擇 | 保留了什麼 | 評語 |
|---|---|---|
| **Sum** | ⭐ **鄰居數量的資訊** | 配 MLP 才是 universal |
| **Mean** | 丟掉鄰居數 | ⭐⭐ **實驗上會垮** |
| **Max / Min** | 只保留極值 | 可實作 Bellman-Ford；像 CNN 的 pooling |
| **對稱歸一化** | GCN（Kipf & Welling） | 用雙方的度數歸一化 |
| ⭐⭐ **MLP-sum-MLP** | 全部 | **multiset 函數族的 universal approximator** |
| **Attention** | 值相關的加權 | ⭐ **就是 transformer** |

### 21.4 ⭐⭐ 理論結果

| 結果 | 內容 |
|---|---|
| **Distinction ⟹ approximation** | 不會對同一等價類給不同標籤的 $H$，就能被 GNN 逼近（Symmetric Stone-Weierstrass） |
| ⭐⭐ **GNN ＝ 1-WL** | GNN **最多**能區分 1-dim WL 能區分的圖；而且**這個上界可以達到** |
| **等價類的判準** | ⭐ **所有節點的樹狀展開結構相同** ⟹ GNN 眼中是同一張圖 |
| **算不出的性質** | 最短／最長 cycle、diameter、motif 計數（Garg 2020, Chen 2020） |
| ⭐ **破解** | **Positional encoding**（Laplacian 特徵向量）——但代價是失去 permutation invariance |

### 21.5 ⭐ Summary 投影片原文（p.45）

- 編碼圖結構與節點／邊屬性
- **重要：permutation invariance / equivariance**
- **主要想法：message passing + aggregations**
- ⭐ **能吃不同大小與結構的圖**（跟 CNN 一樣）
- 連結：graph signal processing、graphical models、distributed computing、isomorphism testing…
- 表示能力的增強：higher-order、node IDs / augmentation

---

## 22. 與其他筆記的連結

| 主題 | 去哪看 |
|---|---|
| **MLP、非線性、為什麼要堆層** | [Lecture 1 筆記](lecture01_intro_note.md) §5.2 |
| **Backprop、computation graph**（GNN 展開後就是這個） | [Lecture 2 筆記](lecture02_backprop_note.md) §7, §13 |
| ⭐⭐ **Universal approximation**（§8 那個「universal 但受限」的論證要用） | [Lecture 3 筆記](lecture03_approximation_note.md) §5, §6 |
| ⭐ **「萬能逼近其實沒那麼了不起」的態度** | [Lecture 3 筆記](lecture03_approximation_note.md) §14.6 —— **本講 §0.1 把它推到極致** |
| **CNN、translation invariance、sinusoidal positional encoding** | [Lecture 4 筆記](lecture04_cnn_note.md) §6, §21 |
| **Transformer ＝ attention 當 AGGREGATE 的 GNN** | 課程後續；另見 `transformers.md` |
| **舊版 GNN 筆記（另一份教材）** | [`gnn_intro.md`](gnn_intro.md)、[`gnn_summary.md`](gnn_summary.md) |
