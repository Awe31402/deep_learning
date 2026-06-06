# 圖神經網路模型 (GNN) 重點摘要

> [!NOTE]
> 本摘要整理自 [gnn.pdf](file:///home/awe/disk/deep_learning/mit-deep-learning/gnn.pdf) (Chapter 5: The Graph Neural Network Model)。本章詳細介紹了圖神經網路 (GNN) 的核心理論、訊息傳遞機制、以及各種變體與優化方法。

## 1. 核心觀念：置換不變性與同變性
- **非歐氏數據的挑戰**：傳統 CNN 僅適用於網格結構（如圖像），RNN 僅適用於序列（如文本）。對於通用圖結構，需要一種與節點順序無關的架構。
- **數學定義**：
  - **置換不變性 (Permutation Invariance)**：函數輸出不隨節點順序而改變。
    $$f(\mathbf{P}\mathbf{A}\mathbf{P}^T) = f(\mathbf{A})$$
  - **置換同變性 (Permutation Equivariance)**：函數輸出隨節點順序以相同方式置換。
    $$f(\mathbf{P}\mathbf{A}\mathbf{P}^T) = \mathbf{P}f(\mathbf{A})$$
  *(其中 $\mathbf{P}$ 為置換矩陣，$\mathbf{A}$ 為鄰接矩陣)*

## 2. 神經訊息傳遞機制 (Neural Message Passing)
在第 $k$ 次疊代中，節點 $u$ 的表徵更新包含兩個主要步驟：
1. **MESSAGE AGGREGATION (訊息聚合)**：
   $$\mathbf{m}_{\mathcal{N}(u)}^{(k)} = \text{AGGREGATE}^{(k)}\left(\{\mathbf{h}_v^{(k-1)}, \forall v \in \mathcal{N}(u)\}\right)$$
2. **STATE UPDATE (狀態更新)**：
   $$\mathbf{h}_u^{(k)} = \text{UPDATE}^{(k)}\left(\mathbf{h}_u^{(k-1)}, \mathbf{m}_{\mathcal{N}(u)}^{(k)}\right)$$

- **基礎 GNN 更新公式**：
  $$\mathbf{h}_u^{(k)} = \sigma \left(\mathbf{W}_{\text{self}}^{(k)}\mathbf{h}_u^{(k-1)} + \mathbf{W}_{\text{neigh}}^{(k)}\sum_{v\in\mathcal{N}(u)}\mathbf{h}_v^{(k-1)} + \mathbf{b}^{(k)}\right)$$
- **自環設計 (Self-loops)**：將節點自身直接納入鄰域中共同聚合，以簡化更新函數：
  $$\mathbf{h}_u^{(k)} = \text{AGGREGATE}\left(\{\mathbf{h}_v^{(k-1)}, \forall v \in \mathcal{N}(u) \cup \{u\}\}\right)$$

---

## 3. 廣義鄰域聚合 (Generalized Neighborhood Aggregation)

> [!TIP]
> 簡單的求和聚合易受到節點度數（Degree）的影響而導致數值不穩定。合適的歸一化對優化至關重要。

### A. 鄰域歸一化 (Normalization)
- **平均歸一化 (Mean)**：
  $$\mathbf{m}_{\mathcal{N}(u)} = \frac{\sum_{v\in\mathcal{N}(u)} \mathbf{h}_v}{|\mathcal{N}(u)|}$$
- **對稱歸一化 (Symmetric)**（如 GCN 採用）：
  $$\mathbf{m}_{\mathcal{N}(u)} = \sum_{v \in \mathcal{N}(u)} \frac{\mathbf{h}_v}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}}$$

### B. 集合聚合器 (Set Aggregators)
- **Set Pooling (DeepSets)**：使用通用集合函數逼近器。
  $$\mathbf{m}_{\mathcal{N}(u)} = \text{MLP}_\theta\left(\sum_{v\in\mathcal{N}(u)}\text{MLP}_\phi(\mathbf{h}_v)\right)$$
- **Janossy Pooling**：利用 LSTM 等序列模型處理所有排列組合（或隨機/規範化子集）後取平均。

### C. 鄰域注意力機制 (Neighborhood Attention)
- **GAT (Graph Attention Network)**：利用注意力權重 $\alpha_{u,v}$ 計算加權和：
  $$\mathbf{m}_{\mathcal{N}(u)} = \sum_{v\in\mathcal{N}(u)} \alpha_{u,v}\mathbf{h}_v$$
  其中 $\alpha_{u,v}$ 通常透過 Softmax 歸一化 bilinear、MLP 或拼接特徵後的點積計算。
- **與 Transformer 的關聯**：Transformer 層實質上等同於在**全連接圖**上執行的多頭注意力 GNN 層。

---

## 4. 解決過平滑問題 (Over-smoothing)

> [!WARNING]
> 當 GNN 層數較深時，節點表徵容易變得非常相似，進而失去區分度。此現象稱為 **過平滑 (Over-smoothing)**。

為了構建更深層的 GNN 架構，文獻提出了以下更新方式：
1. **殘差與拼接連接 (Skip-Connections / Concatenation)**：
   $$\text{UPDATE}_{\text{concat}}(\mathbf{h}_u, \mathbf{m}_{\mathcal{N}(u)}) = [\text{UPDATE}_{\text{base}}(\mathbf{h}_u, \mathbf{m}_{\mathcal{N}(u)}) \oplus \mathbf{h}_u]$$
2. **門控更新 (Gated Updates)**：引入 RNN 門控機制（如 GRU 或 LSTM）來更新節點狀態。
   $$\mathbf{h}_u^{(k)} = \text{GRU}(\mathbf{h}_u^{(k-1)}, \mathbf{m}_{\mathcal{N}(u)}^{(k)})$$
3. **跳躍知識連接 (Jumping Knowledge, JK)**：在最終輸出時結合各層的表徵。
   $$\mathbf{z}_u = f_{\text{JK}}\left(\mathbf{h}_u^{(0)} \oplus \mathbf{h}_u^{(1)} \oplus \dots \oplus \mathbf{h}_u^{(K)}\right)$$

---

## 5. 多關係圖與圖池化 (Multi-relational & Graph Pooling)
- **多關係圖 (RGCN)**：為每一種邊關係 $\tau$ 定義專屬的參數矩陣 $\mathbf{W}_\tau$，並配合基底分解 (Basis decomposition) 減少參數數量。
- **圖池化 (Graph Pooling)**：
  - **Set Pooling**：直接對全圖所有節點進行 Sum/Mean 或基於 Attention 機制的池化。
  - **Graph Coarsening (圖粗糙化)**：使用分群方法（如 DiffPool）階層式地精簡圖結構，生成層次化 GNN。
