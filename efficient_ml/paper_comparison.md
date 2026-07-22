# 論文《Learning both Weights and Connections for Efficient Neural Networks》與《MIT 6.5940 Lab 1 Pruning》知識點比對分析報告

本報告針對 Song Han 等人於 2015 年發表的經典論文 **《Learning both Weights and Connections for Efficient Neural Networks》 (arXiv:1506.02626)** 與您所提供的實驗筆記本 **[Lab1_zh.md](file:///home/awe/disk/deep_learning/efficient_ml/Lab1_zh.md)** 進行了深度知識點比對。

---

## 📌 核心映射與知識點比對摘要表

| 知識點 / 技術概念 | 論文 NIPS 2015 (Han et al.) 的論述與發現 | 實驗 Lab 1 的實作、問題與體現 | 一致性與延伸 |
| :--- | :--- | :--- | :--- |
| **三階段剪枝管線**<br>*(Three-Step Pipeline)* | **提出三階段核心方法**：<br>1. 正常訓練（學習重要連接）<br>2. 剪除小於閾值的權重（轉為稀疏）<br>3. 重新微調（恢復準確率） | **實驗的核心邏輯**：<br>1. 載入官方預訓練 VGG 模型<br>2. 實作並應用大小閾值剪枝（Q2）<br>3. 實作微調（Fine-tuning）進行精度恢復（Q4） | **高度一致**。<br>實驗完美復現了論文中的三階段流程，使學生能在本地觀察剪枝前、剪枝後、微調後的準確率變化。 |
| **細粒度剪枝**<br>*(Fine-grained Pruning)* | **定義非結構化（細粒度）剪枝**：<br>依據單個權重絕對值的大小設定閾值將其遮蔽，使權重矩陣稀疏化。 | **實作細粒度剪枝算法**（Q2）：<br>使用絕對值 $|W|$ 作為 `Importance`。利用 `kthvalue` 依據目標稀疏度 $s$ 計算截斷閾值，乘以遮罩（Mask）。 | **公式完全對應**。<br>論文中的 $W = W \cdot M$（遮罩相乘）即為實驗程式碼之核心。 |
| **L1 與 L2 正則化對比** | **關鍵發現**：<br>- 剪枝前：L1 正則化能將更多權重逼近於 0，因此**直接剪枝後未微調時**，L1 的準確率優於 L2。<br>- 微調時：**L2 的效果顯著優於 L1**，因為微調不需再將剩餘權重推向 0。推薦在微調時使用 L2（Weight Decay）。 | **微調參數配置與程式碼驗證**（Q4/Q5）：<br>在微調訓練程式碼（Cell 43 / hw1.py）中，優化器設定為 `torch.optim.SGD(..., weight_decay=1e-4)`。<br>`weight_decay` 即代表 L2 正則化，完美體現了論文的實踐結論。 | **深度理論印證**。<br>實驗預設的優化器參數直接採用了論文對於微調的最佳實踐（使用 L2/Weight Decay）。 |
| **各層敏感度與閾值設定**<br>*(Layer Sensitivity)* | **關鍵發現（以 AlexNet 為例）**：<br>- **FC（全連接層）敏感度低**，冗餘極大，可剪除達 90% 以上。<br>- **CONV（卷積層）敏感度高**，尤其是第一層卷積（因為僅 3 通道且冗餘極低），剪除極易崩塌。<br>- **閾值設定**：應依據敏感度動態調整各層閾值（乘以標準差 $\sigma$）。 | **敏感度分析實驗**（Q3/Q4）：<br>學生需要對模型進行逐層剪枝，並繪製各層在不同稀疏度下的敏感度曲線。確認第一層卷積層是模型中最敏感的瓶頸，而 FC 則非常 robust。 | **完美契合**。<br>實驗引導學生繪製與論文 Figure 6 完全相同的敏感度分析曲線，印證卷積與全連接層的冗餘度差異。 |
| **稀疏推論與硬體加速** | **指出硬體瓶頸**：<br>細粒度剪枝雖然使參數大量減少（AlexNet 9x, VGG 13x），但因矩陣稀疏化，**在常規通用硬體（CPU/GPU）上因無法利用稠密 GEMM，難以獲得實際的推論加速**，需要專用硬體（如 EIE 晶片）。 | **引出通道剪枝的必要性**（Section 2 / Q6-Q8）：<br>實驗展示細粒度剪枝雖能降低記憶體佔用，但無法直接在一般設備上加速，進而引入**通道剪枝**（Pruning Channels）。通道剪枝生成更小的稠密矩陣，能直接在一般 CPU/GPU 上帶來實際的 Latency 降低（Q8.2）。 | **技術演進的橋樑**。<br>論文指出了稀疏矩陣難以在常規硬體加速的局限；實驗以此為動機，進一步探討了結構化（通道）剪枝及其加速比（Q8.1 & Q8.2）。 |

---

## 🔍 重點知識點深度對比分析

### 1. 三階段剪枝方法（Three-Step Pruning Pipeline）

*   **論文的論述：**
    Han 等人指出，傳統的網路結構在訓練前就是固定的，無法在訓練中調整結構。他們提出：
    > *"Our method consists of a three-step method. First, we train the network to learn which connections are important. Next, we prune the unimportant connections. Finally, we retrain the network to fine tune the weights of the remaining connections."*
*   **實驗對應（Lab 1 - Setup & Fine-grained Pruning）：**
    在實驗中，學生直接載入了官方預訓練好的 VGG 模型（代表 **Step 1**）。在細粒度剪枝程式碼實作中，學生需要實作閾值計算（代表 **Step 2**），隨後在微調程式區塊中（Q4 / Cell 43）訓練數個 Epochs（代表 **Step 3**）。實驗用程式碼完整復現了這一流程。

### 2. 剪枝敏感度與正則化（Pruning Sensitivity & Regularization）

*   **論文的論述：**
    論文透過敏感度分析（Sensitivity Analysis）來決定各層的剪枝閾值。作者發現，第一層卷積層（`conv1`）直接與輸入影像的 3 個通道交互，特徵冗餘度最低，因此最為敏感。
    另外，在正則化方面，論文中提到一個非常有趣的「免費午餐（Free Lunch）」現象：
    > *"L1 regularization gives better accuracy than L2 directly after pruning since it pushes more parameters closer to zero. However, L2 outperforms L1 after retraining..."*
*   **實驗對應（Lab 1 - Question 1 & Question 3）：**
    - 實驗首先在 **Question 1** 中讓學生繪製預訓練權重的直方圖，觀察其分佈。
    - 在 **Question 3** 的敏感度分析中，實驗引導學生探討不同層的剪枝耐受度，這正是論文 Figure 6 的復刻。學生可以親眼觀察到第一層卷積層（僅有 3 通道）在剪枝率提高時，準確率跌落的速度遠快於擁有 512 通道的中後期卷積層與全連接層。

### 3. 計算量與實際加速的落差（MACs vs Latency）

*   **論文的論述：**
    論文提到，細粒度稀疏矩陣（Sparse Matrix）的儲存與運算需要額外的 Indexing 索引開銷（例如儲存相對索引需要 5 bits / 4 bits）。如果要在硬體上獲得真正的加速，必須使用專門設計的稀疏矩陣計算硬體。
*   **實驗對應（Lab 1 - Question 8 & Question 9）：**
    實驗的第二大部分**通道剪枝（Channel Pruning）**直接彌補了這一技術演進。
    - **Question 8.1 / 8.2** 探討了通道剪枝中，移除 30% 的通道為什麼能直接帶來約 50% 的 MACs（運算量）減少，以及為什麼實際的延遲（Latency）減少比例會略低於運算量減少比例（因為記憶體頻寬瓶頸、非卷積層開銷等因素）。
    - **Question 9** 則要求學生從「壓縮率」、「精度保持」、「延遲加速」與「硬體支持」等角度對比兩種剪枝法。這正是對論文所提稀疏化瓶頸的一大延伸思考。

---

## 💡 總結

**《MIT 6.5940 Lab 1》是《NIPS 2015 NNs Pruning 論文》的實踐昇華版。** 

實驗不僅完美復現了論文中提出的**細粒度剪枝三階段流程**、**重要性閾值公式**與**層敏感度分析**，更在此基礎上加入了**結構化通道剪枝**的實作。這填補了論文所提到的「非結構化稀疏矩陣在一般硬體上難以獲得直接加速」的遺憾，使學生能夠在一般 CPU/GPU 上親自測量出**真正的推論延遲加速（Speedup）**。

本報告以 Markdown 文件形式儲存於您的專案根目錄下，方便您隨時查閱與學習！
