# EfficientML.ai 第二講 對話紀錄 - 2026-05-18

## 總覽
本次對話主要聚焦於分析 MIT 6.5940 (2023 秋季班) 第二講：「神經網路基礎」(Basics of Neural Networks)。複習的資料包含課程逐字稿以及隨附的 PDF 投影片 (`Lec02-Basics.pdf`)。

---

## 對話日誌

### 1. 讀取第二講逐字稿
**User:** 讀取 .books/efficient-ml/EfficientML.ai Lecture 2 - Basics of Neural Networks (MIT 6.5940, Fall 2023, Zoom recording).txt'

**Gemini:** 分析了逐字稿並提供全面摘要，涵蓋以下內容：
- **課程先修：** 具備機器學習 (6.036) 與計算機架構 (6.191) 的背景知識。
- **動機：** 深度學習模型的需求超越硬體發展（摩爾定律 vs. 模型每兩年成長 4 倍）。
- **核心組件：** 全連接層、卷積、群組/深度卷積、池化、歸一化（Batch, Layer, Instance, Group）與激活函數。
- **效率指標：** 參數數量、模型大小、激活值（總和 vs. 峰值）、MACs、FLOPs、延遲 (Latency) 與吞吐量 (Throughput)。

### 2. 讀取課程投影片 (PDF)
**User:** 讀取 @.books/efficient-ml/Lec02-Basics.pdf

**Gemini:** 找到 PDF 檔案並使用 `pdftotext` 提取文字。提供了與課程結構對應的投影片摘要：
- 複習術語（神經元、突觸）。
- 構建模塊（全連接層、卷積、池化、歸一化、Transformer）。
- 效率指標的計算公式。

### 3. 詳細投影片分析 - 第 7 頁 (深度神經網路)
**User:** 讀取 @.books/efficient-ml/Lec02-Basics.pdf 第七頁

**Gemini:** 
- **觀念：** 將生物神經元映射到數學模型。
- **術語：** 神經元 = 激活值/特徵；突觸 = 權重/參數。
- **模型維度：** 隱藏層大小 = 寬度 (Width)；層數 = 深度 (Depth)。
- **公式：** $y_j = f(\sum w_i x_i + b)$。

### 4. 詳細投影片分析 - 第 32 頁 (感受野)
**User:** 第七頁想說明什麼觀念讀取 @.books/efficient-ml/Lec02-Basics.pdf 第32頁

**Gemini:**
- **定義：** 感受野 (Receptive Field, RF) 是指輸入影像中能被輸出元素「看到」的區域。
- **成長：** 連續的卷積層會將感受野增加 $k-1$。公式：$RF = L \cdot (k-1) + 1$。
- **問題：** 如果只使用 $s=1$ 的卷積，大型影像需要過多層數才能讓輸出看到整張影像。
- **解決方案：** 下採樣（步長/池化）以更有效率地擴展感受野。

### 5. 詳細投影片分析 - 第 33 頁 (步長卷積)
**User:** 讀取 @.books/efficient-ml/Lec02-Basics.pdf 第33頁

**Gemini:**
- **觀念：** 步長 (Stride) 如何影響感受野。
- **比較：** 在 $k=3$ 的情況下，2 層 $s=2$ 的卷積可以達到 $RF=7$，而 $s=1$ 只能達到 $RF=5$。
- **優勢：** 步長卷積是一種能用較少層數快速增加感受野的方法。

### 6. 群組與深度卷積說明
**User:** 讀取 @.books/efficient-ml/Lec02-Basics.pdf 第34-35頁及逐字稿說明

**Gemini:**
- **群組卷積 (P.34)：** 將輸入通道分成 $g$ 組。參數/MACs 將減少為原本的 $1/g$。
- **深度卷積 (P.35)：** 極端情況，當 $g = c_i$。每個通道都有獨立的濾波器。
- **權衡：** 教授警告減少參數不一定等於提升速度。深度卷積通常需要增加通道數（例如 MobileNetV2 擴張 6 倍），這會增加記憶體數據移動（激活值），而這才是真正的硬體瓶頸。

### 7. 教授對 Transformer 的解釋
**User:** 教授怎麼 解釋 transformer

**Gemini:**
- **產業影響：** 同時革新了 NLP 與視覺領域 (ViT)。
- **機制 (P.40-41)：** 用搜尋引擎來比喻 QKV (Query-Key-Value)（Query = 搜尋字詞，Key = 影片標題，Value = 影片內容）。
- **過程：** 將 Q 與 K 進行矩陣乘法，透過 Softmax 取得注意力權重，再將其乘上 V。
- **架構：** 包含多頭注意力機制 (MHA) 與前饋網路 (FFN/MLP)。

### 8. 延遲 (Latency) vs. 吞吐量 (Throughput) (P.46/48)
**User:** 關於 pdf 第46 頁 latency 的部份說明

**Gemini:**
- **公式：** $Latency \approx \max(T_{computation}, T_{memory})$。
- **組成：** $T_{computation}$ (運算量 / 峰值運算能力) 與 $T_{memory}$ (權重和激活值大小 / 記憶體頻寬)。
- **關鍵見解：** 延遲針對「單一任務」；吞吐量則是「單位時間內的任務數」。兩者不能直接互換。
- **優化：** 將計算與記憶體存取重疊 (Overlapping)，以隱藏數據搬移的時間。

### 9. 步長 (Stride) 對參數量的影響
**User:** CNN 的 stride 對參數量是否有影響

**Gemini:**
- **無影響：** 參數數量僅取決於 $c_o, c_i, k_h, k_w$。
- **顯著影響：** 步長會縮小輸出特徵圖的尺寸 ($h_o, w_o$)，進而大幅減少 **MACs** 和 **激活值**。

### 10. 激活值數量分析 (P.62-67)
**User:** 解釋 Number of Activation 部份

**Gemini:**
- **定義：** 中間特徵圖的大小 ($C \times H \times W$)。
- **真正瓶頸：** 雖然 MobileNet 減少了參數，但其**峰值激活值**仍然很高，這是物聯網/微控制器 (IoT/MCU) 部署的真正瓶頸。
- **總和 vs. 峰值：** 訓練需要*總激活值*（儲存用於計算梯度）；推論只需要*峰值激活值*（所有層中輸入+輸出最大的那層）。
- **U 型分佈：** 激活值在初期很大（高解析度），後期很小（高通道數，但低解析度）。權重則是在初期很小，後期很大。

---
*Archived by Gemini CLI on 2026-05-18*