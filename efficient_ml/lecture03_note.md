# EfficientML.ai 第三講 對話紀錄 (完整版) - 2026-05-20

## 課程主題
**MIT 6.5940 (Fall 2023) Lecture 3: Pruning and Sparsity (Part I)**

---

## 完整對話紀錄

### 第一輪：讀取與摘要
**User:** 讀取 @.books/efficient-ml/EfficientML.ai Lecture 3 - Pruning and Sparsity (Part I) (MIT 6.5940, Fall 2023, Zoom recording).txt

**Gemini:** 這份文件是 MIT 6.5940 (Fall 2023) 課程 "EfficientML.ai Lecture 3 - Pruning and Sparsity (Part I)" 的課堂錄影逐字稿。以下是核心要點總結：
- **核心動機與背景**：模型需求超越硬體能力，DRAM 存取成本極高，剪枝旨在減少記憶體存取以節省能耗。
- **MLPerf 案例**：NVIDIA 在開放組透過剪枝、量化與蒸餾達到 4.5 倍加速。
- **剪枝基本概念**：將稠密網路轉化為稀疏網路。流程為：訓練 -> 剪枝 -> 微調。
- **剪枝粒度**：包含細粒度（Unstructured）與結構化（Structured, 如 Channel Pruning, NVIDIA 2:4 Sparsity）。
- **剪枝準則**：基於量級 (Magnitude)、縮放因子 (BN $\gamma$)、二階導數、活化值分析 (APoZ) 等。

### 第二輪：關於 MLPerf
**User:** MLPerf 的解釋

**Gemini:** 在這份逐字稿中，講者將 MLPerf 解釋為：
- **「AI 運算的奧運會」**：衡量硬體與演算法在給定精準度下執行速度的基準測試。
- **封閉組 (Closed Division)**：禁止修改模型，純比硬體與編譯器優化。
- **開放組 (Open Division)**：允許剪枝、壓縮、量化等各種創新，只要準確率達標。這是為了向學生證明，結合模型壓縮技術能榨出硬體數倍的極限效能。

### 第三輪：剪枝具體作法
**User:** 解釋一下剪枝的具體作法

**Gemini:** 根據課程內容，具體作法可分為三個層面：
1. **執行流程（迭代剪枝）**：預訓練 -> 剪枝 -> 微調 -> 重複循環。採取「剪一點、練一點」策略以恢復精度。
2. **粒度選擇**：決定刪除單位。
   - **非結構化**：隨機刪除單個權重，壓縮率高但難加速。
   - **結構化**：刪除整個 Channel 或 Row，易於現成硬體加速。
   - **模式化**：如 2:4 Sparsity，兼顧靈活性與規律性。
3. **挑選準則**：決定誰該走。
   - **數值大小**：絕對值小於門檻的權重設為 0。
   - **縮放因子**：利用 BN 層的 $\gamma$。
   - **活化值分析**：利用 APoZ（神經元輸出為 0 的頻率）。
   - **局部重建誤差**：針對單層最小化誤差。

### 第四輪：剪枝的對象
**User:** 剪枝是剪掉了那些東西

**Gemini:** 具體被剪掉的東西分為兩大類：
1. **權重 (Weights) / 突觸 (Synapses)**：將矩陣中的權重強制設為 0，切斷神經元間的訊號傳遞。
2. **活化值 (Activations) / 神經元 (Neurons)**：直接拔除整顆神經元或通道。剪掉一個神經元等同於一併剪掉所有與其相連的輸入與輸出權重。

### 第五輪：剪枝粒度定義
**User:** pruning granularity 是什麼

**Gemini:** 剪枝粒度指的是「每次剪枝操作所作用的最小結構單位」：
- **細粒度 (Fine-grained)**：針對單一權重。壓縮極限最高，但因 0 分佈不規律導致硬體極難加速。
- **模式剪枝 (Pattern-based)**：如 2:4 Sparsity，規定在特定範圍內的稀疏比例。
- **粗粒度 (Coarse-grained)**：針對 Channel 或整行/列。對精度傷害較大，但因矩陣依然整齊，所有現成硬體（手機、GPU）都能立刻加速。

---
*Archived by Gemini CLI on 2026-05-20*
