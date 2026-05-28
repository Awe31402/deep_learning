# 作業 1 (Homework 1)

**6.7960 / 6.S898 深度學習 (Deep Learning)**
**2024 年秋季 (Fall 2024)**

---

**說明 (Instructions)**：本作業總分為 29 分。每道題目均標有其分值。部分題目**不計分**，包括所有加分題（bonus questions）。您可以根據需要選擇手繪或電腦生成的圖表，但請務必確保重要趨勢清晰可見。

**符號說明 (Notation)**：我們將使用課程網頁上的數學符號。例如，$c$ 是純量，$\mathbf{b}$ 是向量，$\mathbf{W}$ 是矩陣。鼓勵（但不強求）您在排版提交的作業中遵循此符號規範，或在手寫回答中盡最大努力遵守（將向量加粗可能比較困難 :））。

**數學入門 (Math primer)**：有幾個問題使用了「凸性 (convexity)」、「不連續性 (discontinuity)」和「可微性 (differentiability)」等術語。要解決這些問題，只需要對這些概念有非正式的理解即可。例如，$\text{ReLU}(x)$ 是連續的，因為在畫它時不需要將筆尖從紙上提起。$\text{ReLU}(x)$ 並非處處可微，因為它在 $x = 0$ 處有一個「折點 (kink)」。

---

## 逼近能力 (Approximation) (14分)

在本節中，除非另有說明，我們考慮由具有單個實數輸入和單個實數輸出的 ReLU 網路所表示的函數。令 $l$ 表示層數。例如，一個具有 $l = 2$ 層的網路可以寫成：
$$f(x; \mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2) = \mathbf{W}_2 \text{ReLU}(\mathbf{W}_1 x + \mathbf{b}_1) + \mathbf{b}_2$$
其中輸入 $x$ 和輸出 $f(x; \mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2)$ 均為純量，除非另有說明（而 $l$ 是權重矩陣的數量）。

1. **(1分)** 考慮一個寬度為 $k$（即有 $k$ 個隱藏神經元）、具有權重矩陣 $\mathbf{W}_1, \mathbf{W}_2$ 和偏置向量 $\mathbf{b}_1, \mathbf{b}_2$ 的兩層 ReLU 網路。$\mathbf{W}_1$ 是 $\mathbb{R}^{k \times 1}$ 中的矩陣——寫出 $\mathbf{W}_2, \mathbf{b}_1$ 和 $\mathbf{b}_2$ 的形狀 (shapes)。

2. **(1分)** 寫出具有 $l = 3$ 層的 ReLU 網路的表達式。

3. **(1分)** 回答「是」或「否」：對於一個具有 $l \ge 2$ 層的 ReLU 網路，一般情況下，網路輸出相對於輸入 $x$ 是凸的嗎？是凹的嗎？
   * 凸 (Convex)：(是/否)
   * 凹 (Concave)：(是/否)

4. **(5分)** 考慮一個具有 $l$ 層、每層寬度為 $k$ 的 ReLU 網路。
   * (a) **(1分)** 輸出相對於輸入最多可以有多少個不連續點？
   * (b) **(1分)** 單選題。作為輸入的函數，輸出總是：
     * (A) 線性的 (linear)
     * (B) 分段線性的 (piecewise-linear)
     * (C) 多項式的 (polynomial)
   * (c) **(2分)** 一般情況下，該函數在每個輸入點都可微嗎？如果是，為什麼？如果不是：
     * i. 該函數不可微的最少輸入點數是多少？
     * ii. 對於 $l = 2$ 層，該函數不可微的最大輸入點數是多少？
     * iii. 對於一般的層數 $l$，該函數不可微的最大輸入點數是多少？單選題：
       * (A) 關於 $l$ 為常數 (Constant in $l$)
       * (B) 關於 $l$ 為線性 (Linear in $l$)
       * (C) 關於 $l$ 為多項式 (polynomial in $l$)
       * (D) 關於 $l$ 為指數級 (exponential in $l$)

       **提示 (Hint)**：
       * 很難推導出精確答案，因此請專注於漸近行為 (asymptotics)。
       * 不可微點與線性區域 (linear regions) 有何關係？
       * 每個神經元都是前一層輸出的一個分割超平面 (separating hyperplane)。
       * 假設前一層的每個線性區域都被當前層的某個超平面分割（成兩個）。增加一層會如何影響線性區域的數量？

     * iv. 回想課堂上提到的，具有足夠寬度 $k$ 的 2 層網路（$l = 2$）是一個萬能逼近器 (universal approximator)。基於您對前面問題的回答，更深層的 ReLU 網路（$l > 2$）在逼近某些函數時，是否可能（在神經元 / 隱藏單元總數方面）更有效率？
   * (d) **(1分)** 一般情況下，如果使用 $\tanh$ 非線性激活函數代替 ReLU，該函數在每個輸入點都可微嗎？如果是，為什麼？如果不是，請重新回答 (c) 部分的子問題。

5. **(2分)** 對於一個寬度為 2 且無偏置（即 $\mathbf{b}_1$ 和 $\mathbf{b}_2$ 全為零）的 2 層 ReLU 網路，我們的目標是尋找 $\mathbf{W}_1$ 和 $\mathbf{W}_2$，使得對應的網路具有不同的平滑性質。對於每種情況：
   * 如果存在 $\mathbf{W}_1$ 和 $\mathbf{W}_2$ 使得對應的性質在輸入 $x \in [-5, 5]$ 上成立，請給出此類 $\mathbf{W}_1$ 和 $\mathbf{W}_2$ 的範例，並為該範例提供該函數在 $x \in [-5, 5]$ 上的圖像。
   * 如果不存在這樣的 $\mathbf{W}_1$ 和 $\mathbf{W}_2$，請解釋原因。
   * (a) **(1分)** 該函數是線性的。
   * (b) **(1分)** 該函數具有 2 個不可微點。
   * (c) **(加分題；0分)** 該函數是凸的（且非線性）。
   * (d) **(加分題；0分)** 該函數既非凸也非凹。

6. **(加分題；神經網路的非凸性；0分)** 考慮一個 2 層 ReLU 網路。畫出一個例子，以展示網路輸出相對於**網路參數**並不保證是凸的（或凹的）。您可以自行選擇網路寬度（儘管寬度為 2 就足夠了）。
   特別地，尋找一個固定的輸入以及參數空間中的一條線性路徑，並繪製在保持該固定輸入的同時，沿著該線性路徑變化參數時的網路輸出。該圖像應該既非凸也非凹。

7. **(4分) 邏輯閘 ReLU 網路**
   **提示 (Hint)**：$\text{ReLU}(x)$ 非線性激活函數就像是 $x = 0$ 處的「分支」操作。您能否找到一組權重，使得期望的決策邊界對應於 ReLU 的零輸入？
   * (a) **(OR 閘；2分)** 建構一個 2 層、寬度為 2 的 ReLU 網路，其具有 $\mathbb{R}^2$ 中的 2 維輸入，滿足：
     $$f(\mathbf{x}; \mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2) > 0 \iff x_1 > 0 \text{ 或 } x_2 > 0 \quad (1)$$
     寫出具有顯式 $\mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2$ 的 $f$ 的代數公式。
     **提示 (Hint)**：提醒您，輸入不是布林值（boolean）。
   * (b) **(XOR 閘；2分)** 建構一個最多 3 層、每層寬度最多為 4 且具有 $\mathbb{R}^2$ 中的 2 維輸入的 ReLU 網路，滿足：
     $$f(\mathbf{x}; \mathbf{W}_1, \mathbf{W}_2, \mathbf{b}_1, \mathbf{b}_2) > 0 \iff (x_1 < 0 \text{ 且 } x_2 > 0) \text{ 或 } (x_1 > 0 \text{ 且 } x_2 < 0) \quad (2)$$
     寫出具有顯式權重矩陣和偏置向量的 $f$ 的代數公式。
     **提示 (Hint)**：這是一個更具挑戰性的問題。思考如何實現一個 AND 閘。
   * (c) **(加分題；NAND 閘與功能完備性；0分)** 寫出一個實現 NAND 閘的 ReLU 網路。這對於使用 ReLU 網路表示任何布林函數的可能性告訴了我們什麼？

---

## 反向傳播 (Backpropagation) (3分)

8. **(3分)** 令 $\mathbf{W}$ 表示一個 $d \times d$ 的實數矩陣，並考慮以下方程系統：
   $$\mathbf{y} = \mathbf{W}\mathbf{x} \quad (3)$$
   $$\mathbf{u} = \text{ReLU}(\mathbf{y}) \quad (4)$$
   $$\mathbf{v} = \mathbf{u} + \mathbf{W}\mathbf{u} \quad (5)$$
   $$\mathcal{L} = \frac{1}{2} \|\mathbf{v}\|_2^2 \quad (6)$$
   請注意，$\mathbf{x}, \mathbf{y}, \mathbf{u}$ 和 $\mathbf{v}$ 必須都是 $\mathbb{R}^d$ 中的向量，這些方程才有意義。由於 $\|\mathbf{v}\|_2^2$ 表示向量 $\mathbf{v}$ 的標準平方歐幾里得範數 (squared Euclidean norm)，因此 $\mathcal{L}$ 是一個純量。
   * (a) **(1分)** 證明：
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{ij}} = \sum_{m=1}^d \mathbf{v}_m \cdot \frac{\partial \mathbf{v}_m}{\partial \mathbf{W}_{ij}}$$

   以與 (a) 部分類似的方式，可以推導出以下額外的關係式：
   * $$\frac{\partial \mathbf{v}_m}{\partial \mathbf{W}_{ij}} = \frac{\partial \mathbf{u}_m}{\partial \mathbf{W}_{ij}} + \delta_{im} \cdot \mathbf{u}_j + \sum_{l=1}^d \mathbf{W}_{ml} \frac{\partial \mathbf{u}_l}{\partial \mathbf{W}_{ij}}$$
   * $$\frac{\partial \mathbf{y}_k}{\partial \mathbf{W}_{ij}} = \delta_{ik} \mathbf{x}_j$$
     其中「克羅內克函數 (Kronecker delta)」定義為：
     $$\delta_{ik} = \begin{cases} 1 & \text{若 } i = k \\ 0 & \text{若 } i \neq k \end{cases}$$
   * $$\frac{\partial \mathbf{u}_l}{\partial \mathbf{y}_k} = \delta_{lk} \cdot \Theta(\mathbf{y}_k)$$
     其中 $\Theta$ 表示赫維賽德階躍函數 (Heaviside step function)，定義為：
     $$\Theta(\mathbf{y}_k) = \begin{cases} 1 & \text{若 } \mathbf{y}_k \ge 0 \\ 0 & \text{若 } \mathbf{y}_k < 0 \end{cases}$$

   * (b) **(2分)** 令 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ 表示其元素為 $\left(\frac{\partial \mathcal{L}}{\partial \mathbf{W}}\right)_{ij} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{ij}}$ 的矩陣。使用給定的關係式以及您在 (a) 部分的答案，證明：
     $$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{v} \otimes \mathbf{u} + \text{diag}(\Theta(\mathbf{y}))(\mathbf{I} + \mathbf{W}^\top)\mathbf{v} \otimes \mathbf{x}$$
     其中 $\otimes$ 是外積 (outer product)，而 $\text{diag}$ 將其輸入塑形為對角矩陣 (diagonal matrix)。

   這種對 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ 的表達式優勢在於它很容易在 PyTorch 中編寫程式碼，並且能高效利用矩陣乘法原語，這些原語在 GPU 上有高度優化、並行化的實現。

---

## PyTorch (0分)

9. **(不計分；0分)** 在[此處](https://github.com/davidbau/how-to-read-pytorch/tree/master)完成 PyTorch 教學 Colab 筆記本。在繼續下一節之前，您應該至少完成以下筆記本（「張量算術 Tensor Arithmetic」和「網路模組 Network Modules」）。

---

## CIFAR-10 分類 (CIFAR-10 Classification) (12分)

在本節中，我們將在[此 Colab 筆記本](https://colab.research.google.com/drive/1H_Htddamebxg7trHfaO9uGiG_IuCi2b9?usp=sharing)中進行操作，訓練一個網路來分類手寫數字數據集，CIFAR-10 [Krizhevsky, 2009, Torralba et al., 2008]。
以下問題 10-15 在 Colab 筆記本中有詳細說明。請在同一個 PDF 提交檔案中包含您添加的程式碼行、文字輸出以及所有相關圖表。
要從 Colab 下載圖表，請在按住 Shift 鍵的同時右鍵單擊圖像。

**注意 (Note)**：已為您提供了許多骨架程式碼 (skeleton code)。請務必仔細閱讀並理解它們。隨著您對深度學習程式碼結構越來越熟悉，我們在未來的作業中提供的骨架程式碼將會減少。

10. **(建構神經網路；4分)** 完成模組類別中未完成的 forward 和 backward 定義，每個定義使用 $\le 5$ 行程式碼。我們預計這個問題會花費較多時間，因為您被要求推導並實現多個組件的反向傳播。**提示 (Hint)**：回想一下，對於線性層，前向傳播的形式為：$\text{out} = \mathbf{W}\mathbf{x} + \mathbf{b}$，而反向傳播需要我們求出 $\frac{\partial \mathcal{L}}{\partial \text{dout}}, \frac{\partial \mathcal{L}}{\partial \mathbf{x}}, \frac{\partial \mathcal{L}}{\partial \mathbf{b}}$。ReLU 和損失層可以類似地計算。
    * (a) **(1分)** `Linear.forward`
    * (b) **(1分)** `Linear.backward`
    * (c) **(1分)** `ReLU.backward`
    * (d) **(1分)** `CrossEntropyLoss.backward`

11. **(訓練循環；2分)** 完成 `train_epoch` 和 `evaluate` 函數中缺失的部分，每個函數使用 $\le 5$ 行程式碼。

12. **(訓練曲線 1；2分)** 訓練模型 30 個 epochs 並繪製學習曲線。對圖表中的任何有趣觀察進行評論。

13. **(萬能逼近；1分)** 神經網路是萬能逼近器，這意味著我們總能找到一個擬合訓練集的網路。您認為我們為什麼沒有得到完美的訓練準確率？寫下一些提高訓練準確率的想法。完美的訓練準確率是我們所需要的全部嗎？

14. **(Data augmentation 數據增強；1分)** 編寫程式碼以創建訓練和驗證數據集，其中只有訓練集具有隨機裁剪增強 (random cropping augmentation)（規格在 Colab 中說明）。視覺化數據增強的效果。
    提供您的程式碼（$\le 5$ 行）和您的視覺化結果。

15. **(訓練曲線 2；2分)** 在新的訓練數據集上訓練模型並繪製學習曲線。評論您觀察到的與先前曲線的任何差異。

---

## 參考文獻 (References)

* Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.
* Antonio Torralba, Rob Fergus, and William T Freeman. 80 million tiny images: A large data set for nonparametric object and scene recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2008.

---
MIT OpenCourseWare
https://ocw.mit.edu

6.7960 Deep Learning
Fall 2024

有關引用這些材料或我們的使用條款的資訊，請造訪：https://ocw.mit.edu/terms
