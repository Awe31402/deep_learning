# EfficientML.ai 第三講 對話紀錄 (完整版) - 2026-05-20

## 課程主題
**MIT 6.5940 (Fall 2023) Lecture 3: Pruning and Sparsity (Part I)**

---

## 完整對話紀錄

### 第一輪：讀取與摘要
**User:** 讀取 @.books/efficient-ml/EfficientML.ai Lecture 3 - Pruning and Sparsity (Part I) (MIT 6.5940, Fall 2023, Zoom recording).txt

**Gemini:** 這份文件是 MIT 6.5940 (Fall 2023) 課程 **EfficientML.ai** 課程第三堂課的逐字稿，主題是 **「剪枝與稀疏性 (Pruning and Sparsity) 第一部分」**。

以下是內容摘要：

### 1. 動機與背景 (Motivation)
*   **模型太大：** AI 模型持續增長，硬體供應與模型需求之間存在巨大缺口。
*   **記憶體存取昂貴：** 存取 DRAM 的能耗比運算高出兩個數量級。剪枝可以減少權重和激活值，進而節省能源和延長電池壽命。
*   **生物學啟發：** 人類大腦在幼年時期突觸增加，但在青少年時期會經歷剪枝過程，使大腦更高效。

### 2. 剪枝顆粒度 (Pruning Granularity)
*   **非結構化剪枝 (Unstructured Pruning)：** 隨機移除單個權重，靈活性最高，但硬體加速困難（不規則）。
*   **結構化剪枝 (Structured Pruning)：** 
    *   **通道剪枝 (Channel Pruning)：** 移除整個卷積通道，最容易在現有硬體上加速，但壓縮率相對較低。
    *   **模式剪枝 (Pattern-based Pruning)：** 如 NVIDIA Ampere 架構的 **2:4 稀疏性**，兼顧靈活性與硬體加速。
    *   **向量/卷積核剪枝 (Vector/Kernel-level Pruning)：** 介於兩者之間。

### 3. 剪枝準則 (Pruning Criteria)
*   **權重剪枝 (Weight Pruning)：**
    *   **基於數值大小 (Magnitude-based)：** 移除絕對值較小的權重（$L_1, L_2$ Norm）。這是工業界最常用的方法。
    *   **縮放因子 (Scaling Factor)：** 利用 Batch Normalization 層中的 $\gamma$ 參數來決定通道的重要性。
    *   **二階導數 (Second-order)：** 基於 Hessian 矩陣（Taylor expansion）判斷對損失函數影響最小的參數。
*   **神經元/激活剪枝 (Neuron/Activation Pruning)：**
    *   **APoZ (Average Percentage of Zero activation)：** 統計激活值為零的頻率，移除經常為零的通道。
*   **回歸剪枝 (Regression-based Pruning)：** 最小化單層的重建誤差（Reconstruction Error），這在 LLM 時代特別有用，因為不需要對整個網絡進行端到端的反向傳播。

### 4. 流程與演示 (Workflow & Demo)
*   **流程：** 訓練 (Train) -> 剪枝 (Prune) -> 微調 (Fine-tune) 以恢復精度。
*   **Demo 亮點：**
    *   使用 MNIST 數字識別模型。
    *   剪掉 80% 的權重後精度會下降（例如從 99% 掉到 76%）。
    *   經過微調後，即使只有 20% 的參數，精度也能恢復到 99% 左右。
    *   若剪枝率過高（如 99%），則精度無法恢復。

### 5. 課堂 Q&A 亮點
*   **剪枝 vs. Dropout：** Dropout 是訓練時隨機丟棄以防止過擬合；剪枝是為了推理時減少運算量。
*   **加速問題：** 非結構化剪枝需要特殊的內核 (Kernels) 才能在硬體上產生實際的加速效果，否則只是在運算時套用 Mask。

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

### 第六輪：決定剪枝對象的準則
**User:** 如何決定那些神經元或突觸該被剪枝

**Gemini:** 根據這堂課的內容，決定哪些神經元（Neurons）或突觸（Synapses / Weights）該被剪枝，主要依賴於**「剪枝準則 (Pruning Criteria)」**。課程中介紹了幾種主要的方法：

#### 1. 突觸剪枝 (Synapse / Weight Pruning)
這是針對個別權重或一組權重進行剪枝，判斷標準包括：

*   **基於數值大小 (Magnitude-based)：**
    *   **概念：** 最直觀且工業界最廣泛使用的方法。假設「權重絕對值越小，影響力越低」，因此直接移除絕對值最小的權重。
    *   **進階 (L1 / L2 Norm)：** 如果是進行結構化剪枝（例如剪掉整行或整個矩陣），可以計算該組權重的 L1 Norm（絕對值總和）或 L2 Norm（平方和開根號），然後淘汰數值最小的整組權重。
*   **利用縮放因子 (Scaling Factor)：**
    *   **概念：** 卷積神經網路通常會使用 Batch Normalization (BN) 層。BN 層中有一個可學習的縮放參數 $\gamma$ (Gamma)。
    *   **做法：** 訓練時加入懲罰項，促使 $\gamma$ 趨近於零。剪枝時，直接檢查每個通道對應的 $\gamma$ 值，如果 $\gamma$ 非常小，代表該通道的影響力很低，就可以將整個通道剪除。
*   **基於二階導數 / 泰勒展開 (Second-order based / Taylor Expansion)：**
    *   **概念：** 評估「移除某個權重後，對整體損失函數 (Loss Function) 會造成多大變化」。
    *   **做法：** 透過泰勒展開式並計算 Hessian 矩陣。為了簡化運算，通常會假設一階梯度已收斂為零，並忽略高階項，找出那些對二階導數影響最小的參數進行參數進行剪除。
*   **基於局部重建誤差 (Regression-based)：**
    *   **概念：** 不看整體的 Loss，而是專注於「單一神經網路層」。
    *   **做法：** 嘗試移除某些通道，並調整剩下的權重，使得該層「剪枝前的輸出」與「剪枝後的輸出」誤差最小化。這種方法不需要對整個龐大的網路進行反向傳播，在處理巨大的大型語言模型 (LLM) 時特別有效。

#### 2. 神經元剪枝 (Neuron / Activation Pruning)
剪掉一個神經元，等同於**把與該神經元相連的所有輸入與輸出突觸（權重）全部移除**，這是一種粗顆粒度（Coarse-grained）的結構化剪枝。判斷神經元是否該被剪除的標準：

*   **APoZ (Average Percentage of Zero activation)：**
    *   **概念：** 在使用 ReLU 激活函數的網路中，負數會被轉換為零。如果某個神經元或某個卷積通道的輸出「經常是零」，代表它對後續網路的貢獻極低。
    *   **做法：** 輸入一批資料（Batch），統計各個通道輸出為 0 的比例。找出那些 **「平均零激活率最高」** 的通道並將其剪除，因為把它們設為零對最終預測結果的影響微乎其微。

### 第七輪：二階導數與泰勒展開詳解
**User:** 詳細說明 基於二階導數 / 泰勒展開 (Second-order based / Taylor Expansion)

**Gemini:** 基於二階導數（Second-order derivative）與泰勒展開（Taylor Expansion）的剪枝方法，其核心精神是：**「我們不只看權重本身的絕對值大小，而是評估如果把這個權重移除（設為 0），對整體損失函數（Loss Function）會造成多大的影響。」**

我們希望剪掉那些「對 Loss 影響最小」的權重。這種方法比單純看絕對值（Magnitude-based）更具理論基礎，因為有時候絕對值小的權重，對最終預測結果可能扮演著關鍵的調節角色。

以下是根據課堂內容與數學推導的詳細說明：

#### 1. 核心概念與目標
假設原本神經網路的權重為 $W$，目前的損失為 $L(W)$。
當我們把某個權重 $w_i$ 剪掉，相當於對原本的權重施加了一個擾動（Perturbation）$\Delta W$。因為我們是把它變成 0，所以對於該權重而言，$\Delta w_i = -w_i$。

剪枝後的損失變成 $L(W + \Delta W)$。
我們的目標是找出一個 $\Delta W$，使得**損失的變化量 $\Delta L$ 越小越好**：
$$ \Delta L = L(W + \Delta W) - L(W) $$

#### 2. 利用泰勒展開式 (Taylor Expansion)
為了估算 $\Delta L$，我們對 $L(W + \Delta W)$ 在 $W$ 的位置進行泰勒展開：

$$ \Delta L \approx \underbrace{g^T \Delta W}_{\text{一階項 (First-order)}} + \underbrace{\frac{1}{2} \Delta W^T H \Delta W}_{\text{二階項 (Second-order)}} + \underbrace{O(\|\Delta W\|^3)}_{\text{高階項 (Higher-order)}} $$

這裡的變數定義如下：
*   $g$ 是損失函數對權重的一階導數（Gradient，梯度矩陣）。
*   $H$ 是損失函數對權重的二階導數矩陣，稱為 **海森矩陣 (Hessian Matrix)**。
*   $O(\|\Delta W\|^3)$ 是三階以上的更高階項。

#### 3. 三大假設與簡化
要計算上述公式非常困難，因此在經典演算法（例如 Yann LeCun 在 1990 年提出的 Optimal Brain Damage, OBD）中，引入了三個關鍵假設來大幅簡化公式：

**假設一：忽略高階項 (Higher-order terms)**
假設擾動 $\Delta W$ （即移除的權重）足夠小，因此三階以上的項 $O(\|\Delta W\|^3)$ 可以忽略不計。

**假設二：網路已收斂 (Network is converged)**
我們通常是在網路「訓練完成（收斂）」後才進行剪枝。當網路收斂到局部最小值時，損失函數相對於權重的一階梯度 $g$ 應該趨近於 0（$g \approx 0$）。
因此，**一階項 $g^T \Delta W$ 也被消除了**。

**假設三：參數互相獨立 (Parameters are independent)**
這是最關鍵且最強的假設。假設權重之間互不影響，這意謂著海森矩陣 $H$ 是一個**對角矩陣 (Diagonal Matrix)**，即除了對角線元素外，所有交叉項（Cross-terms）都是 0。

#### 4. 最終的顯著性評分 (Saliency / Importance Score)
經過上述三個簡化步驟後，整個 $\Delta L$ 的方程式被極度簡化，只剩下對角線上的二階項：

$$ \Delta L \approx \frac{1}{2} \sum_i h_{ii} (\Delta w_i)^2 $$

因為我們是把權重 $w_i$ 變成 0，所以 $\Delta w_i = -w_i$。代入後得到每個權重 $w_i$ 對損失函數影響的「顯著性評分（Saliency Score）」：

$$ S_i = \frac{1}{2} h_{ii} w_i^2 $$

*   $w_i^2$：權重原本大小的平方。
*   $h_{ii}$：該權重對應的二階導數（海森矩陣對角線元素），代表該權重所在位置的曲率 (Curvature)。曲率越大，代表稍微移動這個權重，Loss 就會急劇上升。

**結論做法：**
計算所有權重的 $S_i$，然後**剪除 $S_i$ 最小的權重**。

#### 5. 為什麼不用精確的海森矩陣？(課堂提及的限制)
雖然這個數學模型很漂亮，但課堂中提到：**「計算精確的海森矩陣 $H$ 在計算上是非常沉重的 (computationally heavy)。」**

*   現代神經網路動輒幾千萬到數百億個參數。如果網路有 $N$ 個參數，海森矩陣的大小就是 $N \times N$。
*   對於一個 1,000 萬參數的模型，海森矩陣會有 $10^{14}$ 個元素，這在記憶體和運算上都是不可能實現的。

因此，後來許多研究（如 Optimal Brain Surgeon, OBS 或近代的一些二階近似方法）都是在致力於：
1.  提出**各種近似 (Approximations)** 來估算海森矩陣的對角線元素，或者其逆矩陣 $H^{-1}$，而不實際計算完整的矩陣。
2.  這也是為什麼工業界往往更偏好極度簡單的「基於數值大小 (Magnitude-based)」的方法，雖然它在理論上次於二階方法，但它「免費且不需要額外計算成本」。

### 第八輪：APoZ 範例與 Channel 3 剪枝原因
**User:** APoZ 的範例中為什麼Channel 3 被剪枝

**Gemini:** 在 APoZ（Average Percentage of Zeros）的範例中，**Channel 3 被剪枝的原因是因為它的「零激活率 (Zero Activation Rate)」最高**。

APoZ 的邏輯是：如果一個通道輸出的數值幾乎全是 0，代表這個通道對後續的網路幾乎沒有貢獻（因為 0 乘以任何權重還是 0）。

根據課堂逐字稿中的具體計算範例，我們可以拆解如下：

#### 1. 設定與數據
*   **範例結構：** 4x4 的解析度（每層 16 個像素/元素），測試了 2 張圖片（Batch Size = 2）。
*   **總元素量：** $2 \times 4 \times 4 = 32$ 個元素。
*   **統計結果：**
    *   **Channel 1：** 第一張圖有 5 個零，第二張圖有 6 個零 $\rightarrow$ 共 **11** 個零。
    *   **Channel 2：** 第一張圖有 5 個零，第二張圖有 7 個零 $\rightarrow$ 共 **12** 個零。
    *   **Channel 3：** 逐字稿提到共有 **14** 個零（分佈在兩張圖片中）。

#### 2. 計算 APoZ 分數
APoZ 的計算公式是：$\text{總共出現 0 的次數} / \text{總元素個數}$。
*   **Channel 1 的 APoZ：** $11 / 32 \approx 34.4\%$
*   **Channel 2 的 APoZ：** $12 / 32 = 37.5\%$
*   **Channel 3 的 APoZ：** **$14 / 32 \approx 43.8\%$**

#### 3. 為什麼選 Channel 3？
教授在課堂上解釋道：
> 「既然 Channel 3 擁有最多的零（14/32，比例最高），這代表如果你把它**全部設為零（即剪掉它）**，對最終結果造成的改變會是最小的。」

**總結：**
APoZ 越高 $\rightarrow$ 該通道越「廢」（資訊量越少） $\rightarrow$ 優先剪掉。在這個範例中，Channel 3 的零激活佔比最高（43.8%），所以它是首選的剪枝對象。

### 第九輪：局部重建誤差 (Regression-based) 詳解
**User:** 詳細說au/6詳細說明 基於局部重建誤差 (Regression-based)

**Gemini:** 基於**局部重建誤差（Regression-based / Reconstruction-based）**的剪枝方法，其核心精神是：**「與其從頭到尾計算整個網路的誤差，不如只看『單一網路層』，目標是讓這一層剪枝後的輸出，盡可能等於剪枝前的輸出。」**

這種方法在面對當今參數動輒上百億、千億的大型語言模型（LLM）時特別有效。以下是根據課堂內容的詳細說明：

#### 1. 為什麼需要 Regression-based 剪枝？
傳統的微調（Fine-tuning）或剪枝評估，是將資料輸入網路，經過前向傳播（Forward Pass）得到最終結果，計算 Loss，再透過反向傳播（Backpropagation）更新所有層的權重。

*   **痛點：** 對於像 175B（1750億）參數的模型，要執行完整的反向傳播極度消耗運算資源，需要極大量的 GPU 記憶體，非常困難。
*   **解法：** 把問題「局部化（Locally）」。我們一次只專注於一個 Layer（神經網路層），這大幅降低了運算複雜度。

#### 2. 核心邏輯與目標
假設我們有一個卷積層或全連接層：
*   **輸入：** $X$
*   **原本的權重（稠密）：** $W$
*   **原本的輸出：** $Y = X \times W$

現在我們要對 $W$ 進行剪枝，得到一個稀疏的權重 $W_{pruned}$。
**我們的目標是：** 找到一組最好的 $W_{pruned}$，使得新的輸出 $Y_{pruned} = X \times W_{pruned}$ 盡可能接近原本的輸出 $Y$。

換句話說，我們要最小化以下的回歸誤差（Regression Error）：
$$ \text{Minimize} \quad ||Y - X \times W_{pruned}||_2^2 $$
*(這就是為什麼它被稱為 Regression-based，因為它本質上是在解一個最小平方法的回歸問題)*

#### 3. 具體的數學拆解與做法 (以 Channel Pruning 為例)
教授在課堂上舉了一個矩陣乘法的例子來說明如何選擇要剪掉哪個 Channel（通道）：

1.  **拆解矩陣乘法為外積 (Outer Product) 的總和：**
    一個矩陣相乘 $X \times W$ 可以被視為多個局部矩陣（Partial Matrices）加總的結果。
    假設 $X$ 有 4 個通道，每一層的計算可以拆解成 4 個「輸入通道與對應權重的外積」加總。
    $$ Y = \sum_{i=1}^{C_{in}} X_i \otimes W_i $$
    *(這裡的 $X_i$ 是一欄， $W_i$ 是一列，相乘會得到一個局部輸出矩陣)*

2.  **引入可學習的係數 $\beta$ (Learnable Coefficient)：**
    為了解決到底該丟掉哪一個外積（通道），我們在每個局部矩陣前面乘上一個可學習的係數 $\beta_i$：
    $$ Y_{pruned} = \sum_{i=1}^{C_{in}} \beta_i (X_i \otimes W_i) $$
    我們的限制是：**$\beta$ 中非零的數量必須小於我們設定的剪枝閾值**（例如只保留 3 個通道，就只有 3 個 $\beta$ 可以是非零的）。如果 $\beta_i = 0$，就代表那個通道被剪掉了。

3.  **迭代最佳化 (Iterative Optimization)：**
    這是一個很難直接解的問題，所以採用交替最佳化的方式：
    *   **Step A (學習 $\beta$)：** 先固定權重 $W$，去尋找哪幾組 $\beta$ 設為 0 對整體的誤差影響最小，藉此決定「要剪掉哪些通道」。
    *   **Step B (學習 $W$)：** 決定好哪些 $\beta$ 是 0（決定好要剪哪些通道）之後，固定 $\beta$，然後**調整剩下的權重 $W$**，讓新的輸出 $Y_{pruned}$ 更接近原本的 $Y$。
    
    這樣反覆交替執行 Step A 和 Step B，直到誤差收斂為止。

#### 4. Regression-based 剪枝的優缺點
**優點 (Pros)：**
*   **極省運算資源：** 因為不需要計算整個網路的梯度 (Gradient)，也不需要 Backpropagation 到頭。
*   **適合巨型模型：** 是目前對 LLM (Large Language Models) 進行壓縮的關鍵技術之一，像是後來的 SparseGPT 或 OBC 等先進演算法都受到了這種 layer-wise reconstruction 思想的啟發。

**缺點 (Cons)：**
*   **缺乏全局視野 (No holistic picture)：** 因為它只專注於把「當前這一層」的輸出還原，但這不保證所有層組合起來後，最終的分類或生成結果是最好的。局部的最優解不等於全局的最優解。

---
*Archived by Gemini CLI on 2026-05-20*
