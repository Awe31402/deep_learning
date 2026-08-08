import json
import re

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

md_out = []

md_out.append("# Jupyter 筆記本：Lab3.ipynb\n")
md_out.append("# **MIT 6.5940 EfficientML.ai Lab 3: Neural Architecture Search (神經網路架構搜尋)**\n")
md_out.append("by MIT HAN Lab\n")

# Cell-by-cell translations for Markdown cells
translations = {
    1: """## 簡介 (Introduction)

本 Colab 筆記本提供了 Lab 3：神經網路架構搜尋 (Neural Architecture Search, NAS) 的程式碼和框架。在此實驗中，你將學習如何搜尋可以在微控制器 (Microcontroller) 上高效運行的小型神經網路。你可以在這裡完成你的解答。""",

    3: """在很長一段時間裡，研究人員都是手動設計神經網路架構。神經網路架構的設計空間 (Design Space) 非常龐大：包含層數 (#layers)、通道寬度 (#channel width)、分支數 (#branches)、卷積核大小 (kernel sizes) 以及輸入解析度 (input resolutions)。因此，手動微調這些設計選項極其困難。

另一方面，**神經網路架構搜尋 (NAS)** 可以幫助研究人員在各種效率和準確率約束下，自動微調這些設計選項。結果是，它大大節省了神經網路設計的工程成本，並有助於推動 AI 的普及與大眾化。在本次實驗中，我們將從頭開始帶你了解神經網路架構搜尋。""",

    5: """早期的 NAS 方法在設計空間中窮舉訓練候選網路，並使用基於 **RNN 的控制器 (RNN-based controllers)** 結合增強學習 (Reinforcement Learning) 來優化採樣策略。代表性方法包括 [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)、[NASNet](https://arxiv.org/abs/1707.07012) 和 [MNASNet](https://arxiv.org/abs/1807.11626)。這些方法通常計算成本極其昂貴，因為每個候選網路都必須從頭開始訓練，以便基於 RNN 的控制器能夠獲得獎勵訊號 (即候選網路的準確率)。

後來，研究人員開發了**可微分 NAS (Differentiable NAS)** 方法，例如 [DARTS](https://arxiv.org/abs/1806.09055)、[ProxylessNAS](https://arxiv.org/abs/1812.00332) 和 [FBNet](https://arxiv.org/abs/1812.03443)，這些方法大幅降低了訓練候選網路的總成本。DARTS 將每層的輸出建立模型為來自不同候選操作輸出的加權平均值，而 ProxylessNAS 通過在記憶體中僅保留兩條路徑而不是所有路徑，進一步降低了 DARTS 的記憶體成本。後來的 **One-Shot** 方法 (如 [Single Path One Shot](https://arxiv.org/abs/1904.00420)) 進一步發現，在訓練過程中每次僅保留一條路徑也是可行的。

儘管可微分 NAS 和 One-Shot NAS 比基於控制器的做法高效得多，但每次我們設計新神經網路時，仍然需要執行整個訓練、搜尋和微調流程。考慮到大量的邊緣設備 (例如截至 2018 年全球有 [超過 200 億個 IoT 設備](https://www.statista.com/statistics/471264/iot-number-of-connected-devices-worldwide/))，模型客製化 (Model Specialization) 依然會帶來巨大成本 (對於 ImageNet 數據集通常需要 200-300 GPU 小時)。""",

    7: """因此，在本次實驗中，我們參考了 [Once for All](https://arxiv.org/abs/1908.09791) (OFA) 方法，該方法可以大幅降低為不同設備客製化神經網路架構的成本。OFA 訓練一個大型的**超網路 (Super Network)**，其中包含了設計空間內所有的**子網路 (Sub-networks)**。如果我們直接從超網路中擷取子網路，它們無需從頭訓練即可達到與從頭訓練相近的準確率。因此，OFA 支援**無需重新訓練 (No Retrain)** 的直接部署。

此外，OFA 引入了**準率預測器與效率預測器 (Accuracy & Efficiency Predictors)**，以進一步減少子網路評估的成本。在本實驗中，你將學習如何建構這兩種預測器並執行快速的神經網路架構搜尋。""",

    8: """在本實驗中，你將學習如何利用 **OFA** 和**預測器**搜尋可以在極度受限的微控制器資源上高效運行的網路。微控制器是低成本、低功耗的硬體，被廣泛部署且應用極為廣泛。""",

    10: """但是微控制器嚴格的記憶體預算 (比 GPU 小 50,000 倍) 使得深度學習的部署非常困難。""",

    12: """本實驗主要分為兩個部分：**準確率與效率預測器** 以及 **架構搜尋**。

- 對於預測器部分，共有 ***4*** 個問題。其中 **Getting Started** 部分有 1 個問題 (5 分)，其餘 3 個問題 (30 分) 在 **Predictors** 部分。
- 對於架構搜尋部分，共有 ***6*** 個問題。""",

    13: """首先，安裝所需的套件並下載本實驗將使用的 [**Visual Wake Words (VWW)** 數據集](https://arxiv.org/abs/1906.05721)。""",

    16: """## **Getting Started：超網路與 VWW 數據集 (1 個問題，5 分)**""",

    17: """在本實驗中，我們將使用以 **Once-for-All (OFA)** 方式訓練的 **[MCUNetV2](https://arxiv.org/abs/2110.15352)** *超網路*。回想一下，*超網路* 是一個隨機化的大型神經網路，包含了設計空間內所有候選子網路。我們可以直接從超網路中提取子網路並評估其準確率。該準確率可以進一步用作指導神經網路設計的反饋訊號。OFA 超網路的優勢在於直接擷取的子網路可以獲得與從頭訓練相當 (甚至更好) 的效能。

MCUNetV2 是專為資源受限微控制器量身定制的高效神經網路家族。它採用基於 Patch 的推論 (Patch-based inference)、感受野重分配 (Receptive field redistribution) 以及系統-NN 聯合設計 (System-NN Co-design)，大幅提升了 [MCUNet](https://arxiv.org/abs/2007.10319) 的準確率與效率權衡。""",

    18: """我們首先在 VWW 數據集中視覺化一些樣本。這是一個從 [Microsoft COCO](https://arxiv.org/abs/1405.0312) 二次採樣得到的二元圖像分類數據集 (判斷圖像中是否存在人)。我們首先定義一個函式來建立驗證集上的數據載入器 (DataLoader)。

注意：函式 `build_val_data_loader` 有一個 `split` 參數。我們使用 `split = 0` (預設值) 表示驗證集 (不能直接用於架構搜尋)，`split = 1` 將用作保留的 minival 數據集 (用於產生準確率數據集並校正 BN 參數)。""",

    20: """使用該數據載入器建構器，我們能夠瀏覽 VWW 驗證集。你可以多次運行以下單元格以查看數據集中的不同圖像。""",

    22: """太棒了，現在你對數據集有了基本的了解。接下來讓我們構建 OFA 超網路！`OFAMCUNets` 超網路在 MCUNetV2 設計空間中包含了 $>10^{19}$ 個子網路。這些子網路由具有不同卷積核大小 (3, 5, 7) 和擴展比例 (3, 4, 6) 的 [反向 MobileNet 模組 (Inverted MobileNet blocks)](https://arxiv.org/abs/1801.04381) 組成。OFA 超網路還允許所有網路階段具備彈性深度 (基礎深度到 base_depth + 2)。最後，超網路支援 0.5$\times$、0.75$\times$ 或 1.0$\times$ 的全域通道縮放 (由 `width_mult_list` 指定)。""",

    24: """然後我們驗證檢查點 (Checkpoint) 已正確載入。我們將在 MCUNetV2 設計空間中採樣一些網路，並在 VWW 數據集中評估其準確率。評估過程將花費不到一分鐘的時間，預計你將看到大約 83.6-88.7% 的準確率。正如你所看到的，我們可以直接從設計空間中提取這些子網路，並在**無需訓練**的情況下非常快速地獲得其準確率。這是 Once-for-All (OFA) 超網路帶來的獨特優勢。

我們首先定義一個輔助函式 `evaluate_sub_network`，用於測試直接從超網路提取的子網路的準確率。""",

    26: """我們還提供了一個便捷的輔助函式來將子網路的架構視覺化。該函式接收子網路的配置並傳回代表該架構的圖像。""",

    28: """現在，讓我們將一些子網路視覺化並在 VWW 數據集中評估它們！我們提供了一個範例，從設計空間中隨機採樣一個子網路，並獲取其在 VWW 數據集上的準確率、MACs 和參數數量。我們還使用 `visualize_subnet` 將架構視覺化。

在架構視覺化中，每個模組 `MBConv{e}-{k}x{k}` 的圖例表示當前模組是擴展比例為 `e` 且深度可分離卷積層核大小為 `k` 的 Mobile Inverted Block。模組的不同顏色表示不同的卷積核大小，灰色模組是網路階段的分隔符。模組的不同寬度表示不同的擴展比例。我們還在每個模組附近標註了輸出解析度。

請注意，我們假設圖像解析度固定為 96。歡迎在下方添加另一個單元格並嘗試更改輸入解析度。

提示：你可以更改 `sample_active_subnet` 方法的 `sample_function` 參數來控制採樣過程。""",

    30: """### 問題 1 (5 分)：設計空間探索 (Design space exploration)

嘗試通過多次運行上面的單元格手動採樣不同的子網路。你也可以改變輸入解析度。談談你的發現。

提示：哪一個維度對準確率起到了最重要的作用？

**回答：** (請填寫)""",

    31: """## **第一部分：預測器 (3 個問題，30 分)**

神經網路架構搜尋需要從 OFA 超網中採樣大量子網路並評估這些子網路的性能。這種性能評估非常耗時。""",

    33: """在本實驗中，我們使用**效率預測器 (Efficiency Predictors)** 和**準確率預測器 (Accuracy Predictors)** 來探索極速的神經網路搜尋。""",

    34: """### 問題 2 (10 分)：實作效率預測器。

對於效率預測器，我們使用基於 Hook 的分析模型來計算給定網路的 #MACs 和峰值記憶體消耗 (Peak Memory Consumption)。讓我們使用提供的 API 從頭開始建立它。

具體來說，我們定義了一個名為 `AnalyticalEfficiencyPredictor` 的類別。該類別有兩個主要的函式：`get_efficiency` 和 `satisfy_constraint`。

函式 `get_efficiency` 傳入子網路配置，並傳回該子網路的 #MACs 和峰值記憶體。這裡我們假設 #MACs 的單位是百萬 (Million)，峰值記憶體消耗的單位是 KB。

提示：參考上面的 `evaluate_sub_network` 函式。使用 `count_net_flops` 獲取網路的 MACs，使用 `count_peak_activation_size` 獲取網路的激活大小 (Activation Size)。""",

    36: """讓我們通過檢查不久前我們評估的最小和最大子網路的傳回值，來測試你實作的分析效率預測器。效率預測器的結果應該與之前的結果相匹配。""",

    38: """### 問題 3 (10 分)：實作準確率預測器。

對於準確率預測器，它預測給定子網路在 VWW 數據集上的分類準確率，這樣我們在架構搜尋過程中遇到新的子網路時就**不需要**每次都執行高成本的推論。這樣的準確率預測器是一個在用 OFA 網路建立的準確率數據集上訓練的 MLP (多層感知機) 模型。MLP 網路的推論僅需幾毫秒，因此準確率預測器可以將搜尋過程加速**幾個數量級**。""",

    39: """準確率預測器接收子網路的架構，並預測其在 VWW 數據集上的準確率。由於它是一個 MLP 網路，子網路必須編碼為一個**向量 (Vector)**。在本實驗中，我們提供了一個類別 `MCUNetArchEncoder` 來執行將**子網路架構**轉換為**二元向量 (Binary Vector)** 的操作。""",

    41: """我們預先生成了一個準確率數據集，它是存儲在 `acc_datasets` 文件夾下的 `[architecture, accuracy]` 對的集合。

利用架構編碼器，你現在需要定義準確率預測器，它是一個多層感知機 (MLP) 網路，每個中間層有 400 個通道。為簡單起見，我們將層數固定為 **3**。請在以下單元格中實作此 MLP 網路。""",

    43: """讓我們列印出你剛剛定義的 `AccuracyPredictor` 的架構。""",

    45: """讓我們首先在以下單元格中視覺化準確率數據集中的一些樣本。

準確率數據集由 50,000 個 `[architecture, accuracy]` 對組成，其中 40,000 個用作訓練集，其餘 10,000 個用作驗證集。

對於**準確率**，我們計算準確率數據集中所有 `[architecture, accuracy]` 對的平均準確率，並將其定義為 `base_acc`。對於準確率預測器，它的訓練目標不是直接對每個架構的準確率進行回歸，而是 `accuracy - base_acc`。由於 `accuracy - base_acc` 通常比 `accuracy` 本身小得多，這可以使訓練更容易。

對於**架構**，設計空間內的每個子網路都由二元向量唯一表示。二元向量是全域參數 (*例如* 輸入解析度、通道縮放倍率) 和每個反向 MobileNet 模組的參數 (*例如* 卷積核大小和擴展比例) 的 **One-hot 表示 (One-hot representation)** 的拼接。請注意，我們偏好使用 **One-hot** 表示而非**數值**表示，因為所有設計超參數都是**離散 (Discrete)** 值。

例如，我們的設計空間支援：

```python
kernel_size = [3, 5, 7]
expand_ratio = [3, 4, 6]
```

然後，我們將 `kernel_size=3` 表示為 `[1, 0, 0]`，`kernel_size=5` 表示為 `[0, 1, 0]`，`kernel_size=7` 表示為 `[0, 0, 1]`。類似地，對於 `expand_ratio=3`，寫為 `[1, 0, 0]`；`expand_ratio=4` 寫為 `[0, 1, 0]`，`expand_ratio=6` 寫為 `[0, 0, 1]`。每個反向 MobileNet 模組的表示是通過將卷積核大小的嵌入與擴展比例的嵌入拼接得到的。請注意，對於跳過的模組 (Skipped blocks)，我們使用 `[0, 0, 0]` 來表示它們的卷積核大小和擴展比例。運行以下單元格後，你將看到架構嵌入對應關係的詳細說明。""",

    47: """### 問題 4 (10 分)：完成準確率預測器訓練的程式碼。

現在讓我們使用我們提供的數據集來訓練準確率預測器！在這部分中，你負責實作準確率預測器的訓練和驗證。訓練過程大約需要 1-2 分鐘。

提示：你可以參考 Tutorial 2 中關於如何使用 PyTorch 訓練神經網路的內容。""",

    49: """現在讓我們繪制預測準確率與真實 Ground Truth 準確率的相關性圖，以確保我們的預測器是可靠的。要獲得滿分，你預計在這部分會看到線性相關性。""",

    51: """## **第二部分：神經網路架構搜尋 (6 個問題，65 分 + 10 分加分)**""",

    52: """到目前為止，我們已經定義了效率預測器和準確率預測器。讓我們開始使用這兩個強大的預測器進行快速模型客製化！

![nas.png](assets/nas.png)

在這部分中，你需要實作兩種典型的搜尋演算法：**隨機搜尋 (Random Search)** 和 **進化搜尋 (Evolutionary Search)**。搜尋演算法旨在找到在滿足效率約束 (*例如* MACs、峰值記憶體) 的同時提供最佳準確率的模型架構。""",

    53: """### 問題 5 (5 分)：完成以下隨機搜尋代理 (Random Search Agent)。""",

    55: """### 問題 6 (5 分)：完成以下函式。""",

    58: """### 問題 7 (20 分)：完成以下進化搜尋代理 (Evolutionary Search Agent)。""",

    59: """![evolution.png](assets/evolution.png)

現在你已經成功實作了隨機搜尋演算法。在這部分中，我們將實作一個樣本效率更高的搜尋演算法——進化搜尋 (Evolutionary Search)。進化搜尋靈感來自進化演算法 (或遺傳演算法)。首先從設計空間中採樣一個子網路**種群 (Population)**。然後，在每個**世代 (Generation)** 中，我們執行如上圖所示的隨機變異 (Mutation) 和交叉 (Crossover) 操作。將保留具有最高準確率的子網路，並重複此過程，直到世代數達到 `max_time_budget`。與隨機搜尋類似，在整個搜尋過程中，所有無法滿足效率約束的子網路都將被丟棄。""",

    61: """### 問題 8 (10 分)：運行進化搜尋並微調 evo_params 以優化結果。描述你的發現。""",

    63: """### 問題 9 (15 分 + 10 分加分)：在真實世界約束下運行進化搜尋。

在真實世界的應用中，我們可能有多個效率約束：https://blog.tensorflow.org/2019/10/visual-wake-words-with-tensorflow-lite_30.html。
使用進化搜尋來找到滿足以下約束的模型：
- [15 分] 250 KB，60M MACs (準確率 >= 92.5% 可獲得滿分)
- [10 分，**加分**] 200KB，30M MACs (準確率 >= 90% 可獲得滿分)

提示：這兩個任務你不必使用相同的 `evo_params`。""",

    66: """### 問題 10 (10 分)：在目前的設計空間中，是否有可能找到滿足以下效率約束的子網路？
- A: 子網路的激活大小 **最多 256KB** 且子網路的 MACs **最多 15M**。
- B: 子網路的激活大小 **最多 64 KB**。"""
}

for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = ''.join(cell['source'])
    source_clean = re.sub(r'!\[(.*?)\]\(data:image/[^;]+;base64,[^\)]+\)', r'![\1](assets/\1)', source)
    
    md_out.append(f"### [Cell {idx}] ({ctype.capitalize()})\n")
    if ctype == 'markdown':
        if idx in translations:
            md_out.append(translations[idx])
        else:
            md_out.append(source_clean)
        md_out.append("\n")
    elif ctype == 'code':
        md_out.append("```python")
        md_out.append(source_clean)
        md_out.append("```\n")

with open('Lab3_zh.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_out))

print("Saved Lab3_zh.md successfully!")
