---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 19
lecture: 19
date: 2023-11-14
video: https://youtu.be/0YABI9mxG60
notes: Lec19-On-Device-Training-and-Transfer-Learning.pdf（89 頁）
---

# 第 19 集 · Lecture 19：裝置端訓練與遷移學習 [▶ 影片](https://youtu.be/0YABI9mxG60)

← 上一集：[Lecture 18 分散式訓練（下）](lecture18_distributed_training_part2_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 20 高效微調與 Prompt Engineering](lecture20_efficient_finetuning_prompt_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 19 集
> **講者**：Song Han
> **來源**：YouTube 錄影逐字稿（1:16:48，英文自動字幕，Zoom 版）＋ 投影片 `Lec19-On-Device-Training-and-Transfer-Learning.pdf`（89 頁）
> **標記說明**：
> - `【口頭】` = 教授口頭補充、投影片沒有
> - `【投影片補充】` = 投影片有、課堂沒講
> - `⚠️ 逐字稿缺字` = 字幕聽錯，依投影片修正
> - `⚠️ 待確認` = 查不清楚，列在最後的清單
> - `[▶ mm:ss]` = 可點的影片時間點（字幕以分鐘為單位，時間點是大約位置）

---

## 目錄

- [0. 這堂課要做什麼](#0-這堂課要做什麼)
- [1. ⭐ 從梯度洩漏資料（Deep Leakage from Gradients）](#1--從梯度洩漏資料deep-leakage-from-gradients)
- [2. ⭐⭐ 裝置端訓練的記憶體瓶頸](#2--裝置端訓練的記憶體瓶頸)
- [3. ⭐⭐ TinyTL：只更新 bias + 輕量殘差](#3--tinytl只更新-bias--輕量殘差)
- [4. ⭐⭐ 稀疏反向傳播（Sparse BP）](#4--稀疏反向傳播sparse-bp)
- [5. ⭐⭐ 量化訓練：Quantization-Aware Scaling（QAS）](#5--量化訓練quantization-aware-scalingqas)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與後續集數的連結](#7-與後續集數的連結)

---

## 0. 這堂課要做什麼

### 0.1 遷移學習

[▶ 00:00](https://youtu.be/0YABI9mxG60?t=0)

**遷移學習（transfer learning）**：把一個任務學到的知識，拿來**提升相關任務**的表現。

```
來源資料（例如 ImageNet，1000 類、120 萬張）→ 來源模型
                                               ↓ 遷移知識 + 微調
                    下游資料（例如卡車）    →  下游模型
```

【口頭】例子：本來學會辨識各種汽車，現在拿去辨識各種卡車。

### 0.2 為什麼要在裝置上學？

[▶ 01:00](https://youtu.be/0YABI9mxG60?t=60)

| 好處 | 說明 |
|---|---|
| **客製化** | 持續適應感測器收集到的新資料。【口頭】每個人都能有符合自己喜好的模型 |
| **隱私** | **資料不用離開裝置**。【口頭】程式碼、企業資料這種敏感資料不用傳到雲端 |

**難處**：[▶ 02:00](https://youtu.be/0YABI9mxG60?t=120) 邊緣裝置記憶體很小，而訓練比推論更吃計算、**更吃記憶體**。

### 0.3 本講大綱

[▶ 03:00](https://youtu.be/0YABI9mxG60?t=180)

| 段落 | 內容 |
|---|---|
| 1 | **從梯度洩漏資料**：梯度不能隨便分享 → 動機：資料要留在本地 |
| 2 | 裝置端訓練的**記憶體瓶頸**在哪裡 |
| 3 | **TinyTL**：只更新 bias、輕量殘差學習 |
| 4 | **Sparse BP**：只更新一部分的層、一部分的通道 |
| 5 | **量化訓練**：Quantization-Aware Scaling |
| 6 | 【投影片補充】PockEngine（稀疏反向傳播的系統支援）——【口頭】教授說有時間才講，否則下一講 |

---

## 1. ⭐ 從梯度洩漏資料（Deep Leakage from Gradients）

### 1.1 聯邦學習（Federated Learning）

[▶ 05:00](https://youtu.be/0YABI9mxG60?t=300)

**想法**：只分享**梯度或權重**，使用者的資料**留在手機上**。

【口頭】例子：Google 鍵盤從很多使用者的打字紀錄學習，讓大家打字更順。

**FedAvg 演算法**：[▶ 06:00](https://youtu.be/0YABI9mxG60?t=360)

1. 使用者在裝置上產生個人資料，在本地訓練。
2. 每台裝置用本地資料更新模型 $N$ 次迭代。
3. 把更新後的模型送到伺服器。
4. 伺服器把各裝置的模型**平均**，再送回去。

→ 重要的私人資料**從不離開裝置**。……真的嗎？

### 1.2 梯度安全嗎？

[▶ 07:00](https://youtu.be/0YABI9mxG60?t=420)

傳統想法：**訓練資料是私密的**，**梯度是公開的**（可以分享）。但如果能從梯度反推出訓練資料，分享梯度就不安全了。

**已知的攻擊**：[▶ 08:00](https://youtu.be/0YABI9mxG60?t=480)

| 攻擊 | 能知道什麼 |
|---|---|
| **成員推斷（membership inference）** | 某筆資料**有沒有在這個 batch 裡** |
| **屬性推斷（property inference）** | batch 裡**有沒有具備某種屬性**的資料 |

**能不能更進一步，直接偷出原始資料？**

### 1.3 Deep Leakage 攻擊：梯度配對

[▶ 09:00](https://youtu.be/0YABI9mxG60?t=540)

**正常訓練**：真實輸入 $x$ + 真實標籤 $y$ → 模型 $F(x, W)$ → 損失 → 梯度 $\nabla W$。

**攻擊者**（有同一個模型 $W$，也看得到 $\nabla W$）：

1. 隨機初始化一個**假輸入** $x'$ 和**假標籤** $y'$。
2. $x'$ 過模型、跟 $y'$ 算損失 → 得到**假梯度** $\nabla W'$。
3. 定義**梯度配對損失**：

$$
D = \left\Vert \nabla W' - \nabla W \right\Vert^2
$$

4. [▶ 10:00](https://youtu.be/0YABI9mxG60?t=600) **對 $x'$ 和 $y'$ 做反向傳播**（不是對權重）：

$$
x' \leftarrow x' - \eta\frac{\partial D}{\partial x'},\qquad y' \leftarrow y' - \eta\frac{\partial D}{\partial y'}
$$

5. 重複幾次，$\nabla W'$ 越來越接近 $\nabla W$，$x'$ 就越來越像原本的影像。

【口頭】[▶ 11:00](https://youtu.be/0YABI9mxG60?t=660) 攻擊者和使用者之間**只分享了梯度**，卻洩漏了隱私。

### 1.4 攻擊結果

**batch size = 1 的視覺模型**：幾次迭代就還原出原圖。

**batch size = 8**：[▶ 12:00](https://youtu.be/0YABI9mxG60?t=720)

【口頭】batch 裡有更多張圖，攻擊會更難還是更容易？**更難**。直覺：

- **方程式的數量** = 梯度的維度（固定）。
- **未知數** = 每個像素。batch 越大，未知數越多。

但實驗顯示 batch 有 6～8 張時**還是能偷得很好**（猴子、腳踏車都還原出來了），只是順序會變。

**語言模型（BERT）也能攻擊**：[▶ 13:00](https://youtu.be/0YABI9mxG60?t=780) 第 0 次迭代是亂碼，第 20 次已經大致對了，第 30 次幾乎跟原文一樣。

**限制**：[▶ 14:00](https://youtu.be/0YABI9mxG60?t=840)

- 例子裡的影像解析度都很低（CIFAR）。資訊量 = 梯度的維度 = 模型的大小，是**固定**的。要偷的資料越大越難。
- 【口頭】反過來說，**模型越大越容易被偷**（例如 LLM、視覺語言模型），因為方程式更多。

### 1.5 防禦方法

**方法 1：加雜訊**　[▶ 16:00](https://youtu.be/0YABI9mxG60?t=960)

- 加高斯或拉普拉斯雜訊。
- 雜訊小（$10^{-4}$、$10^{-3}$）：**還是會洩漏**。
- 雜訊大到能防住（$10^{-2}$、$10^{-1}$）：模型**準確率大掉**。【口頭】教授說準確率幾乎掉到 0，模型就不能用了。
- → **不是好方法**。

**方法 2：梯度壓縮（剪枝）**　[▶ 17:00](https://youtu.be/0YABI9mxG60?t=1020)

- 剪掉 1%、10% 的梯度：還是會洩漏。
- 【口頭】剪到約 70%：就防住了。
- 上一講的 DGC 能剪到 **99%** 以上準確率都不掉（ResNet-50：75.96% → 76.15%），所以空間很大。
- 【投影片補充】DGC 還有**本地累積**，會進一步混淆梯度，更能保護隱私。

**為什麼剪枝有用？**[▶ 18:00](https://youtu.be/0YABI9mxG60?t=1080) 非零的梯度少 → **方程式少**。而且攻擊者看到一個 0，不知道是剪掉的、還是本來就是 0。

【口頭】[▶ 19:00](https://youtu.be/0YABI9mxG60?t=1140) 實作只要約 20 行 PyTorch，2019 年發表後引用很多次。

> **分享梯度跟分享原始資料一樣危險** → 資料應該**留在本地**，這就是裝置端訓練的動機。

---

## 2. ⭐⭐ 裝置端訓練的記憶體瓶頸

### 2.1 裝置的記憶體非常小

[▶ 19:00](https://youtu.be/0YABI9mxG60?t=1140)

| | 雲端 AI | 手機 AI | 微型 AI（MCU） |
|---|---|---|---|
| 記憶體（放激活值） | 141 GB（【口頭】昨天剛發布的 H200） | 4 GB | **320 KB** |
| 儲存（放權重） | TB ~ PB | 256 GB | **1 MB** |

【口頭】H200 每張卡 141 GB，比 H100 的 80 GB 大很多。MCU 小了好幾個數量級（參考 [Lecture 10 MCUNet](lecture10_mcunet_note.md)）。

→ 跟推論一樣要同時減少**權重和激活值**，但訓練更難。

### 2.2 訓練比推論吃更多記憶體

[▶ 20:00](https://youtu.be/0YABI9mxG60?t=1200)

MobileNetV2：

| | 記憶體 |
|---|---|
| 推論（batch = 1） | 20 MB |
| 訓練（batch = 8） | **452 MB** |
| Raspberry Pi 1 的 DRAM | 256 MB |
| MCU | 2 MB |

### 2.3 為什麼？因為要存中間的激活值

[▶ 21:00](https://youtu.be/0YABI9mxG60?t=1260)

$$
\text{前向：}\ a_{i+1} = a_i W_i + b_i
\qquad
\text{反向：}\ \frac{\partial L}{\partial W_i} = a_i^\top\frac{\partial L}{\partial a_{i+1}}
$$

⭐ 關鍵一步：算**權重的梯度** $\dfrac{\partial L}{\partial W_i}$ 需要**這一層的輸入激活值 $a_i$**。

| | 推論 | 訓練 |
|---|---|---|
| 激活值 | 算完下一層就**丟掉** | 要**存下來**給反向傳播用 |
| batch size | 通常是 1 | 激活值**隨 batch size 線性成長** |

- 【口頭】[▶ 22:00](https://youtu.be/0YABI9mxG60?t=1320) 就算 batch size 是 1，激活值也超出很多邊緣裝置的記憶體。
- **能耗**：[▶ 23:00](https://youtu.be/0YABI9mxG60?t=1380) 邊緣裝置靠電池。訓練放不進省電的 **cache（SRAM）**，就要用 DRAM，能耗大增。

### 2.4 CNN 的瓶頸是激活值，不是參數

[▶ 23:00](https://youtu.be/0YABI9mxG60?t=1380)

| 模型 | 參數 vs 激活值 |
|---|---|
| **ResNet-50** | 激活值比參數大 **6.9 倍** |
| **MobileNetV2-1.4** | 參數比 ResNet-50 **小 4.3 倍**，但激活值只小了 **1.1 倍** |

【口頭】下一講的 LLM 微調結論會不同：LLM 的**參數比激活值多**（batch size 合理時）。

**為什麼 MobileNet 的激活值沒變小？**[▶ 24:00](https://youtu.be/0YABI9mxG60?t=1440)

- MobileNet 專注在減少**參數和 FLOPs**，主要靠 depthwise 卷積。
- 為了補回 depthwise 損失的容量，用了**很大的擴張比（約 6）**：通道數變成 6 倍。
- 激活值是 $H\times W\times C$，**$C$ 變 6 倍**，激活值就大了。
- 但 depthwise 的參數只有 $3\times3\times C$，不是 $3\times3\times C_{\text{in}}\times C_{\text{out}}$，所以**參數和 FLOPs 不受影響**。

### 2.5 參數少 ≠ 記憶體少

[▶ 25:00](https://youtu.be/0YABI9mxG60?t=1500)

| 方法 | 做法 | 可訓練參數 | 問題 |
|---|---|---|---|
| **Full** | 微調整個網路 | 很多 | 準確率好，但效率很差 |
| **Last** | 只微調最後的分類層 | 很少 | 有效率，但**準確率大掉**（容量太小） |
| **BN + Last** | [▶ 26:00](https://youtu.be/0YABI9mxG60?t=1560) 微調 BatchNorm 的 scale、bias + 最後一層（Google 提出） | **少 12 倍** | 參數有效率，但**記憶體只省 1.8 倍**，準確率還差 12% |

⭐ 關鍵觀察：[▶ 27:00](https://youtu.be/0YABI9mxG60?t=1620) **參數效率不等於記憶體效率**（12× vs 1.8×）。要微調 BN 層，還是要存一大堆激活值。

⚠️ 逐字稿缺字：教授口頭說記憶體「只改善約 80%」，投影片是 **1.8 倍**。

---

## 3. ⭐⭐ TinyTL：只更新 bias + 輕量殘差

### 3.1 更新權重很貴，更新 bias 很便宜

[▶ 29:00](https://youtu.be/0YABI9mxG60?t=1740)

MobileNet 的 inverted bottleneck 區塊：

```
C, R ──1×1 Conv──→ 6C, R ──Depthwise──→ 6C, R ──1×1 Conv──→ C, R
```

（$C$ = 通道數、$R$ = 解析度）

三個梯度公式：

$$
\frac{\partial L}{\partial W_i} = a_i^\top\frac{\partial L}{\partial a_{i+1}},\qquad
\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial a_{i+1}},\qquad
\frac{\partial L}{\partial a_{i+1}} = \frac{\partial L}{\partial a_{i+2}}W_{i+1}^\top
$$

⭐ 關鍵一步：[▶ 30:00](https://youtu.be/0YABI9mxG60?t=1800)

- **權重的梯度**要乘上 $a_i$ → **必須存激活值**。
- **bias 的梯度**就等於往回傳的梯度，【口頭】因為 bias 是線性項，前面沒有係數 → **不用存激活值**。
- 往回傳梯度只需要**權重**（本來就在記憶體裡），不需要激活值。

【投影片補充】投影片在 bias-only 這頁註記「LoRA 是一個特例」。

### 3.2 只微調 bias

[▶ 31:00](https://youtu.be/0YABI9mxG60?t=1860)

**凍結權重、只微調 bias** → 比 BN + Last **省 12 倍記憶體**。

但**準確率掉 16.3%**。好的起點，但還不夠。

### 3.3 輕量殘差學習（Lite Residual Learning）

[▶ 31:00](https://youtu.be/0YABI9mxG60?t=1860)

【口頭】bias 只是一個純量，能不能學個更強的「bias」，記憶體又不增加？→ 加一條**輕量的殘差分支**。

```
主分支：C, R ──1×1──→ 6C, R ──DW──→ 6C, R ──1×1──→ C, R    （權重凍結，只更新 bias）
          │                                              ↑
殘差分支： └─降採樣─→ C, 0.5R ──Group Conv──→ ──1×1 Conv──→ 上採樣 ─┘   （可訓練）
```

**關鍵原則：讓激活值保持很小**

1. **降低解析度**：[▶ 32:00](https://youtu.be/0YABI9mxG60?t=1920) 先降採樣到一半解析度。【口頭】計算量和激活值隨解析度平方變化，所以只剩 1/4。
2. **避免 inverted bottleneck**：通道數維持 $C$，**不擴張到 $6C$**。
3. 用 group 卷積（空間建模）+ 1×1 卷積（通道建模），**不用 depthwise**。

**成本**：[▶ 34:00](https://youtu.be/0YABI9mxG60?t=2040)

- 1/6 的通道、1/2 的解析度、2/3 的深度 → 約 **4%** 的激活值大小。
- 只多 **5 MB** 記憶體（相對約 400 MB）。
- 準確率**提升 11.6%**。

⚠️ 待確認：$\dfrac{1}{6}\times\left(\dfrac{1}{2}\right)^2\times\dfrac{2}{3} \approx 2.8\%$，跟投影片和教授說的「約 4%」對不起來。可能是解析度那項的算法不同。

**TinyTL = 只更新 bias + 輕量殘差**：[▶ 35:00](https://youtu.be/0YABI9mxG60?t=2100) 補回準確率差距，記憶體也大幅節省（相對 BN + Last 的 12% 差距，現在補回來了）。

### 3.4 不同任務用不同的模型

[▶ 35:00](https://youtu.be/0YABI9mxG60?t=2100)

- 用 Once-for-All 超級網路（[Lecture 7-8](lecture07_08_nas_note.md)、Lab 3）：難的下游任務用大的子網路，簡單的（例如手寫數字）用小的。
- 【口頭】**專門化（specialization）**是計算機結構裡提高效率的關鍵，這裡也一樣。

**發現**：[▶ 36:00](https://youtu.be/0YABI9mxG60?t=2160) **ImageNet 準確率高的模型，遷移到下游任務不一定好**。

- 只訓練最後一層時，不同預訓練模型在 ImageNet、汽車、花、飛機上的**排名差很多**。
- 【口頭】例如 Inception-v3 在 ImageNet 很好，在其他任務不一定最好；花的資料集上最好的是 ResNet-50，但它在 ImageNet 只排第三。
- → 不同下游任務需要**個人化、專門化的架構**。

【口頭】[▶ 38:00](https://youtu.be/0YABI9mxG60?t=2280) 教授補充：這是去年的投影片，今年局勢變了。現在有**單一基礎模型**能用 zero-shot 或 few-shot 處理各種任務，不用微調。下一講會講 prompt tuning。

### 3.5 結果

[▶ 38:00](https://youtu.be/0YABI9mxG60?t=2280)

- 花、汽車、食物資料集上，TinyTL **最多省 6.5 倍記憶體，準確率不掉**（其他資料集 4.5×、4.6×）。
- [▶ 39:00](https://youtu.be/0YABI9mxG60?t=2340) 比**動態激活值剪枝**（剪 ResNet-50 或 MobileNetV2 的激活值）更有效地省記憶體。
- 【口頭】[▶ 40:00](https://youtu.be/0YABI9mxG60?t=2400) 輕量殘差的想法，跟之後 LLM 微調用的方法很像：用很輕的殘差分支增加容量，記憶體幾乎不增加。

**能在 cache 裡訓練**：[▶ 40:00](https://youtu.be/0YABI9mxG60?t=2400)

- TinyTL 用 **group normalization** 支援 **batch size = 1** 的訓練。
- 加上輕量殘差，訓練記憶體降到 **16 MB**，放得進典型的 L3 cache（AMD EPYC 的 L3 是 128 MB）。
- 在 cache 裡訓練**比在 DRAM 上省電很多**。

⚠️ 逐字稿缺字：字幕寫「160 megabytes」「60 megabytes」「120 MB」，依投影片是 **16 MB**（L3 cache）和 **128 MB**（AMD EPYC L3）。

---

## 4. ⭐⭐ 稀疏反向傳播（Sparse BP）

### 4.1 為什麼要稀疏？

[▶ 41:00](https://youtu.be/0YABI9mxG60?t=2460)

- 完整反向傳播要存**所有**中間激活值，而且隨 batch size 線性成長。
- 如果某些層**不用算權重的梯度**，就**不用存那些層的激活值**，計算和記憶體都省。

**人腦也是稀疏學習**：[▶ 42:00](https://youtu.be/0YABI9mxG60?t=2520)

| 時期 | 每個神經元的突觸數 |
|---|---|
| 新生兒 | 2,500 |
| 2～4 歲 | **15,000** |
| 青春期（K-12 教育） | 越來越**稀疏** |
| 成人 | 約 7,000 |

【口頭】我們一生學最多知識的 K-12 時期，突觸數反而**減少**，暗示某種稀疏可能對學習有幫助。

### 4.2 四種反向傳播比較

[▶ 42:00](https://youtu.be/0YABI9mxG60?t=2520)

| 方式 | 做法 | 記憶體 | 準確率 |
|---|---|---|---|
| **(a) 完整** | 所有層的權重和 bias | 很大 | 最好 |
| **(b) 只更新最後一層** | 不用往前傳 | 小（12× 省） | **大掉** |
| **(c) 只更新 bias** | [▶ 43:00](https://youtu.be/0YABI9mxG60?t=2580) 一路傳到第一層，但只更新 bias | 跟 (b) 差不多 | 比 (b) 好，但還有差距 |
| ⭐ **(d) 稀疏** | **只更新部分層、部分通道** | 小 | 接近 (a) |

【口頭】(b) 和 (c) 都可以看成稀疏學習的特例。(c) 比 (b) 好很多又幾乎不多花記憶體，是很容易摘的果實。

### 4.3 稀疏的兩個層次

[▶ 44:00](https://youtu.be/0YABI9mxG60?t=2640)

1. **稀疏層（sparse layer）**：有些**層**沒那麼重要 → 不更新它的權重。
   - 【口頭】通常**不用傳到最前面幾層**，因為前面幾層學的是低階特徵（邊緣、紋理），不同任務都差不多，凍結就好。
   - 反向傳播停在哪裡，前面的激活值就不用存。
2. **稀疏張量（sparse tensor）**：一層裡有些**通道**沒那麼重要 → 只更新一部分通道。

**省多少？**[▶ 45:00](https://youtu.be/0YABI9mxG60?t=2700)

$$
(dW)^\top = G^\top X
$$

| | 要存的激活值 | FLOPs |
|---|---|---|
| 完整更新 | $X$ 的全部：$(N, M)$ | $M\times H\times N$ |
| 只更新 1/4 的通道 | 只存對應的 1/4：$(N, 0.25M)$ | $0.25\times M\times H\times N$（**少 4 倍**） |

### 4.4 該更新哪些層？

**記憶體的角度**：[▶ 46:00](https://youtu.be/0YABI9mxG60?t=2760)

| 位置 | 激活值 | 權重 | 總記憶體 |
|---|---|---|---|
| 前面的層 | **很大**（解析度高） | 小 | 大 |
| 中間的層 | 中 | 中 | **最小** |
| 後面的層 | 小（降採樣後，隨解析度平方縮小） | **很大**（通道多） | 大 |

→ 後面的層只更新 **bias**（只跟激活值有關），**中間的層**更新權重。

**準確率的角度：貢獻分析（contribution analysis）**　[▶ 48:00](https://youtu.be/0YABI9mxG60?t=2880)

- 在下游任務上**只微調一層**，量準確率提升多少（$\Delta$accuracy），當作這一層的貢獻。
- 只微調 $\Delta$accuracy 大的層。

**MobileNetV2 的結果**：[▶ 49:00](https://youtu.be/0YABI9mxG60?t=2940)

- 出現很規則的**鋸齒狀**：低、低、高，每 3 層重複一次，對應 inverted bottleneck 的 pointwise → depthwise → pointwise。
- 有些前面的層甚至是**負貢獻**；最後一層貢獻很大。
- 【口頭】[▶ 50:00](https://youtu.be/0YABI9mxG60?t=3000) **第一個 pointwise 卷積**值得微調：
  1. 貢獻大（超過 5%）。
  2. 它的**輸入激活值很小**：只有 $C$ 個通道；後面兩層的輸入是 $6C$。

⚠️ 待確認：教授口頭和投影片的貢獻分析圖都說「第一個 **pointwise** 卷積貢獻更多」，但投影片的總結寫「MobileNetV2 偏好第一個 **depthwise** 卷積」，兩處不一致。

**BERT 的結果**：[▶ 51:00](https://youtu.be/0YABI9mxG60?t=3060)

- Transformer 每層有 6 個可學的線性層：Q、K、V、輸出投影、FFN1、FFN2。【口頭】attention 本身沒有權重。
- 每 6 個一個週期：
  - **QKV 投影**：貢獻大。
  - 輸出投影：貢獻小。
  - **FFN1** 比 FFN2 貢獻大。
- [▶ 52:00](https://youtu.be/0YABI9mxG60?t=3120) FFN1 也比較便宜：FFN 有 inverted bottleneck（擴張比約 2～4），**FFN1 的輸入比 FFN2 的輸入小**。

**總結**：[▶ 53:00](https://youtu.be/0YABI9mxG60?t=3180) **不同模型偏好微調不同的層**。這個分析只要做一次，就能用在各種下游任務。

### 4.5 用演化搜尋找最好的稀疏方案

[▶ 54:00](https://youtu.be/0YABI9mxG60?t=3240)

每層還能選更新多少通道：全部、1/2、1/4、1/8。更新越多，準確率越高。

**最佳化問題**：

$$
\max\ \left[\Delta\text{acc}_{\text{bias}}(k) + \sum_{i\in\mathcal{I}}\Delta\text{acc}_{W_i, r_i}\right]
\quad\text{s.t.}\quad \text{Memory}(k, \mathcal{I}, r) \le \text{限制}
$$

- $k$：最後幾層更新 bias。
- $\mathcal{I}$：哪些層更新權重。
- $r_i$：第 $i$ 層更新的通道比例。
- 【口頭】假設各層貢獻獨立（實際上不完全是，但是不錯的近似）。

[▶ 55:00](https://youtu.be/0YABI9mxG60?t=3300) 用**演化搜尋**找，比隨機搜尋更快找到更好的方案。

### 4.6 結果

[▶ 55:00](https://youtu.be/0YABI9mxG60?t=3300)

- DistilBERT、BERT（GLUE 平均）、MCUNet、MobileNetV2、ResNet-50（TinyTL 基準平均）：**Sparse BP 跟完整 BP 準確率相當**，有時還更好；只更新 bias 則明顯較差。
- [▶ 56:00](https://youtu.be/0YABI9mxG60?t=3360) 部署到 MCU：Sparse BP 比「更新最後 K 層」準確率更高，**額外記憶體少 4.5～7.5 倍**。

### 4.7 在邊緣 GPU 上微調 LLM

[▶ 57:00](https://youtu.be/0YABI9mxG60?t=3420)

- 模型：Llama-2 7B。資料：Stanford Alpaca（52K 條指令）。
- **指令微調（instruction tuning，也叫對齊 alignment）**：雖然 LLM 的知識主要是預訓練學的，但要靠微調**教它怎麼跟使用者互動**。

**Jetson Orin 上的結果**：[▶ 59:00](https://youtu.be/0YABI9mxG60?t=3540)

| 框架 | 方法 | 每次迭代 | GPU 記憶體 | Loss | Alpaca-Eval 勝率 | MT-Bench |
|---|---|---|---|---|---|---|
| PyTorch | 完整微調 | 7.7 s | 45.1 GB | 0.761 | 44.1% | 6.1 |
| PyTorch | LoRA（rank 8） | 7.3 s | 30.9 GB | 0.801 | 43.1% | 5.1 |
| PockEngine | 完整微調 | 1.8 s | 43.1 GB | 0.768 | 43.7% | 6.1 |
| PockEngine | **Sparse BP** | **0.9 s** | 31.2 GB | 0.779 | 43.1% | 5.7 |

- Sparse BP 的準確率跟 LoRA 一樣、跟完整微調相當。
- 速度：完整 BP 1.8 s → Sparse BP 0.9 s（快 2 倍）；比 PyTorch 快很多（因為 PockEngine）。
- 記憶體跟 LoRA 差不多。

**質性比較**：[▶ 59:00](https://youtu.be/0YABI9mxG60?t=3540)

| 問題類型 | 原始 Llama-2 | 完整 BP | Sparse BP |
|---|---|---|---|
| 簡單事實（美國首都） | 對 | 對 | 對 |
| 冷門事實（Michael Jackson 最後一張專輯） | **錯**（還說他 2020 年過世） | 對（Invincible，2001） | 對 |
| 推理（「MICRO 去年在哪辦？」） | 答成 2021 年線上 | 對（2022 芝加哥） | 對 |
| 反轉句子「I love the Micro conference」 | **錯**（變成 I hate…） | 對 | 有一點錯（順序弄亂） |
| 數學（割草 16 次每次 33 元，加 3 個 10 元小費） | **錯**（660） | 對：$33\times16 + 10\times3 = 558$ | 對（推理過程比較長） |

【口頭】[▶ 1:02:00](https://youtu.be/0YABI9mxG60?t=3720) Sparse BP 有時還是不如完整 BP（例如反轉句子），但這只是挑出來的例子。數學題 Sparse BP 回答比較長，好處是推理過程完整，壞處是要生更多 token。

---

## 5. ⭐⭐ 量化訓練：Quantization-Aware Scaling（QAS）

### 5.1 假量化 vs 真量化

[▶ 1:04:00](https://youtu.be/0YABI9mxG60?t=3840)

**真量化圖**：輸入、權重都是 INT8 → 卷積 → INT32 累加 → 加 INT32 bias → 縮放 → 再量化成 INT8 給下一層。

| | **假量化（fake quantization，QAT 用的）** | **真量化（real quantization）** |
|---|---|---|
| 權重 | FP32（用 FP32 表示 INT8 的值） | **INT8** |
| 激活值 | FP32 | **INT8** |
| BatchNorm | 有 | **沒有**（已經摺進卷積） |
| 省訓練記憶體？ | ❌ 大部分張量還是 FP32 | ✅ 少 4 倍 |
| 好不好訓練？ | 好 | **很難** |

【口頭】[▶ 1:05:00](https://youtu.be/0YABI9mxG60?t=3900) 假量化（[Lecture 5-6](lecture05_06_quantization_note.md) 的 QAT）目的是訓練出一個推論時好用的量化模型，訓練時記憶體沒省。

**真量化難訓練的原因**：[▶ 1:06:00](https://youtu.be/0YABI9mxG60?t=3960)

- 混合精度：INT8、INT32、FP32 混在一起。
- **沒有 BatchNorm** 可以幫忙調整。

**結果**：直接用 INT8 SGD 訓練，10 個資料集平均 **86.0% → 75.4%（掉 10.6%）**。

### 5.2 問題出在權重和梯度的比例不對

[▶ 1:07:00](https://youtu.be/0YABI9mxG60?t=4020)

畫出每個張量的 $\log_{10}\left(\Vert W\Vert / \Vert G\Vert\right)$：

- FP32 和 INT8 的比例**差非常多**。
- INT8 的**權重很大、梯度很小** → 梯度推不動權重，**什麼都學不到**。
- 權重和 bias 交錯出現鋸齒狀。

### 5.3 ⭐ 推導

[▶ 1:08:00](https://youtu.be/0YABI9mxG60?t=4080)

Per-channel 量化：

$$
W = s_W\cdot\bar{W},\qquad \bar{W}\in[-128, 127]
$$

**Step 1：量化權重的梯度**。用連鎖律：

$$
G_{\bar{W}} = \frac{\partial L}{\partial\bar{W}} = \frac{\partial L}{\partial W}\cdot\frac{\partial W}{\partial\bar{W}} = s_W\cdot G_W
$$

**Step 2：權重 / 梯度的比例**　[▶ 1:10:00](https://youtu.be/0YABI9mxG60?t=4200)

$$
\frac{\Vert\bar{W}\Vert}{\Vert G_{\bar{W}}\Vert} = \frac{\Vert W/s_W\Vert}{\Vert s_W\cdot G_W\Vert} = s_W^{-2}\cdot\frac{\Vert W\Vert}{\Vert G_W\Vert}
$$

$s_W$ 出現在分母兩次，所以**差了 $s_W^{-2}$ 倍**。

**Step 3：$s_W$ 比 1 大還是小？**[▶ 1:11:00](https://youtu.be/0YABI9mxG60?t=4260)

- $\bar{W}$ 在 $-128$ 到 $127$ 之間，原本的 FP32 權重 $W$ 大約在 1 附近。
- 所以 $s_W = W/\bar{W} \ll 1$，$s_W^{-2} \gg 1$。
- → 量化後的**權重 / 梯度比例大很多**，跟觀察一致。

### 5.4 解法：把梯度乘回來

[▶ 1:12:00](https://youtu.be/0YABI9mxG60?t=4320)

**權重**：

$$
\tilde{G}_{\bar{W}} = G_{\bar{W}}\cdot s_W^{-2}
$$

**bias**：【口頭】同樣處理，只是有兩個 scale 合在一起。設 $\bar{b} = b/(s_W s_x)$（$s_x$ 是輸入的 scale），同理

$$
\tilde{G}_{\bar{b}} = G_{\bar{b}}\cdot(s_W\cdot s_x)^{-2}
$$

- $s_W^{-2} \gg 1$ → 梯度變大 → 真的能學到東西。
- 【口頭】**只要一兩行程式碼**，**不用調超參數**，scale 直接用推論時算好的。

### 5.5 結果

[▶ 1:13:00](https://youtu.be/0YABI9mxG60?t=4380)

- 加 QAS 後，INT8 的權重 / 梯度比例**對齊 FP32**。
- 訓練 loss 和驗證 loss 都下降得更快、更穩定。

**10 個資料集平均 Top-1**：

| 方法 | 準確率 | 額外記憶體 |
|---|---|---|
| FP32 SGD | 86.0% | — |
| INT8 SGD | 75.4% | — |
| INT8 LARS | 64.8% | — |
| INT8 Adam | 84.5% | **3 倍**（Adam 多兩份 optimizer state） |
| ⭐ **INT8 QAS** | **86.9%** | 無 |

QAS 比天真的 INT8 好很多，**甚至不輸 FP32**。

---

## 6. 一頁速查表

### 6.1 隱私

| 項目 | 公式或做法 | 關鍵一步 |
|---|---|---|
| **Deep Leakage** | 最小化 $D = \Vert\nabla W' - \nabla W\Vert^2$，對假資料 $x', y'$ 反向傳播 | 梯度維度 = 方程式數量 |
| **防禦：加雜訊** | 要很大才防得住 | 準確率會崩 |
| **防禦：梯度剪枝** | 剪約 70% 以上就防住 | 非零梯度少 → 方程式少；DGC 剪 99% 準確率不掉 |

### 6.2 記憶體

| 項目 | 公式 | 關鍵一步 |
|---|---|---|
| **權重梯度** | $\dfrac{\partial L}{\partial W_i} = a_i^\top\dfrac{\partial L}{\partial a_{i+1}}$ | 要存激活值 $a_i$ |
| **bias 梯度** | $\dfrac{\partial L}{\partial b_i} = \dfrac{\partial L}{\partial a_{i+1}}$ | **不用存激活值** |
| **往回傳** | $\dfrac{\partial L}{\partial a_{i+1}} = \dfrac{\partial L}{\partial a_{i+2}}W_{i+1}^\top$ | 只要權重 |
| **CNN 瓶頸** | ResNet-50 激活值是參數的 6.9 倍 | MobileNet 的 6× 擴張讓激活值很大 |

### 6.3 方法

| 方法 | 做法 | 效果 |
|---|---|---|
| **BN + Last** | 微調 BN 的 scale、bias + 最後一層 | 參數少 12×，記憶體只少 1.8× |
| **Bias only** | 凍結權重，只更新 bias | 記憶體少 12×，準確率掉 16.3% |
| **TinyTL** | bias + 輕量殘差（半解析度、不擴張、group conv） | 最多省 6.5×，準確率不掉；16 MB 放進 L3 cache |
| **Sparse BP** | 貢獻分析 + 演化搜尋，只更新部分層和通道 | 準確率 ≈ 完整 BP；Llama-2 7B 每步 0.9 s |
| **QAS** | $\tilde{G}_{\bar{W}} = G_{\bar{W}}\cdot s_W^{-2}$ | 比例差 $s_W^{-2}$ 倍 → 乘回來；INT8 達 86.9% |

### 6.4 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **320 KB / 1 MB** | MCU 的記憶體 / 儲存 |
| **20 MB → 452 MB** | MobileNetV2 推論（bs=1）→ 訓練（bs=8） |
| **6.9×** | ResNet-50 激活值 / 參數 |
| **12× vs 1.8×** | BN + Last 的參數節省 vs 記憶體節省 |
| **~4% / 5 MB / +11.6%** | 輕量殘差的激活值大小 / 額外記憶體 / 準確率提升 |
| **6.5×** | TinyTL 最多省的記憶體 |
| **16 MB** | TinyTL 加輕量殘差後的訓練記憶體 |
| **4.5 ~ 7.5×** | Sparse BP 比更新最後 K 層少的額外記憶體 |
| **0.9 s** | Llama-2 7B 在 Jetson Orin 上 Sparse BP 一次迭代 |
| **75.4% → 86.9%** | INT8 訓練：天真 SGD → QAS |

---

## 7. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **§2.4 LLM 是參數大於激活值** | **[Lecture 20 高效微調與 Prompt Engineering](lecture20_efficient_finetuning_prompt_note.md)**：LoRA、adapter 等省參數的微調 |
| **§3.4 Zero-shot、few-shot 不用微調** | **Lecture 20**：prompt tuning、prompt engineering |
| **§4.7 PockEngine** | **Lecture 20**：把演算法的節省轉成實測加速 |

---

## 附：本集一句話總結

> **分享梯度就可能洩漏資料，所以資料要留在裝置上訓練；但訓練的記憶體瓶頸是要存的激活值，不是參數：只更新 bias 就不用存激活值，再加一條半解析度、不擴張的輕量殘差補回準確率（TinyTL）；更進一步只挑貢獻大、又便宜的層和通道來更新（Sparse BP）；用 INT8 真量化訓練時，梯度和權重的比例會差 $s_W^{-2}$ 倍，乘回來就好（QAS）。**

---

## ⚠️ 待確認清單

1. [▶ 34:00](https://youtu.be/0YABI9mxG60?t=2040) 輕量殘差的激活值大小：$\frac{1}{6}\times\left(\frac{1}{2}\right)^2\times\frac{2}{3}\approx2.8\%$，跟投影片和教授說的「約 4%」對不起來。
2. [▶ 50:00](https://youtu.be/0YABI9mxG60?t=3000) MobileNetV2 偏好微調哪一層：教授口頭和貢獻分析圖說第一個 **pointwise** 卷積，投影片總結寫第一個 **depthwise** 卷積。

---

*筆記依據 Song Han「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 19 錄影逐字稿（英文自動字幕，1:16:48）與投影片 `Lec19-On-Device-Training-and-Transfer-Learning.pdf`（89 頁）整理。*
