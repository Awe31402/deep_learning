---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 20
lecture: 20
date: 2023-11-16
video: https://youtu.be/vOPwwRCZ8q8
notes: Lec20-Efficient-Fine-tuning-and-Prompt-Engineering.pdf（95 頁）
---

# 第 20 集 · Lecture 20：高效微調與 Prompt Engineering [▶ 影片](https://youtu.be/vOPwwRCZ8q8)

← 上一集：[Lecture 19 裝置端訓練與遷移學習](lecture19_on_device_training_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 21 量子計算基礎](lecture21_quantum_computing_basics_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 20 集
> **講者**：Song Han
> **來源**：YouTube 錄影逐字稿（1:17:33，英文自動字幕，**一般版**；Zoom 版的字幕抓不到）＋ 投影片 `Lec20-Efficient-Fine-tuning-and-Prompt-Engineering.pdf`（95 頁）
> **標記說明**：
> - `【口頭】` = 教授口頭補充、投影片沒有
> - `【Q&A】` = 學生主動提問與回答
> - `【投影片補充】` = 投影片有、課堂沒講
> - `⚠️ 逐字稿缺字` = 字幕聽錯，依投影片修正
> - `⚠️ 待確認` = 查不清楚，列在最後的清單
> - `[▶ mm:ss]` = 可點的影片時間點（字幕以分鐘為單位，時間點是大約位置）

---

## 目錄

- [0. 這堂課要做什麼](#0-這堂課要做什麼)
- [1. ⭐ PockEngine：稀疏反向傳播的系統支援](#1--pockengine稀疏反向傳播的系統支援)
- [2. 為什麼 LLM 微調要「省參數」](#2-為什麼-llm-微調要省參數)
- [3. ⭐ BitFit、Adapter、Prompt Tuning、Prefix Tuning](#3--bitfitadapterprompt-tuningprefix-tuning)
- [4. ⭐⭐ LoRA 家族：LoRA、QLoRA、LongLoRA](#4--lora-家族loraqloralonglora)
- [5. ⭐ Prompt Engineering](#5--prompt-engineering)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與後續集數的連結](#7-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/vOPwwRCZ8q8?t=0)

接著上一講：上一講講了量化感知縮放（QAS）和稀疏更新這些**演算法**，這一講先講怎麼把它們變成**實測的加速**，再講 LLM 專用的微調技巧和 prompt。

【口頭】PockEngine 今天早上剛上了 MIT News 首頁：讓邊緣裝置上的 AI 可以一直學下去。

| 段落 | 內容 | 要不要改模型 |
|---|---|---|
| **I** | **PockEngine**：稀疏反向傳播的系統支援 | — |
| **II** | **高效 LLM 微調**：BitFit、Adapter、Prompt Tuning、Prefix Tuning、LoRA、QLoRA、LongLoRA | 要微調（但只動一小部分） |
| **III** | **Prompt Engineering**：zero-shot、few-shot、chain-of-thought；LLM 和擴散模型的例子 | **完全不用微調** |

---

## 1. ⭐ PockEngine：稀疏反向傳播的系統支援

### 1.1 把自動微分從執行時移到編譯時

[▶ 02:00](https://youtu.be/vOPwwRCZ8q8?t=120)

訓練的流程：

```
前向計算圖 → 自動微分（AutoDiff）→ 反向計算圖 → 執行
```

| | 傳統訓練框架 | **PockEngine** |
|---|---|---|
| 編譯時 | 只拿到前向圖 | 前向圖 + **反向圖** + 圖優化 + 程式碼生成 |
| 執行時 | 自動微分 + 執行 | **只執行** |
| 重點 | 彈性 | 效率 |

【口頭】[▶ 03:00](https://youtu.be/vOPwwRCZ8q8?t=180) 比喻：編譯時就像出發健行**之前仔細規劃路線**；執行時只要照著規劃好的路走。

**好處**：

1. 執行時的額外成本最小。
2. [▶ 04:00](https://youtu.be/vOPwwRCZ8q8?t=240) 編譯時可以做**大量的圖優化**，例如把不用算、不用更新的張量**剪掉**。
3. 最後產生的執行檔**很小**。

### 1.2 流程

[▶ 04:00](https://youtu.be/vOPwwRCZ8q8?t=240)

1. **前端**：PyTorch 定義的模型 → 追蹤出靜態圖 → **前向 IR**。
2. **前向 IR → 反向 IR**：[▶ 05:00](https://youtu.be/vOPwwRCZ8q8?t=300)
   - 【口頭】教授問：前向的 1 個卷積，反向要幾個？答案是 **2 個**：一個算對輸入的梯度（轉置卷積），一個算對權重的梯度。再加上 bias 的梯度。
   - 【口頭】這就是為什麼**反向的 FLOPs 大約是前向的 2 倍**。
3. **圖優化**：[▶ 06:00](https://youtu.be/vOPwwRCZ8q8?t=360) 稀疏層 / 稀疏張量更新、運算子重排、原地（in-place）更新、常數摺疊、死碼消除。
4. **調排程（tune schedules）**。
5. **程式碼生成**：只產生**真的有用到**的運算子的二進位檔。

### 1.3 稀疏更新怎麼做到：標註 + 死碼消除

[▶ 06:00](https://youtu.be/vOPwwRCZ8q8?t=360)

以一個線性層 $y = Wx + b$ 為例，反向要算三個東西：

| 梯度 | 運算 |
|---|---|
| $\partial y/\partial x$ | 乘法 |
| $\partial y/\partial W$ | 乘法（**要用到激活值 $x$**） |
| $\partial y/\partial b$ | 加總 |

| 更新方式 | 做法 | 結果 |
|---|---|---|
| **只更新 bias** | [▶ 07:00](https://youtu.be/vOPwwRCZ8q8?t=420) 把 $W$ 標成 `need_grad = False` | 自動刪掉算 $\partial y/\partial W$ 的程式碼，**不用存激活值** |
| **稀疏層更新** | [▶ 08:00](https://youtu.be/vOPwwRCZ8q8?t=480) 第 1 層的 $W$、$b$ 都標 False，第 2 層標 True | 只算第 2 層 |
| **稀疏張量更新** | [▶ 09:00](https://youtu.be/vOPwwRCZ8q8?t=540) 標成「一半」 | 只**切出一半**的張量（例如 20 維切出 10 維）來乘，計算和記憶體都省 |

- 靠**相依性分析 + 死碼消除**，從 DAG 裡刪掉不必要的計算。
- 【口頭】可以**自由標註任何參數**，PockEngine 會自動修剪計算圖。

### 1.4 運算子重排：梯度算完馬上用掉

[▶ 10:00](https://youtu.be/vOPwwRCZ8q8?t=600)

| | 傳統做法 | **重排後** |
|---|---|---|
| 順序 | 先算完**所有**梯度，再一個一個套用 | 每一層的梯度**一算完就馬上套用** |
| 梯度緩衝區 | 要等到最後才能釋放 → **記憶體浪費** | 馬上釋放 |

```
傳統：F0 F1 F2 F3 → B3 B2 B1 B0 → U3 U2 U1 U0
重排：F0 F1 F2 F3 → B3 U3 → B2 U2 → B1 U1 → B0 U0
```

（F = 前向、B = 反向、U = 更新）

- 峰值記憶體減少 **2.7～3.1 倍**（在 MCU 上，不同模型都一致）。
- [▶ 12:00](https://youtu.be/vOPwwRCZ8q8?t=720) 加上運算子融合和原地梯度更新，記憶體緩衝區的生命週期大幅縮短。

### 1.5 結果

**MCU 上的延遲**：[▶ 13:00](https://youtu.be/vOPwwRCZ8q8?t=780)

- TF-Lite 稀疏更新比完整更新快 2 倍以上。
- **PockEngine 稀疏更新比 TF-Lite 快 21～23 倍**（MobileNetV2、ProxylessNAS、MCUNet）。

**記憶體一路壓到 256 KB 以下**：[▶ 15:00](https://youtu.be/vOPwwRCZ8q8?t=900)

| 框架 / 技術 | 訓練記憶體 | 這一步省 |
|---|---|---|
| TensorFlow（雲端） | 652 MB | — |
| PyTorch（雲端） | 303 MB | — |
| MNN（邊緣） | 41.5 MB | — |
| **PockEngine** | 5.7 MB | 7.3× |
| + 量化感知縮放（INT8） | 2.9 MB | 2.0× |
| + 稀疏層 / 張量更新 | 355 KB | 8.8× |
| + 運算子重排 | **141 KB** | 2.4× |
| **總計** | | **約 2300×** |

【口頭】三個數量級的記憶體節省，而且**不用特殊硬體**。

### 1.6 擴展到更多平台

[▶ 16:00](https://youtu.be/vOPwwRCZ8q8?t=960)

支援 CNN 和 Transformer；前端支援 PyTorch、TensorFlow、JAX；後端支援 Apple M1、Raspberry Pi、手機、DSP、MCU……

| 平台 | 加速 | 原因 |
|---|---|---|
| **邊緣 GPU**（Jetson Nano、Orin） | **2～4×** | 編譯：Python 在低頻 CPU 上很慢。TensorRT 等推論框架也靠編譯，但不支援訓練 |
| **ARM CPU**（Raspberry Pi 4） | [▶ 17:00](https://youtu.be/vOPwwRCZ8q8?t=1020) **13～21×** | 現有框架主要優化推論；ARM CPU 的訓練**非常缺乏優化**；編譯能做 kernel 調校 |
| **Apple M1/M2** | [▶ 18:00](https://youtu.be/vOPwwRCZ8q8?t=1080) 更快、更相容 | 【口頭】PyTorch、TensorFlow 在 M1 上常出錯。PockEngine 直接編譯到 **Metal** |
| **DSP、MCU** | | 整合 Qualcomm SNPE 和 TinyEngine |

⭐ 關鍵：[▶ 18:00](https://youtu.be/vOPwwRCZ8q8?t=1080) PockEngine 的編譯流程**跟 kernel 無關**，前向和反向**共用同一套運算子**。所以本來只支援推論的框架（例如 SNPE、TensorRT），也能拿來訓練。

**展示**：[▶ 20:00](https://youtu.be/vOPwwRCZ8q8?t=1200) MCU 上有紅綠兩個按鈕，對著不同物體按按鈕標註，幾次迭代後就學會分辨兩類，全部在裝置上進行，約每秒 3 幀。另一個展示：在 MCU 上學會「有人 / 沒人」，本來不會，訓練後就會了。

---

## 2. 為什麼 LLM 微調要「省參數」

[▶ 23:00](https://youtu.be/vOPwwRCZ8q8?t=1380)

**本地微調的好處**：

- **安全**：資料留在裝置上，例如寫程式的 copilot。【口頭】很多程式碼是專有的，不想分享。
- **客製化**：一個通用的基礎模型 → 很多專門的 LLM：程式設計師 copilot、金融 LLM、法律 LLM、個人助理……

**跟上一講的差別**：[▶ 24:00](https://youtu.be/vOPwwRCZ8q8?t=1440)

| 模型 | Optimizer | 誰佔最多記憶體 |
|---|---|---|
| ResNet-50、MobileNetV2 | SGD + momentum（1 份 state） | **激活值** |
| BERT、Llama | **Adam**（2 份 state） | **參數 + optimizer state** |

- 【口頭】Llama 7B、70B 的激活值相對很小。Adam 的 optimizer state 是權重的 2 倍以上。
- 尤其 **batch size 小**時，參數更是主導。
- → 要把微調塞進個人裝置，**微調必須省參數（parameter-efficient）**。

---

## 3. ⭐ BitFit、Adapter、Prompt Tuning、Prefix Tuning

### 3.1 BitFit：只微調 bias

[▶ 26:00](https://youtu.be/vOPwwRCZ8q8?t=1560)

- 【口頭】上一講的 TinyTL 已經探索過只微調 bias，在視覺模型上效果很好。
- 只更新模型的 bias（或其中一部分）。
- BERT-base 有 **110M 參數**，但 bias 只有 **0.1M**（少 1000 倍以上）。
- **小到中型資料集**：跟完整微調差不多，有時更好。
- **大資料集**：[▶ 27:00](https://youtu.be/vOPwwRCZ8q8?t=1620) 比完整微調差。【口頭】合理，只動 bias 的容量太小。

### 3.2 Adapter：在 Transformer 裡插入可學習的層

[▶ 28:00](https://youtu.be/vOPwwRCZ8q8?t=1680)

- 插入一些小的層，**初始化成 identity**，只微調這些層。
- 每個任務只加一點參數；**新增任務不用動舊的**。

**省儲存空間**：

| | 1000 個子任務 |
|---|---|
| **完整微調**（每個任務一份 Llama 7B） | 投影片寫 14 PB |
| **Adapter**（每個任務 14 MB） | 14 GB |

⚠️ 待確認：$7\text{B}\times2\ \text{bytes}\times1000 = 14\ \text{TB}$，不是投影片和教授說的 14 PB。

**缺點**：[▶ 29:00](https://youtu.be/vOPwwRCZ8q8?t=1740) adapter 要**插在原本的網路裡**，推論時一定會多跑幾層。

- 【口頭】額外延遲從 2～5% 到 20～30% 都有，**batch size 小、序列短**時特別明顯。
- 【投影片補充】數據來自 GPT-2 medium 在 RTX 8000 上 100 次單次前向的平均。

### 3.3 Prompt Tuning：不改模型，改 prompt

[▶ 30:00](https://youtu.be/vOPwwRCZ8q8?t=1800)

**例子**：一個 11B 的預訓練模型，三個任務：A（例如翻譯）2 筆、B（情感分類）1 筆、C（摘要）2 筆。

| | **傳統微調** | **Prompt Tuning** |
|---|---|---|
| 模型 | 3 個微調過的模型 | **1 個**原本的模型 |
| batch | A 跑 2 筆、B 跑 1 筆、C 跑 2 筆 | **5 筆一起跑** |
| 問題 | 【口頭】GPU 擅長批次處理，每個模型只跑 1～2 筆會讓 GPU「餓肚子」 | — |

**做法**：[▶ 31:00](https://youtu.be/vOPwwRCZ8q8?t=1860)

- 每個任務學一段**連續的 prompt 向量**，接在輸入前面。
- 端到端訓練這段 prompt，**預訓練模型完全不動**。
- 推論時：任務 prompt + 原本的輸入 → Transformer。

| | 離散 prompt（人工） | 連續 prompt（學出來的） |
|---|---|---|
| 例子 | 「Please tell me the sentiment of the following text:」 | 一串可學習的向量 |

**好處**：[▶ 33:00](https://youtu.be/vOPwwRCZ8q8?t=1980)

- 【口頭】模型**從來不用碰**，在模型上做的量化、計算圖優化都不受影響。
- 不同任務的 prompt 可以**混在同一個 batch**。【口頭】從 LLM 的角度看，「A 的 prompt + A1」跟「C 的 prompt + C2」沒有差別。

**結果**：[▶ 34:00](https://youtu.be/vOPwwRCZ8q8?t=2040) 模型越大越好：

- 人工設計 prompt：最差。
- 完整微調：最好。
- **Prompt tuning**：模型夠大時，**幾乎追上完整微調**。

（Lecture 13 §4.4 也講過，見 [Lecture 13 筆記](lecture13_transformer_llm_part2_note.md)。）

### 3.4 Prefix Tuning：每一層都加

[▶ 35:00](https://youtu.be/vOPwwRCZ8q8?t=2100)

| | Prompt Tuning | **Prefix Tuning** |
|---|---|---|
| 加在哪 | 只有**第一層**的輸入 | **每一層**的輸入 |
| 準確率 | 較低 | 一致**比較好** |
| 代價 | 較少 | 每一層都多了 token，計算較多 |

模型本身還是不動。

### 3.5 Prompt / Prefix Tuning 的缺點

[▶ 37:00](https://youtu.be/vOPwwRCZ8q8?t=2220)

兩者都**增加了輸入長度**：

1. **推論延遲變長**。
2. **佔掉可用的序列長度**。【口頭】例如 Llama-2 只有 4K 的上下文，多出來的 token 會擠掉真正能用的長度。

→ **能不能微調又不增加推論延遲？**

---

## 4. ⭐⭐ LoRA 家族：LoRA、QLoRA、LongLoRA

### 4.1 LoRA：可以合併回去的平行分支

[▶ 38:00](https://youtu.be/vOPwwRCZ8q8?t=2280)

在每一層旁邊加一條**低秩分解**的平行分支：

| 矩陣 | 形狀 | 初始化 |
|---|---|---|
| $W$（原本的） | $d\times d$ | 凍結 |
| $A$ | $d\times r$（$d$ → 低秩 $r$） | **高斯分布** |
| $B$ | $r\times d$（$r$ → $d$） | **全部 0** |

$$
h = xW + xAB
$$

**一開始**：$B = 0$ → $AB = 0$ → 輸出跟原本**完全一樣**。

**訓練後**：[▶ 40:00](https://youtu.be/vOPwwRCZ8q8?t=2400) $B$ 不再是 0，分支開始適應任務。

⭐ **關鍵一步：可以合併**

$$
h = xW + xAB = x(W + AB) = xW'
$$

- $AB$ 跟 $W$ 的形狀一樣，直接**逐元素加回去**。
- → **推論時零額外延遲**。

【Q&A】[▶ 39:00](https://youtu.be/vOPwwRCZ8q8?t=2340) 有學生問到 $B$ 初始化為 0 的問題。教授回答的大意：$B$ 一開始是 0，但訓練一陣子之後就不是 0 了。

⚠️ 逐字稿缺字：這段 Q&A 的問題和回答字幕都不完整，看不出學生確切問了什麼（可能是「權重是 0 的話梯度會不會也是 0」）。

**實例（Stable Diffusion v1.5 + 社群做的 LoRA）**：[▶ 41:00](https://youtu.be/vOPwwRCZ8q8?t=2460)

- 「水墨山水」LoRA：同樣的 prompt，畫出真正的水墨風格。
- 「細節增強」LoRA：城市、日出、鐵路，細節更好。
- [▶ 43:00](https://youtu.be/vOPwwRCZ8q8?t=2580) 「熊貓造型的壽司捲」：原本的模型畫不出熊貓，加了 LoRA 就可以。

【Q&A】[▶ 42:00](https://youtu.be/vOPwwRCZ8q8?t=2520) 學生問：LoRA 跟 ControlNet 一樣嗎？教授：不一樣，兩者**互不衝突**。ControlNet（[Lecture 16 §2.4](lecture16_diffusion_model_note.md)）只用在擴散模型，用來注入影像條件；LoRA 是通用技術，LLM 和擴散模型都能用，兩者還可以一起用。

### 4.2 QLoRA：連主幹也量化

[▶ 43:00](https://youtu.be/vOPwwRCZ8q8?t=2580)

【口頭】只微調旁路，能不能**把主幹也壓縮**？

| | LoRA | **QLoRA** |
|---|---|---|
| 主幹 | 16-bit | **4-bit** |
| Adapter | 16-bit | 16-bit |
| Optimizer state | GPU | 需要時**換到 CPU**（paged optimizer） |

**三個技術**：

1. **NF4（NormalFloat 4-bit）**：[▶ 44:00](https://youtu.be/vOPwwRCZ8q8?t=2640) 新的 4-bit 格式。【口頭】類似 K-means，是**非均勻量化**，**0 附近的格點比較多**。
   【投影片補充】16 個值從 $-1.0$、$-0.696$、$-0.525$……到 $0$、$0.0796$……$0.723$、$1.0$。
2. **雙重量化**：量化用的 scale 也量化。
3. **Paged optimizers**：CPU 卸載。

**結果**：[▶ 45:00](https://youtu.be/vOPwwRCZ8q8?t=2700) 讓中低階 GPU 也能微調 LLM。【口頭】教授說記憶體大約從 26 GB 降到 21 GB，準確率相當，**改善約 20～25%，沒有預期的大**。

⚠️ 待確認：教授舉的 26 GB → 21 GB 是哪個模型，字幕聽不清楚。

### 4.3 LongLoRA：便宜地延長上下文

#### 問題

[▶ 45:00](https://youtu.be/vOPwwRCZ8q8?t=2700)

- 想讀**一整本書**（例如 3 萬個 token）再回答問題。
- Llama 只能吃 4K token。直接丟進去會說「不支援這麼長」。
- 用長資料完整微調：**很貴**。【口頭】LongLLaMA 要 128 張 TPU v3，大多數人負擔不起。

**LongLoRA 的目標**：[▶ 47:00](https://youtu.be/vOPwwRCZ8q8?t=2820) 在**單一節點（8 張 A100）**上，把 Llama-2 7B 從 4K 延長到 32K。

#### 技術 1：Shifted Sparse Attention（S²-Attn）

[▶ 47:00](https://youtu.be/vOPwwRCZ8q8?t=2820)

- 序列一長，attention map 隨長度**平方**成長，計算量很大。
- **分組**：把 token 分成幾組，**只在組內**做完整 attention。組之間沒有 attention → 容量會損失。
- **移動半組**：[▶ 48:00](https://youtu.be/vOPwwRCZ8q8?t=2880) 把**一半的 head** 的 token 移動半組。原本 1～4 一組，移動後 3～6 一組，資訊就能跨組流動。

三步驟：[▶ 50:00](https://youtu.be/vOPwwRCZ8q8?t=3000)

1. **切 head**：把 attention head 分成兩半。
2. **移動 token**：第二半的 token 移動半組。
3. **重組**：分組，每組內各自做 attention。

（跟 Swin 的 shifted window 是同一個想法，見 [Lecture 14 §3.1](lecture14_vision_transformer_note.md)。）

- **訓練時**用 S²-Attn，**推論時**用完整 attention。【口頭】[▶ 51:00](https://youtu.be/vOPwwRCZ8q8?t=3060) 這很關鍵。
- 實作很簡單，幾行 PyTorch。
- [▶ 53:00](https://youtu.be/vOPwwRCZ8q8?t=3180) 【口頭】把 pattern 1、pattern 2 **在不同 attention head 之間交錯**效果最好；也比 dilated、block sparse、strided 等其他稀疏 attention 好。

#### 技術 2：強化版 LoRA

[▶ 49:00](https://youtu.be/vOPwwRCZ8q8?t=2940)

- 除了 LoRA 旁路，**也微調輸入 embedding 和正規化層**。
- 這兩種層很便宜：【投影片補充】正規化層 < 0.004%、embedding < 2% 的參數。
- 【口頭】沒有這兩個，連原本的 LoRA 在長上下文都**收斂不好**。

⚠️ 逐字稿缺字：[▶ 55:00](https://youtu.be/vOPwwRCZ8q8?t=3300) 教授口頭說正規化層「少於 0.4%」，投影片寫 **< 0.004%**。

#### 結果

[▶ 52:00](https://youtu.be/vOPwwRCZ8q8?t=3120)

- 困惑度一路到 **64K** 都很低；一般 LoRA（沒有強化）困惑度很高。
- 完整微調在 32K 之後 GPU 記憶體就不夠了，LongLoRA 能推到 64K。
- 訓練時間大幅減少（【口頭】教授說「從 990 小時減到 52 小時」）。

⚠️ 待確認：[▶ 53:00](https://youtu.be/vOPwwRCZ8q8?t=3180) 訓練時間的數字字幕寫「990 hours to 52 hours」，投影片的圖抽不出文字，無法核對。

**檢索測試**：[▶ 56:00](https://youtu.be/vOPwwRCZ8q8?t=3360)

- **Passkey 檢索**：在一大堆無關的句子（「草是綠的、天是藍的……」）中間藏一個密碼，最後問密碼是什麼。
- Llama-2 7B：超過約 5K 長度後，準確率從 100% 掉到 0%。
- **LongLoRA**：一直到 **32K** 都維持住；再加上**位置內插（position interpolation，[Lecture 12](lecture12_transformer_llm_note.md)）**還能更長。
- 【口頭】要總結很長的論文、複雜的財報時很重要。

---

## 5. ⭐ Prompt Engineering

[▶ 58:00](https://youtu.be/vOPwwRCZ8q8?t=3480)

【口頭】前面的方法都要微調模型。這一段**完全不用微調**，只靠設計 prompt 得到想要的輸出。

### 5.1 Zero-shot

[▶ 59:00](https://youtu.be/vOPwwRCZ8q8?t=3540)

| | 以前（非大型語言模型） | LLM |
|---|---|---|
| 模型和任務 | **一個任務一個模型**：英翻法 BERT、情感 BERT、摘要 BERT…… | **一個模型做很多任務** |
| 原因 | — | 模型夠大時出現**湧現能力（emergent abilities）** |

- 同一個基礎模型，只換 prompt：「翻成西班牙文」「翻成韓文」「翻成中文」……甚至「翻成 16 進位 ASCII」。

### 5.2 Few-shot（in-context learning）

[▶ 1:00:00](https://youtu.be/vOPwwRCZ8q8?t=3600)

- 在 prompt 裡**給幾個示範**，LLM 就能做**沒訓練過的新任務**。也叫 **in-context learning（上下文學習）**。

**例子**：

- **編新詞**：先定義一個虛構動物「whatpu」並造一個句子，再定義一個虛構動作「farduddle」→ 模型會照樣造出合理的句子。
- **二元情感**：[▶ 1:02:00](https://youtu.be/vOPwwRCZ8q8?t=3720) 示範「正面 → 0x00」「負面 → 0x01」，再問一句負面的話 → 模型答 0x01。

⚠️ 逐字稿缺字：[▶ 1:01:00](https://youtu.be/vOPwwRCZ8q8?t=3660) 字幕把虛構詞寫成「watero」「fudo」「bingo」，依投影片是 **whatpu**、**farduddle**、**Burringo**。

**Few-shot 的技巧**：[▶ 1:03:00](https://youtu.be/vOPwwRCZ8q8?t=3780)

| 技巧 | 好的做法 | 壞的做法 |
|---|---|---|
| **類別平均** | 分類時，每一類給差不多數量的例子（例如鳥 5 個、魚 5 個） | 鳥 1 個、魚 10 個 |
| **格式一致** | [▶ 1:04:00](https://youtu.be/vOPwwRCZ8q8?t=3840) 每個翻譯例子都用同樣的分隔符號（例如 `-`） | 混用 `-`、`/`、「should be translated into」、換行，甚至顛倒語言順序 |

### 5.3 Chain-of-Thought（思維鏈）

[▶ 1:05:00](https://youtu.be/vOPwwRCZ8q8?t=3900)

**想法**：示範**中間的推理步驟**，讓模型能做複雜推理。從 A 到 D 很遠，就示範 A → B → C → D，而不是直接跳到 D。

**例子**：[▶ 1:06:00](https://youtu.be/vOPwwRCZ8q8?t=3960)

| | 示範的寫法 | 新問題（有 23 顆蘋果，用掉 20 顆，又買了 6 顆） |
|---|---|---|
| **一般 few-shot** | 「Roger 有 5 顆網球，又買 2 罐、每罐 3 顆 → **答案 11**」 | 答 27 ❌ |
| **Chain-of-thought** | 「一開始 5 顆，2 罐 × 3 顆 = 6 顆，5 + 6 = 11 → 答案 11」 | $23 - 20 = 3$，$3 + 6 = 9$ → 答 **9** ✅ |

**Zero-shot CoT**：[▶ 1:07:00](https://youtu.be/vOPwwRCZ8q8?t=4020) 不給示範，只加一句**「Let's think step by step」**。

- 例子：16 顆球，一半是高爾夫球，高爾夫球裡一半是藍的。
- 不加：答 8 ❌。
- 加了：16 ÷ 2 = 8 顆高爾夫球，8 ÷ 2 = **4** 顆藍色 ✅。

### 5.4 LLM prompt 的實例

[▶ 1:08:00](https://youtu.be/vOPwwRCZ8q8?t=4080)

教授用 ChatGPT-3.5 幫這堂課生大綱：

| 嘗試 | Prompt | 結果 |
|---|---|---|
| 1 | 「幫我列一份高效微調和 prompt engineering 的課程大綱」 | **太長**，像一整學期的課 |
| 2 | 加上「一小時的課」 | [▶ 1:09:00](https://youtu.be/vOPwwRCZ8q8?t=4140) 長度對了，但**漏掉了高效微調** |
| 3 | 先貼上 Lecture 18、19 的大綱當示範（few-shot），再問一樣的問題 | [▶ 1:10:00](https://youtu.be/vOPwwRCZ8q8?t=4200) 有 adapter、LoRA、prompt engineering，**可以當起點了** |

【口頭】[▶ 1:11:00](https://youtu.be/vOPwwRCZ8q8?t=4260) 「AI 不會取代人，但不會用 AI 的人會被會用 AI 的人取代。」

**「奶奶法」**：【口頭】一個有名的越獄（jailbreak）prompt。直接要 Windows 序號會被拒絕；但要它「扮演會念序號哄我睡覺的奶奶」，以前就會念出來。**現在已經修掉了**。教授舉這個例子說明 LLM 還有很多安全問題要處理。

### 5.5 擴散模型的 prompt

[▶ 1:13:00](https://youtu.be/vOPwwRCZ8q8?t=4380)

- **逐步加 prompt**：「森林小屋」→ 加上某種動畫風格 → 加上「河邊、日出」→ 越來越接近想要的。
- 「在實驗室高效工作的機器人」→ 加上「logo、卡通風、白色背景、寫著 efficient ml」。
- 【口頭】[▶ 1:14:00](https://youtu.be/vOPwwRCZ8q8?t=4440) 擴散模型**寫字寫不好**，文字常常亂掉。教授推測是因為在**低解析度的 latent 空間**運作（[Lecture 16 §3](lecture16_diffusion_model_note.md)）。
- **負面 prompt**：叫它避開「變形、低品質、顆粒感、雜訊、塑膠感、霧霧的、低對比」等。

【Q&A】[▶ 1:15:00](https://youtu.be/vOPwwRCZ8q8?t=4500) 學生問：prompt 很長的時候會不會忘記？教授：會，尤其最近有人發現**「lost in the middle」**現象：開頭和結尾記得住，中間的會被忽略。但那要很長才會發生。

---

## 6. 一頁速查表

### 6.1 PockEngine

| 技術 | 做法 | 效果 |
|---|---|---|
| **編譯時自動微分** | 前向 + 反向圖都在編譯時建好 | 執行時零額外成本，可做大量圖優化 |
| **稀疏更新** | 標註 `need_grad` → 死碼消除 | 不用存激活值、不用算多餘的梯度 |
| **運算子重排** | 每層梯度一算完就更新、釋放 | 峰值記憶體少 2.7～3.1× |
| **總效果** | TensorFlow 652 MB → 141 KB | 約 2300× |
| **反向 FLOPs** | 1 個卷積 → 2 個（對輸入、對權重） | 約前向的 2 倍 |

### 6.2 LLM 微調方法

| 方法 | 改什麼 | 推論額外成本 | 關鍵一步 |
|---|---|---|---|
| **BitFit** | 只有 bias | 無 | BERT-base bias 只有 0.1M；大資料集不夠 |
| **Adapter** | 插入的小層 | **有**（多跑幾層） | 每任務 14 MB |
| **Prompt Tuning** | 第一層前面的連續 prompt | 輸入變長 | 多任務混在同一個 batch |
| **Prefix Tuning** | **每一層**前面的 prompt | 輸入變長 | 比 prompt tuning 準 |
| **LoRA** | $h = xW + xAB$ | **無** | $A$ 高斯、$B = 0$；$W' = W + AB$ 合併 |
| **QLoRA** | LoRA + 4-bit 主幹 | 無 | NF4（0 附近格點多）+ 雙重量化 + paged optimizer |
| **LongLoRA** | LoRA + embedding + norm | 無 | 訓練用 S²-Attn（分組 + 一半 head 移半組），推論用完整 attention |

### 6.3 Prompt Engineering

| 方法 | 做法 | 例子 |
|---|---|---|
| **Zero-shot** | 直接描述任務 | 「翻成韓文：」 |
| **Few-shot** | 給幾個示範（類別平均、格式一致） | 示範 0x00 / 0x01 |
| **Chain-of-thought** | 示範中間推理步驟 | $23 - 20 + 6 = 9$ |
| **Zero-shot CoT** | 加「Let's think step by step」 | $16/2/2 = 4$ |
| **擴散模型** | 逐步加 prompt + 負面 prompt | 擴散模型寫字不好 |

### 6.4 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **2.7～3.1×** | 運算子重排的峰值記憶體節省 |
| **21～23×** | PockEngine 在 MCU 上比 TF-Lite 快 |
| **141 KB / 2300×** | PockEngine 的最終訓練記憶體 / 總節省 |
| **2～4× / 13～21×** | PockEngine 在 Jetson / Raspberry Pi 上的加速 |
| **110M vs 0.1M** | BERT-base 的參數 vs bias |
| **14 MB** | 每個任務的 adapter |
| **4K → 32K** | LongLoRA 在 8 張 A100 上延長 Llama-2 7B 的上下文 |
| **< 0.004% / < 2%** | 正規化層 / embedding 佔的參數比例 |

---

## 7. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **第 20 講結束「訓練」章節** | **[Lecture 21 量子計算基礎](lecture21_quantum_computing_basics_note.md)** 開始講**量子機器學習** |
| **§1 編譯器把訓練轉成高效執行** | **Lecture 22-23**：量子電路的編譯與優化 |

---

## 附：本集一句話總結

> **PockEngine 把自動微分搬到編譯時，靠標註和死碼消除實現稀疏更新、靠重排讓梯度一算完就釋放，把訓練記憶體壓到 141 KB；LLM 微調的瓶頸是參數，所以只動一小部分：BitFit 只動 bias、adapter 插小層、prompt / prefix tuning 只學 prompt，而 LoRA 的低秩旁路能合併回權重、推論零成本，QLoRA 再把主幹壓成 4-bit，LongLoRA 用分組加移動的稀疏 attention 便宜地延長上下文；最後，prompt engineering 連微調都不用。**

---

## ⚠️ 待確認清單

1. [▶ 29:00](https://youtu.be/vOPwwRCZ8q8?t=1740) 1000 個 Llama 7B 的儲存：$7\text{B}\times2\ \text{bytes}\times1000 = 14\ \text{TB}$，投影片和教授都說 14 PB。
2. [▶ 45:00](https://youtu.be/vOPwwRCZ8q8?t=2700) QLoRA 記憶體 26 GB → 21 GB 是哪個模型，字幕聽不清楚。
3. [▶ 53:00](https://youtu.be/vOPwwRCZ8q8?t=3180) LongLoRA 的訓練時間「990 小時 → 52 小時」，投影片的圖抽不出文字，無法核對。

---

*筆記依據 Song Han「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 20 錄影逐字稿（英文自動字幕，一般版，1:17:33）與投影片 `Lec20-Efficient-Fine-tuning-and-Prompt-Engineering.pdf`（95 頁）整理。*
