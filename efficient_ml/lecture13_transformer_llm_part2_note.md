---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 13
lecture: 13
date: 2023-10-24
video: https://youtu.be/caEDCA3Jo_4
notes: Lec13-Transformer-and-LLM-Part-II.pdf（79 頁）
---

# 第 13 集 · Lecture 13：Transformer 與 LLM（下） [▶ 影片](https://youtu.be/caEDCA3Jo_4)

← 上一集：[Lecture 12 Transformer 與 LLM（上）](lecture12_transformer_llm_note.md) ｜ [總目錄](進度.md) ｜ 下一集：Lecture 14 Vision Transformer（未完成） →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 13 集
> **講者**：Song Han
> **來源**：YouTube 錄影逐字稿（1:17:03，英文自動字幕，Zoom 版）＋ 投影片 `Lec13-Transformer-and-LLM-Part-II.pdf`（79 頁）
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
- [1. ⭐⭐ LLM 量化：SmoothQuant、AWQ、TinyChat](#1--llm-量化smoothquantawqtinychat)
  - [1.1 複習：量化在做什麼](#11-複習量化在做什麼)
  - [1.2 問題：舊方法用在大 LLM 上會壞掉](#12-問題舊方法用在大-llm-上會壞掉)
  - [1.3 SmoothQuant（W8A8）](#13-smoothquantw8a8)
  - [1.4 為什麼還需要 W4A16](#14-為什麼還需要-w4a16)
  - [1.5 AWQ（W4A16）](#15-awqw4a16)
  - [1.6 TinyChat：在邊緣裝置上跑 LLM](#16-tinychat在邊緣裝置上跑-llm)
- [2. ⭐ LLM 剪枝與稀疏](#2--llm-剪枝與稀疏)
- [3. ⭐⭐ LLM 推論系統](#3--llm-推論系統)
  - [3.1 複習：KV cache 有多大](#31-複習kv-cache-有多大)
  - [3.2 vLLM 與 PagedAttention](#32-vllm-與-pagedattention)
  - [3.3 StreamingLLM 與 Attention Sink](#33-streamingllm-與-attention-sink)
  - [3.4 FlashAttention](#34-flashattention)
  - [3.5 Speculative Decoding](#35-speculative-decoding推測解碼)
- [4. ⭐ 高效微調：LoRA、QLoRA、Adapter、Prompt Tuning](#4--高效微調loraqloraadapterprompt-tuning)
- [5. 業界連結：TensorRT-LLM](#5-業界連結tensorrt-llm)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與後續集數的連結](#7-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/caEDCA3Jo_4?t=0)

上一講（[Lecture 12](lecture12_transformer_llm_note.md)）講 LLM **長什麼樣**。這一講講怎麼讓它**變小、變快、好客製**。

| 段落 | 內容 | 投影片 |
|---|---|---|
| **I. 高效推論演算法** | 量化（SmoothQuant、AWQ、TinyChat）、剪枝／稀疏（Wanda、SpAtten、H2O、DejaVu、MoE） | 前段 |
| **II. 高效推論系統** | vLLM、StreamingLLM、FlashAttention、Speculative decoding | 中段 |
| **III. 高效微調** | LoRA／QLoRA、Adapter、Prompt Tuning | 後段 |

【口頭】[▶ 01:00](https://youtu.be/caEDCA3Jo_4?t=60) 教授說這一講「非常即時」：NVIDIA 的 **TensorRT-LLM** 上週剛開源，裡面用的量化方法就是本講的 SmoothQuant 和 AWQ。

---

## 1. ⭐⭐ LLM 量化：SmoothQuant、AWQ、TinyChat

### 1.1 複習：量化在做什麼

[▶ 01:00](https://youtu.be/caEDCA3Jo_4?t=60)

把浮點數的範圍對應到整數格點上（[Lecture 5-6](lecture05_06_quantization_note.md)）：有 $q_{\min}$、$q_{\max}$，還有一個 zero point 對應到浮點的 0。2-bit 的例子就是四個格點：−2、−1、0、1。

### 1.2 問題：舊方法用在大 LLM 上會壞掉

[▶ 02:00](https://youtu.be/caEDCA3Jo_4?t=120)

- W8A8（權重 8-bit、激活值 8-bit）在 CNN 上早就是**業界標準**，但直接用在 LLM 上不行。
- 投影片的圖：OPT 系列從 1.3B 到 175B，**超過 6.7B 之後**，W8A8 的準確率大幅下降，FP16 則維持住。
- 原因：**模型變大後，激活值（activation）裡會出現系統性的離群值（outlier）**。

### 1.3 SmoothQuant（W8A8）

#### 觀察

[▶ 02:00](https://youtu.be/caEDCA3Jo_4?t=120)

| | **激活值 X** | **權重 W** |
|---|---|---|
| **數值範圍** | 0 到 70 以上，少數幾個 channel 特別大 | 很平，大約 0 到 1 |
| **好不好量化** | **很難** | **非常容易** |

好消息：**離群值固定出現在某幾個 channel**，不是隨機亂跳。

#### 核心想法：把量化難度從激活值「搬」到權重

[▶ 03:00](https://youtu.be/caEDCA3Jo_4?t=180)

矩陣乘法是線性的，所以可以在中間插入一個對角矩陣和它的反矩陣：

$$
Y = XW = \left(X\,\mathrm{diag}(s)^{-1}\right)\cdot\left(\mathrm{diag}(s)\,W\right) = \hat{X}\hat{W}
$$

- 激活值的大 channel **除以 $s$**（變平），權重對應的那一列**乘以 $s$**（變得稍微不平）。
- 數學上完全相等，但兩邊現在**都好量化**了。
- 【口頭】教授的直覺版本：激活值的某個 channel 乘 0.1，權重那邊乘 10。權重本來很小，乘 10 還是很小。

#### 縮放係數怎麼算

[▶ 04:00](https://youtu.be/caEDCA3Jo_4?t=240)

$$
s_j = \frac{\max\left(|X_j|\right)^{\alpha}}{\max\left(|W_j|\right)^{1-\alpha}},\quad j = 1, 2, \dots, C_i
$$

- $\alpha$ 控制「搬多少難度」給權重。
- 【口頭】教授講例子時用的是開根號，也就是 $\alpha = 0.5$。

**投影片的小例子**：

```
X（2×4）            W（4×3）            s
 1  -16   2   6      2   1  -2         1  4  1  3
-2    8  -1  -9      1  -1  -1
                     2  -1  -2
                    -1  -1   1
```

第 2 個 channel 一步一步算：

1. $\max|X_2| = 16$，$\max|W_2| = 1$。
2. $s_2 = \sqrt{16/1} = 4$。
3. $\hat{X}$ 第 2 欄：$-16, 8$ 除以 4 → $-4, 2$。
4. $\hat{W}$ 第 2 列：$1, -1, -1$ 乘以 4 → $4, -4, -4$。
5. 檢查：$(-16)\times 1 = (-4)\times 4$，乘積不變。

【口頭】[▶ 05:00](https://youtu.be/caEDCA3Jo_4?t=300)「天下沒有白吃的午餐」：權重本來是 1、2 這種超好量化的數，現在變得**稍微難一點，但還在掌控之中**。

#### 跑的時候有沒有額外成本？沒有

[▶ 06:00](https://youtu.be/caEDCA3Jo_4?t=360)

【口頭】教授問：乘上 $s^{-1}$ 和 $s$，跑的時候要不要多算？他自己回答：

- **權重那邊**：$\mathrm{diag}(s)W$ 可以**離線先算好**，直接存成新權重。
- **激活值那邊**：$s$ 是固定常數，跟輸入無關，所以可以**合併進前一層 LayerNorm 的縮放係數**（呼應 [Lecture 12 §2.7](lecture12_transformer_llm_note.md)）。
- 結論：**數學上相等，跑的時候零額外成本**。

#### 系統實作：哪些層量化？

[▶ 07:00](https://youtu.be/caEDCA3Jo_4?t=420)

- **所有計算密集的運算都量化成 INT8**：Q/K/V 投影、兩個 BMM（$QK^\top$ 和乘 $V$）、輸出投影、FFN 的 FC1/FC2。
- **留在 FP16**：LayerNorm、Softmax。

#### 結果

[▶ 08:00](https://youtu.be/caEDCA3Jo_4?t=480)

| 模型 | 結果 |
|---|---|
| **OPT-175B** | 6.7B 以上不再崩，記憶體**減半**，服務用的 GPU 從 **8 張變 4 張**，延遲也更低 |
| **MT-NLG 530B** | 準確率維持住，GPU 數**減半**、延遲差不多，可以放進**單一節點** |
| **LLaMA 7B～65B** | LLaMA 用了 SwiGLU 和 RoPE，SmoothQuant 一樣幾乎無損（下表） |

| WikiText PPL ↓ | LLaMA 7B | 13B | 30B | 65B |
|---|---|---|---|---|
| FP16 | 11.51 | 10.05 | 7.53 | 6.17 |
| SmoothQuant | 11.56 | 10.08 | 7.56 | 6.20 |

【口頭】教授算了一筆錢：一張 A100 大約 2 萬美元，8 張減到 4 張就**直接省 8 萬美元**。一整個節點大約 18 萬（A100）到 20～30 萬美元（H100），530B 模型可以少用一半的節點。

【口頭】[▶ 09:00](https://youtu.be/caEDCA3Jo_4?t=540) 延遲變低是因為 INT8 比 FP16 小，GPU 之間**要傳的資料變少**。

### 1.4 為什麼還需要 W4A16

[▶ 11:00](https://youtu.be/caEDCA3Jo_4?t=660)

- **W8A8 適合大 batch**（例如 batch 128）的服務。
- **單一使用者（例如在自己電腦上跑）時，LLM 解碼嚴重受限於記憶體頻寬。**
- 投影片的 roofline 圖（A100、LLaMA-65B 解碼）：bs=128 接近峰值，**bs=1 的利用率非常低**。

【口頭】[▶ 12:00](https://youtu.be/caEDCA3Jo_4?t=720) 教授的算術：

1. Llama-2 7B 用 FP16 存：$7\times10^9 \times 2\ \text{bytes} = 14\ \text{GB}$。
2. 自回歸解碼**每生一個 token 就要讀一遍 14 GB**。
3. 權重是 $4\text{K}\times4\text{K}$，單一 batch 的激活值只有 $1\times4\text{K}$，**小了約一千倍**。
4. 所以要壓的是**權重**，激活值留 FP16 → **只量化權重（weight-only quantization）**。
5. 權重變 4-bit：$14\ \text{GB} \div 4 = 3.5\ \text{GB}$。

> 「計算很便宜，記憶體很貴。」—— 同 [Lecture 11](lecture11_tinyengine_note.md) 的結論。

### 1.5 AWQ（W4A16）

AWQ = Activation-aware Weight Quantization。

#### 起點：直接量化會壞

[▶ 13:00](https://youtu.be/caEDCA3Jo_4?t=780)

- OPT-6.7B、INT3、group size 128（每 128 個數共用一個 scale），用最簡單的「四捨五入到最近格點」（round-to-nearest，RTN），**perplexity 大幅上升**。
- Perplexity（困惑度）：衡量語言模型預測下一個字準不準，**越低越好**。

#### 觀察：只有約 1% 的權重很關鍵

[▶ 15:00](https://youtu.be/caEDCA3Jo_4?t=900)

- **只把 1% 的 channel 留在 FP16**，其他照樣量化，perplexity 就幾乎回到原本的水準。
- 【口頭】教授留下兩個待辦：
  1. **怎麼挑出那 1%？**
  2. **混合精度（1% FP16 + 99% INT）很難寫 kernel，能不能全部都用 INT？**

#### 待辦 1：看激活值，不是看權重

[▶ 16:00](https://youtu.be/caEDCA3Jo_4?t=960)

| 挑選依據 | 效果 |
|---|---|
| 隨機挑 | 沒用 |
| **看權重大小**（剪枝時的做法） | 只小幅改善 ❌ |
| ⭐ **看激活值大小** | **大幅改善** ✅ |

- 理由：SmoothQuant 已經發現某些激活 channel 特別大，而且**不同輸入都一樣大**。乘上大激活值的權重，誤差會被放大，所以它們才重要。
- 這就是名字的由來：**量化的是權重，但要看激活值決定誰重要**。

#### 待辦 2：不用混合精度，改用「放大」保護

[▶ 18:00](https://youtu.be/caEDCA3Jo_4?t=1080)

- 把重要 channel 的權重**乘以 $s > 1$**，對應的激活值**除以 $s$**。數學上相等，跟 SmoothQuant 同一招。
- ⚠️ 待確認：下表數字是從投影片長條圖上讀的（OPT-6.7B、Wiki-2 PPL），沒有原始數據可對。

| 做法 | PPL |
|---|---|
| FP16 | 12.29 |
| RTN | 43.16 |
| 重要 channel ×1.5 | 14.49 |
| ×2 | 14.07 |
| ×4 | 14.42 |
| **自動搜尋 $s$** | **13.18** |

- 先降後升 → 有最佳值 → 要**自動搜尋**。

#### ⭐ 為什麼放大 $s$ 倍，誤差會變小？

[▶ 19:00](https://youtu.be/caEDCA3Jo_4?t=1140)

一個 channel $y = wx$。量化函數：

$$
Q(w) = \Delta \cdot \mathrm{Round}\left(\frac{w}{\Delta}\right),\qquad \Delta = \frac{\max(|w|)}{2^{N-1}}
$$

放大後的版本：

$$
Q(w\cdot s)\cdot\frac{x}{s} = \Delta' \cdot \mathrm{Round}\left(\frac{s\,w}{\Delta'}\right)\cdot x \cdot \frac{1}{s}
$$

一步一步看誤差：

1. **Round() 的誤差平均約 0.25**：四捨五入的誤差落在 0 到 0.5 之間，平均是 1/4。放大 $s$ 不改變這件事。
   ⚠️ 逐字稿缺字：字幕寫「The average is 0.225. It's a quarter.」，依投影片應為 **0.25**。
2. **$\Delta$ 幾乎不變**：[▶ 22:00](https://youtu.be/caEDCA3Jo_4?t=1320) $\Delta$ 只看這一組（128 個數）的最大值。只放大一個 channel，**很少剛好碰到最大值**，所以 $\Delta' \approx \Delta$。
3. **外面多了 $1/s$**：誤差約為 $\Delta \cdot 0.25 \cdot |x| \cdot \frac{1}{s}$。$s > 1$，所以誤差**縮小成原本的 $1/s$**。

【Q&A】[▶ 22:00](https://youtu.be/caEDCA3Jo_4?t=1320) 學生問：放大是在量化之前還是之後？教授：**之前**。先放大，再量化。

#### 實作與結果

[▶ 23:00](https://youtu.be/caEDCA3Jo_4?t=1380)

- 放大和縮小可以**合併進前一個運算**，形式跟 SmoothQuant 很像。【口頭】所以業界可以用**同一套基礎設施**跑 SmoothQuant 或 AWQ。
- 用資料驅動的**快速 grid search** 找最佳 $s$。【口頭】後續有論文改用梯度下降去學 $s$。
- LLaMA／Llama-2（7B～70B）INT3-g128：AWQ 一致優於 RTN、GPTQ、GPTQ-R。
- **多模態也有效**：[▶ 24:00](https://youtu.be/caEDCA3Jo_4?t=1440)
  - OpenFlamingo-9B 看圖說話：RTN 把「兩架停在草地上的玩具飛機」說成「一架在天上飛的模型飛機」，AWQ 說對了。
  - LLaVA 視覺推理：AWQ 版能讀懂圖片裡的文字（等於自動做了 OCR）。

#### 業界採用

[▶ 26:00](https://youtu.be/caEDCA3Jo_4?t=1560)

SmoothQuant 和 AWQ 被 NVIDIA FasterTransformer／TensorRT-LLM、Intel Neural Compressor、vLLM、HuggingFace TGI、FastChat、LMDeploy 等採用。

【口頭】Lab 4 就是做 SmoothQuant 和 AWQ，Lab 5 會在你的筆電上真的跑起來。

### 1.6 TinyChat：在邊緣裝置上跑 LLM

[▶ 27:00](https://youtu.be/caEDCA3Jo_4?t=1620)

**為什麼要在本地跑？**

- 程式補全、Office、遊戲聊天這些 copilot 服務，在筆電、車、機器人上跑很有用。
- 這些裝置**資源有限、低功耗、不一定有網路**。
- **隱私**：【口頭】尤其是寫程式，企業資料很敏感，不想上傳雲端。

**怎麼做到的：演算法 × 系統一起設計**

[▶ 29:00](https://youtu.be/caEDCA3Jo_4?t=1740)

| 層面 | 技術 |
|---|---|
| **演算法** | 4-bit AWQ 量化 |
| **系統** | TinyEngine 的技巧（[Lecture 11](lecture11_tinyengine_note.md)）：loop unrolling、tiling、cache locality、多執行緒、CUDA；**4-bit 權重在記憶體中怎麼排**，以及執行時把 4-bit 解碼回 FP16 |

【口頭】[▶ 30:00](https://youtu.be/caEDCA3Jo_4?t=1800) 權重 4-bit、激活值 FP16：讀進來時把權重**即時解碼成 FP16，再用 FP16 算**。瓶頸在記憶體，不在計算。教授說這跟他當年的 **EIE** 是同一個想法。

**數字**：

| 平台 | 模型 | 速度 |
|---|---|---|
| RTX 4090 | Llama-2-7B FP16 | 50 tokens/s |
| RTX 4090 | Llama-2-7B **W4A16 AWQ** | **166 tokens/s（約 3.3×）** |
| RTX 4090 | MPT-7B / Falcon-7B / Vicuna-7B | 31 / 22 / 33 tokens/s |
| Jetson Orin | Llama-2-7B W4A16 | 30 tokens/s |

⚠️ 逐字稿缺字：[▶ 31:00](https://youtu.be/caEDCA3Jo_4?t=1860) 字幕說「Koala 7B」，投影片寫 **Vicuna-7B**，以投影片為準。

【口頭】TinyChatEngine 是 C/C++ 從零寫的，容易移植。教授鼓勵大家把繼續改進 TinyChat 當成**期末專題**的選項之一。

---

## 2. ⭐ LLM 剪枝與稀疏

### 2.1 Wanda：剪權重時也要看激活值

[▶ 32:00](https://youtu.be/caEDCA3Jo_4?t=1920)

- 跟 AWQ 同一個想法：LLM 的激活值有超大離群值，所以要**保護乘上大激活值的權重**。
- 剪枝標準：

$$
\text{重要度}_{ij} = |W_{ij}| \cdot \left\| X_j \right\|
$$

- 一致優於只看權重大小的 magnitude pruning（[Lecture 3-4](lecture03_04_pruning_note.md)）。
- 【口頭】結論：**用「權重 × 激活值」的絕對值當標準，不要只看權重本身。**

### 2.2 SpAtten：剪 token、剪 head

[▶ 34:00](https://youtu.be/caEDCA3Jo_4?t=2040)

【口頭】attention 裡的 $QK^\top \to \mathrm{softmax} \to \times V$ **沒有權重，全部都是激活值**，所以只能剪 token。好消息是能剪的 token 很多。

| 技巧 | 做法 |
|---|---|
| **級聯 token 剪枝** | 用**累積注意力分數**判斷，分數小的 token 丟掉，一層一層慢慢剪 |
| **級聯 head 剪枝** | 整個 attention head 拿掉 |
| **V 剪枝** | $QK^\top$ 很小的話，就**不用去讀對應的 V** |
| **漸進式量化** | 先用低精度算。softmax 分布很「平」代表沒把握，再去讀高精度的位元；很「尖」就不用 |

**例子**（情感分類）：

- 「As a visual treat, the film is almost perfect.」11 個 token。
- 第一層剪成 5 個：treat、film、perfect 等。
- 下一層剪成 2 個：film、perfect。
- 還是判斷得出「正面」。

### 2.3 H2O：剪 KV cache 裡的 token

[▶ 37:00](https://youtu.be/caEDCA3Jo_4?t=2220)

- 【口頭】SpAtten 三年後的論文，把同樣的想法用在**生成式（自回歸）**模式。
- 自回歸只有下三角的 attention map，看每一欄的**累積注意力**。
- 保留兩種 token：**最近的 token** 和 **heavy hitter（H2，累積注意力很高的 token）**，其他從 KV cache 丟掉。

⚠️ 逐字稿缺字：字幕寫「Hydra Heater」，依投影片應為 **Heavy Hitter**。

### 2.4 DejaVu：依輸入決定剪哪裡

[▶ 38:00](https://youtu.be/caEDCA3Jo_4?t=2280)

| | **靜態稀疏** | **情境稀疏（contextual sparsity）** |
|---|---|---|
| **怎麼剪** | 固定剪某些參數 | **依輸入不同**，剪不同的 head 和 FFN channel |
| **效果** | 中高稀疏度時準確率掉很快 | 剪很多也能維持準確率 |

- 用一個**預測器**，根據目前的特徵預測下一層哪些 head／channel 可以跳過。
- 預測器跟 attention／MLP 的計算**平行跑**，把它的成本藏起來。

### 2.5 MoE：每個 token 只用一部分參數

[▶ 39:00](https://youtu.be/caEDCA3Jo_4?t=2340)

- **router（路由器）**決定每個 token 送去哪個 expert（專家，其實就是一個 FFN）。
- 總參數量可以變很大，但**每個 token 的推論成本不變**。
- expert 越多 → 總模型越大 → loss 越低。

**Capacity factor（容量係數）$C$**：[▶ 40:00](https://youtu.be/caEDCA3Jo_4?t=2400)

$$
\text{每個 expert 的上限} = \frac{\text{每批 token 數}}{\text{expert 數}} \times C
$$

| 6 個 token、3 個 expert | 上限 | 結果 |
|---|---|---|
| **$C = 1$** | $6/3 \times 1 = 2$ | 有 3 個 token 都想去 expert 1 → **1 個被跳過** |
| **$C = 1.5$** | $6/3 \times 1.5 = 3$ | 有餘裕 |

**路由方式**：[▶ 41:00](https://youtu.be/caEDCA3Jo_4?t=2460) 每個 token 挑 top-k 個 expert；或每個 expert 挑要處理的 token；或全域一起決定。

---

## 3. ⭐⭐ LLM 推論系統

### 3.1 複習：KV cache 有多大

[▶ 42:00](https://youtu.be/caEDCA3Jo_4?t=2520)

- 自回歸解碼時，要存所有前面 token 的 Key 和 Value；新 token 只需要算自己的 Query。
- KV cache **隨序列長度線性變大**。

投影片的算式（Llama-2-70B，假設用 MHA）：

$$
\underbrace{BS}_{\text{batch}} \times \underbrace{80}_{\text{層數}} \times \underbrace{64}_{\text{KV head}} \times \underbrace{128}_{\text{head 維度}} \times \underbrace{N}_{\text{長度}} \times \underbrace{2}_{K,V} \times \underbrace{2\ \text{bytes}}_{\text{FP16}} \approx 2.5\ \text{MB} \times BS \times N
$$

| 設定 | KV cache |
|---|---|
| bs=1，長度 512 | 1.25 GB |
| bs=1，長度 4096 | 10 GB |
| ⭐ bs=16，長度 4096 | **160 GB（要兩張 A100）** |

⚠️ 待確認：[▶ 43:00](https://youtu.be/caEDCA3Jo_4?t=2580) 兩個來源講法不同。
- **逐字稿**：「Llama 7B 每 token 2.5 MB」「6 個使用者、4K 長度需要 160 GB」。
- **投影片**：**Llama-2-70B（MHA）**、**bs=16**。
- 投影片的算式自己算得通（$80\times64\times128\times2\times2 = 2{,}621{,}440$ bytes ≈ 2.5 MB），也跟 [Lecture 12 §6.2](lecture12_transformer_llm_note.md) 一致，所以上表用投影片。

### 3.2 vLLM 與 PagedAttention

⚠️ 逐字稿缺字：字幕把 vLLM 寫成「VRM」「VROM」「VL LM」。

#### 問題：KV cache 的三種浪費

[▶ 44:00](https://youtu.be/caEDCA3Jo_4?t=2640)

服務多個使用者時，要**事先保留**每個請求的 KV cache 空間（執行時才配置太慢）。但你不知道它會生多長，只好**多保留一點**。

| 浪費 | 原因 |
|---|---|
| **內部碎片** | 不知道輸出多長，**保留太多**（例如 2,000 個 token 的位置沒用到） |
| **預留** | 現在還沒用到，但未來的 token 會用到，現在就得先占著 |
| **外部碎片** | 不同請求長度不同，中間的空隙沒法用 |

#### 解法：借用作業系統的「虛擬記憶體 + 分頁」

[▶ 45:00](https://youtu.be/caEDCA3Jo_4?t=2700)

【口頭】教授問「這讓你想到什麼？」，自己答：**作業系統**。不同 process 的記憶體也有碎片問題，OS 用 **page table（分頁表）**多加一層間接對應來解決。

| 作業系統 | vLLM |
|---|---|
| process | 請求（request） |
| 虛擬頁 | **邏輯 KV block** |
| 實體頁 | **實體 KV block** |
| page table | **block table** |

**運作方式**：[▶ 46:00](https://youtu.be/caEDCA3Jo_4?t=2760)

- 每個 block 放固定數量的 token（投影片例子：每 block 4 個 token）。
- **邏輯上連續，實體上可以不連續**。block table 記錄「邏輯 block → 實體 block」和「已經填了幾格」。
- 一個 block 填滿才配置下一個 → **最多只浪費一個 block**。
- 不同請求的 block 可以**交錯放在同一塊實體記憶體**，不會有外部碎片。[▶ 49:00](https://youtu.be/caEDCA3Jo_4?t=2940)

#### 額外好處：共用 prompt

[▶ 50:00](https://youtu.be/caEDCA3Jo_4?t=3000)

- **平行取樣（parallel sampling）**：同一個 prompt 生好幾個不同的回答。
- prompt 的 KV block 讓多個回答**共用**，不用存好幾份。
- 【口頭】例子：寫程式時 Copilot 給你三個建議讓你挑，這三個共用同一份 prompt 的 KV cache。

### 3.3 StreamingLLM 與 Attention Sink

⚠️ 逐字稿缺字：字幕把 StreamingLLM 寫成「streaming ORAM」「streaming LM」，KV cache 寫成「KB cache」。

#### 問題：串流應用（很長的多輪對話）

[▶ 51:00](https://youtu.be/caEDCA3Jo_4?t=3060)

【口頭】例如想跟聊天機器人一直聊下去（教授舉了「虛擬女友」的例子），要怎麼讓它不忘記、也不爆記憶體？

兩個失敗模式：

1. **記憶體爆掉**：KV cache 一直長大。
2. **模型品質崩壞**：超過訓練時的長度，perplexity 暴衝。

#### 現有做法都不行

[▶ 54:00](https://youtu.be/caEDCA3Jo_4?t=3240)

| 做法 | 複雜度 | PPL | 問題 |
|---|---|---|---|
| (a) **Dense attention** | $O(T^2)$ | 5641 | 超過訓練長度就壞，又慢 |
| (b) **Window attention**（只留最近 $L$ 個） | $O(TL)$ | 5158 | **第一個 token 被丟掉時就崩** |
| (c) **滑動視窗 + 重算** | $O(TL^2)$ | 5.43 | 品質好，但每個新 token 都要重算整個視窗，**太慢** |
| ⭐ (d) **StreamingLLM** | $O(TL)$ | **5.40** | 又快又穩 |

$T$ = 總 token 數，$L$ = cache 裡的 token 數。

複雜度怎麼來的：
- (a) 第 $t$ 個 token 看前面 $t$ 個，加總 $\sum_{t=1}^{T} t = O(T^2)$。
- (b) 每個 token 只看 $L$ 個，共 $T$ 個 → $O(TL)$。
- (c) 每個新 token 都要把 $L$ 個 token 的 attention 從頭算一次，一次 $O(L^2)$，共 $T$ 次 → $O(TL^2)$。

#### 關鍵發現：Attention Sink（注意力匯集點）

[▶ 56:00](https://youtu.be/caEDCA3Jo_4?t=3360)

- **最前面幾個 token 會拿到超大的注意力分數**，即使它們沒什麼意義。從第 2 層之後，很多層都這樣。
- 這種 token 叫 **attention sink**：不管跟內容有沒有關，都吸走大量注意力。

【口頭】[▶ 57:00](https://youtu.be/caEDCA3Jo_4?t=3420) 教授說他們**三年前在 SpAtten（HPCA 2021）就看過這個現象**：

- 「He is a very famous researcher in the computer architecture area.」後面的字都很注意第一個字「He」，但「He」其實沒什麼意義。
- 用 GPT-2 做 token 剪枝時，「Du Fu was a great poet of the Tang Dynasty…」剪掉的是「Fu」而不是「Du」。這不合理，原因是「Du」是第一個 token，是 sink。
- 「三年前就看到了，但直到現在才真正理解為什麼。」

#### 為什麼會有 Attention Sink？

[▶ 58:00](https://youtu.be/caEDCA3Jo_4?t=3480)

1. **softmax 必須加總為 1**（呼應 [Lecture 12 §2.3](lecture12_transformer_llm_note.md)）：就算沒有哪個 token 真的相關，注意力也得分配到某個地方。模型就學會**把多的注意力倒給第一個 token**。
2. **第一個 token 對後面所有 token 都看得到**：在自回歸模型裡，只有它是每個 token 都看得到的。

**是位置還是語意？** [▶ 59:00](https://youtu.be/caEDCA3Jo_4?t=3540) 把前 4 個 token 換成 4 個「`\n`」（換行，沒有任何意義），perplexity 一樣救得回來 → **是位置，不是語意**。

#### StreamingLLM 的做法

[▶ 1:00:00](https://youtu.be/caEDCA3Jo_4?t=3600)

- **永遠保留前 4 個 token（attention sink）的 KV**，其他用滑動視窗。
- 不需要重新訓練。
- **位置編碼用「在 cache 裡的位置」**，不是原本在文章裡的位置：

```
生成 token 9 時：
cache 裡實際是：  0 1 2 3 | 6 7 8 9      （4、5 已經被丟掉）
分配的位置：      0 1 2 3 | 4 5 6 7      ← 不是 6 7 8 9
```

**跟 PagedAttention 整合**：把 **KV cache 的第一頁固定住，永遠不丟**，再改位置編碼就好。【口頭】雖然只需要 4 個 token，一頁可能有 16 個，有點浪費，但很好實作。vLLM 在過去兩週已經整合了。

#### 結果

[▶ 1:01:00](https://youtu.be/caEDCA3Jo_4?t=3660)

- Llama-2、MPT、Falcon、Pythia 都能穩定處理到 **400 萬個 token**。
- 跟唯一能用的基準（滑動視窗 + 重算）比，**最高 22.2× 加速**。

⚠️ 待確認：[▶ 1:02:00](https://youtu.be/caEDCA3Jo_4?t=3720) 兩個來源講法不同。
- **逐字稿**：延遲「從 1,400 降到 65」、「20 倍加速」。
- **投影片**：**22.2×**。
- 1400 / 65 ≈ 21.5，接近但不等於 22.2，可能是不同模型的數字。上面用投影片。

#### 要幾個 attention sink？

[▶ 1:03:00](https://youtu.be/caEDCA3Jo_4?t=3780)

【口頭】教授口頭講的 PPL：0 個 sink 爆掉；1 個約 11；2 個約 10；4 個約 9；8 個差不多 → **4 到 8 之間變平，所以選 4**。

**能不能只用 1 個？可以，但要從頭訓練**：[▶ 1:04:00](https://youtu.be/caEDCA3Jo_4?t=3840)

- 在每個訓練樣本最前面加一個**可學習的專用 sink token**。
- 這樣訓練出來的模型，串流時**只要 1 個 sink 就夠**；一般模型要 4 個。
- 預訓練 loss 甚至稍微好一點（但只在小模型上驗證過）。
- 【口頭】教授在課堂上**徵求合作**，想在更大的模型上驗證。也提到後來有人在 Mistral（用 window attention 訓練的）上驗證，sink 一樣需要。

### 3.4 FlashAttention

[▶ 1:05:00](https://youtu.be/caEDCA3Jo_4?t=3900)

- **問題**：attention 要產生一張 $N\times N$ 的表。全部寫進 GPU 主記憶體（HBM）很慢、很占空間。
- **解法**：
  - **Tiling（切塊）**：一小塊一小塊在快速的 **SRAM** 上算，**不把整張 $N\times N$ 表寫出來**。
  - **Kernel fusion（合併運算）**：$QK^\top$、縮放、softmax、乘 $V$ 合成**一個 kernel**。
- **結果**：[▶ 1:06:00](https://youtu.be/caEDCA3Jo_4?t=3960) 記憶體用量變小、速度變快。投影片有 H100 上的訓練加速圖，有沒有 causal mask 都快很多。
- 跟 [Lecture 11](lecture11_tinyengine_note.md) 同一個思路：**少讀寫慢的記憶體**。

### 3.5 Speculative Decoding（推測解碼）

[▶ 1:06:00](https://youtu.be/caEDCA3Jo_4?t=3960)

**問題**：解碼一次生一個 token，每個 token 都要讀一遍整個模型（7B FP16 就是 14 GB），**小 batch 時嚴重受限於記憶體**。

**兩個模型**：

| 角色 | 例子 | 工作 |
|---|---|---|
| **Draft model（草稿模型）** | 7B 小模型 | **一個一個**先猜 $K$ 個 token |
| **Target model（目標模型）** | 175B 大模型 | 把 $K$ 個 token **一次平行送進去**檢查 |

**流程**：[▶ 1:07:00](https://youtu.be/caEDCA3Jo_4?t=4020)

1. 小模型自回歸地猜出 $K$ 個 token。
2. 大模型一次處理這 $K$ 個（相當於 batch = $K$），算出每個位置的機率。
3. 決定接受哪些。第一個不對的地方換成大模型的答案，再從那裡重新開始。

**為什麼會快？**

- 大模型一次處理 $K$ 個 token，**計算密度（arithmetic intensity）變高**，不再被記憶體卡住。
- 【口頭】$K$ 不太大時，GPU 處理 $K$ 個 token 跟處理 1 個**時間差不多**，因為平行度夠高。
- 【口頭】[▶ 1:08:00](https://youtu.be/caEDCA3Jo_4?t=4080) 運氣好的話，大部分 token 都是小模型猜對的。

【投影片補充】**2～3× 加速，而且輸出完全一樣。**

【口頭】教授特別提醒：**這不是知識蒸餾**（[Lecture 9](lecture09_kd_note.md)）。兩個模型同時存在，小模型負責猜，大模型負責檢查。

---

## 4. ⭐ 高效微調：LoRA、QLoRA、Adapter、Prompt Tuning

[▶ 1:09:00](https://youtu.be/caEDCA3Jo_4?t=4140)

【口頭】問題：怎麼客製一個有幾千億參數的模型？

### 4.1 LoRA（Low-Rank Adaptation）

- **凍結原本的權重 $W$**，在旁邊加一條**低秩的旁路（bypass）**：

$$
h = Wx + BAx,\qquad W\in\mathbb{R}^{d\times k},\ B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r \ll \min(d,k)
$$

- [▶ 1:10:00](https://youtu.be/caEDCA3Jo_4?t=4200) $A$ 用**隨機值**初始化，$B$ 用**零**初始化 → 一開始 $BA = 0$，旁路沒有任何效果。
- 要學的參數從 $dk$ 個變成 $r(d+k)$ 個。

【投影片補充】LoRA 的好處：

1. **微調更快**：不用算大部分權重的梯度。
2. **省記憶體**：optimizer state 變很小。
3. **避免災難性遺忘**（catastrophic forgetting）：原模型沒動。
4. **可以合併**：訓練完把 $BA$ 加回 $W$，推論時零額外成本。

GPT-3 的結果：用少很多的可訓練參數，就能達到很好的準確率。【口頭】教授提醒圖的 x 軸是**對數刻度**，所以省的量非常大。

### 4.2 QLoRA：LoRA + 量化基底模型

[▶ 1:11:00](https://youtu.be/caEDCA3Jo_4?t=4260)

【口頭】LoRA 省了 optimizer state，但**基底模型還是 FP16**。QLoRA 把基底模型也壓成 4-bit。

| 技術 | 內容 |
|---|---|
| **NF4（NormalFloat 4-bit）** | 新的資料格式。【口頭】類似 K-means，為**常態分布**找出最佳格點。需要查表解碼 |
| **雙重量化（double quantization）** | 不只量化權重，**連量化用的 scale 也量化** |
| **Paged optimizers** | optimizer state 放不下時**卸載到 CPU** |

前向時用 4-bit 基底模型，更新時用 FP16 更新 LoRA 旁路。

### 4.3 Adapter

[▶ 1:12:00](https://youtu.be/caEDCA3Jo_4?t=4320)

| | **LoRA** | **Adapter** |
|---|---|---|
| **位置** | 跟主路徑**並聯** | 跟主路徑**串聯**（插在 attention、FFN 後面） |
| **能不能合併** | ✅ 可以 | ❌ **不能**，推論時一定多一層（【投影片補充】） |

### 4.4 Prompt Tuning

[▶ 1:12:00](https://youtu.be/caEDCA3Jo_4?t=4320)

【口頭】上面三種都要動到模型（改旁路、改量化後的權重、改 adapter），不同任務就要不同模型。能不能**完全不動模型，只改 prompt**？

| | **Prompt engineering（人工）** | **Prompt tuning（學出來的）** |
|---|---|---|
| **prompt 形式** | 文字，例如「Please summarize the following text:」 | **連續的向量**，用反向傳播訓練 |
| **怎麼來** | 人寫規則 | 學習 |

**最大好處：不同任務可以放在同一個 batch**　[▶ 1:13:00](https://youtu.be/caEDCA3Jo_4?t=4380)

- 模型都一樣，只是前面接的 task prompt 不同。
- 任務 A、B、C 各自的 prompt 接在輸入前面，**一起批次推論**。
- 以前不同任務要不同權重，現在一套權重就夠。

**結果**（SuperGLUE）：[▶ 1:14:00](https://youtu.be/caEDCA3Jo_4?t=4440) **模型越大，prompt tuning 越接近完整微調**的準確率，也明顯優於人工設計的 prompt。

---

## 5. 業界連結：TensorRT-LLM

[▶ 1:15:00](https://youtu.be/caEDCA3Jo_4?t=4500)

【口頭】NVIDIA 上週四剛發布 TensorRT-LLM，是目前最先進的 LLM 服務框架。投影片把它的功能列表標了顏色：

| 這兩講已經講過 | 之後會講 |
|---|---|
| MQA / GQA（Lecture 12） | Tensor parallelism（張量平行） |
| In-flight batching、Paged KV cache（= vLLM） | Pipeline parallelism（管線平行） |
| SmoothQuant、GPTQ、AWQ、FP8 量化 | |
| RoPE | |

⚠️ 逐字稿缺字：字幕寫「body high attention, body query attention」，依投影片應為 **multi-head attention、multi-query attention**。

---

## 6. 一頁速查表

### 6.1 量化

| 技術 | 公式 | 關鍵一步 |
|---|---|---|
| **SmoothQuant** | $Y = \left(X\,\mathrm{diag}(s)^{-1}\right)\left(\mathrm{diag}(s)W\right)$ | $s_j = \max\vert X_j\vert^{\alpha} / \max\vert W_j\vert^{1-\alpha}$；$s$ 合併進 LayerNorm 和權重，**零額外成本** |
| **AWQ** | $Q(ws)\cdot\dfrac{x}{s}$ | Round 誤差 ≈ 0.25 不變、$\Delta$ 幾乎不變 → 誤差變成 $\dfrac{1}{s}$ 倍 |
| **Weight-only 的理由** | 7B FP16 = 14 GB，每個 token 讀一遍 | bs=1 時激活值比權重小約 1000 倍 → 只壓權重 |

### 6.2 剪枝與稀疏

| 技術 | 標準或做法 |
|---|---|
| **Wanda** | $\vert W_{ij}\vert\cdot\Vert X_j\Vert$ |
| **SpAtten** | 累積注意力小的 token 剪掉；$QK^\top$ 小就不讀 V |
| **H2O** | 只留最近 token + heavy hitter |
| **DejaVu** | 預測器決定每個輸入要跳過哪些 head／channel |
| **MoE** | 每個 expert 上限 $= \dfrac{\text{token 數}}{\text{expert 數}}\times C$ |

### 6.3 推論系統

| 技術 | 關鍵一步 |
|---|---|
| **KV cache 大小** | $BS\times\text{層}\times\text{KV head}\times d_{\text{head}}\times N\times 2\times 2\ \text{bytes}$；70B（MHA）≈ 2.5 MB × BS × N |
| **vLLM** | 邏輯 block → 實體 block 的對應表，**最多浪費一個 block** |
| **StreamingLLM** | softmax 加總為 1 → 多的注意力倒給第一個 token；保留 4 個 sink + 滑動視窗，位置用 cache 內的位置，$O(TL)$ |
| **FlashAttention** | 切塊在 SRAM 算，**不寫出 $N\times N$ 表** |
| **Speculative decoding** | 大模型一次處理 $K$ 個 token，時間 ≈ 處理 1 個 → 不再被記憶體卡住 |

### 6.4 微調

| 技術 | 公式或做法 | 能不能合併 |
|---|---|---|
| **LoRA** | $h = Wx + BAx$，$B$ 初始為 0，參數 $dk \to r(d+k)$ | ✅ |
| **QLoRA** | LoRA + NF4 基底 + 雙重量化 + paged optimizer | ✅ |
| **Adapter** | 串聯的小模組 | ❌ |
| **Prompt tuning** | 只學 prompt 向量，**多任務放同一個 batch** | 不用改模型 |

### 6.5 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **6.7B** | 超過這個大小，激活值離群值出現，W8A8 開始崩 |
| **8 → 4 張 GPU** | SmoothQuant 讓 OPT-175B 服務的 GPU 減半 |
| **14 GB → 3.5 GB** | Llama-2 7B，FP16 → 4-bit |
| **1%** | AWQ：只有約 1% 的權重 channel 是關鍵的 |
| **50 → 166 tokens/s** | TinyChat 在 4090 上 FP16 → W4A16（約 3.3×） |
| **160 GB** | 70B（MHA）、bs=16、4K token 的 KV cache |
| **4** | StreamingLLM 需要的 attention sink 數 |
| **400 萬** | StreamingLLM 測過的 token 數 |
| **22.2×** | StreamingLLM 相對「滑動視窗 + 重算」的加速 |
| **2～3×** | Speculative decoding 的加速（輸出完全一樣） |

### 6.6 ⭐ 本講反覆出現的兩個主題

```
主題一：看激活值，不要只看權重
  SmoothQuant → 激活值離群值是量化的難點
  AWQ         → 激活值大的 channel，對應的權重才重要
  Wanda       → 剪枝標準 = |W| × ‖X‖

主題二：瓶頸是記憶體，不是計算
  AWQ / TinyChat       → 只壓權重，激活值留 FP16
  vLLM / H2O / Stream  → 管好 KV cache
  FlashAttention       → 不寫出 N×N 表
  Speculative decoding → 讓大模型一次處理 K 個 token
```

---

## 7. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **§1.5 AWQ 用在多模態 LLM** | **Lecture 14 Vision Transformer** |
| **§5 Tensor／pipeline parallelism** | **Lecture 17-18 分散式訓練** |
| **§4 LoRA、Prompt tuning** | **Lecture 20 高效微調與 Prompt Engineering** |
| **§4 微調時的記憶體問題** | **Lecture 19 裝置端訓練** |
| **§1.5-1.6 SmoothQuant、AWQ、TinyChat** | **Lab 4**（[Lab4_zh.md](Lab4_zh.md)）、**Lab 5**（在筆電上跑） |

---

## 附：本集一句話總結

> **讓 LLM 變小變快，靠兩個觀察：一，重要的不是權重本身，而是它乘上的激活值；二，瓶頸是記憶體，不是計算。客製時則盡量少動模型：LoRA 只學旁路，prompt tuning 連模型都不動。**

---

## ⚠️ 待確認清單

1. [▶ 18:00](https://youtu.be/caEDCA3Jo_4?t=1080) AWQ 放大倍數的 PPL 表（12.29 / 43.16 / 14.49 / 14.07 / 14.42 / 13.18）是從投影片長條圖上讀的，數字和欄位的對應沒有原始數據可對。
2. [▶ 43:00](https://youtu.be/caEDCA3Jo_4?t=2580) KV cache 例子：逐字稿說「Llama 7B、6 個使用者」，投影片寫「Llama-2-70B（MHA）、bs=16」。筆記用投影片。
3. [▶ 1:02:00](https://youtu.be/caEDCA3Jo_4?t=3720) StreamingLLM 加速：逐字稿說「1,400 → 65、20 倍」，投影片寫 22.2×。筆記用投影片。

---

*筆記依據 Song Han「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 13 錄影逐字稿（英文自動字幕，1:17:03）與投影片 `Lec13-Transformer-and-LLM-Part-II.pdf`（79 頁）整理。*
