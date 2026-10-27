# EfficientML.ai 第十二講完整筆記：Transformer 與 LLM（上）

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2023 — Lecture 12
> **講者**：Song Han
> **來源**：
>   - 課堂逐字稿：[Lecture 12 錄影](https://youtu.be/mR4u6ZaCYe4)（1:19:53）
>   - 官方投影片：`Lec12-Transformer-and-LLM-Part-I.pdf`（91 頁）
>
> **標記說明**（同 [Lecture 11 筆記](lecture11_tinyengine_note.md) 的慣例）
> - `【口頭】` = 教授課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 課堂上教授反問全班／學生提問
> - `【投影片】` = 投影片有、但這堂課因為超時沒講到的段落
> - 未標記者 = 投影片本身的內容

---

## 目錄

- [0. 這一講在整個課程的位置](#0-這一講在整個課程的位置)
- [1. Transformer 之前：RNN／LSTM／CNN 的死結](#1-transformer-之前rnnlstmcnn-的死結)
- [2. ⭐⭐ Transformer 基礎](#2--transformer-基礎)
- [3. ⭐⭐ Transformer 設計變體](#3--transformer-設計變體)
- [4. ⭐ 大型語言模型（LLM）](#4--大型語言模型llm)
- [5. 進階主題：多模態與 MoE](#5-進階主題多模態與-moe)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與其他課程的連結](#7-與其他課程的連結)

---

## 0. 這一講在整個課程的位置

### 0.1 課程換章了

【口頭】教授開場第一句就是「今天我們要跳進一個新的章節，會非常精彩」。

| | 第 1–11 講 | **第 12 講開始** |
|---|---|---|
| **主題** | **通用**的模型壓縮與高效推論 | **應用專屬**（application-specific）最佳化 |
| **內容** | 剪枝、量化、NAS、知識蒸餾、MCUNet、TinyEngine | **Transformer / LLM**、Vision Transformer、GAN/影片/點雲、Diffusion |
| **可用範圍** | 幾乎任何神經網路 | 針對特定架構的特性去挖 |

【口頭】教授說 Transformer 和 LLM「因為太熱門，會花**兩講**」——本講是上半，下半（Lecture 13）講怎麼把它跑快、跑小。

### 0.2 為什麼要學這個：LLM 已經無所不在

投影片開場列了四個應用：**ChatGPT**、**Google Translate**、**GitHub Copilot**、**多模態 GPT-4V**。

【口頭】教授現場拿 ChatGPT 幫這門課生成大綱，結果它產出的內容「**剛好就是我們課上教的東西**」——高效網路架構、量化、硬體 NPU、TFLite。他還放了 GPT-4V 解幾何題的例子：看圖算出邊長 4 和 8，自己算出斜邊 √(16+64)、再算 tanθ = 4/8。

### 0.3 本講四段結構

| 段落 | 內容 |
|---|---|
| **I. Transformer 基礎** | tokenizer、embedding、attention、FFN、LayerNorm、位置編碼 |
| **II. 設計變體** | encoder/decoder 三派、相對位置編碼、KV cache 最佳化、GLU |
| **III. LLM** | scaling、emergent abilities、GPT-3 in-context learning、各家開源模型 |
| **IV. 進階主題** | 多模態 LLM、MoE（**課堂超時沒講到**） |

---

## 1. Transformer 之前：RNN／LSTM／CNN 的死結

### 1.1 NLP 的兩大類任務

| | **判別式任務**（discriminative） | **生成式任務**（generative） |
|---|---|---|
| **例子** | 情感分析、文本分類、文本蘊涵 | 語言模型、機器翻譯、摘要 |
| **例句** | 「As a visual treat, the film is almost perfect.」→ **Positive / Negative** | 「As a visual treat, the film is almost ___」→ **perfect** |
| **能看到什麼** | 整句都能看（**雙向**） | 只能看前面（**單向**） |

這個「雙向 vs 單向」的差別，之後會一路貫穿到 BERT vs GPT、以及 attention mask 的設計。

### 1.2 RNN／LSTM 的兩個致命問題

**問題一：長距離依賴很難建模**

- 兩個 token 之間要互動，需要 **O(seq_len) 步**。
- 【口頭】教授的比喻很關鍵：
  > **「影像有 locality（局部性），但語言不見得有。我現在講的這句話，可能跟我很久以前講的那句話有關。」**

  所以在中間某個字要拿到很前面某個字（例如 "chef"）的資訊，訊號得穿過**正比於句長**的很多層。

**問題二：訓練無法平行**

- 狀態嚴格依賴前一個狀態，要算第 n 個狀態就得先算完前 n−1 個。
- **依賴性殺死平行度**（dependency kills parallelization）。

### 1.3 CNN 也不行

| | 優點 | 缺點 |
|---|---|---|
| **CNN 做 NLP** | token 之間沒有依賴，**可平行、可 scale** | **視野有限**（受限的 receptive field），建模能力較差 |

【口頭】就算疊很多層，也只能看到一個窗口的文字；而語言沒有 locality，所以卷積**不見得是好的 primitive（基本運算單元）**。

### 1.4 於是：Transformer

要同時拿到「**可平行**」和「**任兩個 token 一步就能互動**」——這就是 2017 年 *Attention Is All You Need* 的答案。

---

## 2. ⭐⭐ Transformer 基礎

整個 Transformer 就是這七步，投影片反覆用同一張圖強調：

1. **Tokenize**：文字 → token
2. **Embedding**：token → 向量
3. **Transformer block**：
   - **Multi-Head Attention（MHA）**
   - **Feed-Forward Network（FFN）**
   - **LayerNorm**
   - **Residual connection**
4. **Positional encoding**（位置編碼）
5. **Linear head** 做最終預測

---

### 2.1 Tokenization：字 → token

- Tokenizer 把一個字對應到**一個或多個** token。
- 投影片的例子：**110 個字 → 162 個 token**。

【口頭】教授看著 OpenAI tokenizer 的畫面解釋（每個顏色是一個 token）：

- 「generative」被切成兩個 token：`gener` + `ative`。
- **逗號本身也可以是一個 token**。
- 大致比例是 **1 個字 ≈ 1.5 個 token**（投影片是 110:162）。
- 記憶方法：**「大致可以把 word 想成 token，但一個 token 可能只是 word 的一小塊。」**

### 2.2 Word Representation：從 one-hot 到 embedding

| | **One-Hot Encoding** | **Word Embedding** |
|---|---|---|
| **做法** | 向量長度 = 詞彙表大小，只有一個位置是 1 | 用查表（look-up table）把 token index 映成**連續向量** |
| **問題／好處** | 詞彙表一大，向量**超長且幾乎全 0**（極稀疏、極浪費） | 維度可控，且能**跟模型一起 end-to-end 訓練** |
| **代表** | — | **Word2Vec、GloVe**（早期先驅） |

【Q&A】教授問全班「one-hot 的問題是什麼？」——答案正是**向量太長、幾乎全零、非常冗餘**。

【口頭】投影片例子只有 3 維，但現實中 word embedding 是**幾百到幾千維**。教授的收尾：**「以後就把每個字想成一個向量。」**

---

### 2.3 ⭐⭐ Self-Attention：整堂課的核心

**做法**：把 embedding E 投影成 **Q（query）、K（key）、V（value）** 三個矩陣。

**QKV 的直覺 —— YouTube 搜尋**（投影片和教授都用這個類比）：

| | 對應到 YouTube |
|---|---|
| **Query** | 你在搜尋框打的字 |
| **Key** | 影片的標題／描述 |
| **Value** | 對應的**影片本身**（很大的檔案） |

流程：拿 query 去比對所有 key 找相似度 → 用相似度當權重 → 對 value 做加權平均。

**公式與形狀**：

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d ) · V

Q, K, V : [N, d]        N = token 數，d = hidden dim
Q Kᵀ    : [N, N]        ← ⭐ 注意力矩陣，O(N²)
輸出     : [N, d]
```

投影片的運算圖（由下往上）：`E[N,d]` → Linear → `Q/K/V[N,d]` → Matmul → `Attn[N,N]` → Mask(opt) → Softmax → `Prob[N,N]` → Matmul with V → `Out[N,d]`。

**幾個要點**：

- **除以 √d**：不同 hidden dimension 之間要能正規化（否則內積數值會隨 d 變大而爆掉）。
- 【Q&A】教授問「為什麼要 softmax？」學生答**「要讓注意力加總為 1」**——正解，因為後面是對 V 做**加權平均**。
- ⭐ **複雜度是 O(N²)**。【口頭】教授強調這很重：一篇學術論文可能就是 **6K 個 token**。
- 【口頭】**預告**：softmax 強迫加總為 1 會帶來 **attention sink（注意力匯點）** 現象，**下一講（Lecture 13）會講**。
- 【口頭】**「非常多公司的面試題就是：10 分鐘內在紙上寫出 self-attention。」**

**Attention map 長什麼樣**：投影片放了英法翻譯的注意力矩陣，主要是**對角線**（字大致按順序對應），但有一段是**反向的**——因為英文和法文那三個字的語序剛好相反。**attention 自動找到了字與字的對應關係。**

**【Q&A】學生問：研究者怎麼想到 QKV 這種設計的？感覺有點憑空冒出來。**

教授的回答（兩個理由）：

1. **為了平行**：不要讓每個 token 都依賴前一個 token。QKV 投影**不依賴前一個矩陣**，所以可以平行算。
2. **借用檢索系統的概念**：query / key / value 本來就是資訊檢索領域的詞彙，是從那邊借過來的。

---

### 2.4 Multi-Head Attention（MHA）：一張圖不夠看

- 一個 attention map 只能捕捉一種語意關係，**我們需要多種**。
- MHA = **H 個平行的 QKV attention 分支**（head），每個 head 產生一張自己的 attention map。
- 最後把所有 head 的輸出 **concat**，再過一個 linear projection 合併特徵。

【口頭】實作上的維度分配（以 4K hidden dim 為例）：

> **32 個 head × 每個 head 128 維 = 4096 維**

總維度不變，但切成多個子空間**各自獨立做轉換**，就能產生多張 attention map。

【口頭】看著 8 個 head 的視覺化，教授誠實地說：**「很難解釋為什麼每個 head 會長成這種 pattern，但它是 end-to-end 學出來的。」**

---

### 2.5 Attention Masking：全域 vs 因果

| | **Global（全域）self-attention** | **Causal / Masked（因果）self-attention** |
|---|---|---|
| **mask 形狀** | **全 1** 的方陣 | **下三角**（左下全 1、右上全 0） |
| **誰能看誰** | 每個 token 看得到**所有** token | 第 t 個 token **只能看 0..t** |
| **用在** | 判別式任務 / **encoder（BERT）** | 生成式任務 / **decoder（GPT）** |

【口頭】例子：token 4 只能 attend 到 token 0~4，看不到未來的 token。

---

### 2.6 Feed-Forward Network（FFN）

**為什麼需要它？**

> **Self-attention 建模的是 token 之間的關係，但它裡面沒有 element-wise 的非線性。**

所以要加一個 FFN，對**每個 hidden dimension** 做逐元素的非線性，幫助特徵建模。

**原始實作**：兩層 MLP，中間放大 4 倍（**inverted bottleneck / 倒瓶頸**），中間用 ReLU 或 GeLU。

```
d --Linear--> 4d --ReLU/GeLU--> 4d --Linear--> d
```

【口頭】現實中的 d 有多大：

| 模型 | hidden dimension d |
|---|---|
| **LLaMA 7B** | **4K**（4096） |
| **GPT-3 175B** | **12K**（12288） |

（§3.5 會看到 FFN 被 **GLU** 取代的變體。）

---

### 2.7 LayerNorm：跟 BatchNorm 差在哪

| | **BatchNorm（CNN 用）** | **LayerNorm（Transformer 用）** |
|---|---|---|
| **沿哪個維度統計** | 沿 **batch + 空間（H×W）**，每個 channel 一組 | **完全不碰 batch 維度**，沿**單一 token 的 feature 維度** |
| **結果** | C 個 mean/var | **每個 token 各自**被拉成 mean=0、var=1 |
| **後面接** | 可學的 affine（γ, β） | 同樣接可學的 affine（γ, β） |

【口頭】以 7B 模型為例：feature 維度是 4K，就是**對這 4096 個數字**做正規化，讓**每個 token** 的均值為 0、變異數為 1。

【Q&A】學生問 **ε（epsilon）** 是幹嘛的 → **防止除以零**。

⭐ 【口頭】**γ 和 β 這兩個參數超級有用**，是後面幾講的伏筆：

1. **只微調 γ、β 就能做 fine-tuning**——很便宜（線性運算），**特別適合搭配 LoRA + 長 context**。
2. **可以把量化的 scaling factor 併進 γ、β 裡**——**下一講的 SmoothQuant / AWQ 就是這樣做的**（也就是 Lab 4）。

---

### 2.8 Transformer Block：Post-norm vs Pre-norm

把 MHA + FFN + LayerNorm + residual 疊起來就是一個 block。

| | **Post-norm**（原始論文） | **Pre-norm**（現在主流） |
|---|---|---|
| **順序** | MHA → **LN** → FFN → **LN** | **LN** → MHA → **LN** → FFN |
| **白話** | 先算，再正規化 | **先正規化，再算** |
| **為什麼** | — | **訓練穩定性更好** |

【口頭】教授解釋 pre-norm 為什麼合理：**「我們想先把每個 token 正規化成 mean 0 / var 1，然後才去算 attention 矩陣。」**

> 記法：**pre-norm = 先 norm 再算；post-norm = 先算再 norm。現在大家都用 pre-norm。**

---

### 2.9 ⭐ Positional Encoding（位置編碼）

**問題**：attention 和 FFN **完全分不出 token 的順序**（不像卷積天生有順序）。

> 這叫 **set encoding（集合編碼，壞）**，我們要的是 **sequence encoding（序列編碼，好）**。

【口頭】為什麼不行：**兩個字靠很近**和**隔兩個字**，意思是不一樣的。

**解法（原始版，絕對位置編碼）**：

- 每個位置一組**獨一無二**的編碼，是 token index **t** 的函數（所以叫 absolute PE）。
- 用 **sin(ω_k · t) / cos(ω_k · t)**，直接**加到原始 word embedding 上**。

【口頭】教授現場做的消融分析（看那張彩色熱圖）：

- 圖有兩個軸：**t（token index）** 和 **i（feature dimension）**。
- **i 小的時候** → 分母小 → **ω 大 → 頻率高** → 沿 token 方向變化很快。
- **i 大的時候** → 分母大 → **ω 小 → 頻率低** → 沿 token 方向幾乎不變。
- **用不同頻率的組合，就能區分出很長的序列**——這就是為什麼要用週期函數：**理論上可以處理無限長的輸入**，而多個維度的不同頻率可以避免「轉了 2π 就撞回原值」的問題。

【Q&A】學生問：有沒有研究不是直接相加，而是做加權平均？
教授：**可以，但現在（原始論文）就是等權重直接相加**——word embedding + positional embedding 加起來就好。§3.2 開始講怎麼改進它。

---

### 2.10 結果：原始 Transformer 的成績

**神經機器翻譯（NMT）**：

- Transformer **超越所有先前發表的模型與 ensemble**。
- 而且訓練成本只有競爭對手的一小部分。【口頭】教授說是**低了三個數量級**。
- **BLEU** = 衡量機器翻譯結果與一組參考翻譯的相似度。

---

## 3. ⭐⭐ Transformer 設計變體

投影片的總表——原始論文的設計大多沿用至今，但有四類被改掉了：

| # | 原始設計 | 後來的變體 |
|---|---|---|
| **1** | Encoder-Decoder | **Encoder-only（BERT）**、**Decoder-only（GPT）**、統一 text-to-text（**T5**） |
| **2** | 絕對位置編碼 | **相對位置編碼**（**ALiBi**、**RoPE**） |
| **3** | MHA | **MQA → GQA**（為了縮小 **KV cache**） |
| **4** | FFN | **GLU**（gated linear unit，**SwiGLU**） |

---

### 3.1 三種架構：Encoder-Decoder / Encoder-only / Decoder-only

| | **Encoder-Decoder** | **Encoder-only** | **Decoder-only** |
|---|---|---|---|
| **代表** | 原始 Transformer、**T5** | **BERT** | **GPT** 系列 |
| **Attention mask** | encoder 全域、decoder 因果 | **全 1（雙向）** | **下三角（因果）** |
| **預訓練目標** | text-to-text | **MLM**（+ NSP） | **預測下一個字** |
| **怎麼用** | prompt 餵給 encoder，decoder 生成答案 | 預訓練完**再 fine-tune** 到下游任務 | 小模型 fine-tune；**大模型直接 zero/few-shot** |

**T5（Encoder-Decoder）**：用**統一的 text-to-text** 格式做各種 NLP 任務。投影片四個例子：

| 任務 | 輸入（encoder） | 輸出（decoder） |
|---|---|---|
| **翻譯** | 英文句 | 德文句 |
| **可接受度（CoLA）** | 「the course is jumping well」 | **not acceptable**（文法對、語意不通） |
| **語意相似度（STS-B）** | 兩句關於犀牛吃草的話 | **3.8**（0–5 分） |
| **摘要** | 一段密西西比州風災新聞 | 「six people hospitalized after a storm…」 |

**BERT（Encoder-only）** = Bidirectional Encoder Representations from Transformers

兩個預訓練目標：

1. ⭐ **Masked Language Model（MLM）**：隨機遮掉 **15%** 的 token（換成特殊的 `[MASK]` token），叫模型**把被遮的字猜回來**，用分類 loss 監督。
2. **Next Sentence Prediction（NSP）**：判斷句子 B 是不是句子 A 的下一句。【口頭】**這個現在比較少用了。**

因為是 encoder，**attention mask 是完整的方陣**——每個字都看得到前面和後面。

**GPT（Decoder-only）** = Generative Pre-trained Transformer

- 預訓練目標**只有一個**：**預測下一個字**（這就是「generative」的由來）。
- attention mask 只有**下三角**。
- 【口頭】投影片的例子「a robot must obey the orders…」：預測時，當前的字**很自然地 attend 到前面的 "a robot"**，因為它就是在指涉那個 robot。
- **小模型（GPT-2）**：預訓練後 fine-tune 到下游任務。
- **大模型**：**zero-shot / few-shot，完全不用 fine-tune**（§4.3）。

---

### 3.2 絕對 vs 相對位置編碼

| | **絕對位置編碼（Absolute PE）** | **相對位置編碼（Relative PE）** |
|---|---|---|
| **加在哪** | 加進 **input embedding** → 因此影響 **Q、K、V 全部** | **影響 attention score**（加 bias，或改動 Q 和 K） |
| **V 會被改嗎** | **會** | ⭐ **不會** |
| **傳播範圍** | 資訊傳遍整個 Transformer | 只作用在 N×N 的 attention map |
| **好處** | 簡單 | ⭐ **可以泛化到訓練時沒看過的長度：train short, test long** |

【口頭】為什麼這件事很急：**你要摘要一整篇論文，可能是 6K token；但模型只訓練過 4K。**絕對編碼的位置索引只到 4K，要怎麼延伸到 6K、8K？

⚠️ 【口頭】教授的重要提醒：**「train short, test long 如果天真地做是行不通的，需要一些技巧才會真的 work。」**（技巧見 §3.4）

---

### 3.3 相對位置編碼之一：ALiBi

**Attention with Linear Biases**（*Train Short, Test Long*, Press et al., 2021）

**核心想法**：位置編碼**只跟相對距離有關，跟絕對索引無關**。

**做法**：不是加到 input embedding，而是**在 attention 矩陣上加一個 offset**，offset 就是相對距離：

```
attention_score(i, j) += m · (j − i)      m 是手動設定的常數
```

【口頭】看那張表就懂了：

- Q2-K1 差 1、Q3-K2 也差 1 → **值一樣**。
- Q5-K1 差 4 → 值是 4。
- **整條對角線的值都相同**——絕對位置在哪裡完全不重要。

> 一句話：**絕對編碼是「加在 word embedding 上」；ALiBi 是「加在 attention 矩陣上」。**

---

### 3.4 ⭐⭐ 相對位置編碼之二：RoPE

**Rotary Positional Embedding**（RoFormer, Su et al., 2021）——**LLaMA 用的就是這個**。

【口頭】教授說：**「這是 Llama 2 的關鍵部分，作業裡會看到，現在非常流行。」**

**做法**：**在 2D 空間裡旋轉 embedding**

1. 把 d 維的 embedding **每兩個元素分成一組** → 得到 **d/2 對**，每一對當成一個 **2D 座標 (x₁, x₂)**。
2. 依照位置 **m** 把這個 2D 座標**旋轉 m·θᵢ 角度**。

```
θᵢ = 10000^( −2(i−1)/d )
```

**⭐ 為什麼底數是 10000 這麼大的數？**（【口頭】教授特別解釋）

> 因為**旋轉滿 2π 就會轉回原點，就分不出來了**。
> 所以每次只能轉一點點，才能容納夠長的序列。
> 但也**不能無限大**，否則轉動量太小，一樣分不出來。
> **「我們需要一個夠大的數字，才能區分更多 token。」**

**⭐ 為什麼它是「相對」的？**

兩個經過 RoPE 轉換的向量做內積時：**複數內積的相位角 = 兩者的相位差**，也就是 **(m − n)**。

> **只有 m 和 n 的差值有影響，絕對值沒有影響** → 這就是相對位置編碼。

**注意**：RoPE **只作用在 Q 和 K，不作用在 V**。

【Q&A】學生確認：「所以你不是把一個數字加到內積上，你是做了旋轉——這是一個新的層？」
教授：**「完全正確。**比單純相加複雜一點，它改的是**計算內積的方式**——不是直接 Q·K，而是 transform(Q)·transform(K)。」

**⭐⭐ RoPE 的殺手級應用：延長 context window**

現況：LLM 訓練時有 context 長度限制（**LLaMA 2k、Llama-2 4k、GPT-4 8k**），超過就壞掉。

**天真做法（會失敗）**：θ 不動，直接讓 m 從 0 跑到 4K。
【口頭】教授：**「這行不通。」**

**正確做法：位置內插（Position Interpolation）**

> 序列變兩倍長，**旋轉角度就砍一半**：

```
m ∈ [0, 2048 × 2)  →  θ′ᵢ = θᵢ / 2
```

【口頭】直覺：**「與其轉兩圈，不如還是轉同樣的總量，但把 4K 個點擠進同一個範圍裡。」**

**成果**：把 LLaMA 的 context 從 **2k 延長到 32k**（Chen et al., 2023）。

---

### 3.5 ⭐⭐ KV Cache：這一講最重要的系統觀念

#### 3.5.1 什麼是 KV cache

GPT 式的 decoding（一次生一個 token）時：

- 每個新 token 都要 attend 到**所有先前的 token**。
- 但先前 token 的 **K 和 V 每次都一樣**——不用重算，**存起來就好**。
- **只有當前的 query token 是新的。**

這份存起來的東西就叫 **KV cache**。

【口頭】教授對照影片講的「沒有 KV cache 會多浪費」：

| | 沒有 KV cache | **有 KV cache** |
|---|---|---|
| **每生一個 token** | 整個 QKᵀ 矩陣**全部重算**（4×4 全算） | **只算新的那一行/列**（紫色部分是快取重用的） |
| **代價** | 大量重複計算 | ⭐ **記憶體，而且隨序列長度線性成長** |

#### 3.5.2 ⭐ KV cache 到底多大：算給你看

```
KV cache 大小 = BS × L × H × d × N × 2 × 2 bytes
                 │     │   │   │   │   │      └── FP16
                 │     │   │   │   │   └───────── K 和 V 兩份
                 │     │   │   │   └───────────── 序列長度
                 │     │   │   └───────────────── 每個 head 的維度（各家都是 128）
                 │     │   └───────────────────── head 數
                 │     └───────────────────────── 層數
                 └─────────────────────────────── batch size
```

| 模型 | 層數 | head 數 | 每個 head 維度 | **每 token 每 batch** |
|---|---|---|---|---|
| **Llama-2-7B** | 32 | 32 | 128 | **512 KB** |
| **Llama-2-13B** | 40 | 40 | 128 | **800 KB** |
| **Llama-2-70B**（若用 MHA） | 80 | 64 | 128 | ⭐ **2.5 MB** |

**換算成現實場景**（Llama-2-70B，2.5 MB × BS × N）：

| 情境 | KV cache | 【口頭】教授的白話 |
|---|---|---|
| **bs=1, N=512** | **1.25 GB** | 一段短對話 |
| **bs=1, N=4096** | ⭐ **10 GB** | **「差不多就是一篇論文」**（4–6K token） |
| **bs=16, N=4096** | ⭐⭐ **160 GB** | **「教室裡 16 個人各丟一篇論文進去，就要兩張 A100」** |

【口頭】教授的重擊：

> **「兩張 A100，一張兩萬美金，就是四萬美金——而且這還只是為了放 KV cache，**
> **不是 activation，也不是權重，就只是 KV cache。**
> **如果整個 MIT 的人都要用這個服務，你能想像嗎？」**

⭐⭐ **而且**：投影片那張圖顯示，**batch size 一大，KV cache 就超過模型權重本身的大小**。

> **這就是為什麼「縮小 KV cache」在 LLM 推論裡是頭號問題。**

---

### 3.6 ⭐⭐ MQA 與 GQA：把 KV head 砍掉

**核心觀察**：query 的 head 數要保留（需要多種注意力模式），但 **key/value 的 head 數可以減少**。

【口頭】用 YouTube 類比：**query 可以有很多種，但我們想減少「影片」的數量。**

| | **MHA**（原始） | **MQA** | **GQA** |
|---|---|---|---|
| **全名** | Multi-Head Attention | **Multi-Query Attention** | **Grouped-Query Attention** |
| **query head** | N | N | N |
| **key/value head** | **N** | ⭐ **1** | ⭐ **G**（典型 **G = N/8**） |
| **KV cache 縮小**（以 70B 的 64 head 為例） | 1× | **64× 小** | **8× 小** |
| **論文** | Vaswani 2017 | Shazeer 2019 | Ainslie 2023 |

⭐ 【口頭】教授給了一個超好記的類比：

> **MHA ↔ 普通卷積**
> **MQA ↔ depthwise 卷積**
> **GQA ↔ grouped 卷積**

**精度呢？**（Llama-2, 30B, 150B tokens 的實驗）

| | 精度 |
|---|---|
| **GQA** | ⭐ **在大模型上跟 MHA 一樣好** |
| **MQA** | **明顯掉一點**——因為資訊丟最多 |

【口頭】結論：**選對 group 數（8）就能在大幅省記憶體的同時保住精度**——所以 **Llama-2-70B 用的就是 GQA**（64 個 query head，**只有 8 個 KV head**）。

---

### 3.7 FFN → GLU（Gated Linear Unit）

**改動**：把兩個矩陣的 FFN 換成**三個矩陣 + 逐元素相乘的閘門**。

| | **原始 FFN** | **SwiGLU** |
|---|---|---|
| **矩陣數** | 2 | ⭐ **3** |
| **中間維度** | **4d** | ⭐ **8d/3**（比 4d 小，**為了維持總計算量相同**） |
| **激活函數** | ReLU / GeLU | **Swish** |
| **結構** | d → 4d → d | d → 8d/3（走兩條路，一條過 Swish）→ **逐元素相乘** → d |

【口頭】為什麼中間維度變成 8/3 d：**因為變成三個矩陣了，要把總計算量拉回一樣。**

**結果**：PPL（perplexity，**越低越好**）明顯改善。**SwiGLU 現在最流行**（LLaMA 用的就是它）。

**激活函數對照**：GeLU 和 Swish 的公式只差一個參數，**整體形狀很像**。

---

## 4. ⭐ 大型語言模型（LLM）

### 4.1 規模成長的曲線

LLM 就是**放大版的 Transformer**，在超大語料（自然語言、程式碼…）上訓練。

| 年份 | 模型 | 參數量 |
|---|---|---|
| 2017 | Transformer | 0.05B |
| 2018 | GPT / **BERT** | 0.11B / 0.34B |
| 2019 | **GPT-2** | **1.5B** |
| 2019 | MegatronLM | 8.3B |
| 2020 | T-NLG | 17B |
| 2020 | **GPT-3** | ⭐ **175B** |
| 2022 | MT-NLG | **530B** |

⚠️ 同一張圖上還畫了 **GPU 記憶體**的成長線（V100 16/32GB → TPUv3 32GB → A100 40GB → A100 80GB）：

> ⭐ **模型大小五年漲了至少三個數量級，GPU 記憶體完全沒跟上。**

【口頭】教授的感慨：**「這正好是我來 MIT 開始教書的時候，看到這種進展實在太驚人了……但供給遠遠追不上需求，這中間的鴻溝就是巨大的機會。」**（這也正是這整門課存在的理由。）

### 4.2 Emergent Abilities（湧現能力）

**LLM 有些能力，只有在模型夠大時才會突然出現。**

投影片的例子：

| 任務 | 內容 |
|---|---|
| **(a) 修改過的算術** | 「-> 代表某個簡單數學運算」：`100 + 200 -> 301`、`1 + 1 -> 3`、`2 + 2 -> 5`（其實是加完再 +1） |
| **(c) 單字解謎** | 「`hte` 是某個英文單字的亂序版」→ `the`；「`sohpto`」→ `photos` |

⭐ 【口頭】關鍵數字：**訓練 FLOPs 要大於約 10²²，模型才做得出這些任務**；小於這個門檻的模型**完全解不了**。

### 4.3 ⭐ GPT-3 與 In-Context Learning

| | **傳統 NLP 流程** | **GPT-3 的做法** |
|---|---|---|
| **步驟** | 預訓練 → **對每個下游任務各 fine-tune 一次** | 預訓練**一次**，之後**不再更新權重** |
| **十個任務** | fine-tune 十次，十份權重 | ⭐ **同一個模型，靠 prompt 區分任務** |
| **有梯度嗎** | 有 | ⭐ **完全沒有梯度、沒有 fine-tune** |

兩種模式：

- **Zero-shot（零樣本）**：prompt 只給任務描述。例：`Translate English to French: cheese =>`
- **Few-shot（少樣本）**：任務描述 **+ 幾個示範例子**，再問新問題。

**三個發現**（Brown et al., 2020）：

1. ⭐ **模型越大，越能有效利用 few-shot 示範**——175B 的準確率遠高於 13B、1.3B。
2. **示範例子越多，準確率越高**（zero-shot < one-shot < few-shot）。
3. ⭐ **模型夠大 + 例子夠多時，「不給任務描述」的準確率會追上「有任務描述」**。
   【口頭】教授：**「你甚至不用告訴模型『把英文翻成法文』，光給例子，它自己就會推出來。」**

### 4.4 GPT 家族的規模史

| 版本 | 年份 | 參數 |
|---|---|---|
| **GPT-1** | 2018 | 117M |
| **GPT-2** | 2019 | 1.5B |
| **GPT-3** | 2020 | 175B |
| **GPT-4** | 2023 | **???**（未公開） |

### 4.5 ⭐⭐ 開源 LLM 對照表

這張表是整段的精華——**把 §3 的每個設計變體對到真實模型**：

| 模型 | 出處 | 尺寸 | **Norm** | **FFN 激活** | **位置編碼** | **KV 注意力** | Context |
|---|---|---|---|---|---|---|---|
| **OPT** | Meta, 2022 | 125M…**175B** | Pre-norm<br>（350M 例外用 post-norm） | **ReLU** | 絕對 | MHA | **2K** |
| **BLOOM** | BigScience, 2022 | 560M…**176B** | Pre-norm | **GeLU** | ⭐ **ALiBi** | MHA | **2K** |
| **LLaMA** | Meta, 2023 | 7B/13B/33B/**65B** | Pre-norm | ⭐ **SwiGLU** | ⭐ **RoPE** | MHA | **2K** |
| **Llama 2** | Meta, 2023 | 7B…**70B** | Pre-norm | SwiGLU | RoPE | ⭐ **GQA**（70B） | ⭐ **4K** |
| **Falcon** | TII, 2023 | 1.3B…**180B** | — | — | — | **GQA**（8 kv heads） | — |
| **Mistral-7B** | Mistral, 2023 | 7B | — | — | — | ⭐ **GQA + SWA** | ⭐ **8K** |

**全部都是 decoder-only。**

**細部規格**：

| 模型 | hidden dim | heads | kv heads | layers | vocab |
|---|---|---|---|---|---|
| **OPT-175B** | 12288 | 96 | — | — | 50K |
| **BLOOM-176B** | 14336 | 112 | — | — | **250K**（支援 **59 種語言**） |
| **LLaMA-7B** | **4096** | **32** | — | **32** | 32K |
| **LLaMA-65B** | 8192 | 64 | — | 80 | 32K |
| **Llama-2-70B** | — | **64** | ⭐ **8** | 80 | 32K |
| **Falcon-180B** | 14848 | **232** | **8** | 80 | 65K |
| **Mistral-7B** | 4096 | 32 | **8** | 32 | 32K |

**Llama 2 相對 LLaMA 的四個改進**：

1. context **2K → 4K**
2. 訓練資料 **1T/1.4T → 2T tokens**（⭐ **還沒有飽和的跡象**）
3. 大模型改用 **GQA**
4. 多了 **Llama-2-chat**（指令微調版，對話更好）

**Mistral-7B 的亮點**：7B 卻**打敗 Llama-2-13B 和 Llama-34B**。除了 GQA，還用了 **Sliding Window Attention（SWA，滑動視窗注意力）**：

> 【口頭】**單層只看一個固定視窗（不再是 O(N²)），但把很多層疊起來，receptive field 會擴大，最後仍能看到很遠的 token**——用小代價換到更長的有效 context。
> （這個思路跟 **CNN 疊層擴大感受野**一模一樣。）

### 4.6 ⭐ Chinchilla Law：模型和資料要一起放大

**Chinchilla 定律**（Hoffmann et al., 2022）：要拿到最佳的「**訓練**計算量 vs 精度」權衡，**模型大小和訓練資料量必須一起等比例放大**。

⚠️ **但這是只考慮訓練成本的結論。教授特別強調轉折：**

> ⭐ **如果你把「推論」的成本也算進去，答案就不一樣了：**
> **你會想訓練一個比較小的模型、訓練更久，以節省之後每一次推論的成本。**

**LLaMA 就是這樣做的**：Chinchilla 說 7B 模型大約需要 200B tokens，**LLaMA / Llama-2 卻餵了 1T～2T**——**超出十倍**，換來一個又小又強的模型。

【口頭】教授的延伸：**「所以資料工程、資料收集、資料清理，是非常重要的工作。」**

### 4.7 【口頭】這一講跟作業的關係

教授在最後說：

> **「Lab 4 / HW 4 今天釋出。我們要對 Llama 做量化，然後把它部署到筆電上。」**

對應到本 repo 的 [`Lab4_zh.md`](Lab4_zh.md) / [`lab4.ipynb`](lab4.ipynb)，用的技術是 **AWQ**——會在 **Lecture 13** 講。

---

## 5. 進階主題：多模態與 MoE

> ⚠️ 【投影片】**這一段課堂上超時沒講到**（教授最後說「we're over time, this is so exciting, we're going to continue next time」）。以下整理自投影片。

### 5.1 多模態 LLM：讓 LLM 看得見

**兩種做法**：

| | **Cross-attention 派** | **Visual tokens 派** |
|---|---|---|
| **代表** | **Flamingo** | **PaLM-E** |
| **做法** | LLM **凍結**，在中間層插入 **cross-attention** 層接收視覺特徵 | 把視覺（或其他模態）**直接當成 token** 餵進去 |

**Flamingo**（Alayrac et al., 2022）的兩個關鍵元件：

1. **Perceiver Resampler**：把**大小不定的大特徵圖**壓成**少數固定數量的視覺 token**。
   - 視覺 token `[27, dim]` + 可學習 query `[5, dim]` → K&V `[32, dim]` → attention map `[5, 32]` → 輸出 `[5, dim]`
2. **Gated cross-attention layer**：用 **tanh 閘門**控制視覺資訊流入多少，**閘門初始化為 0**（一開始完全不干擾原本的 LLM，訓練才慢慢打開）。

能力：**in-context learning**、**視覺對話**。

**PaLM-E → RT-2**：支援彈性模態（`<img>` 圖片、`<emb>` 機器人狀態／3D 表徵），**RT-2 甚至能直接輸出控制訊號**去操作機器人。

**應用**：自駕車的**罕見情境處理**（投影片例子：問「這張圖哪裡不對勁？」→ 答「高速公路上有張椅子從卡車後面飛出來」）。

### 5.2 Mixture-of-Experts（MoE，專家混合）

**核心**：每個 token **只稀疏地啟用一部分參數**。

> ⭐ **可以增加模型的總參數量，卻不增加每個 token 的推論成本。**

- 由一個 **router（路由器）** 決定每個 token 交給哪些 expert。
- **expert 越多 → 總模型越大 → loss / perplexity 越低。**

**Capacity factor C**（投影片的 6 token、3 expert 例子）：

| C | 每個 expert 最多處理 | 結果 |
|---|---|---|
| **C = 1** | 6/3 × 1 = **2 個 token** | **有一個 token 被丟掉** |
| **C = 1.5** | 6/3 × 1.5 = **3 個 token** | expert 2、3 有**餘裕**（slack capacity） |

（Switch Transformers, Fedus et al., 2022）

### 5.3 TensorRT-LLM

投影片最後放了 NVIDIA 的 **TensorRT-LLM**，把裡面用到的技巧分成「**本講已涵蓋**」和「**後續講次會介紹**」兩類——它就是這兩講內容在工業界的集大成。

---

## 6. 一頁速查表

### 6.1 Transformer 七件套

| 元件 | 一句話 | 為什麼需要 |
|---|---|---|
| **Tokenizer** | 字 → token（1 字 ≈ 1.5 token） | 電腦只認數字 |
| **Embedding** | token → 連續向量（查表） | one-hot 太稀疏太長 |
| **Self-Attention** | `softmax(QKᵀ/√d)·V` | **任兩 token 一步互動**，且可平行 |
| **MHA** | H 個平行 attention，concat 後投影 | 一張 attention map 抓不完所有語意 |
| **FFN** | d → 4d → d，中間 ReLU/GeLU | attention **沒有 element-wise 非線性** |
| **LayerNorm** | 對**每個 token 的 feature 維度**正規化 | 不碰 batch 維，訓練穩定 |
| **Positional Encoding** | sin/cos，加到 embedding 上 | attention 本身**分不出順序** |

### 6.2 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **O(N²)** | attention 的複雜度（N = token 數） |
| **110 字 → 162 token** | tokenizer 的比例 |
| **15%** | BERT MLM 遮掉的比例 |
| **4d → 8d/3** | FFN → SwiGLU 的中間維度（為了維持計算量） |
| **10000** | RoPE 的 θ 底數（要夠大才能區分長序列） |
| **θ → θ/2** | RoPE 位置內插：序列變兩倍，角度砍一半 |
| **2K → 32K** | RoPE 內插能延長的 LLaMA context |
| **128** | 各家 LLM **每個 head 的維度**（幾乎都一樣） |
| **512 KB / 800 KB / 2.5 MB** | Llama-2 **7B / 13B / 70B** 每 token 每 batch 的 KV cache |
| ⭐ **160 GB** | 70B、bs=16、4K token 的 KV cache（**要兩張 A100**） |
| **64× / 8×** | **MQA / GQA** 相對 MHA 的 KV cache 縮減 |
| **10²²** | 湧現能力出現的訓練 FLOPs 門檻 |
| **175B / 530B** | GPT-3 / MT-NLG 的參數量 |
| **2T tokens** | Llama-2 的訓練資料量（**遠超 Chinchilla 建議**） |

### 6.3 ⭐ 三代注意力的卷積類比

| Attention | 卷積類比 | KV head 數 |
|---|---|---|
| **MHA** | 普通卷積 | N |
| **GQA** | **grouped 卷積** | N/8 |
| **MQA** | **depthwise 卷積** | 1 |

### 6.4 一眼看懂各家設計選擇

```
Decoder-only  ← 全部都是
Pre-norm      ← 全部都是（比 post-norm 穩）
ReLU  →  GeLU  →  SwiGLU        （OPT → BLOOM → LLaMA）
絕對PE →  ALiBi  /  RoPE         （OPT → BLOOM  /  LLaMA）
MHA   →  GQA                     （LLaMA → Llama-2-70B / Falcon / Mistral）
2K    →  4K  →  8K               （LLaMA → Llama-2 → Mistral）
```

---

## 7. 與其他課程的連結

| 本講觀念 | 連到哪裡 |
|---|---|
| **§2.7 LayerNorm 的 γ/β 可以吸收量化 scale** | ⭐ **[Lecture 5-6 量化](lecture05_06_quantization_note.md)** —— SmoothQuant / AWQ 就是靠這招，**下一講詳談** |
| **§2.3 softmax 必須加總為 1 → attention sink** | **Lecture 13**（下一講） |
| **§3.5 KV cache 是記憶體瓶頸** | ⭐ **[Lecture 11 TinyEngine](lecture11_tinyengine_note.md)** —— 同一個結論：**算力不是瓶頸，記憶體才是** |
| **§3.6 MQA/GQA ↔ depthwise/grouped 卷積** | **[Lecture 7-8 NAS](lecture07_08_nas_note.md)、[Lecture 10 MCUNet](lecture10_mcunet_note.md)** —— MobileNet 的 depthwise 是同一個省參數技巧，換到了 attention 上 |
| **§4.5 Mistral 的 sliding window attention** | **[Lecture 2 基礎](lecture02_note.md)** —— 疊層擴大 receptive field，跟 CNN 完全同一套邏輯 |
| **§4.6 訓練 vs 推論的取捨不同** | **[Lecture 9 知識蒸餾](lecture09_kd_note.md)** —— 都是「多花訓練成本換小模型」 |
| **§4.7 Lab 4：Llama 量化 + 筆電部署** | **[Lab4_zh.md](Lab4_zh.md)**、[Lecture 11 §7](lecture11_tinyengine_note.md) TinyChatEngine |
| **§5.2 MoE 稀疏啟用** | **[Lecture 3-4 剪枝](lecture03_04_pruning_note.md)** —— 剪枝是**永久**移除權重，MoE 是**動態**只啟用一部分 |

→ 下一講：**[Lecture 13 Transformer 與 LLM（下）筆記](lecture13_transformer_llm_part2_note.md)**

---

## 附：這一講的一句話總結

> **Transformer 用一招 `softmax(QKᵀ/√d)·V` 同時解掉了 RNN 的兩個死結：**
> **任兩個 token 一步就能互動，而且全部可以平行算。**
> **代價是 O(N²) 的注意力，和一份會隨對話長度線性長大的 KV cache ——**
> **大到 batch 一開，它就比模型權重本身還大。**
> **所以後來所有的設計變體（RoPE、GQA、SwiGLU、SWA）本質上只在做兩件事：**
> **讓 context 更長，讓記憶體更小。**

---

*筆記依據 MIT 6.5940 Fall 2023 Lecture 12（[錄影](https://youtu.be/mR4u6ZaCYe4)，1:19:53）的課堂逐字稿與官方投影片 `Lec12-Transformer-and-LLM-Part-I.pdf`（91 頁）整理。標記【口頭】處為投影片沒有、僅課堂講述的內容；【Q&A】為教授反問全班或學生提問；【投影片】為投影片有、但課堂超時未講到的段落（第 5 節多模態與 MoE）。*
