---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 14
lecture: 14
date: 2023-10-26
video: https://youtu.be/fcmOYHd57Dk
notes: Lec14-Vision-Transformer.pdf（74 頁）
---

# 第 14 集 · Lecture 14：Vision Transformer [▶ 影片](https://youtu.be/fcmOYHd57Dk)

← 上一集：[Lecture 13 Transformer 與 LLM（下）](lecture13_transformer_llm_part2_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 15 GAN、影片與點雲](lecture15_gan_video_pointcloud_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 14 集
> **講者**：Song Han
> **來源**：YouTube 錄影逐字稿（1:14:19，英文自動字幕，Zoom 版）＋ 投影片 `Lec14-Vision-Transformer.pdf`（74 頁）
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
- [1. ⭐ ViT 基礎：把圖片切成 token](#1--vit-基礎把圖片切成-token)
- [2. 為什麼 ViT 需要變快：高解析度密集預測](#2-為什麼-vit-需要變快高解析度密集預測)
- [3. ⭐⭐ 高效 ViT 技術](#3--高效-vit-技術)
  - [3.1 Window attention（Swin Transformer）](#31-window-attentionswin-transformer)
  - [3.2 稀疏 window attention（FlatFormer）](#32-稀疏-window-attentionflatformer)
  - [3.3 ⭐ Linear attention（EfficientViT）](#33--linear-attentionefficientvit)
  - [3.4 Sparse attention（SparseViT）](#34-sparse-attentionsparsevit)
- [4. ⭐ ViT 的自監督學習](#4--vit-的自監督學習)
  - [4.1 對比學習（Contrastive learning）](#41-對比學習contrastive-learning)
  - [4.2 多模態對比學習：CLIP](#42-多模態對比學習clip)
  - [4.3 Masked Autoencoder（MAE）](#43-masked-autoencodermae)
- [5. ⭐ 多模態 LLM](#5--多模態-llm)
  - [5.1 GPT-4V 能做什麼](#51-gpt-4v-能做什麼)
  - [5.2 Flamingo：用 cross attention 把影像注入 LLM](#52-flamingo用-cross-attention-把影像注入-llm)
  - [5.3 PaLM-E：把影像當成 token](#53-palm-e把影像當成-token)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與後續集數的連結](#7-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/fcmOYHd57Dk?t=0)

前兩講把 Transformer 用在語言上。【口頭】教授說 Transformer 的模型容量很大，連視覺任務也能幫上忙，所以今年**新加了這一章**。但 ViT 要夠大才會好，所以馬上就碰到效率問題。

| 段落 | 內容 |
|---|---|
| **I. ViT 基礎** | 圖片切成 patch、每個 patch 是一個 token |
| **II. 高效 ViT** | Window attention、Linear attention、Sparse attention |
| **III. 自監督學習** | 對比學習（含 CLIP）、Masked image modeling（MAE） |
| **IV. 多模態 LLM** | Cross attention（Flamingo）、Visual token（PaLM-E） |

【口頭】[▶ 01:00](https://youtu.be/fcmOYHd57Dk?t=60) 教授提醒：這裡的 sparse attention 跟之前講的**權重稀疏**很不一樣。

---

## 1. ⭐ ViT 基礎：把圖片切成 token

### 1.1 圖片 → patch 序列

[▶ 02:00](https://youtu.be/fcmOYHd57Dk?t=120)

語言可以切成 token，圖片呢？最直接的做法：**把圖片切成一塊一塊的 patch，每個 patch 當一個 token**，排成一維序列送進 Transformer。

投影片的例子，一步一步算：

| 項目 | 算法 | 數字 |
|---|---|---|
| 圖片大小 | — | 96 × 96 |
| Patch 大小 | — | 32 × 32 |
| Token 數 | $(96/32)\times(96/32)$ | **9** |
| 每個 token 的維度 | $3\times32\times32$ | **3072** |
| 投影到 hidden size | — | **768** |
| 投影層參數 | $3072\times768$ | **約 2.36M** |

### 1.2 位置編碼

[▶ 04:00](https://youtu.be/fcmOYHd57Dk?t=240)

【口頭】全部攤平之後，模型**不知道 patch 原本在哪裡**。例如原圖裡不相鄰的兩個 patch，攤平後可能排在一起。所以要加**位置編碼（positional encoding）**。

### 1.3 實作：用卷積做投影

[▶ 04:00](https://youtu.be/fcmOYHd57Dk?t=240)

每個 patch 都乘同一個 $3072\times768$ 的矩陣，等於用**同一組卷積核**掃過每個 patch：

```
Conv2d(in_channels=3, out_channels=768, kernel_size=32, stride=32, padding=0)
```

- 關鍵是 **stride = kernel size = 32**，patch 之間**不重疊**。
- 【口頭】卷積在 GPU 上已經優化得很好，所以這樣實作很快。

之後就跟處理語言**完全一樣**：embedding + 位置編碼 → multi-head attention → LayerNorm → FFN。

### 1.4 模型變體

[▶ 06:00](https://youtu.be/fcmOYHd57Dk?t=360)

【口頭】ViT-Base／Large／Huge：

- 層數 12 到 32。
- Hidden 維度 768 到 1280。⚠️ 逐字稿缺字：字幕寫「700」，應為 768。
- MLP 維度 3072 到 5120。
- 參數量**上億**。對比 ResNet-50 大約 25M，ViT 大了約 10 倍。

**命名方式**：**ViT-L/16** = ViT-Large，patch size 16×16。Patch 可以是 2、4、8、16、32……

【口頭】[▶ 07:00](https://youtu.be/fcmOYHd57Dk?t=420) 教授問：patch 越小，計算量越大還是越小？答案是**越大**，因為 token 數變多了。

### 1.5 ViT vs CNN：資料量決定勝負

[▶ 07:00](https://youtu.be/fcmOYHd57Dk?t=420)

| 訓練資料量 | 誰比較好 |
|---|---|
| **少** | **CNN** 較好 |
| **很多**（大資料集預訓練） | **ViT** 超越 CNN |

→ ViT 需要大量**有標籤**的資料。這就是第 4 節自監督學習的動機。

---

## 2. 為什麼 ViT 需要變快：高解析度密集預測

[▶ 08:00](https://youtu.be/fcmOYHd57Dk?t=480)

### 2.1 高解析度很重要

**密集預測（dense prediction）**：每個像素都要給一個答案。例子：

- **醫學影像分割**：解析度要高，才分得出不同的病變。
- **超解析度（super-resolution）**。
- **自動駕駛**：【口頭】解析度太低，遠處的行人和腳踏車會直接消失。

投影片的圖：Cityscapes 分割，高解析度輸入的 mIoU 明顯比低解析度好。

### 2.2 問題：計算量跟解析度成平方

[▶ 10:00](https://youtu.be/fcmOYHd57Dk?t=600)

- Attention map 是 $n\times n$，$n$ 是 token 數。
- 解析度變兩倍 → token 數變 4 倍 → attention 計算變 16 倍。
- 投影片的圖：解析度從 $224^2$ 到 $2048^2$，**ViT 的 GMACs 暴增，CNN 增加得慢很多**。

**實例**：[▶ 11:00](https://youtu.be/fcmOYHd57Dk?t=660)

| 模型 | 速度（Jetson AGX Orin） | Cityscapes mIoU |
|---|---|---|
| SegFormer | 1.6 FPS | 82.4 |
| **EfficientViT** | **21.8 FPS** | **82.7** |

### 2.3 另一個應用：Segment Anything（SAM）

[▶ 12:00](https://youtu.be/fcmOYHd57Dk?t=720)

【口頭】教授在這裡岔開講了一下 SAM 是怎麼運作的：

- **可以用 prompt 分割**：prompt 可以是一個點、一個框、一個遮罩，或文字（文字版當時還沒釋出）。給一個點，它就分割出那個點所在的物體。
- **能處理模糊**：[▶ 13:00](https://youtu.be/fcmOYHd57Dk?t=780) 同一個點可以是指整把剪刀，也可以只是把手，所以會輸出多個可能的遮罩。
- **自動資料引擎**：用模型標註圖片 → 拿標註再訓練模型 → 模型變好 → 再標註，形成良性循環。
- 規模：**超過 10 億個遮罩、1100 萬張圖**。【口頭】ImageNet 約 120 萬張，所以大約是 10 倍。
- Mask decoder 用 **cross attention** 融合 prompt 和影像特徵。

【口頭】AR/VR 應用（例如看哪裡就分割哪裡）也需要高解析度的密集預測。

---

## 3. ⭐⭐ 高效 ViT 技術

### 3.1 Window attention（Swin Transformer）

#### 做法

[▶ 14:00](https://youtu.be/fcmOYHd57Dk?t=840)

- 不讓所有 token 互相做 attention，**只在固定大小的視窗裡**做（投影片例子：7×7）。
- 每個視窗的 token 數固定 → attention map 大小固定 → **總計算量隨解析度線性成長**。
- 【口頭】每個視窗可以平行算。視窗數量就是 **batch 維度**。
- **逐步降採樣**：原本的 ViT 從頭到尾 token 數不變；Swin 會慢慢降採樣，patch 越來越少。

複雜度：設 token 數為 $n$，每個視窗有 $M^2$ 個 token。

$$
\text{全域 attention}: O(n^2) \qquad \text{window attention}: \frac{n}{M^2}\ \text{個視窗} \times O\left((M^2)^2\right) = O(n M^2)
$$

$M$ 固定，所以對 $n$ 是線性的。

#### 問題：視窗之間沒有資訊交流

[▶ 15:00](https://youtu.be/fcmOYHd57Dk?t=900)

- 【口頭】資訊只在視窗內流動。只有在降採樣的時候，感受野才會變大。

#### 解法：Shifted window（移動視窗）

[▶ 16:00](https://youtu.be/fcmOYHd57Dk?t=960)

- 第 $i$ 層用一般視窗（W-MSA），第 $i+1$ 層**把視窗移動一點**（SW-MSA，例子是移 2 格），第 $i+2$ 層再移回來。
- 移動後，原本在不同視窗的相鄰 token 會落在同一個視窗 → **資訊開始在相鄰視窗之間流動**。

#### 結果

[▶ 17:00](https://youtu.be/fcmOYHd57Dk?t=1020)

- ImageNet 準確率比原本的 ViT 好（【口頭】超過 80% vs 不到 80%），throughput 相當或更好。
- 也適用於物體偵測和分割等下游任務。

### 3.2 稀疏 window attention（FlatFormer）

[▶ 18:00](https://youtu.be/fcmOYHd57Dk?t=1080)

【口頭】下一講（Lecture 15）會詳細講點雲。這裡先講怎麼把 Transformer 用在 3D 點雲上。

| | 2D 影像 | 3D 點雲 |
|---|---|---|
| **稀疏度** | 100% 密集 | **99.9% 稀疏**（只有 0.1% 有東西） |

#### 問題：照 Swin 的方式分組，每組點數不平均

[▶ 19:00](https://youtu.be/fcmOYHd57Dk?t=1140)

- **等視窗分組（equal-window grouping）**：每個 2×2 視窗一組。
  - 優點：同組的點在空間上**一定相鄰**。
  - 缺點：有的視窗 1 個點，有的 3 個點。【口頭】實際上不同視窗的點數**差到兩個數量級**。
- 要批次處理就得**補零（padding）**，浪費很多計算。

#### 解法：等大小分組（equal-size grouping）

[▶ 20:00](https://youtu.be/fcmOYHd57Dk?t=1200)

- 改成**每組點數一樣多**。
- 代價：同組的點不一定相鄰。
- 【口頭】但實測**準確率沒有變差，甚至更好**，追上 CNN 方法，而且在 Jetson Orin 上比 CNN 方法快、比傳統 Transformer 方法快很多。

### 3.3 ⭐ Linear attention（EfficientViT）

#### 想法：拿掉 softmax，換成線性的相似度

[▶ 22:00](https://youtu.be/fcmOYHd57Dk?t=1320)

| | **Softmax attention** | **ReLU linear attention** |
|---|---|---|
| 相似度 | $\mathrm{Sim}(Q,K) = \exp\left(\dfrac{QK^\top}{\sqrt{d}}\right)$ | $\mathrm{Sim}(Q,K) = \mathrm{ReLU}(Q)\,\mathrm{ReLU}(K)^\top$ |
| 成本 | $O(n^2)$ | 照原本順序算一樣是 $O(n^2)$，**換順序後變 $O(n)$** |

#### ⭐ 關鍵一步：矩陣乘法的結合律

[▶ 23:00](https://youtu.be/fcmOYHd57Dk?t=1380)

因為 softmax 不見了，整串都是矩陣乘法，可以用 $(AB)C = A(BC)$ 換順序。記 $\phi = \mathrm{ReLU}$：

$$
\underbrace{\left(\phi(Q)\,\phi(K)^\top\right)}_{n\times n}V
\quad=\quad
\phi(Q)\underbrace{\left(\phi(K)^\top V\right)}_{d\times d}
$$

一步一步看維度：

1. $\phi(K)^\top$ 是 $d\times n$，$V$ 是 $n\times d$ → 乘起來是 **$d\times d$**，成本 $O(nd^2)$。
2. $\phi(Q)$ 是 $n\times d$，乘上 $d\times d$ → 結果是 $n\times d$，成本 $O(nd^2)$。
3. 總成本 $O(nd^2)$。$d$ 固定，所以**對 token 數 $n$ 是線性的**。
4. 原本的順序要先做出 $n\times n$ 的表，成本 $O(n^2 d)$。

【口頭】輸入和輸出的形狀**完全一樣**，只是計算量從 $O(n^2)$ 變成 $O(n)$。

#### 天下沒有白吃的午餐

[▶ 25:00](https://youtu.be/fcmOYHd57Dk?t=1500)

- Softmax attention 的分布很**尖**：會集中在重點上（例如狗的眼睛、貓的鼻子）。
- Linear attention 的分布很**平**。
- 結果：
  - **擅長抓全域資訊**，**不擅長抓局部資訊**。
  - 也**缺少多尺度（multi-scale）學習能力**。
- 直接把 softmax 換成 ReLU linear attention，ImageNet 準確率**掉很多**。

#### 解法：加上卷積分支，專門抓局部

[▶ 26:00](https://youtu.be/fcmOYHd57Dk?t=1560)

【口頭】什麼最擅長抓局部特徵？**卷積**。所以讓 linear attention 跟卷積搭配：

- 在 Q/K/V 上加一條分支：**3×3 depthwise 卷積 + 1×1 group 卷積**，把鄰近 token 聚合起來，得到**多尺度的 Q/K/V token**。
- 原本尺度和聚合後的尺度**各自做 linear attention**，再合起來。
- 【投影片補充】FFN 裡也加 depthwise 卷積，進一步增強局部資訊。
- 【口頭】核心觀念是**分工（specialization）**：attention 負責全域，卷積負責局部。卷積用 depthwise，所以額外成本很小。

【Q&A】[▶ 28:00](https://youtu.be/fcmOYHd57Dk?t=1680) 學生問：如果只用卷積那條分支呢？教授：還沒試過，但做個消融實驗（ablation study）只用那條分支，應該會有幫助。

**結果**：準確率差距不只補回來，還**超過 softmax attention**。

#### EfficientViT 的成績

[▶ 29:00](https://youtu.be/fcmOYHd57Dk?t=1740)

| 任務 | 結果 |
|---|---|
| **Cityscapes 分割** | 在手機 CPU、Jetson AGX Orin、A100 上都明顯加速，mIoU 維持住 |
| **超解析度** | Restormer 3.6 張/秒、SwinIR 4.7 張/秒 → **EfficientViT 17 張/秒**（最高 6.4×），畫質更好 |
| **Segment Anything** | ViT-Huge 12 張/秒 → **EfficientViT-L0 1009 張/秒** |
| **ImageNet 分類** | Jetson AGX Orin 上最高 **4.2×** 加速 |

⚠️ 待確認：Cityscapes 投影片寫了三個加速倍數（2.8×、1.7×、5.8×），分別對比 SegNeXt 和 SegFormer，但從文字看不出哪個數字對應哪個平台。教授口頭說「2 到 5 倍以上」。

⚠️ 逐字稿缺字：教授說超解析度「大約快 5 到 6 倍」，投影片寫「最高 6.4×」。

**SAM 的比較**（COCO-val2017，A100）：[▶ 31:00](https://youtu.be/fcmOYHd57Dk?t=1860)

| Image encoder | mIoU | Throughput |
|---|---|---|
| NanoSAM | 70.6 | 744 張/秒 |
| MobileSAM | 72.8 | 297 張/秒 |
| **EfficientViT-L0** | **74.5** | **1009 張/秒** |

【口頭】ImageNet 那張圖同時測了兩種指標：Jetson 上的**延遲**（給自動駕駛這種即時應用）和 A100 上的 **throughput**（給雲端）。證明 EfficientViT 在邊緣和雲端都好用。

### 3.4 Sparse attention（SparseViT）

#### 問題：稀疏的高解析度，還是密集的低解析度？

[▶ 32:00](https://youtu.be/fcmOYHd57Dk?t=1920)

計算量一樣的兩個選擇：

- **密集、低解析度**：解析度砍一半 → patch 數變 **1/4**。
- **稀疏、高解析度**：保留原解析度，但只留 **1/4** 的 patch。

哪個比較好？答案：**稀疏、高解析度比較好**。

#### Step 1：視窗激活剪枝（非均勻稀疏度）

[▶ 33:00](https://youtu.be/fcmOYHd57Dk?t=1980)

- 用每個視窗激活值的 **L2 大小**排序重要度。
- 留下 **top-K** 重要的視窗，丟掉不重要的。
- 越深的層剪越多，**每層的稀疏度不一樣**。

【口頭】[▶ 34:00](https://youtu.be/fcmOYHd57Dk?t=2040) 為什麼**對硬體很友善**？

- 視窗**裡面**的 attention 還是密集的。
- 剪的是**視窗的數量**，而視窗數量就是 Swin 的 **batch 維度**。
- 只是讓 batch 變小，**不需要特殊的稀疏 kernel**，任何硬體都跑得快。

#### Step 2：稀疏感知的微調

[▶ 35:00](https://youtu.be/fcmOYHd57Dk?t=2100)

- 原本的模型只看過密集的激活值，要讓它**適應剪枝**。
- 跟 **Once-for-All**（[Lecture 7-8](lecture07_08_nas_note.md)）一樣：每次訓練迭代**隨機抽**每層的稀疏度。
- 不用從頭訓練，在原模型上微調就好。

#### Step 3：在資源限制下搜尋

[▶ 35:00](https://youtu.be/fcmOYHd57Dk?t=2100)

- 推論時用**演化搜尋（evolutionary search）**找每層的稀疏度組合，要同時滿足計算預算和準確率。【口頭】跟 Lab 3 做的一樣。
- 演化搜尋比隨機搜尋好。

#### 結果

[▶ 36:00](https://youtu.be/fcmOYHd57Dk?t=2160)

- 同樣 mIoU 下，比密集低解析度的版本**快 30% 到 50%**。
- 視覺化（19 ms／24 ms 兩種延遲預算）：[▶ 37:00](https://youtu.be/fcmOYHd57Dk?t=2220)
  - 剪掉的是**多餘的背景視窗**。
  - 留下的是**有資訊的前景**：路障、車子。
  - 亮的區域每一層都留著；灰的在中間被剪；黑的一開始就被剪。

---

## 4. ⭐ ViT 的自監督學習

[▶ 38:00](https://youtu.be/fcmOYHd57Dk?t=2280)

**動機**：

- ViT 要大量資料才會超越 CNN。
- 標註大量資料**很貴**。
- 解法：用大量**沒有標籤**的資料做**自監督預訓練**，再遷移到下游任務。下游任務的資料量小，標得起。【口頭】跟 LLM 先訓練 foundation model 再用到下游任務是同一個思路。

### 4.1 對比學習（Contrastive learning）

[▶ 39:00](https://youtu.be/fcmOYHd57Dk?t=2340)

- **正樣本**：同一張圖的不同隨機視角（旋轉、換顏色、裁切……）。【口頭】「貓不管怎麼轉，還是貓。」
- **負樣本**：其他圖片的隨機視角（例如狗）。
- 同一個 encoder 編碼後：**同一張圖的不同視角要靠近，不同圖要遠離**。
- 完全不用標籤，只靠對圖片本身做變換。

**結果**：[▶ 41:00](https://youtu.be/fcmOYHd57Dk?t=2460)

| 做法 | ViT-B → ViT-L → ViT-H（在 CIFAR-10/100 上） |
|---|---|
| **直接在小資料集上從頭訓練** | 模型越大，準確率**越低** |
| **先在 ImageNet 上自監督預訓練** | 模型越大，準確率**越高**，也比監督式預訓練好 |

### 4.2 多模態對比學習：CLIP

[▶ 42:00](https://youtu.be/fcmOYHd57Dk?t=2520)

**訓練**：

- 每張圖配一段文字描述。
- 圖片過 image encoder，文字過 text encoder。
- 做出一個 $N\times N$ 的相似度矩陣（$N$ 張圖 × $N$ 段文字）。
- 目標：**對角線**（圖 $i$ 配文字 $i$）相似度要高，其他要低。
- 只需要成對的「圖片 + 描述」，不用人工類別標籤。

**推論：零樣本（zero-shot）、開放詞彙（open-vocabulary）分類**：[▶ 43:00](https://youtu.be/fcmOYHd57Dk?t=2580)

1. 把候選類別填進句子：「a photo of a {dog / cat / plane …}」，全部過 text encoder。
2. 新圖片過 image encoder。
3. 挑相似度最高的那句 → 例如「a photo of a dog」。

- 類別**不用事先看過**，也**不限 1000 類**，**不用微調**。

**Linear probe**：[▶ 44:00](https://youtu.be/fcmOYHd57Dk?t=2640) 用 CLIP 抽的特徵只訓練一層全連接層，用來檢驗特徵品質。CLIP 超越各種基準模型。

### 4.3 Masked Autoencoder（MAE）

#### 想法：遮住圖片的一部分，叫模型猜

[▶ 45:00](https://youtu.be/fcmOYHd57Dk?t=2700)

- 隨機遮住一些 patch，根據看得到的 patch **預測被遮住的 patch**。
- 跟 BERT 的 masked language model 一樣（[Lecture 12](lecture12_transformer_llm_note.md)）：「The movie is very ___」。
- 在 ViT 裡，每個 patch 就是一個 token，所以可以直接套用。

#### 非對稱的 encoder-decoder

[▶ 47:00](https://youtu.be/fcmOYHd57Dk?t=2820)

| | **Encoder（重）** | **Decoder（輕）** |
|---|---|---|
| **處理什麼** | **只處理沒被遮住的 token** | 所有 token（沒遮的 + 遮住的佔位 token，依位置排回去） |
| **目的** | 學好特徵 | 重建原圖 |

- 【口頭】跟 NLP 不同：BERT 會把 [MASK] token 也送進 encoder；MAE **不送**，所以 encoder 要處理的 token 少很多，算得更快。
- Decoder 要處理全部 token，所以做得很輕。

#### 遮罩比例要很高

[▶ 48:00](https://youtu.be/fcmOYHd57Dk?t=2880)

| | BERT | MAE |
|---|---|---|
| 遮罩比例 | **15%** | **75%** |

- 【口頭】完全不遮學不到東西，全部遮又猜不出來，所以有個最佳點。教授說大約在 **70% 到 80%** 之間，看是 linear probing 還是 fine-tuning。
- 因為遮這麼多，如果把遮住的 token 也送進 encoder 會非常浪費，這就是 encoder 只處理沒遮住 token 的原因。
- 【投影片補充】原因：**影像的資訊密度比語言低**，所以要遮更多才夠難。

#### 結果

[▶ 50:00](https://youtu.be/fcmOYHd57Dk?t=3000)

- 重建效果很好（公車、檯燈、南瓜、鳥等例子），而且完全沒用標籤。
- 部分微調和完整微調下，MAE 都比對比學習的 MoCo v3 好。

---

## 5. ⭐ 多模態 LLM

[▶ 51:00](https://youtu.be/fcmOYHd57Dk?t=3060)

【口頭】LLM 和 ViT 用的是同一個基本單元（Transformer），能不能把語言和視覺合起來？

### 5.1 GPT-4V 能做什麼

[▶ 52:00](https://youtu.be/fcmOYHd57Dk?t=3120)

【口頭】GPT-4V 幾週前剛發布。教授舉了幾個例子：

- **三張收據**：問「我付了多少稅？」它找出每張收據上很小的稅額、說出位置，再加總。
- **跨圖推理**：[▶ 53:00](https://youtu.be/fcmOYHd57Dk?t=3180) 一張是桌上的啤酒，一張是菜單。它要認出啤酒的牌子、在菜單上找到價格（每瓶 6 美元），再數出桌上有兩瓶，算出總價。
  ⚠️ 逐字稿缺字：字幕寫「$2」，依 6 × 2 應為 **12 美元**。
- **讀駕照轉成 JSON**：[▶ 55:00](https://youtu.be/fcmOYHd57Dk?t=3300) 姓名、生日、地址、到期日都抓得很準。但也會錯：髮色欄寫了棕色，它卻說沒有。

**兩種做法**：[▶ 56:00](https://youtu.be/fcmOYHd57Dk?t=3360)

| | **Cross attention（Flamingo 風格）** | **Visual token（PaLM-E 風格）** |
|---|---|---|
| **怎麼放進影像** | 在 LLM 中間層加 cross attention，把影像資訊注入 | 影像轉成 token，**跟文字 token 接在一起**送進 LLM |

### 5.2 Flamingo：用 cross attention 把影像注入 LLM

#### 架構

[▶ 57:00](https://youtu.be/fcmOYHd57Dk?t=3420)

| 元件 | 狀態 | 作用 |
|---|---|---|
| **Vision encoder**（例如 CLIP 的） | **凍結** | 把圖片變特徵 |
| **LLM** | **凍結** | 原本的語言模型 |
| **Perceiver Resampler** | **從頭訓練** | 不同大小的影像特徵 → **固定數量**的視覺 token |
| **Gated cross-attention + FFN** | **從頭訓練** | 插在 LLM 的層之間，把視覺 token 注入 |

#### Perceiver Resampler

[▶ 58:00](https://youtu.be/fcmOYHd57Dk?t=3480)

**問題**：圖片解析度不同、影片幀數不同，視覺 token 數就不同。但 cross attention 希望輸入大小固定。

**做法**：用一組**可學習的 query（latent query）**去做 attention。投影片的例子：

| 東西 | 形狀 | 怎麼來 |
|---|---|---|
| 視覺 token | $[27, d]$ | 影片 3 幀 × 每幀 9 個 patch |
| 可學習的 query $Q$ | $[5, d]$ | 端到端學出來 |
| $K$、$V$ | $[32, d]$ | **視覺 token 接上 query**：$27 + 5 = 32$ |
| Attention map | $[5, 32]$ | $QK^\top$ |
| **輸出** | **$[5, d]$** | 形狀跟著 $Q$ |

⭐ 關鍵一步：[▶ 1:01:00](https://youtu.be/fcmOYHd57Dk?t=3660) $\mathrm{softmax}(QK^\top)V$ 的輸出列數**等於 $Q$ 的列數**。$Q$ 固定是 5 個，所以**不管輸入幾張圖、幾幀，輸出永遠是 5 個 token**。

#### Gated cross-attention

[▶ 1:02:00](https://youtu.be/fcmOYHd57Dk?t=3720)

在原本的 attention、FFN 之間插入新的一層，用 **tanh 閘門**控制要注入多少影像資訊：

$$
y = x + \tanh(\alpha_{\text{xattn}})\cdot\mathrm{CrossAttn}\left(Q = x,\ KV = \text{視覺 token}\right)
$$

$$
y' = y + \tanh(\alpha_{\text{ffw}})\cdot\mathrm{FFW}(y)
$$

- $\alpha$ **初始化為 0** → $\tanh(0) = 0$ → 一開始**完全不用影像資訊**，模型行為跟原本的 LLM 一樣。
- 訓練時 $\alpha$ 慢慢變大，逐漸把影像資訊加進來。

#### 能力

[▶ 1:03:00](https://youtu.be/fcmOYHd57Dk?t=3780)

- **In-context learning**（few-shot）：給兩三個「圖 + 描述」的例子，它就會照樣接下一張。例如「這是龍貓，主要在智利……」「這是柴犬，在日本很受歡迎……」→ 接「這是紅鶴……」。也能看例子學會讀圖上的算式並算出答案（2+1=3、5+6=11）。
- **視覺對話**：[▶ 1:05:00](https://youtu.be/fcmOYHd57Dk?t=3900) 蘋果上貼著寫了「iPod」的貼紙，能回答貼紙寫什麼、是手寫還是印的、什麼顏色。月球上的泰迪熊：知道「泰迪熊通常不會在月球上」，有常識。認得出芝加哥（Shedd 水族館）和東京（東京鐵塔）。

### 5.3 PaLM-E：把影像當成 token

[▶ 1:06:00](https://youtu.be/fcmOYHd57Dk?t=3960)

- 影像、**機器人狀態**、神經 3D 表示……**全部轉成 token**，跟文字 token 接在一起送進 LLM。
- 好處：**支援很多種模態**，不只視覺和語言。
- 對**機器人**很有用：[▶ 1:07:00](https://youtu.be/fcmOYHd57Dk?t=4020) 輸入「怎麼拿藍色積木」，輸出「先拿黃色積木……」，再轉成控制訊號。
- 也能做視覺問答（例如用 emoji 回答圖裡有什麼水果）、純語言任務、桌面操作、任務規劃。

**RT-2**：[▶ 1:08:00](https://youtu.be/fcmOYHd57Dk?t=4080) 從 PaLM-E 延伸，**直接輸出控制訊號**。【口頭】語言模型帶來**常識**：例如知道草莓要放進草莓的桶子、認得出德國國旗，這些只靠視覺資料學不到。

**應用：自動駕駛的罕見情況**　[▶ 1:09:00](https://youtu.be/fcmOYHd57Dk?t=4140)

- 圖：高速公路上，一張椅子從卡車後面飛出來。
- 模型能說出「哪裡不尋常」，也知道「應該立刻停車、保持安全距離」。
- 【口頭】訓練資料裡沒有這種圖，視覺語言模型也能幫上忙。

【Q&A】[▶ 1:10:00](https://youtu.be/fcmOYHd57Dk?t=4200) 學生問：這個能用 LiDAR 嗎？教授：那篇論文不確定，但 LiDAR 的點雲**一定可以編碼成 token**，前面 FlatFormer（§3.2）就講過怎麼處理點雲的稀疏性。

---

## 6. 一頁速查表

### 6.1 ViT 基礎

| 項目 | 公式或做法 | 關鍵一步 |
|---|---|---|
| **Token 數** | $\dfrac{H}{P}\times\dfrac{W}{P}$ | 96×96、patch 32 → 9 個 |
| **Token 維度** | $3\times P\times P$ | patch 32 → 3072 |
| **Patch 投影** | `Conv2d(3, D, kernel=P, stride=P)` | stride = kernel → patch 不重疊 |
| **ViT-L/16** | Large、patch 16 | patch 越小 → token 越多 → 計算越大 |
| **ViT vs CNN** | 資料少 CNN 贏，資料多 ViT 贏 | → 需要自監督學習 |

### 6.2 高效 ViT

| 技術 | 複雜度 | 關鍵一步 |
|---|---|---|
| **Window attention** | $O(nM^2)$ | 每個視窗 $M^2$ 個 token，$n/M^2$ 個視窗 |
| **Shifted window** | 同上 | 隔層移動視窗 → 相鄰視窗交換資訊 |
| **FlatFormer** | — | 點雲 99.9% 稀疏，改成**每組點數相等** |
| **Linear attention** | $O(nd^2)$ | $\left(\phi(Q)\phi(K)^\top\right)V = \phi(Q)\left(\phi(K)^\top V\right)$，先算 $d\times d$ |
| **EfficientViT** | $O(n)$ | linear attention 抓全域 + depthwise 卷積抓局部 |
| **SparseViT** | — | 剪**視窗數 = batch 維度** → 不用稀疏 kernel |

### 6.3 自監督學習

| 技術 | 做法 | 關鍵數字 |
|---|---|---|
| **對比學習** | 同圖不同視角靠近、不同圖遠離 | 自監督後模型越大越好 |
| **CLIP** | 圖文配對，$N\times N$ 相似度矩陣對角線要高 | 零樣本、開放詞彙 |
| **MAE** | 遮住 patch 再重建；encoder 只看沒遮的 | 遮罩比例 **75%**（BERT 15%） |

### 6.4 多模態 LLM

| 技術 | 關鍵一步 |
|---|---|
| **Perceiver Resampler** | 輸出列數 = $Q$ 的列數 → 不管輸入多大，輸出固定（例子：5 個） |
| **Gated cross-attention** | $y = x + \tanh(\alpha)\cdot\mathrm{CrossAttn}(x, \text{視覺})$，$\alpha$ 初始為 0 |
| **PaLM-E** | 所有模態都當 token，跟文字接在一起 |
| **RT-2** | 直接輸出機器人控制訊號 |

### 6.5 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **10×** | ViT 參數量約是 ResNet-50（25M）的 10 倍 |
| **1.6 → 21.8 FPS** | SegFormer → EfficientViT（Jetson AGX Orin，Cityscapes） |
| **10 億 / 1100 萬** | SAM 資料集的遮罩數 / 圖片數 |
| **99.9%** | 3D 點雲的稀疏度 |
| **12 → 1009 張/秒** | SAM image encoder：ViT-Huge → EfficientViT-L0 |
| **17 張/秒（最高 6.4×）** | EfficientViT 超解析度 |
| **30%～50%** | SparseViT 相對密集低解析度的加速 |
| **75%** | MAE 遮罩比例 |
| **27 → 5** | Perceiver Resampler 例子：27 個視覺 token → 5 個輸出 token |

---

## 7. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **§3.2 點雲的稀疏性（FlatFormer）** | **[Lecture 15 GAN、影片與點雲](lecture15_gan_video_pointcloud_note.md)**，點雲會詳細講 |
| **§3.3 EfficientViT、高解析度** | **[Lecture 16 Diffusion Model](lecture16_diffusion_model_note.md)**，高解析度影像生成 |
| **§4 自監督預訓練 + 遷移** | **Lecture 19 裝置端訓練與遷移學習** |
| **§5 多模態 LLM、prompt** | **Lecture 20 高效微調與 Prompt Engineering** |

---

## 附：本集一句話總結

> **ViT 把圖片切成 patch 當 token，但 attention 的計算量跟解析度成平方，所以要靠視窗（Swin）、線性化（EfficientViT）、剪掉不重要的視窗（SparseViT）讓它跑得動高解析度；ViT 又很吃資料，所以用對比學習、CLIP、MAE 從沒有標籤的資料學；最後把影像接進 LLM，就有了能看圖的多模態模型。**

---

## ⚠️ 待確認清單

1. [▶ 29:00](https://youtu.be/fcmOYHd57Dk?t=1740) Cityscapes 分割的三個加速倍數（2.8×、1.7×、5.8×）各自對應哪個平台、對比哪個基準模型，從投影片文字看不出來。

---

*筆記依據 Song Han「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 14 錄影逐字稿（英文自動字幕，1:14:19）與投影片 `Lec14-Vision-Transformer.pdf`（74 頁）整理。*
