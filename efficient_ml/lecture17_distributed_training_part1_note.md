---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 17
lecture: 17
date: 2023-11-07
video: https://youtu.be/u6NEbhNiyAE
notes: Lec17-Distributed-Training-Part-I.pdf（93 頁）
---

# 第 17 集 · Lecture 17：分散式訓練（上） [▶ 影片](https://youtu.be/u6NEbhNiyAE)

← 上一集：[Lecture 16 Diffusion Model](lecture16_diffusion_model_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 18 分散式訓練（下）](lecture18_distributed_training_part2_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 17 集
> **講者**：Song Han
> **來源**：YouTube 錄影逐字稿（1:00:45，英文自動字幕，Zoom 版）＋ 投影片 `Lec17-Distributed-Training-Part-I.pdf`（93 頁）
> **標記說明**：
> - `【口頭】` = 教授口頭補充、投影片沒有
> - `【投影片補充】` = 投影片有、課堂沒講
> - `⚠️ 逐字稿缺字` = 字幕聽錯，依投影片修正
> - `[▶ mm:ss]` = 可點的影片時間點（字幕以分鐘為單位，時間點是大約位置）

---

## 目錄

- [0. 這堂課要做什麼](#0-這堂課要做什麼)
- [1. 為什麼需要分散式訓練](#1-為什麼需要分散式訓練)
- [2. 三種平行方式](#2-三種平行方式)
- [3. ⭐ 資料平行（Data Parallelism）](#3--資料平行data-parallelism)
- [4. ⭐⭐ 通訊原語](#4--通訊原語)
- [5. ⭐⭐ 減少資料平行的記憶體：ZeRO-1/2/3 與 FSDP](#5--減少資料平行的記憶體zero-123-與-fsdp)
- [6. ⭐ 管線平行（Pipeline Parallelism）](#6--管線平行pipeline-parallelism)
- [7. ⭐⭐ 張量平行（Tensor Parallelism）](#7--張量平行tensor-parallelism)
- [8. 一頁速查表](#8-一頁速查表)
- [9. 與後續集數的連結](#9-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/u6NEbhNiyAE?t=0)

【口頭】課程進入**第三章：從推論到訓練**。

| 章 | 主題 |
|---|---|
| 一 | 高效推論（剪枝、量化、蒸餾……） |
| 二 | 領域專屬優化（LLM、ViT、GAN、影片、點雲、擴散） |
| **三** | **訓練**：2 講雲端分散式訓練 + 2 講裝置端訓練 |

【口頭】[▶ 01:00](https://youtu.be/u6NEbhNiyAE?t=60) 下一講之後會帶大家去地下室參觀機房。

**本講大綱**：

1. 背景與動機
2. 三種平行方式
3. 資料平行
4. 通訊原語
5. 減少資料平行的記憶體：ZeRO-1/2/3 與 FSDP
6. 管線平行
7. 張量平行

【口頭】[▶ 02:00](https://youtu.be/u6NEbhNiyAE?t=120) 上完這講，你就會知道怎麼在**有限的 GPU 記憶體**裡訓練超大的模型。

---

## 1. 為什麼需要分散式訓練

### 1.1 模型越來越大，訓練越來越久

[▶ 02:00](https://youtu.be/u6NEbhNiyAE?t=120)

不管是視覺還是 NLP，**更好的模型都要更多計算**。NLP 模型大小呈指數成長：Transformer（0.05B）→ BERT（0.34B）→ GPT-2（1.5B）→ Megatron-LM（8.3B）→ T-NLG（17B）→ GPT-3（175B）。

| 模型 | 參數（M） | 訓練時間（A100 GPU 小時） |
|---|---|---|
| ResNet-50 | 26 | 31 |
| ResNet-101 | 45 | 44 |
| BERT-Base | 108 | 84 |
| Turing-NLG | 17,000 | — |
| GPT-3 | 175,000 | **3,100,000** |

- 沒有分散式訓練的話，單一 GPU 訓練 GPT-3 要 **355 年**。

⚠️ 逐字稿缺字：[▶ 03:00](https://youtu.be/u6NEbhNiyAE?t=180) 字幕說 BERT-Base「84 天」，投影片是 **84 GPU 小時**。

【口頭】教授放了一個笑話：老闆問「你上個月做了什麼？」「訓練了一個 epoch。」「那下個月的計畫呢？」「再訓練一個 epoch？」**訓練時間嚴重限制了研究者的產出**。

### 1.2 用更多 GPU

[▶ 03:00](https://youtu.be/u6NEbhNiyAE?t=180)

- **研究者的時間比硬體值錢**。
- 一個要 10 GPU 天的訓練，用 1024 張 GPU 平行（理想的線性擴展）：

$$
\frac{10\times24\times60\ \text{分鐘}}{1024} \approx 14\ \text{分鐘}
$$

- 【口頭】喝杯咖啡的時間，研究週期大幅加快。

### 1.3 實例：SUMMIT 超級電腦

[▶ 04:00](https://youtu.be/u6NEbhNiyAE?t=240)

- 每個節點 6 張 V100、雙軌 EDR InfiniBand 200 Gbit/s。
- 【口頭】這大約是 2018～2019 年的機器，V100 已經落後兩代（之後有 A100、H100）。現在 InfiniBand 很容易做到 400 Gbit/s。
- 訓練 TSM（[Lecture 15 §6.5](lecture15_gan_video_pointcloud_note.md)）：1 個節點 49 小時 50 分 → 256 個節點（1536 張 GPU）**14 分鐘（211×）**。

⚠️ 逐字稿缺字：[▶ 05:00](https://youtu.be/u6NEbhNiyAE?t=300) 字幕寫「140 minutes」「40 minutes」，投影片是 **14 分鐘**。

---

## 2. 三種平行方式

[▶ 05:00](https://youtu.be/u6NEbhNiyAE?t=300)

| 方式 | 切什麼 | 每張 GPU 有什麼 |
|---|---|---|
| **資料平行** | **切資料** | 完整的模型副本 + 一部分資料 |
| **管線平行** | **按層切模型** | 同一份資料 + 模型的某幾層 |
| **張量平行** | **在一層裡面切張量** | 同一份資料 + 每一層的一部分（例如 1/3 的通道） |

- **資料平行**：[▶ 06:00](https://youtu.be/u6NEbhNiyAE?t=360) 每張 GPU 有同一個模型，各自處理不同的資料。
- **管線平行**：第 1 層放 GPU 1，第 2 層放 GPU 2……
- **張量平行**：[▶ 07:00](https://youtu.be/u6NEbhNiyAE?t=420) 連單一個矩陣乘法都由多張 GPU 一起做，切完要同步。

---

## 3. ⭐ 資料平行（Data Parallelism）

### 3.1 參數伺服器架構

[▶ 08:00](https://youtu.be/u6NEbhNiyAE?t=480)

| 角色 | 工作 |
|---|---|
| **參數伺服器（parameter server）** | 整個訓練的中央控制者。收各 worker 的梯度，彙總後送回 |
| **Worker** | 用自己那份資料算梯度，送給參數伺服器 |

### 3.2 五個步驟

[▶ 09:00](https://youtu.be/u6NEbhNiyAE?t=540)

1. **Replicate / Pull**：把模型權重複製到每個 worker。
2. **Split data**：把訓練資料**隨機但平均地**切給每個 worker。
3. **Compute gradients**：每個 worker 用自己的資料做前向、反向，算出**本地梯度**。
4. **Push & Sum**：[▶ 10:00](https://youtu.be/u6NEbhNiyAE?t=600) 每個 worker 把本地梯度推給參數伺服器，加總。
5. **Update**：參數伺服器用加總的梯度更新權重。然後回到步驟 1，worker 再拉新的權重，保持**同步**。

【口頭】[▶ 10:00](https://youtu.be/u6NEbhNiyAE?t=600) 教授問：每個 worker 的梯度一樣嗎？**維度一樣**（都跟權重一樣大），但**值不一樣**，因為用的是不同的資料。

### 3.3 單機 vs 分散式

[▶ 11:00](https://youtu.be/u6NEbhNiyAE?t=660)

| 單機訓練（每次迭代） | 分散式訓練（每次迭代） |
|---|---|
| | **1. 從參數伺服器拉權重** |
| 1. 抽一批資料 $x^{(i)}, y^{(i)}$ | 2. 每個 worker $j$ 抽自己的資料 $x^{(i,j)}, y^{(i,j)}$ |
| 2. 算梯度 $\nabla w^{(i)}$ | 3. 算本地梯度 $\nabla w^{(i,j)}$ |
| | **4. 推到參數伺服器加總** |
| 3. 更新 $w \leftarrow w - \eta\nabla w$ | 5. 參數伺服器更新 |

→ 多了**兩個同步步驟**（1 和 4）。

---

## 4. ⭐⭐ 通訊原語

### 4.1 一對一：Send / Recv

[▶ 12:00](https://youtu.be/u6NEbhNiyAE?t=720)

- 把資料從一個節點直接傳到另一個。
- 最常用的分散式通訊方式。實作在 Socket、MPI、Gloo、NCCL 裡。

### 4.2 一對多

[▶ 13:00](https://youtu.be/u6NEbhNiyAE?t=780)

| 原語 | 做什麼 | 例子（4 個節點） |
|---|---|---|
| **Scatter** | 一個節點把**不同的資料**分給每個節點 | [A,B,C,D] → 節點 0 拿 A、節點 1 拿 B…… |
| **Gather** | 每個節點的資料**收集**到一個節點（Scatter 的反向） | → [1, 2, 3, 4] |
| **Reduce** | 跟 Gather 類似，但收集時**加總或平均** | [1]、[2]、[3]、[4] → [10] |
| **Broadcast** | 一個節點把**同一份資料**送給所有節點 | 每個人都拿到一樣的 |

【口頭】[▶ 14:00](https://youtu.be/u6NEbhNiyAE?t=840) Gather 和 Reduce 的差別：Gather 得到一個陣列 [1, 2, 3, 4]，Reduce 得到一個純量 10。

### 4.3 多對多

[▶ 15:00](https://youtu.be/u6NEbhNiyAE?t=900)

| 原語 | 做什麼 |
|---|---|
| **All-Reduce** | **每個節點**都做一次 Reduce → 每個人都拿到加總結果 |
| **All-Gather** | **每個節點**都做一次 Gather → 每個人都拿到完整陣列 |

### 4.4 參數伺服器用了哪些原語？

[▶ 17:00](https://youtu.be/u6NEbhNiyAE?t=1020)

| 步驟 | 原語 | Worker 頻寬 | 參數伺服器頻寬 |
|---|---|---|---|
| **Replicate / Pull** | **Broadcast**（伺服器 → 所有 worker） | $O(1)$ | $O(N)$ |
| **Push & Sum** | **Reduce**（所有 worker → 伺服器） | $O(1)$ | $O(N)$ |

**問題**：[▶ 19:00](https://youtu.be/u6NEbhNiyAE?t=1140) 參數伺服器的頻寬隨 worker 數 $N$ **線性成長**，機器一多就是瓶頸，**不可擴展**。

**能不能不要中央伺服器？** → **All-Reduce**。

### 4.5 All-Reduce 的四種實作

#### 循序（Sequential）

[▶ 20:00](https://youtu.be/u6NEbhNiyAE?t=1200)

```
for i = 0 to N−1:
    對 worker[i] 做一次 Reduce
```

- 第 1 步把大家的梯度 reduce 到節點 0，第 2 步 reduce 到節點 1……
- 時間 $O(N)$，頻寬 $O(N)$。

#### ⭐ 環狀（Ring）

[▶ 21:00](https://youtu.be/u6NEbhNiyAE?t=1260)

**只跟左右鄰居講話**：

```
for i = 0 to N−1:
    send(src=i, dst=(i+1) % N, data=worker[i])
    obj = recv(src=(i+1) % N, dst=i)
    worker[i] = merge(obj, worker[i])
```

4 個節點的例子：

- **Step 1**：0 → 1、1 → 2、2 → 3、3 → 0。現在每個節點有**自己 + 前一個**的資料。
- **Step 2**：[▶ 22:00](https://youtu.be/u6NEbhNiyAE?t=1320) 再傳一次。每個節點有**自己 + 前兩個**的資料。
- **Step 3**：再傳一次。每個節點都有**全部 4 個**的資料。

- 時間還是 $O(N)$，但**每個節點的頻寬只要 $O(1)$**，因為每次只跟鄰居講話。
- 【口頭】這是分散式訓練**非常常用**的做法。

#### 平行（Parallel）

[▶ 23:00](https://youtu.be/u6NEbhNiyAE?t=1380)

- 所有 Reduce 之間沒有相依性，可以**同時做**（只要頻寬夠）。
- 時間 $O(1)$，但總頻寬 $O(N^2)$（每個節點都要跟所有節點講話），**更糟**。

#### ⭐ 遞迴減半（Recursive Halving）

[▶ 26:00](https://youtu.be/u6NEbhNiyAE?t=1560)

結合 Ring 和 Parallel 的優點，用**樹狀結構**。8 個節點的例子：

| 步驟 | 跟誰交換 | 交換後每個節點有 |
|---|---|---|
| **Step 1** | 距離 **1** 的鄰居（1↔2、3↔4、5↔6、7↔8） | 2 個節點的資料 |
| **Step 2** | [▶ 27:00](https://youtu.be/u6NEbhNiyAE?t=1620) 距離 **2** 的鄰居（1↔3、2↔4、5↔7、6↔8） | 4 個節點的資料 |
| **Step 3** | 距離 **4** 的鄰居（1↔5、2↔6、3↔7、4↔8） | **全部 8 個** |

- ⭐ 關鍵一步：每一步，每個節點擁有的資料量**加倍**（1 → 2 → 4 → 8），所以 $N$ 個節點只要 **$\log_2 N$ 步**。8 個節點只要 3 步。

### 4.6 比較

[▶ 24:00](https://youtu.be/u6NEbhNiyAE?t=1440)

| 方法 | 時間 | 單一節點最大頻寬 | 總頻寬 |
|---|---|---|---|
| 參數伺服器 | $O(1)$ | $O(N)$ | $O(N)$ |
| All-Reduce 循序 | $O(N)$ | $O(N)$ | $O(N)$ |
| ⭐ All-Reduce 環狀 | $O(N)$ | **$O(1)$** | $O(N)$ |
| All-Reduce 平行 | **$O(1)$** | $O(N)$ | $O(N^2)$ |
| ⭐ 遞迴減半 | **$O(\log N)$** | — | — |

【口頭】[▶ 25:00](https://youtu.be/u6NEbhNiyAE?t=1500) 教授講這張表時一度以為有地方寫錯，後來確認是對的：平行版的**總頻寬**是 $O(N^2)$，**單一節點**還是 $O(N)$。

---

## 5. ⭐⭐ 減少資料平行的記憶體：ZeRO-1/2/3 與 FSDP

【口頭】[▶ 29:00](https://youtu.be/u6NEbhNiyAE?t=1740) 這是今天的重點，尤其是在 LLM 的時代。

### 5.1 問題：大模型根本放不進一張 GPU

[▶ 29:00](https://youtu.be/u6NEbhNiyAE?t=1740)

- GPT-3 175B，FP16：$175\times10^9\times2\ \text{bytes} = 350\ \text{GB}$。
- 一張 A100 只有 **80 GB**。**連權重都放不下**。
- 訓練時還要存梯度和 optimizer state。

### 5.2 每個參數要幾個 byte？

[▶ 30:00](https://youtu.be/u6NEbhNiyAE?t=1800)

用 FP16 混合精度訓練 + Adam optimizer：

| 項目 | 精度 | bytes |
|---|---|---|
| 權重 | FP16 | 2 |
| 梯度 | FP16 | 2 |
| **Optimizer state** | | **12** |
| └ FP32 權重主副本 | FP32 | 4 |
| └ Adam 的 momentum | FP32 | 4 |
| └ Adam 的 variance | FP32 | 4 |
| **合計** | | **16** |

【口頭】optimizer state（綠色那塊）**比權重和梯度大很多**。

一張 80 GB 的 GPU 最多能訓練：

$$
\frac{80\ \text{GB}}{16\ \text{bytes}} = 5\text{B 個參數}
$$

[▶ 33:00](https://youtu.be/u6NEbhNiyAE?t=1980) 離 175B 差太遠了。而且**一般的資料平行，每張 GPU 都存一整份**，有 $N$ 份重複。

【口頭】[▶ 34:00](https://youtu.be/u6NEbhNiyAE?t=2040) 可以從演算法、硬體、系統一起下手：

- 梯度也能用低精度（Lecture 19 裝置端訓練會講）。雲端訓練和裝置端訓練的很多技巧是共通的。
- 最大的是 optimizer state，可以設計**需要更少 state 的 optimizer**，例如 **Lion** 只要 4 bytes。

### 5.3 ZeRO：把東西切開（shard）分給各 GPU

DeepSpeed 的 ZeRO 方法。**Shard** 就是**切開分給不同 worker**。設 $N$ 張 GPU、$\psi$ 個參數，下面用 $N = 64$ 算。

#### ZeRO-1：切 optimizer state

[▶ 35:00](https://youtu.be/u6NEbhNiyAE?t=2100)

- 每個 worker 只存**自己那 $1/N$** 的 optimizer state，所以只負責更新自己那部分的權重。
- 需要別人的資料時，用通訊交換。

$$
\left(2 + 2 + \frac{12}{N}\right)\psi = \left(4 + \frac{12}{64}\right)\psi \approx 4.2\psi\ \text{bytes}
$$

$$
\frac{80\ \text{GB}}{4.2\ \text{bytes}} \approx 19\text{B}
$$

【口頭】從 5B 變 19B，進步很大。

#### ZeRO-2：再切梯度

[▶ 37:00](https://youtu.be/u6NEbhNiyAE?t=2220)

- 每個 worker 只存**自己負責的梯度**，只更新自己那部分的權重。

$$
\left(2 + \frac{2}{N} + \frac{12}{N}\right)\psi = \left(2 + \frac{14}{64}\right)\psi \approx 2.2\psi\ \text{bytes}
\quad\Rightarrow\quad
\frac{80\ \text{GB}}{2.2\ \text{bytes}} \approx 36\text{B}
$$

【口頭】[▶ 38:00](https://youtu.be/u6NEbhNiyAE?t=2280) 實務上 **ZeRO-2 比 ZeRO-1 常用很多**，幾乎是白吃的午餐。

#### ZeRO-3：連權重也切

[▶ 38:00](https://youtu.be/u6NEbhNiyAE?t=2280)

- 比較麻煩：前向和反向都需要**完整的權重**。
- 所以要用時，先用 **All-Gather** 從其他 GPU 把權重收集過來。

$$
\left(\frac{2}{N} + \frac{2}{N} + \frac{12}{N}\right)\psi = \frac{16}{64}\psi = 0.25\psi\ \text{bytes}
\quad\Rightarrow\quad
\frac{80\ \text{GB}}{0.25\ \text{bytes}} = 320\text{B}
$$

**取捨**：[▶ 39:00](https://youtu.be/u6NEbhNiyAE?t=2340) **通訊越多，要存的越少**。需要什麼就從別的伺服器收集過來。

【口頭】所以現代分散式訓練的基礎設施，**高速互連（例如 InfiniBand）**非常重要。

#### 比較與 FSDP

[▶ 40:00](https://youtu.be/u6NEbhNiyAE?t=2400)

| 方法 | 切什麼 | 每個參數 bytes（$N=64$） | 80 GB 能訓練 |
|---|---|---|---|
| 一般資料平行 | 不切 | 16 | 5B |
| **ZeRO-1** | optimizer state | 4.2 | 19B |
| **ZeRO-2** | + 梯度 | 2.2 | 36B |
| **ZeRO-3** | + 權重 | 0.25 | 320B |

- PyTorch 把 ZeRO-3 實作成 **FSDP（Fully Sharded Data Parallel，完全切分的資料平行）**。
- 「Fully sharded」就是**全部都切**：optimizer state、梯度、權重。
- 【口頭】[▶ 41:00](https://youtu.be/u6NEbhNiyAE?t=2460) 訓練 LLM 非常常用。

---

## 6. ⭐ 管線平行（Pipeline Parallelism）

### 6.1 按層切模型

[▶ 41:00](https://youtu.be/u6NEbhNiyAE?t=2460)

- 資料平行切資料，**管線平行切模型**：8 層的網路，第 1 層放 GPU 1、第 2 層放 GPU 2……
- 350 GB 的模型分到 8 張卡：$350/8 = 43.75\ \text{GB} < 80\ \text{GB}$。
- 【口頭】[▶ 42:00](https://youtu.be/u6NEbhNiyAE?t=2520) 這不完全準，因為還有 optimizer state 和梯度，不一定剛好放得下。

【口頭】[▶ 43:00](https://youtu.be/u6NEbhNiyAE?t=2580) 管線平行**很適合 LLM**：每一層的架構都一樣、大小一樣，每張 GPU 的工作量平均。如果某一層特別小、某一層特別大，小的那張 GPU 就會一直閒著等。

### 6.2 天真版：利用率只有 25%

[▶ 44:00](https://youtu.be/u6NEbhNiyAE?t=2640)

4 層網路、4 張 GPU（F = 前向、B = 反向）：

```
GPU 3:          F3 B3
GPU 2:       F2       B2
GPU 1:    F1             B1
GPU 0: F0                   B0  Update
```

- **同一時間只有一張 GPU 在算**，其他都在等。
- 【口頭】[▶ 45:00](https://youtu.be/u6NEbhNiyAE?t=2700) GPU 0 算完 F0 之後要閒置很久才輪到 B0。
- 理論利用率：**25%**。

### 6.3 ⭐ GPipe：切成 micro-batch

[▶ 46:00](https://youtu.be/u6NEbhNiyAE?t=2760)

- 把一個 batch 切成幾個 **micro-batch**：$[16, 10, 512] \to 4\times[4, 10, 512]$。
- GPU 0 算完 micro-batch 0 的 F0 就**馬上交給 GPU 1**，自己接著算 micro-batch 1。
- 這樣不同 GPU 的計算就**重疊**了。

```
GPU 3:          F3,0 F3,1 F3,2 F3,3 B3,0 B3,1 B3,2 B3,3
GPU 2:     F2,0 F2,1 F2,2 F2,3           B2,0 B2,1 B2,2 B2,3
GPU 1: F1,0 F1,1 F1,2 F1,3                    B1,0 B1,1 B1,2 B1,3
GPU 0: F0,0 F0,1 F0,2 F0,3                         B0,0 ...
```

【投影片補充】動機：一個 batch 計算期間模型參數不會變，所以可以把計算和通訊做成管線。

**利用率**：[▶ 48:00](https://youtu.be/u6NEbhNiyAE?t=2880) **57%**（約 2.5 倍提升）。

以下**不是逐字稿內容**，是依投影片的時間軸圖推論的：設 $P$ 個 stage、$M$ 個 micro-batch，每個 stage 前向要 $M$ 格，加上管線填滿需要的 $P-1$ 格空檔：

$$
\text{利用率} = \frac{M}{M + P - 1} = \frac{4}{4 + 4 - 1} = \frac{4}{7} \approx 57\%
$$

$M = 1$ 時就是 $1/4 = 25\%$，跟天真版一致。

**micro-batch 越多，利用率越高**。但【口頭】[▶ 49:00](https://youtu.be/u6NEbhNiyAE?t=2940) 太小的話，一個 micro-batch 可能**填不滿整張 GPU**，所以不能切太細。

---

## 7. ⭐⭐ 張量平行（Tensor Parallelism）

### 7.1 想法：在一層裡面切

[▶ 49:00](https://youtu.be/u6NEbhNiyAE?t=2940)

- 管線平行還是有空檔。能不能切得更細？
- **張量平行**：把**一層的權重矩陣**切成 $N$ 塊，多張 GPU 同時算。

$$
Y = XA = X\left[A_1\ A_2\ A_3\right] = \left[XA_1\ \ XA_2\ \ XA_3\right]
$$

- 【口頭】[▶ 50:00](https://youtu.be/u6NEbhNiyAE?t=3000) 要**沿著沒有 reduction 的維度切**，這樣才能減少通訊。
- 例子：NVIDIA 的 **Megatron-LM**。

### 7.2 FFN（MLP）層：先切欄、再切列

[▶ 51:00](https://youtu.be/u6NEbhNiyAE?t=3060)

例子：$[1, 2] @ [2, 6] @ [6, 2]$，兩張 GPU。$A$ 是第一層權重，$B$ 是第二層。

**第一層：按欄（column）切 $A$**

$$
A = \left[A_1\ A_2\right],\quad A_1, A_2 \in \mathbb{R}^{2\times3}
$$

1. 用 **Broadcast** 把輸入 $X$（$1\times2$）送到兩張 GPU。
2. GPU 1 算 $Y_1 = XA_1$（$1\times3$），GPU 2 算 $Y_2 = XA_2$（$1\times3$）。
3. ⭐ $Y_1$ 和 $Y_2$ **完全獨立**，不需要任何 reduction，**不用通訊**。

**第二層：按列（row）切 $B$**　[▶ 53:00](https://youtu.be/u6NEbhNiyAE?t=3180)

$$
B = \begin{bmatrix}B_1\\ B_2\end{bmatrix},\quad B_1, B_2 \in \mathbb{R}^{3\times2}
$$

1. GPU 1 算 $Z_1 = Y_1 B_1$（$1\times2$），GPU 2 算 $Z_2 = Y_2 B_2$（$1\times2$）。
2. $Z_1$、$Z_2$ 是**部分和**：

$$
Z = \left[Y_1\ Y_2\right]\begin{bmatrix}B_1\\ B_2\end{bmatrix} = Y_1B_1 + Y_2B_2 = Z_1 + Z_2
$$

3. 最後做一次 **All-Reduce** 加總，得到 $Z$（$1\times2$）。

**為什麼這樣切？**[▶ 54:00](https://youtu.be/u6NEbhNiyAE?t=3240)

- 第一層按欄切，輸出**剛好**是第二層按列切需要的輸入，**中間不用通訊**。
- 整個 MLP **只在最後**做一次 All-Reduce。
- 【口頭】剛好 Transformer 的 FFN 就是兩層，非常適合。
- 圖中的 $f$ 和 $g$：$f$ 前向是 identity、$g$ 前向是 All-Reduce（反向時兩者對調）。

### 7.3 Attention 層：按 head 切

[▶ 55:00](https://youtu.be/u6NEbhNiyAE?t=3300)

**Step 1：QKV 投影**，按欄切（按 head 切）

- 輸入 $X$ broadcast 到所有 GPU。
- GPU 1 算 head 1 的 $Q'_1 = XQ_1$、$K'_1 = XK_1$、$V'_1 = XV_1$；GPU 2 算 head 2 的。

**Step 2：Attention 本身**　[▶ 56:00](https://youtu.be/u6NEbhNiyAE?t=3360)

$$
Y_i = \mathrm{softmax}\left(Q'_i K'^\top_i\right)V'_i
$$

- 每個 head 獨立計算（multi-head attention 本來就是這樣），**不用通訊**。

**Step 3：輸出投影**，按列切　[▶ 57:00](https://youtu.be/u6NEbhNiyAE?t=3420)

- 輸出投影矩陣負責把不同 head 的結果混在一起。
- 每張 GPU 算 $Z_i = Y_iB_i$（部分和），最後 **All-Reduce** 得到 $Z$。

【口頭】[▶ 58:00](https://youtu.be/u6NEbhNiyAE?t=3480) 很神奇：QKV 投影和 attention 都不用通訊，**只在輸出投影之後**做一次 All-Reduce。

---

## 8. 一頁速查表

### 8.1 三種平行

| 方式 | 切什麼 | 通訊在哪 | 適合 |
|---|---|---|---|
| **資料平行** | 資料 | 每次迭代同步梯度（All-Reduce） | 模型放得進一張 GPU |
| **管線平行** | 按層切模型 | 相鄰 stage 之間傳激活值 | 每層一樣大（LLM） |
| **張量平行** | 一層裡面切張量 | 每個 MLP / attention 最後一次 All-Reduce | 單層太大；需要高速互連 |

### 8.2 通訊原語

| 原語 | 方向 | 一句話 |
|---|---|---|
| Send / Recv | 一對一 | 直接傳 |
| Scatter / Gather | 一對多 / 多對一 | 分不同的資料 / 收集成陣列 |
| Reduce / Broadcast | 多對一 / 一對多 | 收集並加總 / 送同一份 |
| All-Reduce / All-Gather | 多對多 | 每個人都拿到加總 / 完整陣列 |
| **Ring All-Reduce** | — | 只跟鄰居講話：時間 $O(N)$，單節點頻寬 $O(1)$ |
| **Recursive Halving** | — | 每步資料量加倍 → $\log_2 N$ 步 |

### 8.3 ZeRO

| 方法 | 每個參數 bytes | $N=64$ | 80 GB 能訓練 |
|---|---|---|---|
| 一般 | $2 + 2 + 12$ | 16 | 5B |
| ZeRO-1 | $2 + 2 + 12/N$ | 4.2 | 19B |
| ZeRO-2 | $2 + (2+12)/N$ | 2.2 | 36B |
| ZeRO-3（FSDP） | $(2+2+12)/N$ | 0.25 | 320B |

關鍵一步：Adam + 混合精度的 optimizer state = FP32 權重 + momentum + variance = $4\times3 = 12$ bytes，是最大的一塊。

### 8.4 管線與張量平行

| 項目 | 公式或做法 | 關鍵一步 |
|---|---|---|
| **管線利用率** | $\dfrac{M}{M+P-1}$（推論） | $P=4$：$M=1$ → 25%，$M=4$ → 57% |
| **MLP 張量平行** | $A$ 按欄切、$B$ 按列切 | $XA_i$ 獨立 → $Z = \sum_i Y_iB_i$，只在最後 All-Reduce |
| **Attention 張量平行** | 按 head 切 QKV，輸出投影按列切 | 每個 head 獨立，只在輸出投影後 All-Reduce |

### 8.5 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **355 年** | 單一 GPU 訓練 GPT-3 |
| **14 分鐘** | 10 GPU 天的訓練用 1024 張 GPU（理想） |
| **350 GB** | GPT-3 175B 的 FP16 權重 |
| **16 bytes** | 混合精度 + Adam 每個參數的記憶體 |
| **5B → 19B → 36B → 320B** | 80 GB GPU：一般 → ZeRO-1 → ZeRO-2 → ZeRO-3（$N=64$） |
| **25% → 57%** | 管線平行：天真版 → 4 個 micro-batch |

---

## 9. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **§5.3 ZeRO-3 用通訊換記憶體** | **[Lecture 18 分散式訓練（下）](lecture18_distributed_training_part2_note.md)**：怎麼**節省通訊頻寬**、怎麼**容忍通訊延遲** |
| **§5.2 梯度也能用低精度** | **Lecture 19 裝置端訓練** |
| **§5.2 optimizer state 太大** | **Lecture 20 高效微調**（LoRA 等只訓練少量參數） |

【口頭】[▶ 1:00:00](https://youtu.be/u6NEbhNiyAE?t=3600) 下一講講怎麼在頻寬有限、延遲很高時省通訊、容忍延遲，課後帶大家參觀機房。

---

## 附：本集一句話總結

> **模型太大、訓練太久，就要拆給很多 GPU：資料平行切資料、管線平行按層切、張量平行在一層裡切；同步梯度用 Ring 或遞迴減半的 All-Reduce 避免中央瓶頸；記憶體不夠就用 ZeRO 把 optimizer state、梯度、權重切開分給大家，用更多通訊換更少記憶體。**

---

*筆記依據 Song Han「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 17 錄影逐字稿（英文自動字幕，1:00:45）與投影片 `Lec17-Distributed-Training-Part-I.pdf`（93 頁）整理。*
