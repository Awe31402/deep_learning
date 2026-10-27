---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 23
lecture: 23
date: 2023-11-30
video: https://youtu.be/kCTzlodCZII
notes: Lec23-Noise-Robust-Quantum-ML.pdf（100 頁）
---

# 第 23 集 · Lecture 23：對雜訊穩健的量子機器學習 [▶ 影片](https://youtu.be/kCTzlodCZII)

← 上一集：[Lecture 22 量子機器學習](lecture22_quantum_machine_learning_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 24 期末專題報告](lecture24_final_project_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 23 集
> **講者**：Hanrui Wang（MIT 博士生）
> **來源**：YouTube 錄影逐字稿（1:14:37，英文自動字幕；這一講只有一般版）＋ 投影片 `Lec23-Noise-Robust-Quantum-ML.pdf`（100 頁）
> **標記說明**：
> - `【口頭】` = 講者口頭補充、投影片沒有
> - `【Q&A】` = 學生主動提問與回答
> - `【投影片補充】` = 投影片有、課堂沒講
> - `⚠️ 逐字稿缺字` = 字幕聽錯，依投影片修正
> - `⚠️ 待確認` = 查不清楚，列在最後的清單
> - `[▶ mm:ss]` = 可點的影片時間點（字幕以分鐘為單位，時間點是大約位置）

---

## 目錄

- [0. 這堂課要做什麼](#0-這堂課要做什麼)
- [1. TorchQuantum 的用法](#1-torchquantum-的用法)
- [2. ⭐⭐ 穩健的量子態準備（RobustState）](#2--穩健的量子態準備robuststate)
- [3. ⭐⭐ 穩健的電路架構搜尋（QuantumNAS）](#3--穩健的電路架構搜尋quantumnas)
- [4. 其他架構搜尋方法](#4-其他架構搜尋方法)
- [5. ⭐⭐ 穩健的參數訓練（QuantumNAT）](#5--穩健的參數訓練quantumnat)
- [6. 一頁速查表](#6-一頁速查表)
- [7. 與後續集數的連結](#7-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/kCTzlodCZII?t=0)

上一講建好了整個量子機器學習的流程：編碼 → 架構 → 訓練參數。這一講照同樣的結構，講每一步怎麼**對雜訊更穩健**：

| 段落 | 內容 | 對應流程 |
|---|---|---|
| 1 | TorchQuantum 範例（續） | 工具 |
| 2 | **穩健的量子態準備** | 編碼 |
| 3 | **穩健的電路架構搜尋**（【口頭】就像前面講的神經架構搜尋） | 架構 |
| 4 | **穩健的參數訓練** | 訓練 |

---

## 1. TorchQuantum 的用法

[▶ 01:00](https://youtu.be/kCTzlodCZII?t=60)

### 1.1 狀態向量模擬

核心就是**閘矩陣 × 狀態向量**。

**建立裝置、加閘**（【投影片補充】的寫法）：

```python
import torchquantum as tq
import torchquantum.functional as tqf

q_dev = tq.QuantumDevice(n_wires=5)   # 存狀態向量
tqf.h(q_dev, wires=1)                 # 方法 1：函式式
h_gate = tq.H(); h_gate(q_dev, wires=3)  # 方法 2：先建立閘物件
```

- 【口頭】方法 1 像 PyTorch 的 `torch.nn.functional as F`。方法 2 是給習慣 **Qiskit** 寫法的物理學家用的。
- `tq.QuantumState` 也能存狀態，還能直接 `q_state.h(wires=1)`、`q_state.rx(wires=1, params=0.2*np.pi)`。

**資料結構**：[▶ 02:00](https://youtu.be/kCTzlodCZII?t=120) 都是 PyTorch 原生的 tensor。狀態向量長度 $2^n$，初始化時第一個元素是 1（也就是 $|00\dots0\rangle$），或依需要初始化成別的狀態。

### 1.2 四個特色

| 特色 | 說明 |
|---|---|
| **動態計算圖** | [▶ 03:00](https://youtu.be/kCTzlodCZII?t=180) 加一個閘之後 `print(q_state)` 就能看到狀態變了。【口頭】Qiskit、TensorFlow Quantum 是先建圖再編譯的靜態圖 |
| **批次模式** | [▶ 04:00](https://youtu.be/kCTzlodCZII?t=240) `tq.QuantumState(n_wires=5, bsz=64)`：一次建立 64 個狀態，一次處理一整批影像 |
| **GPU** | `q_state.to(torch.device('cuda'))` |
| **自動微分** | 例如量子態準備的損失 $1 - \left\vert\langle\psi\vert\psi_{\text{target}}\rangle\right\vert$，直接 `loss.backward()` |

**編碼器**：[▶ 05:00](https://youtu.be/kCTzlodCZII?t=300) `tq.AmplitudeEncoder()`（振幅編碼）、`tq.PhaseEncoder()`（用旋轉閘編碼）。

**轉換**：`tq2qiskit(q_dev, q_model)` 轉成 Qiskit 電路，還能畫出來。

**速度**：[▶ 06:00](https://youtu.be/kCTzlodCZII?t=360)

- 跟 Xanadu 公司的 **PennyLane** 比，TorchQuantum 快很多。
- 【口頭】PennyLane 不支援批次和 GPU。量子位元越多、層數越多，差距越大：PennyLane 要一層一層循序做，TorchQuantum 加層數時執行時間只增加一點點。

---

## 2. ⭐⭐ 穩健的量子態準備（RobustState）

### 2.1 什麼是量子態準備，為什麼重要

[▶ 07:00](https://youtu.be/kCTzlodCZII?t=420)

**量子態準備**：把量子裝置初始化成想要的狀態。

【口頭】用途很廣：

- 量子機器學習的**振幅編碼**：把影像編進量子態。
- [▶ 08:00](https://youtu.be/kCTzlodCZII?t=480) 解 **PDE**：要把系統的初始狀態放進去。
- **量子錯誤更正碼**（color code、surface code）：量子位元有特定的初始結構。
- 起點的 fidelity 好，後面的運算才會好。

### 2.2 問題一：需要很多閘

[▶ 08:00](https://youtu.be/kCTzlodCZII?t=480)

| 方法 | 兩量子位元閘數量 |
|---|---|
| **分解式（decomposition）**：把目標狀態用數學分解成固定的閘序列 | **$O(2^N)$** |
| **變分式（variational）**：用含參數的電路去「訓練」出目標狀態 | **少很多** |

【口頭】$N$ 個量子位元要覆蓋整個空間，就要約 $2^N$ 個兩量子位元閘；6、12、20 個量子位元時閘數急遽增加。

### 2.3 問題二：雜訊讓 fidelity 大掉

[▶ 09:00](https://youtu.be/kCTzlodCZII?t=540)

- 分解的時候**沒考慮硬體雜訊**。
- 無雜訊模擬時 fidelity 接近 1。
- 【口頭】[▶ 10:00](https://youtu.be/kCTzlodCZII?t=600) 放到真的機器上，4、5 個量子位元就很低了；6 個量子位元時，**光是準備狀態這一步 fidelity 就接近 0**。

### 2.4 現有方法的兩個維度

[▶ 10:00](https://youtu.be/kCTzlodCZII?t=600)

| | **不考慮雜訊**（在無雜訊模擬器上準備） | **考慮雜訊**（有雜訊的模擬器或真機） |
|---|---|---|
| **不用梯度**（演化、基因演算法、貝氏最佳化） | 不穩健、效率低 | 穩健、效率低 |
| **用梯度**（parameter-shift、反向傳播） | 不穩健、效率高 | ⭐ **兩者都要** |

【口頭】[▶ 12:00](https://youtu.be/kCTzlodCZII?t=720) 考慮雜訊 → **穩健**；用梯度 → **訓練效率高**（想像訓練一個大神經網路卻不能反向傳播、只能用零階方法，會非常沒效率）。

### 2.5 RobustState 的三個做法

[▶ 13:00](https://youtu.be/kCTzlodCZII?t=780)

1. **梯度代理（gradient proxy）**
2. **硬體高效的 ansatz**（ansatz = 電路結構：幾層、用什麼閘、閘放哪）
3. **原生脈衝（native pulse）**

#### ⭐ 做法 1：梯度代理——前向在真機，反向在模擬器

[▶ 14:00](https://youtu.be/kCTzlodCZII?t=840)

```
真機：      電路（4 層）→ 有雜訊的最終密度矩陣 ρ_real
模擬器（平行）：電路（4 層）→ 無雜訊的中間狀態 … → 最終密度矩陣 ρ_sim
反向傳播：用 ρ_real 取代 ρ_sim（straight-through estimator），再沿著模擬器的中間狀態往回傳
```

**為什麼這樣做？**[▶ 15:00](https://youtu.be/kCTzlodCZII?t=900)

- 真機上**拿不到中間結果**：【口頭】因為**不可複製定理（no-cloning theorem）**，不能把量子位元複製出來看狀態。唯一拿得到的、受雜訊影響的結果，就是**最後的輸出**。
- 但反向傳播需要中間結果 → 由**模擬器**提供。
- 結果：最終狀態**有雜訊**（穩健）+ 能**反向傳播**（有效率）。

**估出來的梯度準不準？**[▶ 16:00](https://youtu.be/kCTzlodCZII?t=960)

- 跟最準的方法比：**parameter-shift**（每個參數跑 2 次）。【口頭】100 個參數就要跑 200 次。
- 梯度代理：**只在真機上跑 1 次**，所有參數的梯度一次得到。
- [▶ 17:00](https://youtu.be/kCTzlodCZII?t=1020) 兩者的**餘弦相似度一直高於 0.9**。

#### 做法 2：硬體高效的 ansatz

[▶ 18:00](https://youtu.be/kCTzlodCZII?t=1080)

不同硬體的量子位元連接方式不同：

| 拓撲 | 連接 |
|---|---|
| 線性 | 0–1–2–3 |
| 環狀 | 0–1–2–3–0 |
| 重六角（heavy hexagon） | 六角形，每條邊上也有量子位元 |
| 網格 | 2D 網格 |

- 讓兩量子位元閘**只放在有連線的地方**。例如線性拓撲：0–1、1–2、2–3 之間有閘，**0–3 之間沒有**，否則要付 SWAP 的代價。

#### 做法 3：直接在脈衝層級調參數

[▶ 19:00](https://youtu.be/kCTzlodCZII?t=1140)

- 【口頭】回顧 [Lecture 21 §7.2](lecture21_quantum_computing_basics_note.md)：**旋轉角度正比於脈衝曲線下的面積**。
- 想轉小一點的角度，就**直接縮小脈衝的振幅**，不用把 $R_X$ 分解成好幾個閘。
- [▶ 20:00](https://youtu.be/kCTzlodCZII?t=1200) 兩量子位元閘（例如 CNOT 由兩個 CR 脈衝組成）也能依參數調整**持續時間**。
- → 整個程式的**時間變短**。

### 2.6 結果

**測試的狀態**：[▶ 20:00](https://youtu.be/kCTzlodCZII?t=1200)

- **量子機器學習**：MNIST 降到 4×4 = 16 個值 → 用 4 個量子位元振幅編碼（數值例如 0.048 到 0.41）。
- **PDE 狀態**：有正弦函數的形狀。
- **任意狀態**：有實部和虛部。

**減少相干錯誤（coherent error）**：[▶ 21:00](https://youtu.be/kCTzlodCZII?t=1260)

【口頭】例子：想轉 40°，硬體因為控制不完美**總是多轉 10°**（變成 50°）。那就**只送 30°**，實際就會轉 40°。用真機的回饋訓練角度，就能自動學會這種補償，**過轉、欠轉的錯誤都減少**。

**其他比較**：

| 比較 | 結果 |
|---|---|
| vs 真機上的 parameter-shift、不用梯度的方法 | [▶ 22:00](https://youtu.be/kCTzlodCZII?t=1320) 準確度提升得更快 → 訓練效率更高 |
| vs 分解式（Qiskit 的算術分解、Möttönen 分解） | [▶ 23:00](https://youtu.be/kCTzlodCZII?t=1380) 真機上 **fidelity 更高** |
| vs 其他穩健的變分方法（不考慮雜訊的搜尋、parameter-shift、不用梯度） | 準確度更好 |

⚠️ 逐字稿缺字：字幕寫「maton decomposition」，依上下文應是 **Möttönen** 分解（一種量子態準備的分解方法）。

**脈衝 ansatz vs 閘 ansatz**：[▶ 24:00](https://youtu.be/kCTzlodCZII?t=1440)

| 訓練階段 | 誰比較好 | 原因 |
|---|---|---|
| **一開始** | **閘層級** | 【口頭】閘已經被 IBM 團隊仔細校正過；一改脈衝就會引入額外錯誤 |
| **訓練久了** | **脈衝層級** | 引入的錯誤被梯度訓練修掉，而脈衝層級的**時間更短** |

---

## 3. ⭐⭐ 穩健的電路架構搜尋（QuantumNAS）

### 3.1 兩個挑戰

[▶ 26:00](https://youtu.be/kCTzlodCZII?t=1560)

PQC 用在 VQE、量子神經網路、QAOA 等變分演算法上。

**挑戰 1：雜訊**

- 無雜訊模擬和真機的結果差很多。
- **參數越多，無雜訊的準確率越高，但真機的準確率反而下降**：【口頭】[▶ 27:00](https://youtu.be/kCTzlodCZII?t=1620) 加閘帶來的學習容量，被加閘帶來的雜訊抵銷了。
- → 存在一個**最佳點**，所以**電路架構非常關鍵**。

**挑戰 2：設計空間很大**

[▶ 28:00](https://youtu.be/kCTzlodCZII?t=1680)

| 維度 | 例子 |
|---|---|
| **閘的種類** | U3、或 X + CZ……【口頭】因為參數可以訓練，不同的閘組合可能做到類似的事 |
| **閘的數量** | 一個區塊、或疊很多層（【口頭】層數越多表達力越好，[Lecture 22 §2.3](lecture22_quantum_machine_learning_note.md)） |
| **閘的位置** | 先 CU3 再 U3，或其他排法 |

### 3.2 目標與想法

[▶ 29:00](https://youtu.be/kCTzlodCZII?t=1740)

- **自動**在大設計空間中找最好的架構，而且搜尋時**考慮真實的量子雜訊**。
- **兩層最佳化**：上層是電路結構，下層是電路參數。
- 【口頭】目標是**針對某個硬體**找到最好的結構；找到後，任何任務都能用這個結構（參數再依任務訓練）。

**三個重點**：

1. **參數共享**：[▶ 30:00](https://youtu.be/kCTzlodCZII?t=1800) 不要每個候選都從頭訓練（候選可能有上千個），只訓練一次。
2. **真機回饋**：在真機上評估候選，看哪個最好。
3. **跨層共同設計**：[▶ 30:00](https://youtu.be/kCTzlodCZII?t=1800) 【口頭】傳統計算設計好模型，不管用 H100、A100 還是筆電，結果都一樣；量子不行，**同樣的結構放在不同的實體量子位元上表現也不同**（每個量子位元雜訊不同），所以結構要和量子位元映射**一起搜尋**。

**四個步驟**：

1. 建立並訓練 **SuperCircuit**。
2. **雜訊自適應的演化搜尋**：同時搜尋 SubCircuit 和量子位元映射。
3. 訓練搜尋到的 SubCircuit。
4. **迭代式量子閘剪枝**。

（跟 [Lecture 7-8](lecture07_08_nas_note.md) 的 Once-for-All、[Lecture 14 §3.4](lecture14_vision_transformer_note.md) 的 SparseViT 是同一套思路。）

### 3.3 SuperCircuit 和 SubCircuit

[▶ 31:00](https://youtu.be/kCTzlodCZII?t=1860)

- 先定義設計空間，例如：**第一層最多 4 個 U3、第二層最多 4 個 CU3**，可以疊很多層。
- **SuperCircuit**：設計空間裡**閘最多**的那個電路。
- **SubCircuit**：設計空間裡的每個候選，都是 SuperCircuit 的**子集合**。
- SubCircuit **繼承** SuperCircuit 的參數。

**為什麼要用 SuperCircuit？**[▶ 32:00](https://youtu.be/kCTzlodCZII?t=1920)

- 【口頭】天真的做法：每個候選都訓練一次再上真機評估（例如 10%、20%、90%，選 90% 那個），成本太高。
- 用共享參數：繼承的參數不一定跟從頭訓練一樣好（例如 90% 變 70%、10% 變 5%），但**只要排名不變**就能選出最好的。
- **假設**：用繼承參數排名好的 SubCircuit，從頭訓練也會好（**排名保持，order preserving**）。

### 3.4 SuperCircuit 怎麼訓練

[▶ 34:00](https://youtu.be/kCTzlodCZII?t=2040)

每一步：

1. **取樣**一個 SubCircuit。
2. 只用這個子集合做任務，**只更新它的參數**。
3. 參數的更新會**跨步驟累積**：沒被取樣到的參數保持上一次的值。

【口頭】[▶ 35:00](https://youtu.be/kCTzlodCZII?t=2100) 這樣讓設計空間裡的所有候選都被訓練到**差不多的程度**；也許都只有從頭訓練的一半好，但足夠排名了。

**兩個讓訓練更穩定的取樣技巧**：[▶ 35:00](https://youtu.be/kCTzlodCZII?t=2100)

| 技巧 | 做法 |
|---|---|
| **前段取樣（front sampling）** | 先取樣區塊數，再取樣每個區塊裡的閘；**只能取前面幾個**區塊和前面幾個閘（例如取 2 個閘就取前 2 個） |
| **受限取樣（restricted sampling）** | [▶ 36:00](https://youtu.be/kCTzlodCZII?t=2160) **限制連續兩步的 SubCircuit 差異**，例如最多 4 層不同；差太多就拒絕、重新取樣 |

【口頭】[▶ 37:00](https://youtu.be/kCTzlodCZII?t=2220) 直覺：連續兩步的差異小，訓練在整個景觀上的軌跡就比較連續。

⚠️ 逐字稿缺字：講者口頭舉例時說「最多 3 層不同」，投影片寫「最多 4 層不同」。

**排名真的保持嗎？**[▶ 38:00](https://youtu.be/kCTzlodCZII?t=2280)

- 畫圖：x 軸是**繼承參數**的表現（便宜的代理），y 軸是**從頭訓練**的表現（真正的答案）。
- 代理上最好的那個，在 y 軸上**也是最好的** → 可以用便宜的代理來挑。

### 3.5 雜訊自適應的演化搜尋

[▶ 39:00](https://youtu.be/kCTzlodCZII?t=2340)

**同時搜尋 SubCircuit 結構和量子位元映射**。

- 【口頭】[▶ 40:00](https://youtu.be/kCTzlodCZII?t=2400) 例如 IBM Yorktown 有 5 個量子位元，映射不好就要多加 SWAP。
- 就算不用訓練，**每個候選都上真機評估**還是很貴（設計空間可能有 $10^{10}$ 個候選）→ 用**樣本效率高的演化搜尋**。

**基因（gene）**：[▶ 41:00](https://youtu.be/kCTzlodCZII?t=2460) 一個向量，同時記錄結構和映射：

```
[第 1 層幾個閘, 第 2 層幾個閘, …, 邏輯量子位元 0 → 實體量子位元 ?, 邏輯 1 → ?, …]
```

**演化流程**：[▶ 42:00](https://youtu.be/kCTzlodCZII?t=2520)

| 操作 | 做法 |
|---|---|
| **突變（mutation）** | 改基因裡的某些元素（例如第 1 個數字改成 3） |
| **交叉（crossover）** | [▶ 43:00](https://youtu.be/kCTzlodCZII?t=2580) 從兩個親代各取一部分基因，產生子代 |
| **評估** | 在**真機**上測，用表現更新族群 |

幾輪之後得到最好的結構 + 映射。

**為什麼結構會跟硬體有關？**[▶ 44:00](https://youtu.be/kCTzlodCZII?t=2640) 【口頭】例如某兩個量子位元錯誤率特別低，就能在它們上面放更多閘（更多學習容量）；差的量子位元就放淺一點的電路。

### 3.6 訓練 + 迭代式閘剪枝

[▶ 44:00](https://youtu.be/kCTzlodCZII?t=2640)

1. 固定結構和映射，依具體任務（影像分類、量子化學……）訓練參數（parameter-shift 等）。
2. [▶ 45:00](https://youtu.be/kCTzlodCZII?t=2700) **閘剪枝**：
   - 旋轉角度**接近 0**（或接近 $2\pi$）的閘 ≈ identity，對結果影響很小 → 拿掉。
   - **迭代地**剪掉小角度的閘，再微調剩下的參數。

【Q&A】[▶ 51:00](https://youtu.be/kCTzlodCZII?t=3060) 學生問：剪掉多少閘？講者：這個例子大約 **30%**，看任務而定。做法是迭代剪：先剪 10% 測一下表現，再剪 10%……表現會先升後降，找最好的剪枝比例。

**為什麼剪枝反而會提升準確率？**[▶ 50:00](https://youtu.be/kCTzlodCZII?t=3000)

- 【口頭】傳統的剪枝通常準確率會持平或稍降，因為容量變小。
- 量子的剪枝：拿掉不必要的閘、重新訓練剩下的，**無雜訊的表現差不多**，但**雜訊變少了** → 更接近無雜訊的結果，**準確率反而提升**。
- 【投影片補充】MNIST-4 上，閘剪枝**平均提升 3%** 準確率。

### 3.7 結果

**測試**：[▶ 47:00](https://youtu.be/kCTzlodCZII?t=2820)

- QML 分類：MNIST 10 類 / 4 類 / 2 類、Fashion-MNIST 4 類 / 2 類、Vowel 4 類。
- VQE 分子：H₂、H₂O、LiH、CH₄、BeH₂。
- 【投影片補充】IBMQ 硬體：5～65 個量子位元，quantum volume 8～128。

**跟誰比**：

| 方法 | 說明 |
|---|---|
| ⭐ **QuantumNAS**（考慮雜訊的搜尋） | — |
| 不考慮雜訊的搜尋 | 搜尋時沒用真機回饋，結構沒有針對硬體雜訊特化 |
| 人工設計 | 專家設計的最佳結構 |

**MNIST-4（IBMQ Yorktown）**：[▶ 48:00](https://youtu.be/kCTzlodCZII?t=2880)

- 閘越多，表現**先升後降**（容量 vs 雜訊）。
- QuantumNAS 的**峰值高很多**，而且能承受更多閘。【口頭】以前人工設計只能用約 40 個閘，用搜尋能用到 60～70 個。
- 【口頭】不用等物理學家做出更好的硬體，靠**系統層級的優化**就能從現有硬體榨出更多算力。

**VQE（H₂）**：[▶ 49:00](https://youtu.be/kCTzlodCZII?t=2940) 跟理論設計（UCCSD）、不考慮雜訊的搜尋、隨機結構、人工設計比，QuantumNAS **最接近基態能量**；在不同設計空間都一致比較好。

⚠️ 待確認：講者口頭說 H₂ 基態能量的正確值是「−1.85」，但投影片的數字在圖上抽不出來，無法核對。

**大型硬體上的 MNIST-10**：[▶ 49:00](https://youtu.be/kCTzlodCZII?t=2940)

- 其他方法在大機器上都約 **10%**（10 類，等於亂猜）。
- QuantumNAS：約 **30%**。【口頭】還不高，但至少有在學；無雜訊模擬可以超過 90%。

⚠️ 待確認：[▶ 50:00](https://youtu.be/kCTzlodCZII?t=3000) 字幕這段還提到「60%」，跟前面的 30% 對不起來，看不出 60% 指的是什麼。

**搜尋成本**：[▶ 52:00](https://youtu.be/kCTzlodCZII?t=3120) 共享參數的 SuperCircuit 讓搜尋成本**減少約 3 個數量級**（10 類分類）。【投影片補充】測量在 1 張 GPU 上。

---

## 4. 其他架構搜尋方法

### 4.1 更快地估計真機表現：QuEst

[▶ 52:00](https://youtu.be/kCTzlodCZII?t=3120)

**問題**：要在真機上評估，【口頭】送出工作後常常要**排隊好幾個小時**。

**做法**：用**資料驅動的模型**預測電路的 fidelity。

1. **建資料集**：[▶ 53:00](https://youtu.be/kCTzlodCZII?t=3180) 產生很多隨機電路，**接上它的反電路**（理想結果就是回到初始狀態），在真機上跑一次就知道 fidelity。
2. **模型**：電路本身是一張**相依圖** → 用 **graph transformer**：
   - [▶ 54:00](https://youtu.be/kCTzlodCZII?t=3240) 每層用 attention 讓每個節點收集鄰居的資訊。
   - 最後 global average pooling → FC 層 → 預測 fidelity。

### 4.2 其他做法

[▶ 54:00](https://youtu.be/kCTzlodCZII?t=3240)

| 方法 | 做法 |
|---|---|
| **神經預測器** | 把電路畫成一張「影像」（例如 SWAP 用兩個像素表示），用神經網路預測表現 |
| **強化學習** | [▶ 55:00](https://youtu.be/kCTzlodCZII?t=3300) agent 一次放一個閘到某個量子位元上，拿到 reward，慢慢長出電路 |
| **可微分搜尋** | [▶ 56:00](https://youtu.be/kCTzlodCZII?t=3360) 對結構算梯度，用梯度更新 |
| **另一種 SuperCircuit 方法** | 結構比較受限，而且**沒用真機回饋**，找到的淺電路不一定適合目標硬體 |

---

## 5. ⭐⭐ 穩健的參數訓練（QuantumNAT）

[▶ 57:00](https://youtu.be/kCTzlodCZII?t=3420)

【口頭】前面講結構的穩健性，這一段講**參數**的穩健性。

### 5.1 多節點量子神經網路

[▶ 57:00](https://youtu.be/kCTzlodCZII?t=3420)

- 每個**節點**有：編碼器 → 可訓練層 → 測量。
- 測量後回到傳統領域（圖上的雙線），可以做**傳統的處理**（神經網路、雜訊緩解），再編碼回量子進下一個節點，最後分類。

**取捨**：[▶ 58:00](https://youtu.be/kCTzlodCZII?t=3480)

| | 很多層接在一起 | 中間切開做測量 + 緩解 |
|---|---|---|
| 理論容量 | 高 | 低（電路不深） |
| 雜訊 | **很大** | 每段都被緩解 |

→ 要仔細設計區塊數和用哪種緩解方法。

### 5.2 測量後正規化（Post-measurement normalization）

[▶ 59:00](https://youtu.be/kCTzlodCZII?t=3540)

**做法**：

- 例如 4 個量子位元、batch = 50 → 測量結果是 $50\times4$ 個值。
- ⭐ **沿著 batch 維度**，對**每個量子位元**算平均和標準差（不是沿著量子位元維度）。
- 用這組平均和標準差正規化：

$$
\hat{y}_{b,q} = \frac{y_{b,q} - \mu_q}{\sigma_q},\qquad \mu_q = \frac{1}{B}\sum_{b=1}^{B}y_{b,q},\quad \sigma_q = \sqrt{\frac{1}{B}\sum_{b=1}^{B}\left(y_{b,q} - \mu_q\right)^2}
$$

**效果**：[▶ 1:00:00](https://youtu.be/kCTzlodCZII?t=3600)

- 期望值範圍是 $-1$ 到 $1$。無雜訊和真機的分布**位移、縮放**都不一樣，峰值對不上，訊雜比（SNR）很低。
- 兩者都正規化後，分布**對齊了**。
- 【口頭】SNR 從約 **0.8 提升到 5 以上**。有理論證明：正規化能消除一部分**相干錯誤**對測量的影響。

### 5.3 雜訊注入（Noise injection）

[▶ 1:02:00](https://youtu.be/kCTzlodCZII?t=3720)

**想法**：在傳統模擬器上訓練時**加入雜訊**，讓參數提早適應真機。【口頭】就像傳統學習任務也會注入雜訊。

**注入兩種錯誤**：

| 錯誤 | 說明 | 例子（IBM 真機的校正資料） |
|---|---|---|
| **Pauli 錯誤** | 【口頭】再複雜的錯誤都能分解成 Pauli 錯誤（X、Y、Z） | [▶ 1:03:00](https://youtu.be/kCTzlodCZII?t=3780) X 閘之後：99.7% 沒錯，X、Y、Z 錯誤各約 0.1% |
| **讀出錯誤** | 測量時讀錯 | [▶ 1:04:00](https://youtu.be/kCTzlodCZII?t=3840) 真的是 0 時，98% 讀到 0、約 2% 讀成 1 |

- 每一步訓練**依機率重新取樣**要注入哪些錯誤（真機本來就是隨機的）。
- 【口頭】讀出錯誤大約是單一量子位元閘錯誤的 **10 倍**，不能忽略。知道轉移矩陣的話，可以**反過來修正**讀出錯誤。

### 5.4 測量後量化（Post-measurement quantization）

[▶ 1:05:00](https://youtu.be/kCTzlodCZII?t=3900)

- 把測量結果**量化**到最近的格點（centroid）→ **去雜訊**：小的誤差直接被拉回正確的格點。
- 但誤差太大時，可能被拉到**錯的格點**。
- → 訓練時加一個**損失項**，讓測量結果**盡量靠近格點**：

$$
\mathcal{L}_{\text{quant}} = \left\Vert y - Q(y)\right\Vert^2
$$

（$y$ 是原本的值、$Q(y)$ 是量化後的值；靠近格點的話，小誤差不會把它推到別的格點。）

- [▶ 1:06:00](https://youtu.be/kCTzlodCZII?t=3960) 量化前，誤差有大有小（例如 1.1、0.1）；量化後，**大部分誤差變成 0**。

### 5.5 結果（IBMQ Santiago）

[▶ 1:07:00](https://youtu.be/kCTzlodCZII?t=4020)

| 方法 | MNIST-4 | FMNIST-4 | Vowel-4 | MNIST-2 | FMNIST-2 | CIFAR-2 |
|---|---|---|---|---|---|---|
| 基準 | 0.30 | 0.32 | 0.28 | 0.84 | 0.78 | 0.51 |
| + 正規化 | 0.41 | 0.61 | 0.29 | 0.87 | 0.68 | 0.56 |
| + 雜訊注入 | 0.61 | 0.70 | 0.44 | 0.93 | 0.86 | 0.57 |
| + 量化 | **0.68** | **0.75** | **0.48** | **0.94** | **0.88** | **0.59** |

- 【口頭】Santiago 是當時錯誤率最低的機器之一。
- 正規化 → 約 +10%；加雜訊注入 → 更接近真機環境；最後量化 → 大部分誤差被拉回格點。
- MNIST-4 從 30% 提升到 68%；兩類分類能做到約 94%。

⚠️ 逐字稿缺字：字幕寫「IBM San Diego」，依投影片是 **IBMQ Santiago**。

**特徵空間的視覺化**（MNIST-2：數字 3 vs 6）：[▶ 1:09:00](https://youtu.be/kCTzlodCZII?t=4140)

| 階段 | 樣本分布 |
|---|---|
| 基準 | 雜訊讓所有樣本**擠在一起**，全被分到同一類，等於亂猜 |
| + 正規化 | [▶ 1:10:00](https://youtu.be/kCTzlodCZII?t=4200) 分布被**拉開**，但很多還擠在分界線上 |
| + 雜訊注入 | [▶ 1:11:00](https://youtu.be/kCTzlodCZII?t=4260) 再被拉到左上和右下兩個角落，大部分分對 |

### 5.6 Q&A

【Q&A】[▶ 1:13:00](https://youtu.be/kCTzlodCZII?t=4380) 學生問：很深的電路怎麼辦？編譯器能不能自動找切開的位置？講者：

- 對 ML 來說，很深的電路**確實要切成幾段**，中間做緩解。
- 有研究在找**最好的切點**。
- 中間的測量也是一個設計空間：要**完整重建狀態（tomography）**需要**指數多次**的測量；**shadow tomography** 只要量一小部分，就能抓到狀態中有用的資訊，再處理、緩解、編碼回去。

---

## 6. 一頁速查表

### 6.1 TorchQuantum

| 功能 | 寫法 |
|---|---|
| 建裝置 | `tq.QuantumDevice(n_wires=5)` |
| 加閘 | `tqf.h(q_dev, wires=1)` 或 `tq.H()(q_dev, wires=3)` 或 `q_state.h(wires=1)` |
| 批次 / GPU | `tq.QuantumState(n_wires=5, bsz=64).to('cuda')` |
| 自動微分 | `loss.backward()` |
| 轉 Qiskit | `tq2qiskit(q_dev, q_model)` |

### 6.2 三個穩健化方法

| 方法 | 對應流程 | 關鍵一步 |
|---|---|---|
| **RobustState** | 量子態準備 | 前向在真機（有雜訊的最終狀態）、反向在模擬器（中間狀態）；跟 parameter-shift 的餘弦相似度 > 0.9，只跑 1 次 |
| **QuantumNAS** | 電路架構 | SuperCircuit 共享參數（前段取樣 + 受限取樣）→ 演化搜尋結構 + 映射（真機回饋）→ 訓練 → 剪掉角度接近 0 的閘 |
| **QuantumNAT** | 參數訓練 | 沿 batch 正規化測量結果 + 注入 Pauli / 讀出錯誤 + 量化測量結果（加損失項靠近格點） |

### 6.3 關鍵一步

| 觀念 | 一句話 |
|---|---|
| **為什麼前向要在真機** | 不可複製定理 → 真機只拿得到最終結果 |
| **為什麼參數越多不一定越好** | 容量的好處被雜訊抵銷 → 有最佳點 |
| **為什麼剪枝會提升準確率** | 少一個閘就少一份雜訊，無雜訊表現不變 |
| **為什麼沿 batch 正規化** | 雜訊造成整體位移、縮放 → 用整批的統計量抵銷 |
| **SuperCircuit 的假設** | 繼承參數的排名 ≈ 從頭訓練的排名 |

### 6.4 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **$O(2^N)$** | 分解式量子態準備需要的兩量子位元閘數 |
| **> 0.9** | 梯度代理跟 parameter-shift 的餘弦相似度 |
| **1 vs $2N$** | 梯度代理 vs parameter-shift 每次要在真機上跑的次數 |
| **3 個數量級** | SuperCircuit 減少的搜尋成本 |
| **~30%** | 例子中剪掉的閘比例 |
| **+3%** | 閘剪枝在 MNIST-4 上的平均提升 |
| **0.8 → > 5** | 正規化後的 SNR |
| **0.30 → 0.68** | MNIST-4 在 IBMQ Santiago 上：基準 → QuantumNAT 全部 |

---

## 7. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **量子章節結束** | **[Lecture 24 期末專題報告](lecture24_final_project_note.md)** |
| **§3 QuantumNAS 的 SuperCircuit** | 回顧 [Lecture 7-8 NAS](lecture07_08_nas_note.md) 的 Once-for-All |
| **§3.6 閘剪枝** | 回顧 [Lecture 3-4 剪枝](lecture03_04_pruning_note.md) |
| **§5.4 測量後量化** | 回顧 [Lecture 5-6 量化](lecture05_06_quantization_note.md) |

---

## 附：本集一句話總結

> **NISQ 的雜訊會毀掉量子機器學習的每一步，所以每一步都要穩健化：量子態準備用「前向在真機、反向在模擬器」的梯度代理同時拿到雜訊感知和訓練效率（RobustState）；電路架構用共享參數的 SuperCircuit 加真機回饋的演化搜尋，連量子位元映射一起找，再剪掉沒用的閘（QuantumNAS）；參數訓練則靠測量後正規化、雜訊注入、測量後量化把雜訊壓下去（QuantumNAT）——用的其實都是這門課前面教過的 NAS、剪枝、量化。**

---

## ⚠️ 待確認清單

1. [▶ 49:00](https://youtu.be/kCTzlodCZII?t=2940) H₂ 基態能量的正確值：講者說「−1.85」，投影片的數字在圖上抽不出來，無法核對。
2. [▶ 50:00](https://youtu.be/kCTzlodCZII?t=3000) MNIST-10 大型硬體的結果：字幕提到 30% 和「60%」，看不出 60% 指的是什麼。

---

*筆記依據 Hanrui Wang「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 23 錄影逐字稿（英文自動字幕，1:14:37）與投影片 `Lec23-Noise-Robust-Quantum-ML.pdf`（100 頁）整理。*
