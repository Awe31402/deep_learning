---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 21
lecture: 21
date: 2023-11-21
video: https://youtu.be/Z8GKkgE2840
notes: Lec21-Basics-of-Quantum-Computing.pdf（82 頁）
---

# 第 21 集 · Lecture 21：量子計算基礎 [▶ 影片](https://youtu.be/Z8GKkgE2840)

← 上一集：[Lecture 20 高效微調與 Prompt Engineering](lecture20_efficient_finetuning_prompt_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 22 量子機器學習](lecture22_quantum_machine_learning_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 21 集
> **講者**：Hanrui Wang（Song Han 實驗室的博士生；Song Han 開場介紹）
> **來源**：YouTube 錄影逐字稿（1:16:12，英文自動字幕，**一般版**；Zoom 版的字幕抓不到）＋ 投影片 `Lec21-Basics-of-Quantum-Computing.pdf`（82 頁）
> **標記說明**：
> - `【口頭】` = 講者口頭補充、投影片沒有
> - `【課堂討論】` = 講者問、學生答
> - `【投影片補充】` = 投影片有、課堂沒講
> - `⚠️ 逐字稿缺字` = 字幕聽錯，依投影片修正
> - `⚠️ 待確認` = 查不清楚，列在最後的清單
> - `[▶ mm:ss]` = 可點的影片時間點（字幕以分鐘為單位，時間點是大約位置）

⚠️ 逐字稿缺字：字幕把講者的名字寫成「Hybrid」「Henry」，投影片上是 **Hanrui Wang**。

---

## 目錄

- [0. 這堂課要做什麼](#0-這堂課要做什麼)
- [1. 為什麼要關心量子計算](#1-為什麼要關心量子計算)
- [2. ⭐⭐ 單一量子位元：狀態與測量](#2--單一量子位元狀態與測量)
- [3. ⭐⭐ 單一量子位元的閘](#3--單一量子位元的閘)
- [4. ⭐⭐ 多量子位元：狀態、閘、糾纏](#4--多量子位元狀態閘糾纏)
- [5. 量子電路](#5-量子電路)
- [6. ⭐ NISQ 時代與編譯問題](#6--nisq-時代與編譯問題)
- [7. 超導量子電腦的實際流程](#7-超導量子電腦的實際流程)
- [8. 一頁速查表](#8-一頁速查表)
- [9. 與後續集數的連結](#9-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/Z8GKkgE2840?t=0)

【口頭】Song Han 介紹講者：實驗室的最後一年博士生，2018 年加入。先做 ViT 和 LLM 的效率（例如 hardware-aware transformer、SpAtten 的 token 剪枝），後來轉做**機器學習 × 量子計算**，開發了 **TorchQuantum**（PyTorch 的量子計算函式庫，已整合進 IBM Qiskit、NVIDIA 的量子平台）。

【口頭】接下來三講是量子：這講先講**量子計算的基礎**，下一講講量子機器學習。

| 段落 | 內容 |
|---|---|
| 1 | 單一量子位元的狀態與閘 |
| 2 | 多量子位元的狀態與閘 |
| 3 | 量子電路 |
| 4 | NISQ 時代與編譯問題 |
| 5 | 超導量子電腦的實際流程 |

---

## 1. 為什麼要關心量子計算

### 1.1 硬體進步非常快

[▶ 03:00](https://youtu.be/Z8GKkgE2840?t=180)

- 【口頭】1998 年第一台量子機器只有 **2 個量子位元**，花了約 20 年才做到 10 個以上。
- 真正的加速發生在**最近三年**：從約 50 個量子位元到超過 **1,000 個**，兩年兩個數量級。
- [▶ 04:00](https://youtu.be/Z8GKkgE2840?t=240) 【口頭】一個月前，一家在科羅拉多的新創公司 **Atom Computing** 發表了 **1,181 個量子位元**的機器；IBM 也要發表超過 1,000 個量子位元的機器。
- 量子位元一多，就不能手動設計，**需要自動化的編譯和工具鏈**。

⚠️ 逐字稿缺字：字幕把公司名寫成「IonQ Computing」。但講者說它用的是**中性原子**（IonQ 是做離子阱的），而且下一講（[Lecture 22 §1.1](lecture22_quantum_machine_learning_note.md)）講同一台約 1,200 個量子位元的機器時，字幕寫的是「item Computing」，所以應是 **Atom Computing**。

**多種技術在競爭**：[▶ 05:00](https://youtu.be/Z8GKkgE2840?t=300)

| 技術 | 例子 |
|---|---|
| **超導（superconducting）** | IBM、Google |
| **離子阱（trapped ion）** | 在線性阱裡控制離子 |
| **中性原子（neutral atom）** | 在真空腔裡用雷射控制原子 |
| **光子（photonics）** | — |

【口頭】[▶ 06:00](https://youtu.be/Z8GKkgE2840?t=360) 就像 70 年前大家在爭論該用真空管還是電晶體，現在還不知道哪一種最好。

### 1.2 「雙重指數」的潛力

[▶ 06:00](https://youtu.be/Z8GKkgE2840?t=360)

1. **量子位元數**隨時間**指數成長**。
2. **計算能力**隨量子位元數**指數成長**：$n$ 個量子位元的狀態空間大小是 $2^n$。

→ 兩層指數，所以叫**雙重指數（doubly exponential）**。

【口頭】[▶ 07:00](https://youtu.be/Z8GKkgE2840?t=420) 例子：

- 人類每天產生約 $2.5\times10^{18}$ bytes 的資料，**64 個量子位元**的狀態空間（$2^{64}\approx1.8\times10^{19}$）就裝得下。
- [▶ 08:00](https://youtu.be/Z8GKkgE2840?t=480) 100 個量子位元 → $2^{100}\approx10^{30}$。
- GPU 叢集要再多 1 萬張才能把算力加倍；量子只要**多 1 個量子位元**就加倍（但要有聰明的演算法才能用上）。

### 1.3 歷史

[▶ 09:00](https://youtu.be/Z8GKkgE2840?t=540)

| 時期 | 里程碑 | 假設 |
|---|---|---|
| **1980～90 年代** | **量子模擬**（Feynman：「世界是量子的，為什麼不用量子電腦模擬它？」）、**Shor 因數分解演算法** | **無雜訊、大規模**的量子電腦 |
| **1995** | 理論證明：**有雜訊**的大規模量子電腦，靠**量子錯誤更正（QEC）**也能有實際加速 | 有雜訊、大規模 |
| **2019～** | 實驗證明在**有雜訊、小規模（NISQ）**的電腦上有加速（Google 2019、USTC 2020、Xanadu）——但**是否實用有問號** | 有雜訊、小規模 |
| **未來** | 兩條路：**NISQ 計算** 和 **量子錯誤更正** | |

**Shor 演算法**：[▶ 11:00](https://youtu.be/Z8GKkgE2840?t=660)

- 能有效分解很大的整數，而 RSA 加密就是建立在「大數很難分解」上。
- 【口頭】[▶ 12:00](https://youtu.be/Z8GKkgE2840?t=720) 放心，明天不會有量子電腦破解你的銀行帳戶：現在的機器不可靠，而且要做有意義的事需要約 **100 萬個量子位元**。但密碼學界已經開始改用**抗量子（quantum-safe）**的加密。

**量子錯誤更正的直覺**：[▶ 14:00](https://youtu.be/Z8GKkgE2840?t=840)

- 【口頭】跟傳統的 ECC 一樣，**用冗餘換可靠度**。
- 例子：送 1 個 bit 時重複送 10 次，有些被雜訊翻轉了，只要翻轉的不超過一半，**多數決**就能還原。

**為什麼有 NISQ？**[▶ 15:00](https://youtu.be/Z8GKkgE2840?t=900)

- 【口頭】20 年前跟創投說「給我 100 億美元，我做一台 1,000 個量子位元、容錯的機器」，大概拿不到錢。
- 所以先做**中等規模**（幾十到幾百個量子位元）的機器，試著證明這個技術**現在**就有價值。
- [▶ 16:00](https://youtu.be/Z8GKkgE2840?t=960) 但目前證明的加速（例如 boson sampling）不一定對實際應用有用。

---

## 2. ⭐⭐ 單一量子位元：狀態與測量

### 2.1 狀態向量

[▶ 17:00](https://youtu.be/Z8GKkgE2840?t=1020)

量子計算的基本元件是**量子位元（qubit）**，用**狀態向量（statevector）**描述它的狀態。

**比喻**：[▶ 18:00](https://youtu.be/Z8GKkgE2840?t=1080)

- 傳統系統：一條一維的路上有一台車，用一個數字（例如「在 4 的位置」）就能描述。
- 用「車在每個位置的**機率**」來描述，對傳統系統**很沒效率**（只有一格是 1 的 one-hot 向量），但對量子狀態**非常有效**，因為量子位元可以**同時**在不同狀態。

### 2.2 Bra-ket 記號

[▶ 19:00](https://youtu.be/Z8GKkgE2840?t=1140)

| | 傳統 | 量子 |
|---|---|---|
| 狀態 | $a = 0$ 或 $a = 1$ | 用**正交向量**表示 |

$$
|0\rangle = \begin{bmatrix}1\\0\end{bmatrix},\qquad |1\rangle = \begin{bmatrix}0\\1\end{bmatrix}
$$

- 這叫 **bra-ket 記號**（也叫 Dirac 記號）。$|0\rangle$、$|1\rangle$ 構成一組**正交單位基底（orthonormal basis）**。
- **Ket** $|x\rangle$：行向量（column vector）。
- **Bra** $\langle x|$：列向量（row vector），是 $|x\rangle$ 的**共軛轉置**。

更複雜的狀態，例如：

$$
|q_0\rangle = \begin{bmatrix}\frac{1}{\sqrt{2}}\\[2pt] \frac{i}{\sqrt{2}}\end{bmatrix}
$$

【口頭】[▶ 21:00](https://youtu.be/Z8GKkgE2840?t=1260) 一個量子位元的所有資訊，就是一個有**兩個複數**的向量。

### 2.3 疊加（Superposition）

[▶ 21:00](https://youtu.be/Z8GKkgE2840?t=1260)

把上面的狀態拆成兩個基底的組合：

$$
|q_0\rangle = \frac{1}{\sqrt{2}}|0\rangle + \frac{i}{\sqrt{2}}|1\rangle
$$

- **疊加**：兩個基底狀態的**線性組合**。
- 【口頭】[▶ 22:00](https://youtu.be/Z8GKkgE2840?t=1320) 傳統的 1 個 bit 只能是 0 或 1；量子位元可以同時有 $|0\rangle$ 和 $|1\rangle$ 的成分。

### 2.4 測量

[▶ 23:00](https://youtu.be/Z8GKkgE2840?t=1380)

【口頭】比喻：量子位元像一枚**在桌上旋轉的硬幣**，停下來時有一定機率正面、一定機率反面。測量時量子狀態會**塌縮（collapse）**成 0 或 1。

**測量到 $|x\rangle$ 的機率**：

$$
p\left(|x\rangle\right) = \left|\langle x|\psi\rangle\right|^2
$$

**例子**：[▶ 24:00](https://youtu.be/Z8GKkgE2840?t=1440) 測量上面的 $|q_0\rangle$，得到 0 的機率：

$$
\langle 0|q_0\rangle = \begin{bmatrix}1 & 0\end{bmatrix}\begin{bmatrix}\frac{1}{\sqrt{2}}\\[2pt] \frac{i}{\sqrt{2}}\end{bmatrix} = \frac{1}{\sqrt{2}}
\quad\Rightarrow\quad
p(0) = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}
$$

同理 $p(1) = \left|\frac{i}{\sqrt{2}}\right|^2 = \frac{1}{2}$。

【口頭】[▶ 25:00](https://youtu.be/Z8GKkgE2840?t=1500) 要知道量子位元裡的資訊，就準備同樣的狀態 1,000 次、測 1,000 次，大約 500 次是 0、500 次是 1。要知道確切的係數（是 $\frac{1}{\sqrt{2}}$ 還是 $\frac{i}{\sqrt{2}}$），要在**不同的基底**上測量，才能完整重建狀態。

**機率加總為 1（正規化）**：[▶ 26:00](https://youtu.be/Z8GKkgE2840?t=1560) 對狀態 $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$，測量自己一定得到自己：

$$
\langle\psi|\psi\rangle = 1 \quad\Rightarrow\quad |\alpha|^2 + |\beta|^2 = 1
$$

### 2.5 其他基底

[▶ 26:00](https://youtu.be/Z8GKkgE2840?t=1560)

- 除了 $|0\rangle$、$|1\rangle$，還有**無限多組**正交單位基底可以拿來測量。
- 依演算法或硬體選一組。測量後，狀態會塌縮到那組基底中的其中一個。

### 2.6 全域相位（Global phase）

[▶ 28:00](https://youtu.be/Z8GKkgE2840?t=1680)

- 測量 $|1\rangle$ 一定得到 1（機率 100%）。那 $i|1\rangle$ 呢？

$$
p(1) = \left|\langle1|\left(i|1\rangle\right)\right|^2 = |i|^2 = 1
$$

- $i$ 在取絕對值時**消失了**。
- → $i|1\rangle$ 和 $|1\rangle$ 在**所有物理上有意義的方面都等價**。
- 【口頭】我們沒辦法知道系統有沒有全域相位，但這**不影響**演算法的設計。

### 2.7 觀察者效應

[▶ 28:00](https://youtu.be/Z8GKkgE2840?t=1680)

- 測量後，就**確定**知道量子位元的狀態了：量子狀態**塌縮成傳統狀態**。
- 所以測量通常放在**計算的最後**，否則資訊會遺失。
- 【口頭】[▶ 29:00](https://youtu.be/Z8GKkgE2840?t=1740) 這跟傳統計算很不一樣：傳統程式可以在中間用 `assert` 檢查暫存器；量子不行，一測就塌縮了。量子錯誤更正的做法是**只檢查同位（parity），不看全部的資訊**（之後的課會講）。
- 測量多個量子位元，只會得到一串傳統 bit。

**薛丁格的貓**：[▶ 30:00](https://youtu.be/Z8GKkgE2840?t=1800) 盒子裡的貓，在打開之前處於「活著」和「死了」的疊加狀態。

### 2.8 Bloch 球

[▶ 31:00](https://youtu.be/Z8GKkgE2840?t=1860)

**一個量子位元有幾個自由變數？**

1. $\alpha$、$\beta$ 是兩個複數 → 4 個實數。
2. 去掉全域相位 → 3 個。
3. [▶ 32:00](https://youtu.be/Z8GKkgE2840?t=1920) 加上 $|\alpha|^2 + |\beta|^2 = 1$ → **剩 2 個**。

所以可以寫成：

$$
|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle
$$

- 把 $\theta$、$\phi$ 當作**球座標**：
  - $\theta$：跟 Z 軸（$|0\rangle$ 在北極）的夾角。
  - $\phi$：投影到赤道平面後，跟 X 軸的夾角。
- [▶ 33:00](https://youtu.be/Z8GKkgE2840?t=1980) **任何單一量子位元的狀態都在這顆球的表面上**，球面上每一點對應一個唯一的狀態。這顆球叫 **Bloch 球**。

---

## 3. ⭐⭐ 單一量子位元的閘

### 3.1 量子閘都是可逆的

[▶ 34:00](https://youtu.be/Z8GKkgE2840?t=2040)

【課堂討論】講者問：傳統計算裡**最簡單的可逆閘**是什麼？（XOR 有兩個輸入一個輸出，不可逆。）學生答 **NOT 閘**。講者說很接近，但還有更簡單的：**identity（什麼都不做）**。

- 傳統計算也依賴**不可逆**的閘（XOR、AND……）。
- **量子閘全部都是可逆的**，可以表示成**矩陣**，或是 **Bloch 球上的旋轉**。

### 3.2 Pauli 閘

**X 閘（NOT）**：[▶ 35:00](https://youtu.be/Z8GKkgE2840?t=2100)

$$
X = \begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix},\qquad
X|0\rangle = \begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix}\begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}0\\1\end{bmatrix} = |1\rangle
$$

- 跟傳統的 NOT 一樣：0 變 1。
- 在 Bloch 球上，是繞 X 軸**轉 180°**，從北極轉到南極。

**Y 閘、Z 閘**：[▶ 37:00](https://youtu.be/Z8GKkgE2840?t=2220)

$$
Y = \begin{bmatrix}0 & -i\\ i & 0\end{bmatrix},\qquad
Z = \begin{bmatrix}1 & 0\\ 0 & -1\end{bmatrix}
$$

**Z 閘的效果**：

- $Z|0\rangle = |0\rangle$：不變。
- [▶ 38:00](https://youtu.be/Z8GKkgE2840?t=2280) $Z|1\rangle = -|1\rangle$：多了一個負號，但那是全域相位，**物理上跟 $|1\rangle$ 分不出來**。
- 所以 $|0\rangle$、$|1\rangle$ 是 Z 的**特徵向量（eigenstate）**。用它們當基底的計算叫 **Z 基底**計算。

### 3.3 X、Y、Z 三組基底

[▶ 38:00](https://youtu.be/Z8GKkgE2840?t=2280)

| 基底 | 兩個狀態 | Bloch 球上的位置 |
|---|---|---|
| **Z 基底** | $\vert0\rangle$、$\vert1\rangle$ | 北極、南極 |
| **X 基底** | $\vert+\rangle = \dfrac{\vert0\rangle + \vert1\rangle}{\sqrt{2}}$、$\vert-\rangle = \dfrac{\vert0\rangle - \vert1\rangle}{\sqrt{2}}$ | X 軸正負兩端 |
| **Y 基底** | $\vert R\rangle = \dfrac{\vert0\rangle + i\vert1\rangle}{\sqrt{2}}$、$\vert L\rangle = \dfrac{\vert0\rangle - i\vert1\rangle}{\sqrt{2}}$ | Y 軸正負兩端 |

【口頭】[▶ 39:00](https://youtu.be/Z8GKkgE2840?t=2340) 了解這些基底，對之後理解**含參數的閘**很重要。

### 3.4 Hadamard 閘：製造疊加

[▶ 40:00](https://youtu.be/Z8GKkgE2840?t=2400)

$$
H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix}
$$

$$
H|0\rangle = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\1\end{bmatrix} = |+\rangle,\qquad
H|1\rangle = \frac{1}{\sqrt{2}}\begin{bmatrix}1\\-1\end{bmatrix} = |-\rangle
$$

【口頭】Hadamard 閘就像**讓硬幣在桌上轉起來**的小技巧：把確定的 $|0\rangle$ 變成一半 0、一半 1 的疊加。

### 3.5 含參數的閘

[▶ 41:00](https://youtu.be/Z8GKkgE2840?t=2460)

【口頭】做量子機器學習或量子化學時，要用**含參數的閘**當可訓練的參數，提供很大的設計空間。

**Phase 閘**：

$$
P(\phi) = \begin{bmatrix}1 & 0\\ 0 & e^{i\phi}\end{bmatrix},\qquad S = P\left(\frac{\pi}{2}\right),\qquad S^\dagger = S\text{ 的反閘}
$$

**U 閘（最一般的單一量子位元閘）**：[▶ 42:00](https://youtu.be/Z8GKkgE2840?t=2520) 有 **3 個參數**，能實作**所有**可能的單一量子位元閘。

投影片上的 U 閘公式是圖片、文字抽不出來。以下**不是逐字稿內容**，是依投影片引用的 Qiskit 教科書的標準定義寫的：

$$
U(\theta, \phi, \lambda) = \begin{bmatrix}\cos\frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2}\\[2pt] e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}\end{bmatrix}
$$

講者舉的兩個例子可以驗證這個定義：

- $U\left(\frac{\pi}{2}, 0, \pi\right)$：$\cos\frac{\pi}{4} = \sin\frac{\pi}{4} = \frac{1}{\sqrt{2}}$，$e^{i\pi} = -1$，得到 $\frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix} = H$ ✓。
- $U(0, 0, \lambda)$：$\cos0 = 1$、$\sin0 = 0$，得到 $\begin{bmatrix}1 & 0\\ 0 & e^{i\lambda}\end{bmatrix} = P(\lambda)$ ✓。

---

## 4. ⭐⭐ 多量子位元：狀態、閘、糾纏

### 4.1 多量子位元的狀態

[▶ 42:00](https://youtu.be/Z8GKkgE2840?t=2520)

| | 2 個傳統 bit | 2 個量子位元 |
|---|---|---|
| 可能的狀態 | 00、01、10、11，**同一時間只能是其中一個** | 可以**同時**有四個狀態的成分 |
| 描述 | 2 個 bit | 4 個複數的向量 |

$$
|\psi\rangle = a_{00}|00\rangle + a_{01}|01\rangle + a_{10}|10\rangle + a_{11}|11\rangle
$$

**正規化**：[▶ 44:00](https://youtu.be/Z8GKkgE2840?t=2640) 一樣要 $\sum|a_i|^2 = 1$。【口頭】就算有 1,000 個量子位元，$2^{1000}$ 個係數的平方也要加總為 1。

### 4.2 Kronecker 積

[▶ 44:00](https://youtu.be/Z8GKkgE2840?t=2640)

兩個量子位元 $|a\rangle = \begin{bmatrix}a_0\\a_1\end{bmatrix}$、$|b\rangle = \begin{bmatrix}b_0\\b_1\end{bmatrix}$ 的合併狀態：

$$
|ba\rangle = |b\rangle\otimes|a\rangle = \begin{bmatrix}b_0a_0\\ b_0a_1\\ b_1a_0\\ b_1a_1\end{bmatrix}
$$

- $n$ 個量子位元 → 狀態向量有 **$2^n$** 個複數。
- 【投影片補充】所以在傳統電腦上**很難模擬**。

**例子：三個量子位元都加 H**　[▶ 45:00](https://youtu.be/Z8GKkgE2840?t=2700)

$$
H|0\rangle\otimes H|0\rangle\otimes H|0\rangle = |{+}{+}{+}\rangle = \frac{1}{\sqrt{8}}\sum_{x=000}^{111}|x\rangle
$$

8 個狀態**機率都一樣**。

### 4.3 單一量子位元閘作用在多量子位元系統

[▶ 45:00](https://youtu.be/Z8GKkgE2840?t=2700)

在 $q_1$ 上加 X、$q_0$ 上加 H → 整體的矩陣是兩者的 Kronecker 積：

$$
X\otimes H = \frac{1}{\sqrt{2}}\begin{bmatrix}0 & 0 & 1 & 1\\ 0 & 0 & 1 & -1\\ 1 & 1 & 0 & 0\\ 1 & -1 & 0 & 0\end{bmatrix}
$$

只在 $q_1$ 加 X、$q_0$ 不動：[▶ 46:00](https://youtu.be/Z8GKkgE2840?t=2760) 就是 $X\otimes I$。這叫**聯合么正矩陣（joint unitary matrix）**。

### 4.4 CNOT 閘

[▶ 47:00](https://youtu.be/Z8GKkgE2840?t=2820)

**CNOT = Controlled-NOT**：

- **控制位元是 0**：目標位元**不動**。
- **控制位元是 1**：目標位元做 **NOT**。

| 控制 | 目標（前） | 目標（後） |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

→ 目標位元變成 **控制 XOR 目標**。

### 4.5 ⭐ 糾纏（Entanglement）

[▶ 48:00](https://youtu.be/Z8GKkgE2840?t=2880)

【口頭】前面的 Hadamard 製造了**疊加**，現在 CNOT 製造另一個最重要的量子原理：**糾纏**。

**一步一步**：

1. 一開始 $|q_1q_0\rangle = |00\rangle$。
2. 對 $q_0$ 加 H：$|q_1q_0\rangle = |0\rangle\otimes|+\rangle = \dfrac{1}{\sqrt{2}}\left(|00\rangle + |01\rangle\right)$。
3. [▶ 49:00](https://youtu.be/Z8GKkgE2840?t=2940) CNOT（$q_0$ 控制、$q_1$ 目標）：
   - $|00\rangle$：控制是 0 → 不變 → $|00\rangle$。
   - $|01\rangle$：控制是 1 → $q_1$ 翻轉 → $|11\rangle$。
4. 結果：

$$
\frac{1}{\sqrt{2}}\left(|00\rangle + |11\rangle\right)
$$

這是一個**糾纏態**：兩個量子位元**永遠同時是 0 或同時是 1**，測量結果一定一樣。

**鬼魅般的超距作用（spooky action at a distance）**：[▶ 50:00](https://youtu.be/Z8GKkgE2840?t=3000)

- 【口頭】把兩個糾纏的量子位元分開，一個帶到銀河另一端。我測到 0，對方一定也是 0。
- 物理學還沒解釋為什麼，但實驗上確實觀察到了。

**不能用來超光速通訊（no-communication theorem）**：[▶ 51:00](https://youtu.be/Z8GKkgE2840?t=3060)

- 測量結果是**隨機的**，事先不知道會是 0 還是 1。
- 對其中一個量子位元做任何操作，**不影響**另一個的測量統計。
- → 連 **1 個 bit** 的資訊都傳不過去。

### 4.6 相位反衝（Phase kickback）

[▶ 52:00](https://youtu.be/Z8GKkgE2840?t=3120)

**單一量子位元**：$X|-\rangle = -|-\rangle$（$|-\rangle$ 是 X 的特徵向量，特徵值 $-1$）。

**CNOT，控制 $|+\rangle$、目標 $|-\rangle$**：

$$
\mathrm{CNOT}\left(|+\rangle|-\rangle\right) = \frac{1}{\sqrt{2}}\left(|0\rangle|-\rangle + |1\rangle X|-\rangle\right) = \frac{1}{\sqrt{2}}\left(|0\rangle - |1\rangle\right)|-\rangle = |-\rangle|-\rangle
$$

- **目標沒變**（還是 $|-\rangle$），但**控制位元從 $|+\rangle$ 變成 $|-\rangle$**。
- 【口頭】[▶ 53:00](https://youtu.be/Z8GKkgE2840?t=3180) 傳統的控制位元不會被目標影響；量子裡相位會從目標「反衝」回控制位元。這是很多量子演算法的重要子程序。

**其他常用的多量子位元閘**：CZ、SWAP、CRX、CRZ。

---

## 5. 量子電路

### 5.1 量子電路的組成

[▶ 54:00](https://youtu.be/Z8GKkgE2840?t=3240)

1. **初始化**：把量子位元設成某個初始狀態（例如 $|0\rangle$）。
2. **量子閘**：H、CNOT……
3. **測量**：得到傳統 bit。
4. 有時有**傳統條件**：根據測量結果（已經塌縮成 0 或 1）決定要不要對其他量子位元加閘。

### 5.2 加法器

[▶ 55:00](https://youtu.be/Z8GKkgE2840?t=3300)

【口頭】數位邏輯課學的第一個電路通常是加法器，量子電路怎麼做？

| 位元 | 傳統 | 量子做法 |
|---|---|---|
| **和（sum）** | $A\oplus B$ | 用 **CNOT**：CNOT 會把目標變成「控制 XOR 目標」，控制位元本身不變 |
| **進位（carry）** | $A\cdot B$ | 用 **CCX（Toffoli）**：[▶ 56:00](https://youtu.be/Z8GKkgE2840?t=3360) 兩個控制位元**都是 1** 才翻轉目標 |

- 用 CNOT 做出**半加器**，加上 CCX 處理進位，就變**全加器**。
- 【口頭】這也呼應量子電路的結構：編碼、量子層、測量。

⚠️ 逐字稿缺字：[▶ 56:00](https://youtu.be/Z8GKkgE2840?t=3360) 加法器這段字幕很破碎，上表是依投影片的標題和講者提到的 CNOT（XOR）、CCX 整理的。

### 5.3 【投影片補充】Deutsch 演算法

講者沒有講這一段，投影片有：

- **問題**：有一個黑盒子計算 $f: \{0,1\}\to\{0,1\}$，算一次要 **1 天**。只想知道 $f$ 是：
  - **常數（constant）**：$f(0) = f(1)$；還是
  - **平衡（balanced）**：$f(0)\ne f(1)$。
- **傳統**：要算 $f(0)$ 和 $f(1)$，**2 天**。
- **量子**：量子計算必須可逆，所以把 $f$ 做成作用在兩個量子位元上的 $U_f$。把第二個量子位元準備成 $|-\rangle$、第一個準備成 $|+\rangle$。投影片說**能縮短時間**。

以下**不是逐字稿內容**，是依投影片的設定和 §4.6 的相位反衝推論的：

1. $U_f|x\rangle|y\rangle = |x\rangle|y\oplus f(x)\rangle$。目標是 $|-\rangle$ 時，$f(x) = 1$ 等於對 $|-\rangle$ 做 X，相位 $-1$ 反衝回第一個量子位元：$U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle$。
2. 第一個量子位元是 $|+\rangle$：

$$
U_f|+\rangle|-\rangle = \frac{1}{\sqrt{2}}\left((-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle\right)|-\rangle
$$

3. 常數 → 第一個量子位元是 $\pm|+\rangle$；平衡 → 是 $\pm|-\rangle$。
4. 在 X 基底測一次（或先加 H 再在 Z 基底測），**只跑 1 次 $U_f$（1 天）**就知道答案。

---

## 6. ⭐ NISQ 時代與編譯問題

### 6.1 什麼是 NISQ？

[▶ 57:00](https://youtu.be/Z8GKkgE2840?t=3420)

**NISQ = Noisy Intermediate-Scale Quantum（有雜訊的中等規模量子）**

| 特點 | 說明 |
|---|---|
| **有雜訊** | 量子位元對環境很敏感，閘不可靠。【口頭】錯誤率約 $10^{-3}$～$10^{-2}$ |
| **量子位元有限** | 幾十到幾百個 |
| **連接有限** | 不是任兩個量子位元都能直接互動 |

【口頭】IBM 127 個量子位元的機器上：

- CNOT 錯誤率約 $10^{-3}$～$10^{-2}$，單一量子位元閘約 $10^{-4}$～$10^{-3}$。
- [▶ 58:00](https://youtu.be/Z8GKkgE2840?t=3480) **不同量子位元的表現差很多**。傳統計算不管在筆電還是手機算，結果都一樣；量子要**仔細挑選**要用哪些量子位元。

### 6.2 演算法和硬體之間的差距

[▶ 58:00](https://youtu.be/Z8GKkgE2840?t=3480)

| | 現在的硬體 | Shor 演算法需要 |
|---|---|---|
| 錯誤率 | 從 $10^{-1}$ 進步到 $10^{-3}$，最好的離子阱約 $10^{-4}$ | 約 $10^{-5}$（**差 1 個數量級**） |
| 規模 | 從 2 個進步到約 1,000 個量子位元 | 再大 2～3 個數量級 |

- 【口頭】[▶ 59:00](https://youtu.be/Z8GKkgE2840?t=3540) 變分（variational）演算法比較近，也許 1,000 個量子位元就夠。
- [▶ 1:00:00](https://youtu.be/Z8GKkgE2840?t=3600) 量子電腦架構和系統研究者要幫忙縮小差距：
  - 不要只等硬體廠商（就像不只是等 NVIDIA 出下一代 GPU，也要優化模型）。
  - **雜訊緩解**能把有效錯誤率改善好幾倍。
  - [▶ 1:01:00](https://youtu.be/Z8GKkgE2840?t=3660) 有 1,000 個量子位元，可能只敢用 100 個（用多了結果錯太多）；要想辦法**用滿現有的量子位元**。

### 6.3 量子位元映射（Qubit mapping）

[▶ 1:01:00](https://youtu.be/Z8GKkgE2840?t=3660)

**問題**：

- 硬體有 5 個實體量子位元，只有某些對有連線（例如 P0–P1、P1–P2……共 6 條邊）。**有連線才能做兩量子位元閘**。
- [▶ 1:02:00](https://youtu.be/Z8GKkgE2840?t=3720) 寫演算法的人通常不管硬體拓撲。電路要在 Q0 和 Q3 之間做閘，但它們沒連線，怎麼辦？

**解法：插入 SWAP**　[▶ 1:03:00](https://youtu.be/Z8GKkgE2840?t=3780)

- 交換兩個實體量子位元上的資訊，讓需要互動的邏輯量子位元變成相鄰，再做原本的閘。

**怎麼決定什麼時候 SWAP、SWAP 誰？**

### 6.4 Sabre 映射

**貪婪演算法**：[▶ 1:03:00](https://youtu.be/Z8GKkgE2840?t=3780)（IBM 量子電腦預設用的）

1. 先隨便做初始映射（例如 Q0 → P0、Q1 → P1……）。
2. 能做的閘先全部做掉。
3. [▶ 1:04:00](https://youtu.be/Z8GKkgE2840?t=3840) 碰到做不了的閘（例如 Q0–Q3），列出所有可以做的 SWAP（例如 6 個候選）。
4. [▶ 1:05:00](https://youtu.be/Z8GKkgE2840?t=3900) 每個候選 SWAP 算一個**分數**：做了這個 SWAP 之後，**後面有多少閘能直接做**。
5. 選分數最好（成本最低）的 SWAP，更新映射，回到步驟 2。

**初始映射很重要**：[▶ 1:06:00](https://youtu.be/Z8GKkgE2840?t=3960) 用**可逆性**來找：

1. 從某個初始映射開始，正向跑一次，得到最終映射。
2. 把電路**反過來**（量子電路可逆），用最終映射當起點再跑一次 → 得到新的初始映射。
3. [▶ 1:07:00](https://youtu.be/Z8GKkgE2840?t=4020) 來回跑很多輪，每一輪都改善一點。

### 6.5 QuantumNAS：用演化搜尋找初始映射

[▶ 1:07:00](https://youtu.be/Z8GKkgE2840?t=4020)

```
演化搜尋引擎 → 產生初始映射候選 → 在真的量子硬體上評估 → 有雜訊的回饋 → 更新候選
```

【口頭】除了用可逆性，也可以用演化搜尋或其他機器學習方法找初始映射（[Lecture 7-8 NAS](lecture07_08_nas_note.md) 的想法）。下一講會細講 QuantumNAS。

---

## 7. 超導量子電腦的實際流程

### 7.1 從筆電到量子晶片

[▶ 1:08:00](https://youtu.be/Z8GKkgE2840?t=4080)

1. 在筆電上用高階語言（例如 Qiskit、TorchQuantum）寫電路，例如 `qc.measure(0)`。
2. 編譯，送到傳統伺服器產生**硬體控制訊號**。
3. [▶ 1:09:00](https://youtu.be/Z8GKkgE2840?t=4140) 訊號一路送進**稀釋冷凍機（dilution refrigerator）**，從 300 K 一路降到極低溫（【口頭】約 4 mK），到達量子晶片。
4. 讀出測量結果，一路傳回去，得到傳統的 0 和 1。

### 7.2 單一量子位元控制

[▶ 1:10:00](https://youtu.be/Z8GKkgE2840?t=4200)

控制訊號怎麼產生：

| 元件 | 功能 |
|---|---|
| **本地振盪器** | 載波頻率 $\omega_{LO}$ |
| **任意波形產生器（AWG）** | 產生控制形狀（包絡 $s(t)$，有時帶一個小頻率 $\omega_{AWG}$） |
| **混頻器（mixer）** | 把兩者混起來 → 頻率 $\omega_{LO}\pm\omega_{AWG}$，包絡跟 AWG 一樣 |

- 載波頻率要接近量子位元的**驅動頻率**，才能**共振**改變它的狀態。
- [▶ 1:11:00](https://youtu.be/Z8GKkgE2840?t=4260) ⭐ **旋轉的角度由脈衝曲線下的面積決定**：轉 $\pi/2$ 面積小，轉 $\pi$ 面積大。【口頭】這是超導量子位元的特性；離子阱、中性原子的控制訊號不一樣。
- [▶ 1:13:00](https://youtu.be/Z8GKkgE2840?t=4380) 脈衝形狀看起來像高斯，其實是高斯的一個變形，叫 **DRAG**。
- 微波脈衝讓量子位元在 Bloch 球上**旋轉到目標位置**。

### 7.3 兩量子位元控制

[▶ 1:14:00](https://youtu.be/Z8GKkgE2840?t=4440)

- IBM 超導機器最基本的兩量子位元控制是 **cross resonance（CR）**。
- CR 的 Hamiltonian 有很多項，但 IBM 的 CX 閘**只需要 ZX 項**，其他項在**校正**時消掉。
- 一個 CNOT 被編譯成：X 方向的旋轉、virtual Z 旋轉、兩個 CR 脈衝……

**閘層級 vs 脈衝層級**：[▶ 1:12:00](https://youtu.be/Z8GKkgE2840?t=4320) 兩個不同的抽象層。【口頭】就像寫 Python 跟寫組合語言：看組合語言很難看出原本是什麼閘，但底層硬體只能用這些基本操作。

【口頭】[▶ 1:15:00](https://youtu.be/Z8GKkgE2840?t=4500) 有興趣可以去註冊 IBM Quantum 的帳號，免費使用真的量子電腦。

---

## 8. 一頁速查表

### 8.1 單一量子位元

| 項目 | 公式 | 關鍵一步 |
|---|---|---|
| **基底** | $\vert0\rangle = \begin{bmatrix}1\\0\end{bmatrix}$、$\vert1\rangle = \begin{bmatrix}0\\1\end{bmatrix}$ | 正交單位基底 |
| **一般狀態** | $\alpha\vert0\rangle + \beta\vert1\rangle$ | $\vert\alpha\vert^2 + \vert\beta\vert^2 = 1$ |
| **測量機率** | $p(\vert x\rangle) = \vert\langle x\vert\psi\rangle\vert^2$ | 測量後塌縮 |
| **全域相位** | $e^{i\gamma}\vert\psi\rangle \equiv \vert\psi\rangle$ | 取絕對值時消失 |
| **Bloch 球** | $\cos\frac{\theta}{2}\vert0\rangle + e^{i\phi}\sin\frac{\theta}{2}\vert1\rangle$ | 4 個實數 − 全域相位 − 正規化 = 2 |

### 8.2 單一量子位元閘

| 閘 | 矩陣 | 效果 |
|---|---|---|
| **X** | $\begin{bmatrix}0 & 1\\ 1 & 0\end{bmatrix}$ | NOT：$\vert0\rangle\leftrightarrow\vert1\rangle$ |
| **Y** | $\begin{bmatrix}0 & -i\\ i & 0\end{bmatrix}$ | 繞 Y 軸轉 180° |
| **Z** | $\begin{bmatrix}1 & 0\\ 0 & -1\end{bmatrix}$ | $\vert1\rangle\to-\vert1\rangle$ |
| **H** | $\frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\ 1 & -1\end{bmatrix}$ | $\vert0\rangle\to\vert+\rangle$、$\vert1\rangle\to\vert-\rangle$（疊加） |
| **P(φ)** | $\begin{bmatrix}1 & 0\\ 0 & e^{i\phi}\end{bmatrix}$ | $S = P(\pi/2)$ |
| **U(θ,φ,λ)** | 3 個參數 | $U(\frac{\pi}{2}, 0, \pi) = H$、$U(0, 0, \lambda) = P(\lambda)$ |

### 8.3 多量子位元

| 項目 | 公式 | 關鍵一步 |
|---|---|---|
| **合併狀態** | $\vert b\rangle\otimes\vert a\rangle$ | $n$ 個量子位元 → $2^n$ 個係數 |
| **合併閘** | $A\otimes B$ | 例如 $X\otimes I$ |
| **CNOT** | 控制 1 → 目標翻轉 | 目標 = 控制 XOR 目標 |
| **Bell 態** | $H$ 再 CNOT：$\frac{1}{\sqrt{2}}(\vert00\rangle + \vert11\rangle)$ | 糾纏：兩個結果一定相同 |
| **相位反衝** | $\mathrm{CNOT}\vert+\rangle\vert-\rangle = \vert-\rangle\vert-\rangle$ | $X\vert-\rangle = -\vert-\rangle$，相位反衝到控制位元 |
| **加法器** | sum 用 CNOT，carry 用 CCX | CCX：兩個控制都是 1 才翻轉 |

### 8.4 NISQ 與硬體

| 項目 | 數字或做法 |
|---|---|
| **NISQ** | 有雜訊、幾十到幾百個量子位元、連接有限 |
| **錯誤率** | CNOT $10^{-3}$～$10^{-2}$；單量子位元 $10^{-4}$～$10^{-3}$ |
| **Shor 需要** | 約 100 萬個量子位元、錯誤率約 $10^{-5}$ |
| **Sabre** | 貪婪選 SWAP（看之後能做多少閘）；用電路可逆性來回改善初始映射 |
| **QuantumNAS** | 演化搜尋 + 真實硬體回饋找初始映射 |
| **超導控制** | 旋轉角度 = 脈衝曲線下面積；CX 用 cross resonance 的 ZX 項 |

---

## 9. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **§3.5 含參數的閘** | **[Lecture 22 量子機器學習](lecture22_quantum_machine_learning_note.md)**：用參數化量子電路當模型 |
| **§6.5 QuantumNAS** | **Lecture 22-23**：搜尋對雜訊穩健的量子電路架構 |
| **§6.1 NISQ 的雜訊** | **Lecture 23 對雜訊穩健的量子機器學習** |
| **§1.3、§2.7 量子錯誤更正** | 之後的課 |

---

## 附：本集一句話總結

> **量子位元的狀態是兩個複數的向量（可以畫在 Bloch 球上），測量時依機率塌縮成 0 或 1；量子閘都是可逆的矩陣，H 製造疊加、CNOT 製造糾纏，$n$ 個量子位元有 $2^n$ 維的空間；但現在是 NISQ 時代，量子位元有雜訊、數量少、連接有限，所以需要編譯器幫忙做量子位元映射，最後再把閘轉成實際的微波脈衝。**

---

*筆記依據 Hanrui Wang「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 21 錄影逐字稿（英文自動字幕，一般版，1:16:12）與投影片 `Lec21-Basics-of-Quantum-Computing.pdf`（82 頁）整理。*
