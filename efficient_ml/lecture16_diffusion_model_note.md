---
course: MIT 6.5940 TinyML and Efficient Deep Learning Computing（Fall 2023）
episode: 16
lecture: 16
date: 2023-11-02
video: https://youtu.be/7s2YAGoLcCA
notes: Lec16-Diffusion-Model.pdf（108 頁）
---

# 第 16 集 · Lecture 16：Diffusion Model [▶ 影片](https://youtu.be/7s2YAGoLcCA)

← 上一集：[Lecture 15 GAN、影片與點雲](lecture15_gan_video_pointcloud_note.md) ｜ [總目錄](進度.md) ｜ 下一集：[Lecture 17 分散式訓練（上）](lecture17_distributed_training_part1_note.md) →

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing，Fall 2023 — 第 16 集
> **講者**：Song Han
> **來源**：YouTube 錄影逐字稿（1:16:27，英文自動字幕，Zoom 版）＋ 投影片 `Lec16-Diffusion-Model.pdf`（108 頁）
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
- [1. ⭐⭐ DDPM：去雜訊擴散模型](#1--ddpm去雜訊擴散模型)
  - [1.1 兩個過程](#11-兩個過程)
  - [1.2 前向過程：一步跳到第 t 步](#12-前向過程一步跳到第-t-步)
  - [1.3 反向過程與損失函數](#13-反向過程與損失函數)
  - [1.4 訓練與取樣演算法](#14-訓練與取樣演算法)
- [2. ⭐ 有條件的擴散模型](#2--有條件的擴散模型)
  - [2.1 三種條件](#21-三種條件)
  - [2.2 純量條件](#22-純量條件)
  - [2.3 文字條件：cross attention](#23-文字條件cross-attention)
  - [2.4 逐像素條件：ControlNet](#24-逐像素條件controlnet)
  - [2.5 Classifier guidance](#25-classifier-guidance)
  - [2.6 ⭐ Classifier-free guidance](#26--classifier-free-guidance)
- [3. Latent Diffusion](#3-latent-diffusion)
- [4. 影像編輯](#4-影像編輯)
- [5. 個人化](#5-個人化)
- [6. ⭐⭐ 快速取樣](#6--快速取樣)
  - [6.1 生成模型的三難](#61-生成模型的三難)
  - [6.2 DDIM](#62-ddim)
  - [6.3 Progressive distillation](#63-progressive-distillation)
  - [6.4 Guided distillation](#64-guided-distillation)
- [7. ⭐ 加速技術：稀疏與量化](#7--加速技術稀疏與量化)
- [8. 一頁速查表](#8-一頁速查表)
- [9. 與後續集數的連結](#9-與後續集數的連結)

---

## 0. 這堂課要做什麼

[▶ 00:00](https://youtu.be/7s2YAGoLcCA?t=0)

| | **判別模型（discriminative）** | **生成模型（generative）** |
|---|---|---|
| 輸入 → 輸出 | 資料 → 標籤（類別、框、遮罩、情緒） | 標籤、類別、文字 → 資料（影像、文字） |
| 學的是 | 決策邊界 | **資料的機率分布** |
| 例子 | 邏輯回歸、SVM | GAN、VAE、**Diffusion** |

【口頭】[▶ 01:00](https://youtu.be/7s2YAGoLcCA?t=60) Adobe Firefly、Midjourney、DALL·E 3 等內容生成工具，背後都是這一講的技術。

| 段落 | 內容 |
|---|---|
| **I. 基礎** | DDPM、有條件擴散、Latent diffusion、影像編輯、個人化 |
| **II. 快速取樣** | DDIM、Progressive distillation、Guided distillation |
| **III. 加速技術** | 稀疏、量化 |

【口頭】[▶ 02:00](https://youtu.be/7s2YAGoLcCA?t=120) 雖然也叫「稀疏」和「量化」，但用在擴散模型上的技巧**跟前面講的很不一樣**，有領域專屬的做法。

---

## 1. ⭐⭐ DDPM：去雜訊擴散模型

DDPM = Denoising Diffusion Probabilistic Models。

### 1.1 兩個過程

[▶ 03:00](https://youtu.be/7s2YAGoLcCA?t=180)

| 過程 | 方向 | 做什麼 | 要不要學 |
|---|---|---|---|
| **前向擴散（forward）** | 資料 → 雜訊 | **一點一點加高斯雜訊**，「破壞」資料，最後變成純雜訊 | **固定**，不用學 |
| **反向去雜訊（reverse）** | 雜訊 → 資料 | **一點一點去雜訊**，「創造」資料 | **要學** |

**跟 GAN 比**：GAN 一次就從雜訊生成影像；擴散模型是**一步一步**來，每一步只學一點細節。

### 1.2 前向過程：一步跳到第 t 步

[▶ 04:00](https://youtu.be/7s2YAGoLcCA?t=240)

**相鄰兩步的關係**：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
$$

- $\beta_t$ 是**事先定好的常數**，控制擴散的速度，只跟 $t$ 有關。【口頭】隨 $t$ 線性變大，雜訊越加越多。

**問題**：[▶ 05:00](https://youtu.be/7s2YAGoLcCA?t=300) 能不能從 $x_0$ **直接**得到 $x_t$，不用一步一步算？

**推導**：令 $\alpha_t = 1 - \beta_t$。

**Step 1**：用重參數化寫成

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1},\qquad \epsilon_{t-1}\sim\mathcal{N}(0, I)
$$

**Step 2**：把 $x_{t-1} = \sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-2}$ 代進去：

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\,\epsilon_{t-2}}_{\text{變異數 }\alpha_t - \alpha_t\alpha_{t-1}} + \underbrace{\sqrt{1-\alpha_t}\,\epsilon_{t-1}}_{\text{變異數 }1-\alpha_t}
$$

**Step 3**（⭐ 關鍵一步）：[▶ 06:00](https://youtu.be/7s2YAGoLcCA?t=360) 兩個獨立的零平均高斯相加，**變異數相加**：

$$
\mathcal{N}(0, \sigma_1^2 I) + \mathcal{N}(0, \sigma_2^2 I) = \mathcal{N}\left(0, (\sigma_1^2 + \sigma_2^2) I\right)
$$

所以變異數是 $(\alpha_t - \alpha_t\alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t\alpha_{t-1}$，合成一項：

$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar{\epsilon}_{t-2}
$$

**Step 4**：[▶ 07:00](https://youtu.be/7s2YAGoLcCA?t=420) 一路重複到 $x_0$。定義 $\bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i$：

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0, I)
$$

也就是**擴散核（diffusion kernel）**：

$$
q(x_t \mid x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar{\alpha}_t}\,x_0,\ (1-\bar{\alpha}_t) I\right)
$$

- $\beta_t$ 設計成讓 $\bar{\alpha}_T \to 0$，所以 $q(x_T \mid x_0) \approx \mathcal{N}(0, I)$：最後就是純雜訊。

### 1.3 反向過程與損失函數

[▶ 09:00](https://youtu.be/7s2YAGoLcCA?t=540)

**反向過程**：$\beta_t$ 很小時，$q(x_{t-1} \mid x_t)$ 也是高斯。所以設

$$
p(x_T) = \mathcal{N}(0, I),\qquad p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\ \mu_\theta(x_t),\ \sigma_t^2 I\right)
$$

- $\mu_\theta$ 是**要學的平均值**。【口頭】這是整堂課的核心。
- 例如取 $\sigma_t^2 = \beta_t$。

**損失函數**：[▶ 10:00](https://youtu.be/7s2YAGoLcCA?t=600)

【口頭】教授說數學會講快一點，因為這門課重點在工程面。

用**證據下界（ELBO，Evidence Lower Bound）**：

$$
\mathbb{E}_{q(x_0)}\left[-\log p_\theta(x_0)\right] \le \mathbb{E}_q\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right] = \mathcal{L}
$$

省略一些代數後，拆成三項：

$$
\mathcal{L} = \mathbb{E}_q\left[\underbrace{D_{KL}\left(q(x_T\mid x_0)\,\Vert\,p(x_T)\right)}_{\mathcal{L}_T\text{：跟 }\theta\text{ 無關}} + \sum_{t>1}\underbrace{D_{KL}\left(q(x_{t-1}\mid x_t, x_0)\,\Vert\,p_\theta(x_{t-1}\mid x_t)\right)}_{\mathcal{L}_{t-1}} \underbrace{- \log p_\theta(x_0\mid x_1)}_{\mathcal{L}_0}\right]
$$

- $\mathcal{L}_T$ 跟 $\theta$ 無關，不用管。
- [▶ 11:00](https://youtu.be/7s2YAGoLcCA?t=660) $\mathcal{L}_{t-1}$ 裡的兩個分布**都是高斯**，所以 KL 散度可以化簡成平方差：

$$
\mathcal{L}_{t-1} = \lambda_t\left\Vert \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon\right) - \mu_\theta(x_t, t)\right\Vert^2 + C
$$

⭐ **關鍵一步**：[▶ 12:00](https://youtu.be/7s2YAGoLcCA?t=720) 括號裡唯一**不知道的是加進去的雜訊 $\epsilon$**。所以讓神經網路去**預測雜訊**：

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t)\right)
$$

$\epsilon_\theta$ 是一個 **U-Net**。代回去，再用擴散核把 $x_t$ 換成 $x_0$ 和 $\epsilon$：

$$
\mathcal{L}_{t-1} = \lambda_t\left\Vert \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t\right)\right\Vert^2 + C
$$

$\mathcal{L}_0$ 也是同樣的形式。[▶ 13:00](https://youtu.be/7s2YAGoLcCA?t=780) 最後**直接設 $\lambda_t = 1$**，感知品質比較好：

$$
\mathcal{L} = \mathbb{E}_{t\sim\mathcal{U}(1,T),\ x_0\sim q(x_0),\ \epsilon\sim\mathcal{N}(0,I)}\left[\left\Vert \epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\ t\right)\right\Vert^2\right]
$$

| 符號 | 意思 |
|---|---|
| $t\sim\mathcal{U}(1,T)$ | 從 1 到 $T$ 均勻抽一個時間步 |
| $x_0\sim q(x_0)$ | 抽一張真實影像 |
| $\epsilon\sim\mathcal{N}(0,I)$ | 抽一個高斯雜訊，**這就是正確答案** |
| $\epsilon_\theta(\cdot, t)$ | U-Net 預測的雜訊 |

### 1.4 訓練與取樣演算法

#### 直覺圖（教授說要記住的是這個，不是公式）

[▶ 14:00](https://youtu.be/7s2YAGoLcCA?t=840)

**訓練**：

```
x0（乾淨的貓） × √ᾱt
       +                    →  xt（有雜訊的貓）  →  U-Net  →  預測的雜訊 εθ
ε（高斯雜訊） × √(1−ᾱt)                                       ↕ MSE
                                                            真正加的雜訊 ε
```

【口頭】整堂課最獨特的地方：這是一個**逐像素的預測**。給有雜訊的影像，**預測加進去的雜訊**，再跟真正加的雜訊比較。

**Algorithm：Training**　[▶ 15:00](https://youtu.be/7s2YAGoLcCA?t=900)

```
repeat
    t  ~ Uniform({1, …, T})
    x0 ~ q(x0)
    ε  ~ N(0, I)
    xt ← √ᾱt · x0 + √(1−ᾱt) · ε
    對 ∇θ ‖ε − εθ(xt, t)‖² 做一步梯度下降
until 收斂
```

- U-Net 也吃 $t$ 當輸入，因為不同時間步的行為不同。【口頭】U-Net 有數千萬個參數。

⚠️ 逐字稿缺字：[▶ 17:00](https://youtu.be/7s2YAGoLcCA?t=1020) 這裡有學生提問，但字幕只剩「you want to predict the noise but would it be the same image」，問題和回答都聽不清楚。

**取樣**：[▶ 17:00](https://youtu.be/7s2YAGoLcCA?t=1020)

```
xt（有雜訊） → U-Net → 預測的雜訊 → 從 xt 減掉（乘上一些常數）→ 再加一點新雜訊 → xt−1（雜訊少一點）
重複 T 次 → x0（乾淨的影像）
```

**Algorithm：Sampling**　[▶ 19:00](https://youtu.be/7s2YAGoLcCA?t=1140)

```
xT ~ N(0, I)
for t = T, …, 1:
    ε ~ N(0, I)
    μ ← (1/√αt) · (xt − (1−αt)/√(1−ᾱt) · εθ(xt, t))
    xt−1 ← μ + σt · ε
return x0
```

- 【口頭】$\alpha_t$ 隨 $t$ 線性下降，$\bar{\alpha}_t$ 最後接近 0，$\sigma_t^2$ 遞增。這些都只是常數，**核心就是從有雜訊的輸入減掉預測的雜訊**。
- 【口頭】有趣的是，每一步減完雜訊**還要再加一點新的高斯雜訊**。

**計算量在哪裡？**[▶ 20:00](https://youtu.be/7s2YAGoLcCA?t=1200) 幾乎全在 **U-Net**。輸入和輸出都是高解析度、逐像素預測；其他加減都是逐像素運算，很便宜。

**結果**：擴散模型已經成為很強的生成模型，**超越 GAN**。

⚠️ 待確認：[▶ 21:00](https://youtu.be/7s2YAGoLcCA?t=1260) 教授口頭說 FID 是「量生成影像和真實影像的像素差異」，但 FID 的確切定義這兩講都沒講清楚。

---

## 2. ⭐ 有條件的擴散模型

### 2.1 三種條件

[▶ 22:00](https://youtu.be/7s2YAGoLcCA?t=1320)

| | 無條件 | 有條件 |
|---|---|---|
| 反向過程 | $p_\theta(x_{t-1}\mid x_t)$ | $p_\theta(x_{t-1}\mid x_t, \boldsymbol{c})$ |
| U-Net 輸入 | $\epsilon_\theta(x_t, t)$ | $\epsilon_\theta(x_t, t, \boldsymbol{c})$ |

**唯一的差別**：U-Net 多了一個輸入 $c$（條件）。

| 條件類型 | 例子 | 注入方式 |
|---|---|---|
| **純量條件** | 類別 ID「貓」 | 廣播相加、adaptive normalization |
| **文字條件** | 「photo of a moon gate」 | **cross attention** |
| **逐像素條件** | 分割圖、Canny 邊緣、人體姿勢 | 直接接起來、**ControlNet** |

【口頭】[▶ 23:00](https://youtu.be/7s2YAGoLcCA?t=1380) 教授說投影片上那一格寫錯了，應該是「text condition」，請助教修正。

### 2.2 純量條件

[▶ 24:00](https://youtu.be/7s2YAGoLcCA?t=1440)

**方法 1：廣播相加**

1. 類別 ID $[B]$ → 編碼 → $[B, C']$ → MLP → $[B, C]$（$C$ 跟特徵圖的通道數一樣）。
2. **廣播加**到特徵圖 $[B, C, H, W]$ 的每個位置：第 1 個值加到第 1 個通道的所有位置，第 2 個值加到第 2 個通道……

**方法 2：Adaptive normalization（改良版）**　[▶ 25:00](https://youtu.be/7s2YAGoLcCA?t=1500)

- MLP 輸出 $[B, 2C]$，切成 **scale** 和 **bias** 各 $[B, C]$。
- 特徵圖先**乘**再**加**：$\text{feature}\times(\text{scale} + 1) + \text{bias}$。
- 【投影片補充】scale 會先加 1 再乘。

### 2.3 文字條件：cross attention

[▶ 25:00](https://youtu.be/7s2YAGoLcCA?t=1500)

| | 來源 |
|---|---|
| **Q** | **影像**特徵（切成 patch，每個 patch 一個 token，同 [Lecture 14](lecture14_vision_transformer_note.md)） |
| **K、V** | **文字**條件（例如「photo of a moon gate」的每個字） |

$$
\text{Output} = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

- 【口頭】叫 **cross** attention 是因為 Q 來自影像，K、V 來自文字。
- [▶ 27:00](https://youtu.be/7s2YAGoLcCA?t=1620) 每個影像 token 跟所有文字 token 算相似度 → softmax → 對 V 加權平均 → 得到一個融合了文字資訊的影像 token。
- 這就是 Midjourney 處理文字輸入的方式。

### 2.4 逐像素條件：ControlNet

**最簡單：直接接起來**　[▶ 28:00](https://youtu.be/7s2YAGoLcCA?t=1680)

$$
\epsilon_\theta(\mathrm{concat}(x_t, c), t)
$$

把分割圖跟雜訊影像在通道維度接起來，再預測雜訊。

**ControlNet**：【口頭】一個月前剛在 ICCV 發表，用得非常廣。

- 輸入 Canny 邊緣 + 文字（例如童話風格的鹿）→ 生成的圖都照著邊緣。
- [▶ 29:00](https://youtu.be/7s2YAGoLcCA?t=1740) 輸入人體姿勢 → 生成的人都照這個姿勢。

**做法**：[▶ 30:00](https://youtu.be/7s2YAGoLcCA?t=1800)

1. **凍結**原本的網路區塊。
2. **複製一份可訓練的副本**，初始化跟原本一樣。
3. 副本前後各加一個 **zero convolution**。
4. 把多個這樣的區塊疊起來。

**Zero convolution**：

- 【口頭】名字有點誤導，其實是**初始化為 0 的 1×1 卷積**。
- 一開始是 0 → 副本這條分支**完全沒有影響**，模型行為跟原本一樣。
- 訓練後不再是 0 → 條件開始被注入。
- （跟 Flamingo 的 $\tanh(\alpha)$、$\alpha$ 初始為 0 是同一個想法，見 [Lecture 14 §5.2](lecture14_vision_transformer_note.md)。）

**接到 Stable Diffusion 上**：[▶ 31:00](https://youtu.be/7s2YAGoLcCA?t=1860)

- **降採樣（encoder）階段**：複製 Stable Diffusion 的 encoder 區塊，吃逐像素條件（姿勢、分割、邊緣），經過 zero conv。
- 文字 prompt 照樣用 cross attention 注入。
- **上採樣（decoder）階段**：不複製，直接用 zero conv 把副本的輸出加回原本的 decoder。
- 訓練過程中**主幹完全不動**。

### 2.5 Classifier guidance

[▶ 33:00](https://youtu.be/7s2YAGoLcCA?t=1980)

**目的**：用**多樣性換品質**，讓生成結果更像指定的類別。

**做法**：另外訓練一個分類器 $p_\phi(c \mid x_t)$，取樣時加上分類器的梯度。

新的取樣分布：

$$
\tilde{p}_\theta(x_t \mid c) \propto p_\theta(x_t \mid c)\,p_\phi(c \mid x_t)^{\omega}
$$

取 log，[▶ 34:00](https://youtu.be/7s2YAGoLcCA?t=2040) 乘法變加法，$\omega$ 變成前面的係數：

$$
\log\tilde{p}_\theta(x_t \mid c) = \log p_\theta(x_t \mid c) + \omega\log p_\phi(c \mid x_t) + C
$$

**取樣時**：[▶ 35:00](https://youtu.be/7s2YAGoLcCA?t=2100)

$$
\mu \leftarrow \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t, t, c)\right) + \underbrace{\omega\,\sigma_t\,\nabla_{x_t}\log p_\phi(c\mid x_t)}_{\text{分類器梯度}}
$$

$\omega$ 是**引導強度（guidance strength）**。

**結果**：

| $\omega$ | FID | 效果 |
|---|---|---|
| 1 | 33.0 | 有些看起來不像狗 |
| 10 | **12.0** | 每張都很像狗 |

$\omega$ 越大 → 品質越好，但**多樣性越低**。

**限制**：[▶ 36:00](https://youtu.be/7s2YAGoLcCA?t=2160)

1. 只能用在**類別條件**，文字、逐像素條件不好用。
2. 要**另外訓練一個分類器**。

### 2.6 ⭐ Classifier-free guidance

[▶ 36:00](https://youtu.be/7s2YAGoLcCA?t=2160)

**目的**：支援各種條件，**不用另外訓練分類器**。

**推導**：

**Step 1**：用貝氏定理把分類器換掉：

$$
p_\theta(c \mid x_t) \propto \frac{p_\theta(x_t \mid c)}{p_\theta(x_t)}
$$

分子是**有條件模型**，分母是**無條件模型**。

**Step 2**：[▶ 37:00](https://youtu.be/7s2YAGoLcCA?t=2220) 代進 classifier guidance 的定義：

$$
\tilde{p}_\theta(x_t \mid c) \propto p_\theta(x_t \mid c)\left[\frac{p_\theta(x_t \mid c)}{p_\theta(x_t)}\right]^{\omega} = \frac{p_\theta(x_t \mid c)^{\omega+1}}{p_\theta(x_t)^{\omega}}
$$

**Step 3**：取 log：

$$
\log\tilde{p}_\theta(x_t \mid c) = (\omega+1)\log p_\theta(x_t \mid c) - \omega\log p_\theta(x_t) + C
$$

**訓練**：[▶ 38:00](https://youtu.be/7s2YAGoLcCA?t=2280) 跟一般有條件訓練幾乎一樣，**唯一的改變**：

```
c ← ∅  以機率 p_uncond（隨機丟掉條件，訓練無條件版本）
```

同一個網路就同時學會有條件和無條件。

**取樣**：[▶ 38:00](https://youtu.be/7s2YAGoLcCA?t=2280)

$$
\tilde{\epsilon} \leftarrow (\omega+1)\,\epsilon_\theta(x_t, t, c) - \omega\,\epsilon_\theta(x_t, t)
$$

再用 $\tilde{\epsilon}$ 取代原本的 $\epsilon_\theta$。

⚠️ **代價**：[▶ 39:00](https://youtu.be/7s2YAGoLcCA?t=2340) 每一步要**跑兩次**網路（有條件一次、無條件一次）→ **FLOPs 變兩倍**。【口頭】後面的 guided distillation 會解決這個問題。

**結果**：$\omega = 4$ 時品質比沒有引導好很多，但多樣性也降低。

**優點**：

1. **任何條件都能用**：$c$ 可以是類別、文字（cross attention）、遮罩（ControlNet），用前面學的任何方法注入都行。
2. **不用另外訓練網路**。

---

## 3. Latent Diffusion

[▶ 41:00](https://youtu.be/7s2YAGoLcCA?t=2460)

【口頭】休息回來後，教授先總結：

> 「分類網路預測類別，偵測網路預測框，分割網路預測遮罩，**擴散模型預測雜訊**。為什麼要預測雜訊？因為這樣才能去掉雜訊。」

**問題**：高解析度影像直接在像素空間做擴散，計算量太大。

**做法**：[▶ 42:00](https://youtu.be/7s2YAGoLcCA?t=2520) 結合 **VAE（變分自編碼器）** 和擴散模型。

```
影像 x0 → VAE Encoder → latent z（解析度 1/4 ~ 1/8）→ 在 latent 空間做擴散 → VAE Decoder → 影像 x0'
```

**好處**：維度小很多 → **去雜訊比較簡單、生成比較快**。

**演算法的改變**：[▶ 43:00](https://youtu.be/7s2YAGoLcCA?t=2580)

- **訓練**：多一行 $z_0 \leftarrow E(x_0)$，之後全部把 $x$ 換成 $z$。
- **取樣**：最後多一行 $x_0 \leftarrow D(z_0)$。
- 其他**完全一樣**。

**例子**：Stable Diffusion XL（「叢林裡的太空人，冷色調」「漂浮的深海潛水員」），解析度和品質都比前面的貓狗高很多。

【Q&A】[▶ 44:00](https://youtu.be/7s2YAGoLcCA?t=2640) 學生問：怎麼編輯這些圖？教授：這就是下一段要講的。

---

## 4. 影像編輯

### 4.1 用筆觸編輯：SDEdit

[▶ 44:00](https://youtu.be/7s2YAGoLcCA?t=2640)

隨手畫一個塔、幾朵雲、底下的地面，就能變成一張真實的照片。

**做法**：

1. 筆觸圖離「真實影像的分布」有一段距離。
2. **先加一點雜訊**：把筆觸圖加噪 → 它就像是擴散過程中間某一步的樣子。
3. [▶ 45:00](https://youtu.be/7s2YAGoLcCA?t=2700) **再照常反向去雜訊**：預測雜訊、減掉，一步步做 → 分布的中心慢慢移向真實影像的分布。
4. 輸出看起來就像真的照片。

也能做影像轉換：原圖 + 「A fantasy landscape, trending on artstation」→ 奇幻風景。

### 4.2 用文字編輯：Prompt-to-Prompt

[▶ 46:00](https://youtu.be/7s2YAGoLcCA?t=2760)

【口頭】關鍵是 **cross attention map 決定了每個字的東西「畫在哪裡」**，V 決定了「畫什麼」。[▶ 47:00](https://youtu.be/7s2YAGoLcCA?t=2820)

| 編輯 | 例子 | 做法 |
|---|---|---|
| **換字** | 「一隻貓騎~~腳踏車~~**汽車**」 | **用原本的 cross attention map**，只換新的 V → 位置不變，只換東西 |
| **加字** | 「蛋糕加上**雷根糖**裝飾」 | 舊的字用原本的 map，**新的字用新的 map** |
| **調強弱** | 「大道今天很**擁擠**」→ 少一點人 | [▶ 49:00](https://youtu.be/7s2YAGoLcCA?t=2940) 把那個字對應的 attention map 通道**放大或縮小** |

---

## 5. 個人化

### 5.1 DreamBooth

[▶ 49:00](https://youtu.be/7s2YAGoLcCA?t=2940)

**問題**：想畫**我的那隻狗**（不是隨便一隻狗）在不同場景。一般的文字生成模型很難**同時保留細節又換場景**。

【口頭】[▶ 50:00](https://youtu.be/7s2YAGoLcCA?t=3000) 例子：一個很有特色的黃色鬧鐘，「3」寫在右邊。就算 prompt 寫得很詳細，生成的鬧鐘「3」也不在對的位置。

**做法**：[▶ 51:00](https://youtu.be/7s2YAGoLcCA?t=3060)

| | |
|---|---|
| **輸入** | 幾張這個物體的照片 + 類別名稱 |
| **過程** | 用這些照片**微調**文字生成模型（配上「a [V] dog」這類 prompt，[V] 是專屬識別字） |
| **輸出** | 一個能在各種場景畫出這個物體的新模型 |
| **限制** | **一個物體就要一個微調過的模型** |

### 5.2 FastComposer

[▶ 52:00](https://youtu.be/7s2YAGoLcCA?t=3120)

**現有方法的問題**（例如 Midjourney 畫不出「李飛飛騎馬」）：

1. 要**額外微調**，成本高。
2. **多人同框**很差：【口頭】「牛頓和愛因斯坦坐在公園」，兩個人的臉會**混在一起**，看起來很像彼此。
3. **過度擬合參考照**：只畫出李飛飛，沒有馬。

**FastComposer：不用微調，支援多人**　[▶ 53:00](https://youtu.be/7s2YAGoLcCA?t=3180)

**訓練**：

1. 一張有兩個人的照片 + 每個人的分割遮罩（【口頭】可以用 SAM 自動標）。
2. 文字「a woman and a man are standing together」過 text encoder。
3. [▶ 54:00](https://youtu.be/7s2YAGoLcCA?t=3240) 每個人的照片過 image encoder（例如 CLIP 的）。
4. 把「woman」這個字的 text embedding **接上**那位女性的影像 embedding，「man」也一樣 → 得到條件 embedding。
5. 用 cross attention 注入 U-Net。
6. ⭐ **Cross attention 定位**：讓「woman」這個 token 的 cross attention map **跟她的分割遮罩越接近越好**。這樣兩個人就不會混在一起。

**推論：延遲注入主體條件（delayed subject conditioning）**　[▶ 55:00](https://youtu.be/7s2YAGoLcCA?t=3300)

- **前面幾步**：只用純文字 prompt → 先確定「有人坐在公園」的構圖。
- **後面幾步**：才加入主體條件 → 細修成牛頓和愛因斯坦。

**為什麼要延遲？**[▶ 56:00](https://youtu.be/7s2YAGoLcCA?t=3360) 在**可編輯性和身份保留之間取捨**：

| 主體條件加得 | 結果 |
|---|---|
| 太早 | 像李飛飛，但**沒有馬** |
| 太晚 | 有馬，但**不像李飛飛** |
| 中間 | 兩者兼顧 |

---

## 6. ⭐⭐ 快速取樣

### 6.1 生成模型的三難

[▶ 57:00](https://youtu.be/7s2YAGoLcCA?t=3420)

【口頭】教授說生成模型要在三件事之間取捨，很難三個都有：

| 模型 | 教授說的優點 | 教授說的缺點 |
|---|---|---|
| **GAN** | 品質高、取樣快 | 難收斂、多樣性不好 |
| **擴散模型** | 品質好、收斂好 | **很慢** |
| **VAE** | 快、收斂好 | 品質差 |

⚠️ 待確認：投影片的三難圖文字抽不出來，無法確認三個軸的確切名稱。教授口頭有時說「多樣性」、有時說「收斂」。

**擴散模型的問題**：原版要約 **1000 次網路評估**。下面三個方法都在減少步數或每步的成本。

### 6.2 DDIM

DDIM = Denoising Diffusion Implicit Models。

#### 想法

[▶ 58:00](https://youtu.be/7s2YAGoLcCA?t=3480)

- DDPM 每一步只去一點點雜訊。DDIM 用**更大的步伐**，直接跳過中間的步驟。
- DDPM 是 **Markov** 的：$x_t$ 只跟 $x_{t-1}$ 有關。
- 但訓練時其實**只用到兩個性質**：
  1. 擴散核 $q(x_t\mid x_0)$。
  2. 損失函數 $\mathcal{L}$。
- **問題**：[▶ 59:00](https://youtu.be/7s2YAGoLcCA?t=3540) 能不能定義一個**非 Markov** 的前向和反向過程，擴散核和損失都一樣？這樣就能**直接用同一個訓練好的模型，不用重訓**。
- **答案**：可以。讓 $x_{t-1}$ 不只跟 $x_t$ 有關，也跟 $x_0$ 有關。

#### ⭐ DDIM 取樣：先估計 $x_0$，再用擴散核跳回去

[▶ 59:00](https://youtu.be/7s2YAGoLcCA?t=3540)

**Step 1**：從擴散核 $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$ 反解 $x_0$，用預測的雜訊代替 $\epsilon$：

$$
\bar{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

**Step 2**：再用擴散核從 $\bar{x}_0$ 走到 $x_{t-1}$：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\,\bar{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\,\epsilon_\theta(x_t, t)
$$

| | DDPM | DDIM |
|---|---|---|
| 每步 | $x_{t-1} = \mu + \sigma_t\epsilon$（**加新雜訊**，隨機） | 先估 $\bar{x}_0$，再用擴散核（**確定性**） |

#### 減少步數

[▶ 1:00:00](https://youtu.be/7s2YAGoLcCA?t=3600)

因為 Step 2 可以跳到**任何**時間步，所以取一個子序列：

$$
[1, \dots, T] \Longrightarrow [\tau_0 = 0, \dots, \tau_S = T],\quad\text{例如 } \tau = [0, 10, 20, 30, \dots, 1000]
$$

每次從 $t = \tau_s$ 直接跳到 $t' = \tau_{s-1}$：

$$
x_{t'} = \sqrt{\bar{\alpha}_{t'}}\,\bar{x}_0 + \sqrt{1-\bar{\alpha}_{t'}}\,\bar{\epsilon}
$$

**結果**：

- **CIFAR-10**：DDIM 100 步就有很好的 FID；DDPM 20 步**幾乎不能用**。
- 【口頭】步數一樣多時，DDIM 最後不見得收斂到一樣好，但**步數少時非常有優勢**。
- [▶ 1:01:00](https://youtu.be/7s2YAGoLcCA?t=3660) **CelebA**：約 20 步就大勝 DDPM。

**為什麼 DDIM 步數少也行？** DDPM 的反向過程是高斯，這只在 **$\beta_t$ 很小**時成立（每步不能走太遠）。DDIM **不依賴這個假設**。

【投影片補充】更進階的取樣器：DPM-Solver（約 10 步）、Exponential Integrator。

### 6.3 Progressive distillation

[▶ 1:01:00](https://youtu.be/7s2YAGoLcCA?t=3660)

【口頭】蒸餾（[Lecture 9](lecture09_kd_note.md)）本來是學生模仿老師的輸出。這裡是**學生用 1 步模仿老師的 2 步**。

```
階段 1：老師走 2 步  →  學生 1 步學會
階段 2：上一個學生當老師（2 步）→  新學生 1 步學會
……
每個階段步數減半
```

- **條件**：取樣器必須是**確定性的**。【口頭】這很重要，所以用 DDIM 而不是 DDPM。
- 學生跟老師是**同一個架構**。
- 結果：[▶ 1:03:00](https://youtu.be/7s2YAGoLcCA?t=3780) 少很多步就能追上老師。

### 6.4 Guided distillation

[▶ 1:03:00](https://youtu.be/7s2YAGoLcCA?t=3780)

**問題**：classifier-free guidance 每步要跑**兩次**網路。

**Stage 1：蒸餾 CFG 的輸出**

- 訓練一個學生 $\epsilon_\psi(x_t, t, c, \omega)$，**直接預測**：

$$
\epsilon_\psi(x_t, t, c, \omega) \approx (\omega+1)\,\epsilon_\theta(x_t, t, c) - \omega\,\epsilon_\theta(x_t, t, \varnothing)
$$

- 現在**只要跑一次**。$\omega$ 也是學生的輸入。

**Stage 2：再做 progressive distillation**　[▶ 1:04:00](https://youtu.be/7s2YAGoLcCA?t=3840)

- 結合兩者的好處。
- 結果：**只要 2 步**或 **4 步**去雜訊，品質就很好。

---

## 7. ⭐ 加速技術：稀疏與量化

### 7.1 空間稀疏推論（SIGE）

#### 動機

[▶ 1:05:00](https://youtu.be/7s2YAGoLcCA?t=3900)

- 編輯影像時，通常只改一小塊（例如在天空加個太陽），**只有 1.7% 的區域被編輯**。
- 但原本的 Stable Diffusion **整張圖重新生成**。
- 沒被編輯的區域，特徵圖幾乎不變。

**想法**：**快取（cache）原本的激活值，只更新被編輯的區域**。

| | 計算量（每步） | 步數 |
|---|---|---|
| 原本的 Stable Diffusion | 1855G MACs | ×40 |
| **SIGE** | **225G MACs（少 8.2 倍）** | ×40 |

⚠️ 逐字稿缺字：[▶ 1:06:00](https://youtu.be/7s2YAGoLcCA?t=3960) 字幕寫「1,800 big flops」「1.7 region」，依投影片是 **1855G MACs**、**1.7% 的區域**。

#### 做法：Sparse Incremental Generative Engine

[▶ 1:06:00](https://youtu.be/7s2YAGoLcCA?t=3960)

1. 原圖和編輯後的圖相減 → **差異遮罩**（很稀疏）。
2. [▶ 1:07:00](https://youtu.be/7s2YAGoLcCA?t=4020) **Gather**：只把有變化的區塊收集起來（疊到 batch 維度）。
3. 只對這些區塊做卷積。
4. **Scatter**：把結果放回原本的特徵圖。
5. 原本快取的特徵 + scatter 回來的更新 → 下一層的輸入。

【投影片補充】用 tiling 做卷積、自己寫 gather 和 scatter、**合併 kernel** 減少額外成本。

#### 結果（RTX 3090）

[▶ 1:07:00](https://youtu.be/7s2YAGoLcCA?t=4020)

| 任務 | 編輯比例 | MACs | 延遲 |
|---|---|---|---|
| 原本的 Stable Diffusion | — | 1855G | 369 ms |
| **Inpainting**（「草原上的一匹馬」） | 11.6% | 514G（3.6×） | **95.0 ms（3.9×）** |
| **影像編輯**（「奇幻海灘風景」） | 2.9% | 353G（5.3×） | **76.4 ms（4.8×）** |

- 【口頭】[▶ 1:08:00](https://youtu.be/7s2YAGoLcCA?t=4080) 海灘的例子，原版畫出兩個椰子，SIGE 版反而是對的，教授覺得效果更好。
- 在 **MacBook Pro（M1 Pro GPU）**上可以即時互動：畫一朵雲，馬上生成。

### 7.2 量化：Q-Diffusion

#### 問題：激活值的分布隨時間步變化

[▶ 1:09:00](https://youtu.be/7s2YAGoLcCA?t=4140)

- 【口頭】以前做分類、分割、偵測時，**沒有時間步的概念**，只有一次前向。
- 擴散模型不同時間步的激活值範圍**差很多**：
  - 相鄰的時間步，分布比較像。
  - 差很遠的時間步（例如第 1 步和第 100 步），分布很不一樣。

#### 解法 1：依時間步校正（time step-aware calibration）

[▶ 1:10:00](https://youtu.be/7s2YAGoLcCA?t=4200)

- 每隔固定的時間步，收集一次中間的輸入 → **不同時間步各有校正資料**。
- 以前偵測、分割只有**一組**校正資料（[Lecture 5-6](lecture05_06_quantization_note.md)）。
- 量化激活值時要看時間步：同樣的激活值，在不同時間步要用**不同的 scale**。

#### 解法 2：Shortcut 拆開量化（shortcut-splitting quantization）

[▶ 1:11:00](https://youtu.be/7s2YAGoLcCA?t=4260)

- U-Net 有 **skip connection**：解析度相同時，把降採樣那邊的特徵接到上採樣那邊。
- 接起來的特徵，一半來自**上採樣層自己**，一半來自**降採樣層**，兩邊的**分布差很多**。
- 用同一組 scale 和 bias 量化兩者不好 → **拆開，各自量化**：

$$
X * W = X_1 * W_1 + X_2 * W_2
$$

| | 量化粒度 |
|---|---|
| 激活值 $X_1$、$X_2$ | Tensor-wise（每個 tensor 一組） |
| 權重 $W_1$、$W_2$ | Channel-wise（每個通道一組） |

#### 結果

[▶ 1:12:00](https://youtu.be/7s2YAGoLcCA?t=4320)

Prompt：「a photograph of an astronaut riding a horse」。

| 設定 | 品質 |
|---|---|
| 全精度 | 基準 |
| Q-Diffusion **W4A32**（只量化權重） | 很好 |
| Q-Diffusion **W4A8**（權重和激活值都量化） | 還是很好 |
| 一般的線性量化 | **很差**，即使完全不量化激活值 |

⚠️ 待確認：[▶ 1:13:00](https://youtu.be/7s2YAGoLcCA?t=4380) 一般線性量化那組的位元設定，字幕寫「W4 832」，看不出是 W4A32 還是 W8A32。

---

## 8. 一頁速查表

### 8.1 DDPM

| 項目 | 公式 | 關鍵一步 |
|---|---|---|
| **前向一步** | $q(x_t\mid x_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right)$ | $\beta_t$ 事先定好 |
| **擴散核** | $x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon$，$\bar{\alpha}_t = \prod_{i\le t}(1-\beta_i)$ | 兩個獨立高斯相加，**變異數相加** |
| **預測的平均** | $\mu_\theta = \dfrac{1}{\sqrt{\alpha_t}}\left(x_t - \dfrac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\,\epsilon_\theta(x_t,t)\right)$ | 唯一的未知是 $\epsilon$ → 讓 U-Net 預測它 |
| **損失** | $\left\Vert\epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t\right)\right\Vert^2$ | ELBO 化簡 + $\lambda_t = 1$ |
| **取樣一步** | $x_{t-1} = \mu_\theta + \sigma_t\epsilon$ | 減掉預測雜訊，再加一點新雜訊 |

### 8.2 條件與引導

| 技術 | 公式或做法 | 關鍵一步 |
|---|---|---|
| **純量條件** | MLP → scale、bias，廣播乘加 | 維度對齊特徵圖通道 |
| **文字條件** | $\mathrm{softmax}(QK^\top/\sqrt{d})V$，Q 來自影像、K/V 來自文字 | cross attention |
| **ControlNet** | 凍結主幹 + 可訓練副本 + zero conv | 1×1 卷積初始為 0 → 一開始不影響 |
| **Classifier guidance** | $\mu + \omega\sigma_t\nabla_{x_t}\log p_\phi(c\mid x_t)$ | 要另外訓練分類器 |
| **CFG** | $\tilde{\epsilon} = (\omega+1)\epsilon_\theta(x_t,t,c) - \omega\epsilon_\theta(x_t,t)$ | 貝氏定理把分類器換成「有條件 / 無條件」；**FLOPs 兩倍** |
| **CFG 訓練** | 以機率 $p_{\text{uncond}}$ 把 $c$ 換成 $\varnothing$ | 同一個網路學兩種 |

### 8.3 加速

| 技術 | 公式或做法 | 效果 |
|---|---|---|
| **Latent diffusion** | 在 VAE latent（1/4~1/8 解析度）做擴散 | 生成更快 |
| **DDIM** | $\bar{x}_0 = \dfrac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$，$x_{t'} = \sqrt{\bar{\alpha}_{t'}}\bar{x}_0 + \sqrt{1-\bar{\alpha}_{t'}}\epsilon_\theta$ | 確定性、可跳步，**不用重訓** |
| **Progressive distillation** | 學生 1 步 = 老師 2 步，反覆做 | 每階段步數減半 |
| **Guided distillation** | $\epsilon_\psi(x_t,t,c,\omega)$ 直接預測 CFG 結果 | 一次前向 + 2~4 步 |
| **SIGE** | 差異遮罩 → gather → 卷積 → scatter | 1855G → 225G MACs（8.2×） |
| **Q-Diffusion** | 依時間步校正；$X*W = X_1*W_1 + X_2*W_2$ | W4A8 品質仍好 |

### 8.4 ⭐ 關鍵數字

| 數字 | 是什麼 |
|---|---|
| **~1000** | 原版 DDPM 的網路評估次數 |
| **33.0 → 12.0** | Classifier guidance，$\omega$ 從 1 到 10 的 FID |
| **2×** | CFG 每步的 FLOPs |
| **1/4 ~ 1/8** | Latent diffusion 的解析度 |
| **100 步 / 20 步** | DDIM 在 CIFAR-10 / CelebA 上就很好 |
| **2 ~ 4 步** | Guided distillation 後的步數 |
| **1.7%** | SIGE 例子的編輯區域 |
| **8.2×** | SIGE 的計算量節省 |
| **3.9× / 4.8×** | SIGE 在 inpainting / 編輯的延遲加速（RTX 3090） |
| **W4A8** | Q-Diffusion 仍保持品質的設定 |

---

## 9. 與後續集數的連結

| 本集觀念 | 後面在哪裡用到 |
|---|---|
| **第 16 講結束應用專屬章節** | **[Lecture 17 分散式訓練（上）](lecture17_distributed_training_part1_note.md)** 開始講**訓練** |
| **§5.1 DreamBooth 微調一整個模型** | **Lecture 19 裝置端訓練**、**Lecture 20 高效微調**（LoRA 等省成本的微調方法） |
| **§2.3 cross attention 注入文字** | **Lecture 20 Prompt Engineering** |

【口頭】[▶ 1:15:00](https://youtu.be/7s2YAGoLcCA?t=4500) 這一講結束了學期的第二部分（應用專屬優化：LLM、ViT、GAN、點雲、擴散模型）。下週開始新章節：**訓練**。怎麼在大叢集上訓練、怎麼在邊緣裝置上微調。還會帶大家參觀機房看 GPU 伺服器。

---

## 附：本集一句話總結

> **擴散模型就是訓練一個 U-Net 去預測「加進去的雜訊」，再一步步把雜訊減掉；條件用 cross attention、ControlNet 注入，CFG 用有條件減無條件來加強；要變快就減少步數（DDIM、蒸餾）、減少每步的計算（latent 空間、只算被編輯的區域）、再依時間步量化。**

---

## ⚠️ 待確認清單

1. [▶ 21:00](https://youtu.be/7s2YAGoLcCA?t=1260) FID 的定義：教授口頭說是「像素差異」，但課堂沒講清楚 FID 怎麼算。
2. [▶ 57:00](https://youtu.be/7s2YAGoLcCA?t=3420) 生成模型三難圖的三個軸名稱：投影片文字抽不出來，教授口頭有時說「多樣性」、有時說「收斂」。
3. [▶ 1:13:00](https://youtu.be/7s2YAGoLcCA?t=4380) Q-Diffusion 對照組（一般線性量化）的位元設定：字幕寫「W4 832」，看不出是 W4A32 還是 W8A32。

---

*筆記依據 Song Han「MIT 6.5940 EfficientML.ai, Fall 2023」Lecture 16 錄影逐字稿（英文自動字幕，1:16:27）與投影片 `Lec16-Diffusion-Model.pdf`（108 頁）整理。*
