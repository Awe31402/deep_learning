# Neural Architecture Search 完整筆記（MIT 6.5940 Lecture 7 & 8 + Lab 3）

> 來源：EfficientML.ai Fall 2023 Lecture 07/08 的逐字稿與投影片（`Lec07-Neural-Architecture-Search-I.pdf` 76 頁、`Lec08-Neural-Architecture-Search-II.pdf` 105 頁）、`Lab3_zh.md`、`hw3.py`。
> 論文連結指向同目錄的 `nas-papers/`，編號即該目錄的建議閱讀順序。

---

## 目錄

- [0. 為什麼需要 NAS](#0-為什麼需要-nas)
- [1. 基本運算單元與 MAC 計算](#1-基本運算單元與-mac-計算)
- [2. 經典 Building Blocks](#2-經典-building-blocks)
- [3. NAS 框架三要素](#3-nas-框架三要素)
- [4. 搜尋空間（Search Space）](#4-搜尋空間search-space)
- [5. 搜尋策略（Search Strategy）](#5-搜尋策略search-strategy)
- [6. 效能估計（Performance Estimation）](#6-效能估計performance-estimation)
- [7. Zero-shot NAS](#7-zero-shot-nas)
- [8. Hardware-aware NAS](#8-hardware-aware-nas)
- [9. Once-for-All](#9-once-for-all)
- [10. 神經網路 × 加速器協同搜尋（NAAS）](#10-神經網路--加速器協同搜尋naas)
- [11. NAS 的實際應用](#11-nas-的實際應用)
- [12. NAS 演進史](#12-nas-演進史)
- [13. Lab 3 實作對照](#13-lab-3-實作對照)
- [14. 易錯點與實作細節](#14-易錯點與實作細節)
- [附錄 A：公式速查](#附錄-a公式速查)
- [附錄 B：關鍵數字速查](#附錄-b關鍵數字速查)

---

## 0. 為什麼需要 NAS

Pruning 與 quantization 都是在**既有模型上動刀**；NAS 問的是另一個問題——**能不能一開始就設計出小而精的模型？**

效率不是單一目標，而是一組互相衝突的維度：

| 維度 | 為什麼重要 | 課堂舉的實例 |
|---|---|---|
| Latency | 自駕、語音互動等延遲敏感場景 | GPT-4 語音介面要等 5–10 秒 |
| Throughput | 資料中心批次處理 | YouTube 影片自動字幕／版權比對 |
| Accuracy | 第一公民，太差則其他免談 | — |
| Energy | 電費與碳排 | 一個 50A/250V 插座只餵得動 4 個 A100 節點、2 個 H100 節點 |
| Storage | App 上架、OTA 更新 | 175B 模型 FP16 = 350 GB，使用者流量直接爆掉 |
| Communication | 分散式訓練梯度量 ≈ 模型大小 | 400 Gbps InfiniBand ×8 卡仍可能不夠 |

硬體成本：A100 節點約 16 萬美元、H100 節點 25–30 萬美元。這些維度手工難以同時最佳化，於是有了「**用 AI 設計更好的 AI**」。

---

## 1. 基本運算單元與 MAC 計算

MAC = Multiply-Accumulate（一次乘法 + 一次累加算一個 MAC）。以下均為 batch size = 1、忽略 bias。

| 層 | 權重形狀 | #MACs |
|---|---|---|
| Linear | `(c_o, c_i)` | `c_o · c_i` |
| Convolution (2D) | `(c_o, c_i, k_h, k_w)` | `c_o · c_i · k_h · k_w · h_o · w_o` |
| Grouped Conv | `(g · c_o/g, c_i/g, k_h, k_w)` | `c_o · c_i · k_h · k_w · h_o · w_o / g` |
| Depthwise Conv | `(c, k_h, k_w)` | `c_o · k_h · k_w · h_o · w_o` |
| 1×1 Conv | `(c_o, c_i)` | `c_o · c_i · h_o · w_o` |

**維度演化的直覺**：
- Grouped conv：把通道切成 g 組，組間互不相連 → 權重與 MAC 都除以 g。
- Depthwise 是 grouped 的極端情形（`g = c_i`）→ `c_i/g = 1`，**整個維度消失**，kernel 從 4 維降為 3 維。
- 1×1 conv：`k_h = k_w = 1`，六項退化為四項。

輸出尺寸：`h_o = (h_i + 2p − k_h)/s + 1`。

---

## 2. 經典 Building Blocks

### 2.1 ResNet-50 Bottleneck（1×1 → 3×3 → 1×1）

3×3 conv 在同通道數下比 1×1 貴 **9 倍**，所以先用 1×1 把通道從 2048 壓到 512，做完 3×3 再擴回 2048。

```
#MACs:  2048×512×HW×1  +  512×512×HW×9  +  2048×512×HW×1  =  512×512×HW×17
單層 3×3 對照:           2048×2048×HW×9  =  512×512×HW×144
                                                     → 8.5× 縮減（參數同倍數）
```

### 2.2 ResNeXt（grouped convolution）

把 128→128 的 3×3 換成 32 組 grouped conv（每組 4 通道）。由於後續是線性投影，等價於 32 條獨立路徑各自 `4 → 256` 再相加，總輸出仍是 256 通道。

### 2.3 MobileNet：Depthwise Separable Block

**職責分離**：
- 3×3 depthwise conv → 只做**空間建模**，通道之間互不往來
- 1×1 conv → 只做**通道混合**，像個「信使」，但沒有空間建模能力

### 2.4 MobileNetV2：Inverted Bottleneck

Depthwise 容量不足 → 反過來先**擴張**通道（`N → 6N`）再壓回：

```
#MACs (N=160):  160×960×HW×1 + 960×HW×9 + 160×960×HW×1 = 960×HW×329
單層 3×3 對照:   160×160×HW×9 = 960×HW×240              → 比例 1 : 1.37
```

**注意這個反直覺結論**：inverted bottleneck 的 MACs 與參數其實比同規格的 dense 3×3 **多 37%**，它划算的地方在別處（模型容量／參數量的權衡），不是計算量。

#### 記憶體代價（本講最重要的反例）

| 場景 | 比較對象 | 參數 | Activation |
|---|---|---|---|
| Inference (INT8, bs=1) | ResNet-18 vs MobileNetV2-0.75 | **4.6× 少** | peak activation **1.8× 多** |
| Training (FP32, bs=16) | ResNet-50 vs MobileNetV2-1.4 | 102 MB → 24 MB（4.3×） | 只降 **1.1×** |

原因：bottleneck 把通道縮 4×，inverted bottleneck 把通道擴 6×，兩者差 **24 倍**。參數與計算量都與通道數線性相關（depthwise 只有 `3×3×C`），但 **peak activation 由最寬的那一層決定**，於是 on-chip SRAM 需求反而變大。

> **Take-away**：「參數小」≠「省記憶體」。做 TinyML 時 peak activation 常常才是真正的牆。

### 2.5 ShuffleNet：Channel Shuffle

Grouped conv 的問題是組間永遠不通訊。ShuffleNet 讓每組各取一部分通道交換位置，資訊得以跨組流動，**且沒有任何 MAC 成本**（只是資料重排，但要注意實作上的搬移開銷）。

### 2.6 Transformer：Multi-Head Self-Attention

- Q/K/V 三個投影矩陣可**合併成一個大矩陣乘法**（效率考量）
- `softmax(QKᵀ/√d) · V`，`Q: N×D` → attention map `N×N` → 輸出 `N×D`
- 除以 `√d` 是為了讓不同 hidden dim 下的 attention 分布尺度一致
- 複雜度 **O(N²D)**；FlashAttention 能讓記憶體線性，但**計算量仍是 O(N²)**
- 多頭：各頭獨立算完 concat，再過一個 output projection

---

## 3. NAS 框架三要素

```
        ┌──────────────┐   sample    ┌──────────────┐
        │ Search Space │ ──────────► │   Strategy   │
        └──────────────┘             └──────┬───────┘
               ▲                            │ architecture
               │                            ▼
               │                    ┌──────────────────┐
               └────── feedback ────│ Perf. Estimation │
                  (acc / latency)   └──────────────────┘
```

| 要素 | 選項 | 對應章節 |
|---|---|---|
| Search Space | cell-level / network-level | §4 |
| Search Strategy | grid / random / RL / gradient / evolution | §5 |
| Performance Estimation | train from scratch / inherit weight / hypernetwork / zero-shot | §6, §7 |

論文：[03-nas-survey.pdf](nas-papers/03-nas-survey.pdf)（Elsken et al., JMLR 2019）

---

## 4. 搜尋空間（Search Space）

### 4.1 Cell-level（NASNet）

搜一個 **normal cell** + 一個 **reduction cell**（後者降解析度），再重複堆疊 ×N 構成整個網路。

RNN controller 每個 block 依序做 **5 個選擇**：
1. 選第一個 hidden state（來自前一層 or 前前一層）
2. 選第二個 hidden state
3. 選第一個輸入的變換操作（3×3 conv / 5×5 sep conv / avg pool / identity / …）
4. 選第二個輸入的變換操作
5. 選合併方式（add / concat）

重複 B 次構成一個 cell。**搜尋空間大小**：

```
(2 × 2 × M × M × N)^B  =  4^B · M^{2B} · N^B
M=5, N=2, B=5  →  3.2 × 10^11
```

**缺點**（課堂特別強調）：機器設計出的 cell 連線極複雜，人難以理解，而且**依賴關係盤根錯節導致 buffer 無法提早釋放**，記憶體 footprint 比理論值大，硬體 mapping 效率差。

> 課堂 Q&A：為何每個 block 固定取兩個 operand？答：純經驗——add/multiply/concat 天然是二元運算，也可以三元但沒必要。skip connection 已包含在內（兩個 op 都選 identity 即是）。

論文：[02-nasnet.pdf](nas-papers/02-nasnet.pdf)

### 4.2 Network-level

| 維度 | 候選值（OFA/ProxylessNAS 設定） |
|---|---|
| Depth | 每 stage `[2, 3, 4]` |
| Resolution | `128, 160, 192, 224, 256` |
| Width | 各 stage 如 `[48,64,96] / [192,256,384] / [384,512,768] / [640,1024,1600] / [1280,2048,3200]` |
| Kernel size | `3, 5, 7`（每個 depthwise conv 可獨立選） |
| Topology | 下採樣／上採樣的路徑（Auto-DeepLab 的網格） |

**3×3 vs 5×5 的取捨**（課堂討論）：兩層 3×3 與一層 5×5 感受野相同，但——
- 兩層 3×3：參數較少，但**訓練時要存兩份 activation**
- 一層 5×5：參數較多，但 activation 只有一份，且**平行度可能更好**（7×7 更甚）

沒有普適答案 → 正是該交給 NAS 決定的事。

**投影片用的骨幹已經是 hybrid CNN-Transformer**（p48–50）：4 個 stage，解析度 `C1×160×160 → C2×80×80 → C3×40×40 → C4×20×20`，**前段用 MBConv（depthwise），後段換成 Attention + FFN**，每個 stage 重複 `×L1 … ×L4`。三張投影片分別把 **depth（改 L）／resolution（改 H×W）／width（改 C）** 標成可搜維度——也就是說 network-level 搜尋空間的示範對象已經不是純 CNN，而是**混合架構**。

**Topology 搜尋（Auto-DeepLab）**：每層可以選擇「繼續下採樣 / 維持解析度 / 上採樣」，構成一張網格；沿藍色節點的每條路徑都對應一個架構。U-Net、Stacked Hourglass、DeepLabV3 都只是這張網格中的特定路徑。

論文：[33-auto-deeplab.pdf](nas-papers/33-auto-deeplab.pdf)

### 4.3 設計搜尋空間本身：TinyNAS

搜尋空間太大時，「搜得好」不如「一開始就框對範圍」。MCU 場景沒有任何先驗設計經驗：

| 平台 | 記憶體 | 儲存 |
|---|---|---|
| NVIDIA V100 | 16 GB | — |
| iPhone 11 | 4 GB | 64+ GB |
| STM32H743 | 512 KB SRAM | 2 MB Flash |
| STM32F746 | 320 KB SRAM | 1 MB Flash |
| STM32F412 | 256 KB SRAM | 1 MB Flash |

**為什麼 MCU 要另外設計搜尋空間**（投影片 p54 的框架）：

| 場景 | 約束 |
|---|---|
| **Mobile AI** | Latency + Energy |
| **TinyML** | Latency + Energy + **Memory** ← **多出來的這一項才是關鍵** |

**TinyNAS 的啟發式**：`computation is cheap, memory movement is expensive`
→ 在**相同記憶體約束**下，能塞進越多 FLOPs 的設計空間，容量越大、越可能高準確率。

作法：對每個候選設計空間（width × resolution 的組合）隨機取樣一堆架構，畫 **FLOPs 的累積分布（CDF）**，比較各空間的 top-20% FLOPs。完全**不需要訓練**，只要算 MAC 公式。

實測：`width 0.5 / resolution 144` 的空間 top-20% 可達 50M FLOPs（對照組僅 32M），最終準確率 **78% vs 74%**。

> **投影片的討論題（p57）**：RegNet 與 MCUNet 兩種搜尋空間設計方法，各有什麼優缺點？
> - **RegNet**：用**已訓練**模型的統計去推導設計原則（如 width 隨深度線性成長），結論可解釋、可跨任務沿用，**但要先付出大量訓練成本**。
> - **MCUNet / TinyNAS**：**完全不訓練**，只用 FLOPs CDF 在給定記憶體約束下比較空間容量，**極快且直接對準目標硬體**，但這個 proxy（FLOPs 越大越好）只在「同一記憶體約束下」成立，換場景未必適用。
>
> 共同的立論：**「搜尋空間設計」比「搜尋策略」更關鍵。**

論文：[25-mcunet.pdf](nas-papers/25-mcunet.pdf)、對照組 [23-regnet.pdf](nas-papers/23-regnet.pdf)

---

## 5. 搜尋策略（Search Strategy）

### 5.1 Grid Search
設計空間的笛卡兒積，逐格訓練評估。暴力但衍生出重要洞見——

**Compound Scaling（EfficientNet）**：放大模型時不要只動單一維度。
```
depth: d = α^φ      width: w = β^φ      resolution: r = γ^φ
s.t.  α · β² · γ² ≈ 2        （FLOPs 對 depth 線性、對 width/resolution 平方）
```
用 grid search 找 α/β/γ，再用單一係數 φ 控制總倍率。

論文：[08-efficientnet.pdf](nas-papers/08-efficientnet.pdf)

### 5.2 Random Search
**主要用途是 sanity check**：跑幾輪之後如果 reward 沒有變好、loss 沒有下降，八成是實作有 bug。任何搜尋演算法都應該先用 random search 建立 baseline。（Lab 3 問題 5 就是做這件事。）

### 5.3 Reinforcement Learning
RNN controller 採樣架構 → 訓練子網路得 accuracy R → 因為架構本身不可微，只有取樣機率 P 可微，所以用 `R · ∇log P` 更新 controller。**每次迭代都要完整訓練一個模型，極慢**。

論文：[01-nas-rl.pdf](nas-papers/01-nas-rl.pdf)

### 5.4 Gradient Descent（DARTS 類）
把每層輸出寫成各候選操作輸出的加權平均，權重 `α/β/γ` 經 softmax 後可微，訓練完取機率最大的路徑。

**缺點**：所有候選路徑的 activation 都要留在記憶體中 → GPU 記憶體爆炸（DARTS 直接搜 ImageNet 需要 100 GB），所以早期只能先在 CIFAR 上搜、再遷移。

**讓 latency 也可微**（ProxylessNAS）：離線量測每個 block 的 latency（靜態、可預先建表），則
```
E[latency] = Σ_i E[latency_i],   E[latency_i] = α·F(conv3x3) + β·F(conv5x5) + … 
Loss = Loss_CE + λ₁‖w‖² + λ₂ · E[latency]
```
梯度可以直接回傳到架構參數。

論文：[06-darts.pdf](nas-papers/06-darts.pdf)、[21-proxylessnas.pdf](nas-papers/21-proxylessnas.pdf)

### 5.5 Evolutionary Search（Lab 3 要實作的）
1. 從設計空間隨機取樣一個 **population**
2. 用 **fitness function** 評分：`F(accuracy, efficiency)`——可以是 MACs、參數量、準確率或組合
3. **Mutation**：改深度（stage 從 3 層變 2 層）、改運算子（MB6 3×3 → MB6 5×5 → MB6 7×7）
4. **Crossover**：每一層隨機從 parent 1 或 parent 2 取值
5. 選最佳者作為下一代 parent，重複直到預算用盡

論文：[04-amoebanet.pdf](nas-papers/04-amoebanet.pdf)、[24-once-for-all.pdf](nas-papers/24-once-for-all.pdf)

---

## 6. 效能估計（Performance Estimation）

這是 Lecture 8 的主軸：**如何不用完整訓練就知道一個架構好不好**。

### 6.1 Train from Scratch（最笨的做法）
NAS-RL 在 CIFAR-10 上訓練 **12,800 個架構 / 22,400 GPU-hours**。要擴展到 ImageNet、COCO 完全不可行。

### 6.2 Inherit Weight（權重繼承）
**Net2Net** 提供兩種函數等價的擴張：
- **Net2Wider**：把一個神經元拆成兩個，輸出權重各半 → 函數不變
- **Net2Deeper**：插入一層 identity → 函數不變

於是 controller 的動作空間從「產生一個完整架構」變成「**把現有網路變寬 / 變深**」，每次都站在前一個模型的肩膀上，不用從頭訓練。

論文：[09-net2net.pdf](nas-papers/09-net2net.pdf)、[10-net-transformation.pdf](nas-papers/10-net-transformation.pdf)、[11-path-level-transformation.pdf](nas-papers/11-path-level-transformation.pdf)

### 6.3 Hypernetwork（SMASH）
用另一個網路直接**預測目標網路的權重**：

```
初始 node embedding ──GNN──► 最終 node embedding ──MLP(hypernetwork)──► 目標網路權重
```

為什麼用 GNN？因為神經網路本身就是計算圖，光有「尺寸」資訊不夠，**拓撲關係**才是關鍵，而 GNN 能萃取鄰居、鄰居的鄰居等結構特徵。

換架構時：embedding 變、預測出的權重變，但**hypernetwork 本身留著繼續訓練** → 資訊不再被丟棄。

論文：[13-smash.pdf](nas-papers/13-smash.pdf)

---

## 7. Zero-shot NAS

**完全不訓練**，只靠分析架構本身給分。

### 7.1 ZenNAS

```
1. x ~ N(0,1)，x′ = x + ε          （小擾動）
2. 所有權重初始化為 N(0,1)
3. z₁ = log‖f(x′) − f(x)‖          （好模型應對輸入擾動敏感）
4. z₂ = Σ_i log σ̄_i                 （BN 各層的方差項）
   其中 σ̄_i = Σ_j σ_{i,j} / c_out
5. Zen score  z = z₁ + z₂
```

**直覺**：如果輸入從 A 變成 B 而輸出幾乎不動，這個模型連 A/B 都分不開，一定不好。

**課堂質疑**：那對抗樣本呢？微小擾動不是「不應該」改變輸出嗎？
**回應**：第二項就是為了壓住「對任何輸入都亂跳」的不穩定模型——BN 的作用正是穩定跨樣本的特徵分布。

**重要的邏輯限制**（講者親自強調）：這類指標只能**排除**明顯的壞架構（輸入變了輸出不變 → 幾乎確定不好），**不能反推**「敏感 ⇒ 好」。

### 7.2 GradSign

**直覺**：好模型的 sample-wise local minima 更密集（不同樣本的最佳解彼此靠近）→ 在隨機初始化點上，不同樣本的梯度**符號更容易一致**。

```
for i in samples:
    for k in layers:
        g[i,k] = sign(∇_θ l(f_θ(x_i), y_i)_k)
τ_f = Σ_k |Σ_i g[i,k]|
```

`|Σ_i g[i,k]| ≤ n`，只有當所有樣本梯度同號時取到上限。分數越高越好。

論文：[14-zen-nas.pdf](nas-papers/14-zen-nas.pdf)、[15-gradsign.pdf](nas-papers/15-gradsign.pdf)
同源的 pruning-at-init：[16-snip.pdf](nas-papers/16-snip.pdf)、[17-grasp.pdf](nas-papers/17-grasp.pdf)、[18-synflow.pdf](nas-papers/18-synflow.pdf)

---

## 8. Hardware-aware NAS

### 8.1 為什麼要專用化

通用模型在所有硬體上都不夠好，因為不同平台的瓶頸完全不同：
- **GPU**：平行度極高。通道數加倍可能**完全不影響延遲**（還沒填滿 CUDA core）
- **Raspberry Pi / MCU**：平行度低、cache 小。模型剛好塞進 SRAM 與超出一點點，效能天差地別

### 8.2 MACs ≠ Latency（實測數據）

在 Titan Xp 上分別縮放兩個維度：

| 縮放方式 | FLOPs (M) | GPU Latency |
|---|---|---|
| 改 hidden dim | 159 → 326 | 52.4 → 52.5 ms（**幾乎不變**） |
| 改層數 | 159 → 326 | 52.4 → 261 ms（**線性上升**） |

同樣的 FLOPs，延遲差 5 倍。而在 Raspberry Pi ARM CPU 上結論相反——hidden dim 一漲延遲立刻上升。

另一個例子：NASNet 的 MACs 比 MobileNet **少**，實測延遲卻從 ~140 ms 漲到 ~180 ms。

> **Take-away**：FLOPs / 參數量只能作為一階近似，**必須把真實 latency 拉進 loop**。

### 8.3 ProxylessNAS：拋棄所有 Proxy

| 傳統做法（proxy） | ProxylessNAS |
|---|---|
| 在 CIFAR-10 上搜，遷移到 ImageNet | **直接在 ImageNet 上搜** |
| 縮小架構空間（低深度、重複 block） | **開放完整架構空間** |
| 只訓練 20 個 epoch 就評分 | **完整訓練** |
| 用 FLOPs / 參數量當效率指標 | **profiled latency** |

**關鍵技術**：建一個 over-parameterized network 包含所有候選路徑，訓練時**二值化架構參數，只讓一條路徑的 activation 活著** → 記憶體從 **O(N) 降到 O(1)**。

兩組參數交替更新：
- **weight parameters**：依架構參數的機率取樣一條路徑後更新
- **architecture parameters**：更新各路徑的取樣機率

最後保留機率最大的路徑，剪掉其餘。

（背景數字：NASNet 48,000 GPU-hours ≈ 單卡 5 年；DARTS 直接搜 ImageNet 要 100 GB GPU 記憶體。）

### 8.4 Latency 的取得方式

| 方法 | 優點 | 缺點 |
|---|---|---|
| 直接上機量測 | 最準 | 慢、手工、裝置會發熱降頻 |
| **Layer-wise lookup table** | 快、準確度高（實測近 y=x） | 假設各層延遲**可加** |
| **Network-wise predictor** | 通用，可處理不可加的情形 | 需要收集 latency 資料集訓練 |

**Lookup table 的可行性**：每層的選擇是有限的（kernel ∈ {3,5}、width ∈ 64..1024 step 64、resolution 有限），組合數雖多但可以**一次性預先量測**。

**可加性假設何時失效**：patch-based inference 這類排程會把數層融合在一起執行（MCUNetV2 的做法），此時各層延遲不能簡單相加。

**Predictor 的輸入特徵**：kernel size、layer num、width、embed dim、heads num、resolution → 幾層 MLP → 單一 latency 數值。HAT 在 Raspberry Pi 上的預測值幾乎落在 `y = x`。也可以先用 GNN 萃取架構特徵再預測。

### 8.5 專用化的效果

- 專用模型在 mobile 上比非專用快 **1.83×**，GPU 上差距更大
- **對角線現象**：為 GPU 搜的模型在 GPU 最快、為 CPU 搜的在 CPU 最快、為 mobile 搜的在 mobile 最快
- **GPU 專用模型的特徵**：明顯**更淺更寬**，偏好 **7×7 kernel**（紅色）而非 3×3（藍色）——平行度高的硬體寧可一層做多一點，也不要堆很多層

論文：[20-mnasnet.pdf](nas-papers/20-mnasnet.pdf)、[21-proxylessnas.pdf](nas-papers/21-proxylessnas.pdf)、[22-fbnet.pdf](nas-papers/22-fbnet.pdf)、[27-hat.pdf](nas-papers/27-hat.pdf)

---

## 9. Once-for-All

### 9.1 動機：設計成本的爆炸

```
for device in devices:               # 裝置越多，成本線性爆炸
    for episode in search_episodes:
        for iter in training_iters:
            forward_backward()       # 昂貴！
        if good_model: break
    for iter in post_search_iters:
        forward_backward()           # 又一次昂貴
```

單一裝置 40K GPU-hours → 多裝置 160K → Cloud/Mobile/Tiny 全覆蓋 1600K。而且要支援的不只是「新舊手機」，還有**同一台手機的滿電 / 省電模式**。

### 9.2 核心想法：Train Once, Get Many

訓練**一個** super network，其中包含 **10¹⁹ 個子網路**，彼此**共享權重、聯合訓練**、稀疏啟動。搜尋時只要 sample 子網路（秒級）並評估，不需要重新訓練。

```
傳統：train(一天) → 評估 → 不滿意 → 再 train(一天) → …  重複 1000 次
OFA ：train once  → sample subnet(秒) → 評估 → 不滿意 → 再 sample(秒) → …
```

### 9.3 Progressive Shrinking（四個彈性維度）

| 維度 | 作法 |
|---|---|
| **Elastic Resolution** | 每個 batch 隨機取樣輸入尺寸 |
| **Elastic Kernel Size** | 從 7×7 開始；5×5 取中心權重經 **25×25 transform matrix**、3×3 經 **9×9 matrix** 轉換而來 |
| **Elastic Depth** | 先用完整深度訓練，再逐步允許每個 unit 的**後段層被跳過**（O1→O2→O3） |
| **Elastic Width** | 依 L1/L2 norm 做 **channel sorting**，縮減時保留最重要的通道（同 channel pruning 的思路） |

訓練順序是「先全開，再漸進收縮」，每個 iteration 訓練的是**網路的不同部分**——這與訓練固定網路有本質差異。

> 課堂類比：這種稀疏啟動的精神和 **Mixture-of-Experts** 相通（訓練與推論時都只啟動一部分），但不完全等同於 ensemble。

### 9.4 效果與代價

- **Roofline 分析**：OFA 搜出的模型在 Xilinx ZU3EG/ZU9EG FPGA 上 **arithmetic intensity (Ops/Byte) 與 GOPS/s 都更高**，代表更少 memory-bound、利用率更高——而這**不是明確教它的**，是 latency feedback 端到端學出來的
- 在多種平台（Samsung S7 Edge / Google Pixel2 / LG G8 / Intel Xeon CPU / NVIDIA 1080Ti / Xilinx ZU3EG FPGA）上都優於 MobileNetV2/V3
- 同樣 72.6% top-1 準確率，延遲從約 28 ms 降到約 11 ms

**三支手機上的 ImageNet Top-1 上限**（投影片 p72，同一延遲預算下比較）：

| 平台 | **OFA** | MobileNetV3 | MobileNetV2 |
|---|---|---|---|
| Samsung S7 Edge | **76.3** | 75.2 | 73.3 |
| Google Pixel2 | **76.3** | 75.2 | 73.3 |
| LG G8 | **76.4** | 75.2 | 73.3 |

（三張圖的 x 軸各自是該裝置的實測延遲；OFA 曲線在**每一個延遲點**都在另外兩條之上，不是只有端點贏。）

**成本的誠實面**（課堂 Q&A 重點）：
- OFA 的**訓練時間是單一模型的 2–3 倍**——它省的是**推論與部署**，不是訓練
- 真正的價值有很大一部分是 **storage**：裝置上只需存一個模型，執行期再決定跑哪個子網路，而不是下載大中小好幾份

**有趣的現象**：從 OFA 抽出的子網路**不 fine-tune 就勝過同架構從頭訓練的模型**。推測原因是權重共享帶來的正則化——同一組權重既要獨立工作，又要作為其他子網路的一部分工作，因此更具泛化性。（若確定只用某一個子網路，再 fine-tune 仍可更好。）

論文：[24-once-for-all.pdf](nas-papers/24-once-for-all.pdf)、對照組 [07-spos.pdf](nas-papers/07-spos.pdf)

---

## 10. 神經網路 × 加速器協同搜尋（NAAS）

摩爾定律放緩 → 「plenty of room at the top」，靠演算法與硬體協同設計繼續拿效能。

### 10.1 三層設計空間

| 層級 | 可搜參數 |
|---|---|
| **Accelerator** | Local buffer size、Global buffer size、#PEs、Compute array size、**PE connectivity** |
| **Compiler** | **Loop order**、Loop tiling size、Dataflow |
| **Neural Network** | #Layers、#Channels、Kernel size、Bypass、量化精度 |

分成兩類：**sizing（數值型）** 容易搜；**connectivity（非數值型）** 才是難點。

### 10.2 卷積迴圈的兩種平行

```
For _R … _S … _C … _Y' … _X':          ← 時間上的 tiling（temporal mapping）
    For r … s … _k … _y' … _x' … _c:   ← loop order
        Parallel-For _m in range(16):  ← 空間上的平行（spatial parallelism / HW）
        Parallel-For _n in range(16):
            psum[b,k,y',x'] += acts[…] * wgts[…]
```

### 10.3 Importance-based Encoding（本節精華）

Loop order（`CRXKYS` vs `CXYRSK` …）與 parallel dims 這類參數，用 **index-based encoding 是錯的**——index 從 1 變成 2 不帶任何物理意義，既不是分類問題也不是回歸問題。

**解法**：改成學習每個維度的「重要性分數」（數值！）：
1. 固定各維度在編碼向量中的位置（K, C, Y', X', R, S）
2. Optimizer 依多變量常態分布取樣，賦予每個維度一個**數值 importance**
3. 依 importance **降序排序**
4. Parallel dims 取 top-k（例如硬體只能平行 2 個維度就取前 2）；loop order 則把重要的放**外層**（迭代次數少），不重要的放**內層**

如此非數值參數就轉成了可學習的數值參數，與 sizing 參數一起進同一個演化搜尋迴圈。

### 10.4 結果

**EDP Reduction：只搜 sizing vs. 完整 NAAS**（投影片 p90，兩種硬體資源預算 × 兩個模型）：

| 硬體資源 | 模型 | 只搜 Architectural Sizing | **NAAS（+ connectivity + mapping）** |
|---|---|---|---|
| **EdgeTPU** | VGG | 2.1 | **7.4** |
| **EdgeTPU** | MobileNetV2 | 1.2 | **6.0** |
| **NVDLA1024** | VGG / MobileNetV2 | 1.3 / 1.7 | **2.3 / 2.1** |

> ⚠️ NVDLA 那一列的配對是從投影片長條圖讀出的，配對順序不如 EdgeTPU 那兩組明確；**EdgeTPU 的 2.1 → 7.4 與 1.2 → 6.0 是投影片直接標示的。**
> 共同結論：**只搜硬體尺寸（sizing）遠遠不夠，把 connectivity 與 mapping 一起搜才拿得到大幅 EDP 縮減**——而這正是 §10.3 importance-based encoding 要解決的問題。

| 設定 | 效果 |
|---|---|
| 純硬體架構搜尋 vs 人工設計 | **4.4× EDP 縮減** |
| 硬體搜尋 + OFA NAS 協同 | 額外 **+2.7% 準確率** |

搜出的 dataflow 平行化 **output height × output channel**（`K-X'` parallel、array 18×10、L1 496 B、L2 107 KB），與人工設計非常不同。

論文：[26-naas.pdf](nas-papers/26-naas.pdf)

---

## 11. NAS 的實際應用

> 這是 Lecture 8 總結投影片明列的一項（`NAS applications`），投影片 p93–p100。
> 核心訊息：**Once-for-All 不是只為 ImageNet 分類服務的技巧，它是一套「訓練一次、按硬體取用」的通用範式**，已經被搬到 NLP、點雲、GAN、姿態估計、量子電路，甚至大型語言模型上。

### 11.1 NLP / Transformer — HAT

**Hardware-Aware Transformers**（Wang et al., ACL 2020）：把 OFA 的想法套到 Transformer，訓練一個 **once-for-all Transformer**，再依目標硬體抽子網路。

**WMT'14 En-Fr，跑在 Raspberry Pi 上**（對照 Evolved Transformer）：

| 指標 | Evolved Transformer | **HAT** | 倍數 |
|---|---|---|---|
| **Latency** | 20.9 s | **7.8 s** | **2.7× 快** |
| **Model Size** | 175 MB | **48 MB** | **3.7× 小** |
| FLOPs | — | — | **3.2× 少** |
| **搜尋成本** | — | — | **10,148× 低** |
| BLEU | — | — | **還高 0.1** |

> HAT 也是 §8.4 那張 latency predictor「預測值幾乎落在 $y=x$」圖的來源。

論文：[27-hat.pdf](nas-papers/27-hat.pdf)

### 11.2 點雲 — SPVNAS

**3D Neural Architecture Search with Point-Voxel Convolution**（Liu et al., TPAMI 2021）。

投影片畫出的是**本講三大工具的完整組裝**：

```
【超網路訓練】                          【演化搜尋】
Fine-Grained Channel + Elastic Depth      Sample → Latency Predictor
Stage I   (depth: 3)                        │  t=12ms → 超標，Re-sample
Stage II  (depth: 2, 3)                     │  t=10ms → 符合，Keep Arch.
Stage III (depth: 1, 2, 3)                  │
        ↓ Uniform Sampling                  ├─ Mutate
   權重共享（GPU#1 … GPU#N）                └─ Crossover
```

- **Fine-grained channel + elastic depth** = §9.3 的 elastic width / depth
- **Uniform sampling** = §12 演進史裡 Single Path One-Shot 的取樣策略
- **Latency predictor + 演化搜尋** = §8.4 + §5.5

**效果**：MinkowskiNet **3.4 FPS → SPVNAS 9.1 FPS**。

### 11.3 GAN — Anycost GAN

**Anycost GANs for Interactive Image Synthesis and Editing**（Lin et al., CVPR 2021）。

**問題**：GAN 運算極重、很慢，**在 iPad 上做互動式修圖根本卡住**。

**OFA 式解法** —— 訓練一次，得到一整條成本光譜：

| 子網路 | 用途 |
|---|---|
| **小子網路** | **低成本、快** → **拖動滑桿時的即時預覽（fast prototyping）** |
| **大子網路** | 高品質 → **放手後的最終渲染（finalization）** |

> 這是 OFA 一個很漂亮的變形：**同一個使用者、同一個 session，不同「時刻」動態切換子網路** —— 不是為不同裝置各挑一個。

### 11.4 姿態估計 — LitePose

**Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation**（Wang et al., CVPR 2022）：用 hardware-aware NAS 做**端上（on-device）**姿態估計。

### 11.5 端上部署 demo

投影片展示三個實機 demo（都是 OFA 設計的輕量模型）：

- **on-device car/person detection**
- **on-device segmentation**
- **on-device gaze estimation**（跑在 Raspberry Pi 上）

### 11.6 ⭐ 量子 AI — QuantumNAS

**QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits**（Wang et al., HPCA 2022）。

**問題**：**量子雜訊（quantum noise）是量子神經網路的瓶頸** —— 準確率從 **87% 掉到 47%**。

**做法**（和 OFA 結構完全對應）：

| OFA 的概念 | QuantumNAS 的對應 |
|---|---|
| 訓練 super **network** | 訓練 super **circuit** |
| 搜尋 sub-network | 搜尋**對雜訊穩健**的 sub-circuit |
| Channel pruning（依 L1 norm） | **剪掉 magnitude 小的 quantum gate** |

**結果**（MNIST-4，**真實量子電腦**上）：**47% → 85%**

相關工作：QuantumNAT（雜訊感知訓練）、QOC（量子晶片上訓練）、**TorchQuantum**（開源函式庫，`qmlsys.mit.edu`）

### 11.7 ⭐ 大型語言模型 — Flextron

**Flextron: Many-in-One Flexible Large Language Model**（NVIDIA）—— **OFA 的思想走到 LLM。**

**核心宣稱**：**Same model, same weights, adaptivity during inference.**（同一個模型、同一份權重，推論時才決定用多少）

**部署時「一個模型變多個」：**

| 目標平台 | MLP 用量 | Attention 用量 |
|---|---|---|
| **Mobile** | 50% | 10% |
| **Laptop** | 80% | 40% |
| **Cloud GPU** | **max** | **max** |

**把一個訓練好的 LLM 轉成 Flextron 的四步：**

```
Step 0.  拿一個 Pretrained LLM
Step 1.  Rank heads and neurons        ← 對照 §9.3 的 channel sorting（依重要性排序）
Step 2.  Group（分組）
Step 3.  Train router（訓練路由器，決定推論時啟用哪些）
```

> ⚠️ **這張投影片是後續學期更新的內容**（Flextron 是 2024 年的論文），**2023 秋季的課堂逐字稿裡沒有這一段**。但它正好把整條主線收尾：
> **2020 OFA（CNN）→ 2020 HAT（Transformer）→ 2024 Flextron（LLM）**，是同一個「train once, get many」的想法在三代模型上的延續。

### 11.8 這一節的 take-away

> **OFA 真正的貢獻不是「一個更好的 ImageNet 模型」，而是把「為每個硬體重新設計＋重新訓練」這件事，
> 從一個 $O(\text{裝置數})$ 的成本，變成 $O(1)$ 訓練 + $O(\text{裝置數})$ 次「秒級取樣」。**
>
> 一旦成本結構改變，它就能被搬到任何「需要多種尺寸、但訓練很貴」的領域 —— 這就是 §11.1–11.7 全部應用的共同邏輯。

---

## 12. NAS 演進史

每一代都是被**上一代的瓶頸**逼出來的：

| 年代 | 代表工作 | 突破 | 遺留的瓶頸 |
|---|---|---|---|
| 2017 | **NAS-RL** [01] | 首次證明機器能設計出超越人工的架構 | 22,400 GPU-hours；每個候選都要從頭訓練 |
| 2018 | **NASNet** [02] | Cell-level 搜尋空間，可跨資料集遷移 | 空間仍達 10¹¹；架構複雜難以硬體 mapping |
| 2018 | **AmoebaNet** [04] | 演化搜尋（aging evolution）取代 RL | 仍需大量訓練 |
| 2016–18 | **Net2Net / Network Transformation / Path-Level** [09][10][11] | 權重繼承：不再從零開始 | 仍是一個模型一次訓練 |
| 2018 | **SMASH** [13] / **ENAS** [05] | Hypernetwork 預測權重 / **權重共享**（成本降 1000×） | 共享權重的排名可信度存疑 |
| 2019 | **DARTS** [06] | 連續鬆弛 → 梯度直接搜架構 | 所有路徑同時在記憶體中，100 GB；只能在 CIFAR 上搜 |
| 2019 | **ProxylessNAS** [21] / **FBNet** [22] | 二值化路徑 O(N)→O(1)；**latency 可微**；直接在目標任務+硬體上搜 | 換一個硬體就要重搜一次 |
| 2019 | **MnasNet** [20] | 實測 latency 進 reward | 每個架構都要真的量測，慢 |
| 2020 | **Single Path One-Shot** [07] | 每步只採樣一條路徑、均勻取樣避免偏差 | — |
| 2020 | **Once-for-All** [24] | **訓練一次，得到 10¹⁹ 個子網路**；跨裝置零額外搜尋成本 | 訓練成本 2–3×；仍需訓練 super network |
| 2020 | **MCUNet / TinyNAS** [25] | 先用 FLOPs CDF **設計搜尋空間**，再搜模型 | — |
| 2021–22 | **Zen-NAS** [14] / **GradSign** [15] | **完全不訓練**就評分架構 | 只能排除壞架構，不能保證選出最好的 |
| 2021 | **NAAS** [26] | 神經網路 × 編譯器 × 加速器**協同搜尋** | — |
| 2024 | **Flextron**（投影片新增） | 把 OFA 搬到 **LLM**：同一份權重，推論時決定用多少 MLP / Attention | — |

**兩條主軸**貫穿始終：
1. **降低評估單一架構的成本**：從頭訓練 → 繼承權重 → 共享權重 → 完全不訓練
2. **讓回饋訊號更貼近現實**：accuracy → +FLOPs → +實測 latency → +能耗/EDP → +硬體架構本身也是變數

**反思**：[35-randomly-wired.pdf](nas-papers/35-randomly-wired.pdf) 指出隨機連線的網路就能打平許多 NAS 結果——提醒我們「**搜尋空間的設計**」可能比「搜尋策略」更關鍵，這也正是 TinyNAS 與 RegNet 的立論基礎。

---

## 13. Lab 3 實作對照

Lab 3 是把 §9（OFA）+ §5.5（演化搜尋）+ §8.4（效率預測）串起來，目標是搜出能跑在 MCU 上的 VWW 模型。

### 13.1 任務與資料集

- **VWW (Visual Wake Words)**：從 MS-COCO 二次採樣的**二元分類**任務（畫面中有沒有人）
- `build_val_data_loader(..., split)`：`split=0` 是真正的 val set（不可用於搜尋），`split=1` 是 **holdout minival**，用於產生 accuracy dataset 與 **BN 校正**——這個切分是防止搜尋過程洩漏測試集的關鍵設計

### 13.2 OFA 超網路的設計空間

```python
OFAMCUNets(
    n_classes=2,
    base_stage_width="mcunet384",
    width_mult_list=[0.5, 0.75, 1.0],   # 全域通道縮放
    ks_list=[3, 5, 7],                  # elastic kernel size
    expand_ratio_list=[3, 4, 6],        # inverted bottleneck 的擴展比例
    depth_list=[0, 1, 2],               # base_depth ~ base_depth+2
    base_depth=[1, 2, 2, 2, 2],
    fuse_blk1=True,
    se_stages=[...],                    # Squeeze-and-Excitation
)
```

對照 §9.3 的四個彈性維度：

| 課堂概念 | Lab 3 對應 |
|---|---|
| Elastic Resolution | `image_size_list = [96, 112, 128, 144, 160]` |
| Elastic Kernel Size | `ks_list = [3, 5, 7]` |
| Elastic Depth | `depth_list = [0, 1, 2]`（每 stage 在 base_depth 上加 0–2 層） |
| Elastic Width | `width_mult_list = [0.5, 0.75, 1.0]` + `expand_ratio_list` |

子網路總數 **> 10¹⁹**，用的是 **MCUNetV2** 超網路（patch-based inference + 感受野重分配 + system-NN co-design）。

**驗證 OFA 的核心宣稱**：直接抽取的子網路**不需訓練**即可達到 83.6–88.7% 準確率——這就是 §9 所說的「no retrain, direct deployment」。

### 13.3 `evaluate_sub_network` 的六個步驟

```python
ofa_network.set_active_subnet(**cfg)        # 1. 依 config 啟動子網路
subnet = ofa_network.get_active_subnet()    # 2. 連同權重擷取出來
peak_memory = count_peak_activation_size()  # 3. 效率統計
macs = count_net_flops(); params = count_parameters()
calib_bn(subnet, ...)                       # 4. BN 重新校正 ★
val_loader = build_val_data_loader(...)     # 5.
acc = validate(subnet, val_loader)          # 6.
```

★ **為什麼一定要 `calib_bn`**：super network 的 BN running mean/var 是在「所有子網路混合」的統計下累積的，一旦抽出特定子網路，其 activation 分布與超網路不同。不重新校正，準確率會顯著偏低。這是 OFA 類方法的必要步驟，投影片沒特別強調但實作上必踩。

### 13.4 問題 1：設計空間探索

**結論（`hw3.py` 的作答）**：四個維度中 **輸入解析度對準確率影響最大**。
- 解析度 96 → 160：準確率提升數個百分點，MACs 與 peak memory 約以**解析度平方**成長
- 固定解析度下，最小子網路 → 最大子網路（ks 3→7、e 3→6、d 全開、width 0.5→1.0）：準確率僅約 83.6% → 88.7%，但參數與 MACs 大上數倍

→ **ks / e 的邊際效益最低**，深度與寬度居中。在 MCU 場景「先給足解析度、再壓 ks/e」通常比「維持大模型但降解析度」划算。這也正是後面搜尋演算法會自動找到的取捨。

### 13.5 問題 2：效率預測器（對應 §8.4）

```python
class AnalyticalEfficiencyPredictor:
    def get_efficiency(self, spec):
        self.net.set_active_subnet(**spec)
        subnet = self.net.get_active_subnet()
        image_size = spec["image_size"]
        data_shape = (1, 3, image_size, image_size)
        macs = count_net_flops(subnet, data_shape)
        peak_memory = count_peak_activation_size(subnet, data_shape)
        return dict(millionMACs=macs / 1e6, KBPeakMemory=peak_memory / 1024)

    def satisfy_constraint(self, measured, target):
        for key in measured:
            if key not in target:      # 未指定的約束直接跳過 ← 見 §14 的坑
                continue
            if measured[key] > target[key]:
                return False
        return True
```

這是課堂 latency lookup table 的**解析版（analytical）**：不用實測，直接用 hook 統計 MAC 與 peak activation。對 MCU 而言，**peak activation 就是能不能塞進 SRAM 的直接判準**，比 latency 更關鍵。

### 13.6 問題 3–4：準確率預測器（對應 §6.3 的精神）

**架構編碼（one-hot）**：因為所有設計超參數都是**離散值**，用數值表示會引入不存在的序關係（同 §10.3 的 index-based encoding 問題）。

```
kernel_size: 3 → [1,0,0]   5 → [0,1,0]   7 → [0,0,1]
expand_ratio: 3 → [1,0,0]  4 → [0,1,0]   6 → [0,0,1]
被跳過的 block →  [0,0,0]   ← 深度的表達方式
```

整個子網路 = 全域參數（解析度、width mult）+ 每個 block 的 (ks, e) one-hot 拼接成一個二元向量。

**預測器本體**：3 層 MLP，每層 400 通道 + ReLU，最後 `Linear(400, 1, bias=False)`。

```python
for i in range(self.n_layers):
    layers.append(nn.Sequential(
        nn.Linear(self.arch_encoder.n_dim if i == 0 else self.hidden_size,
                  self.hidden_size),
        nn.ReLU(inplace=True),
    ))
layers.append(nn.Linear(self.hidden_size, 1, bias=False))
```

**`base_acc` 技巧**：資料集有 50,000 組 `[architecture, accuracy]`（40k 訓練 / 10k 驗證）。預測目標不是 accuracy 本身，而是 **`accuracy − base_acc`**（`base_acc` = 全體平均準確率）。因為殘差的數值範圍遠小於絕對值，訓練更容易收斂。`base_acc` 存成 `requires_grad=False` 的 Parameter，forward 時加回去。

**訓練**：L1 Loss + Adam，10 個 epoch，約 1–2 分鐘。驗收標準是「預測 vs 實測」散點圖呈**線性相關**（貼近 `y = x`）——和 §8.4 HAT latency predictor 的驗證方式完全一樣。

**為什麼值得**：MLP 推論只要幾毫秒，讓搜尋過程加速**數個數量級**。這正是 OFA 論文中 accuracy predictor 的角色。

### 13.7 問題 5–6：隨機搜尋

```python
def random_valid_sample(self, constraint):
    while True:                                    # rejection sampling
        sample = self.accuracy_predictor.arch_encoder.random_sample_arch()
        efficiency = self.efficiency_predictor.get_efficiency(sample)
        if self.efficiency_predictor.satisfy_constraint(efficiency, constraint):
            return sample, efficiency

def run_search(self, constraint, n_subnets=100):
    subnet_pool = [self.random_valid_sample(constraint)[0] for _ in range(n_subnets)]
    accs = self.accuracy_predictor.predict_acc(subnet_pool)
    best_idx = int(torch.argmax(accs).item())
    return accs[best_idx], subnet_pool[best_idx]
```

注意約束是用 **rejection sampling** 處理的：一直重抽直到滿足效率約束為止。這也意味著**若可行域極小，這個迴圈會非常慢甚至近乎卡死**（見問題 10）。

### 13.8 問題 7–8：演化搜尋（對應投影片 p71–74）

| 超參數 | 意義 | 課堂對應 |
|---|---|---|
| `population_size` | 族群大小 | population |
| `max_time_budget` | 世代數 | generations |
| `parent_ratio` | 每代保留的 top-K 比例 | select best fit |
| `mutation_ratio` | 子代中變異 vs 交叉的比例 | mutation / crossover |
| `arch_mutate_prob` | 架構（ks/e/d/width）變異機率 | mutation on depth / operator |
| `resolution_mutate_prob` | 解析度變異機率 | elastic resolution |

**主迴圈**（`hw3.py` 補完的部分）：

```python
population = sorted(population, key=lambda x: x[0], reverse=True)  # 依預測準確率降序
population = population[:parents_size]                             # 只留 top-K 當父代

for j in range(mutation_numbers):          # 一部分子代由變異產生
    par = population[np.random.randint(parents_size)][1]
    child_pool.append(self.mutate_sample(par, constraint)[0])

for j in range(population_size - mutation_numbers):   # 其餘由交叉產生
    p1 = population[np.random.randint(parents_size)][1]
    p2 = population[np.random.randint(parents_size)][1]
    child_pool.append(self.crossover_sample(p1, p2, constraint)[0])

accs = self.accuracy_predictor.predict_acc(child_pool)   # 用預測器批次評分（毫秒級）
```

**Crossover 的實作**——逐 key、逐元素隨機取自兩個 parent 之一，正是投影片 p74 的「Randomly choose one operator among two choices (from the parents) for each layer」：

```python
if not isinstance(new_sample[key], list):
    new_sample[key] = random.choice([sample1[key], sample2[key]])
else:
    for i in range(len(new_sample[key])):
        new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])
```

**調參結論（`hw3.py` 的作答）**：

| 參數 | 預設 | 調整後 | 理由 |
|---|---|---|---|
| `population_size` | 10 | 100 | 族群太小多樣性不足，很快退化成局部搜尋 |
| `max_time_budget` | 10 | 100 | 評估靠預測器（毫秒級），世代數的成本很低 |
| `parent_ratio` | 0.1 | 0.25 | 只留 1 個父代（10×0.1）會讓 crossover 退化成複製 |
| `mutation_ratio` | 0.1 | 0.5 | 變異/交叉各半，兼顧局部微調與大步跳躍 |
| `resolution_mutate_prob` | 0.1 | 0.5 | 解析度是影響最劇烈的維度，值得多探索 |

**觀察**：相同約束下，演化搜尋找到的子網路準確率普遍優於隨機搜尋，且所需候選數少得多（樣本效率高）。隨機搜尋是在 10¹⁹ 的空間裡均勻抽樣，落在高準確率區域的機率極低；演化搜尋則持續在當前最佳解附近探索。

**另一個一致的觀察**：搜尋結果幾乎總是把**解析度推到約束允許的上限**，再靠縮小 ks/e/d 把 MACs 與 peak memory 壓回約束內——與問題 1 的結論互相印證。

### 13.9 問題 9：真實世界的多重約束

| 目標 | 約束 | 準確率門檻 |
|---|---|---|
| 主要 (15 分) | 250 KB peak memory、60M MACs | ≥ 92.5% |
| 加分 (10 分) | 200 KB peak memory、30M MACs | ≥ 90% |

`hw3.py` 對兩者用**不同的 evo_params**：第二個約束可行域窄很多，隨機取樣命中率低，因此族群縮小（100→64）、世代拉長（100→150）、提高 `arch_mutate_prob`（0.1→0.2）——在狹窄可行域內做細緻的局部搜尋，比大範圍交叉更有效。

### 13.10 問題 10：可行性分析

問：設計空間中是否存在滿足下列約束的子網路？
- **A**：peak activation ≤ 256 KB **且** MACs ≤ 15M
- **B**：peak activation ≤ 64 KB

**推導方法**（`hw3.py` 的解法，比暴力搜尋更有說服力）：

`#MACs` 與 `peak activation` 在這個設計空間中都是**單調**的——兩者都隨解析度、寬度、深度、擴展比例、kernel size 遞增。因此**整個設計空間的下界**就是「最小解析度 96 + 最小 width_mult 0.5 + 最小 ks/e/d」這個子網路。

```
若下界已違反約束  →  整個設計空間都不可能滿足（不需要搜）
若下界滿足約束    →  可行，再用取樣實際找出一個來驗證
```

**B 為何不可行的物理原因**：peak activation 由 **stem / 早期 block 的輸入+輸出張量**支配（早期解析度最大），即使把 ks/e/d/width 全部取最小、解析度取 96，也降不到 64 KB。這與 §2.4 的結論同源——**peak activation 由最寬/最早的那一層決定，不是把架構縮小就能線性下降**。

---

## 14. 易錯點與實作細節

### 14.1 `millonMACs` 拼字錯誤會讓約束**靜默失效** ★

原始 notebook（`Lab3_zh.md` Cell 57）寫的是：

```python
search_constraint = dict(millonMACs=millonMACs)     # ← 少了一個 i
```

而 `get_efficiency` 回傳的 key 是 `millionMACs`。配合 `satisfy_constraint` 的邏輯：

```python
for key in measured:
    if key not in target:
        continue          # ← 找不到對應約束就直接跳過，視為通過
```

結果是 **MACs 約束完全不生效**，搜尋會退化成「無約束搜尋」，還不會報任何錯。`hw3.py` 已修正為 `millionMACs`。

> 這類「拼錯 key → 約束靜默失效」是設定字典型 API 的通病。若要防禦，可在 `satisfy_constraint` 開頭檢查 `target` 的每個 key 是否都出現在 `measured` 中，否則拋錯。

### 14.2 BN 重新校正不可省略
見 §13.3。抽出子網路後不做 `calib_bn`，準確率會明顯偏低，且這種錯誤不會有任何例外訊息。

### 14.3 搜尋不可以用真正的 val set
`split=1` 的 holdout minival 用於產生 accuracy dataset 與 BN 校正，`split=0` 才是最終評估。混用等同於在測試集上做架構選擇。

### 14.4 Rejection sampling 在窄可行域會卡死
`random_valid_sample` 是 `while True` 無限迴圈。若約束嚴到隨機取樣幾乎不可能命中（例如問題 10 的 B），程式不會報錯而是**永遠跑不完**。`hw3.py` 的 `find_subnet_under_constraint` 因此加上 `n_trials` 上限，回傳 `(None, None)` 表示未命中。

### 14.5 `population` 的累積行為
主迴圈中 `population` 先被截斷成 top-K 父代，再把新產生的 `population_size` 個子代 append 進去，所以下一輪排序時的族群大小是 `parents_size + population_size`。這是刻意的（父代參與下一輪競爭 = elitism），不是 bug。

### 14.6 效率統計的單位
`get_efficiency` 回傳的是 `millionMACs`（百萬）與 `KBPeakMemory`（KB），與 `count_net_flops` / `count_peak_activation_size` 的原始單位（次數 / bytes）差了 `1e6` 與 `1024`。約束值也要用同樣單位。

---

## 附錄 A：公式速查

```
# MACs（batch=1，忽略 bias）
Linear              c_o · c_i
Conv2D              c_o · c_i · k_h · k_w · h_o · w_o
Grouped Conv        c_o · c_i · k_h · k_w · h_o · w_o / g
Depthwise Conv      c_o · k_h · k_w · h_o · w_o
1×1 Conv            c_o · c_i · h_o · w_o

# 輸出尺寸
h_o = (h_i + 2p − k_h) / s + 1

# NASNet cell-level 搜尋空間
(2 · 2 · M · M · N)^B = 4^B · M^{2B} · N^B

# Compound scaling (EfficientNet)
d = α^φ,  w = β^φ,  r = γ^φ,   s.t. α · β² · γ² ≈ 2

# ProxylessNAS 可微 latency
E[latency] = Σ_i E[latency_i]
Loss = Loss_CE + λ₁‖w‖² + λ₂ · E[latency]

# Zen score
z = log‖f(x+ε) − f(x)‖ + Σ_i log σ̄_i

# GradSign
τ_f = Σ_k |Σ_i sign(∇_θ l(f_θ(x_i), y_i))_k|          （≤ n，全同號時取到）

# Attention
複雜度 O(N²D)；scale = 1/√d
```

---

## 附錄 B：關鍵數字速查

| 項目 | 數字 |
|---|---|
| ResNet bottleneck 相對單層 3×3 | 8.5× 縮減（MACs 與參數） |
| MobileNetV2 inverted bottleneck vs 單層 3×3 | 1 : 1.37（**更貴**） |
| ResNet-18 vs MobileNetV2-0.75（推論） | 參數 4.6× 少，peak activation 1.8× 多 |
| ResNet-50 vs MobileNetV2-1.4（訓練 bs=16） | 參數 102→24 MB，activation 只降 1.1× |
| NASNet cell 搜尋空間（M=5,N=2,B=5） | 3.2 × 10¹¹ |
| NAS-RL on CIFAR-10 | 12,800 個模型 / 22,400 GPU-hours |
| NASNet 搜尋成本 | 48,000 GPU-hours ≈ 單卡 5 年 |
| DARTS 直接搜 ImageNet | 需 100 GB GPU 記憶體 |
| ProxylessNAS 記憶體 | O(N) → O(1) |
| Titan Xp：改 hidden dim | FLOPs 159→326M，latency 52.4→52.5 ms |
| Titan Xp：改層數 | FLOPs 159→326M，latency 52.4→261 ms |
| 專用化加速（mobile） | 1.83× |
| OFA 子網路數 | 10¹⁹ |
| OFA 訓練成本 | 單一模型的 2–3 倍 |
| OFA 同 72.6% top-1 的延遲 | ~28 ms → ~11 ms |
| TinyNAS 設計空間比較 | 78% vs 74%（top-20% FLOPs 50M vs 32M） |
| MCU 規格 | STM32F746：320 KB SRAM / 1 MB Flash |
| NAAS vs 人工設計 | 4.4× EDP 縮減，+2.7% 準確率 |
| HAT (WMT'14 En-Fr, Raspberry Pi) | 20.9 → 7.8 s（2.7×）、175 → 48 MB（3.7×）、3.2× 少 FLOPs、搜尋成本 10,148× 低、BLEU +0.1 |
| SPVNAS | MinkowskiNet 3.4 → 9.1 FPS |
| QuantumNAS (MNIST-4, 真實量子電腦) | 雜訊導致 87% → 47%；QuantumNAS 救回 **85%** |
| OFA vs MobileNetV3 vs V2（Top-1 上限） | S7 Edge / Pixel2：76.3 vs 75.2 vs 73.3；LG G8：76.4 vs 75.2 vs 73.3 |
| NAAS EDP Reduction（EdgeTPU） | VGG 2.1 → **7.4**；MobileNetV2 1.2 → **6.0**（只搜 sizing → 完整 NAAS） |
| Flextron（LLM）部署配置 | Mobile：MLP 50% / Attn 10%；Laptop：80% / 40%；Cloud GPU：max / max |
| Lab 3 子網路直接抽取準確率 | 83.6–88.7%（無需訓練） |
| Lab 3 accuracy dataset | 50,000 組（40k 訓練 / 10k 驗證） |

---

## 延伸閱讀

完整論文清單與建議順序見 [`nas-papers/README.md`](nas-papers/README.md)（35 篇，分 8 個階段）。

若只補讀三篇：
1. [05-enas.pdf](nas-papers/05-enas.pdf) — 權重共享的起點，理解 NAS 如何從「幾萬 GPU-hours」變成可負擔
2. [07-spos.pdf](nas-papers/07-spos.pdf) — OFA 的主要對照組，看清 supernet 訓練的取樣偏差問題
3. [35-randomly-wired.pdf](nas-papers/35-randomly-wired.pdf) — 對整個 NAS 範式的批判性觀點

**下一講**：Lecture 9 Knowledge Distillation。
