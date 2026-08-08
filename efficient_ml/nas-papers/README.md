# NAS 閱讀清單（MIT 6.5940 Lecture 7 & 8）

出處：EfficientML.ai Fall 2023 Lecture 07/08 的參考書目（Lec07 p76–78、Lec08 p101–103）。
本目錄下的 35 份 PDF 全部由 arXiv 取得，檔名前綴即建議閱讀順序。

- ✅ = 課堂有講解內容
- ⬜ = 只出現在書目，投影片沒展開（補讀價值高）

---

## Stage 0 — 先建立地圖

| # | 論文 | arXiv | 讀它的理由 |
|---|---|---|---|
| 03 ✅ | Neural Architecture Search: A Survey — Elsken et al., JMLR 2019 | [1808.05377](https://arxiv.org/abs/1808.05377) | 建立「搜尋空間 / 搜尋策略 / 效能估計」三要素框架，後面每一篇都能掛回這張圖 |

---

## Stage 1 — 經典三支柱：RL、演化、梯度

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 01 ✅ | Neural Architecture Search with Reinforcement Learning — Zoph & Le, ICLR 2017 | [1611.01578](https://arxiv.org/abs/1611.01578) | RNN controller 逐步吐出超參數，用 accuracy 當 reward 做 policy gradient。NAS 的起點，也是「貴」的起點 |
| 02 ✅ | Learning Transferable Architectures (NASNet) — Zoph et al., CVPR 2018 | [1707.07012](https://arxiv.org/abs/1707.07012) | Normal cell / reduction cell 的 cell-level 搜尋空間；Lec07 p46–49 的空間大小 `4^B·M^{2B}·N^B` 出自這裡 |
| 04 ⬜ | Regularized Evolution (AmoebaNet) — Real et al., AAAI 2019 | [1802.01548](https://arxiv.org/abs/1802.01548) | 演化式 NAS 的原始論文，aging evolution。Lab 3 要實作的 mutation/crossover 的理論來源 |
| 06 ✅ | DARTS: Differentiable Architecture Search — Liu et al., ICLR 2019 | [1806.09055](https://arxiv.org/abs/1806.09055) | 連續鬆弛 + softmax 混合運算，把離散搜尋變成雙層最佳化。代價：所有候選路徑都要留在記憶體 |

---

## Stage 2 — 讓搜尋變便宜：權重繼承與共享

這一段是 Once-for-All 的前史，按時間順序讀最有感。

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 09 ✅ | Net2Net: Accelerating Learning via Knowledge Transfer — Chen et al., ICLR 2016 | [1511.05641](https://arxiv.org/abs/1511.05641) | Net2Wider（拆神經元、權重各半）與 Net2Deeper（插 identity 層），函數等價的擴張 |
| 10 ✅ | Efficient Architecture Search by Network Transformation — Cai et al., AAAI 2018 | [1707.04873](https://arxiv.org/abs/1707.04873) | controller 改成輸出「變寬／變深」動作而非整個架構，不用從頭訓 |
| 11 ⬜ | Path-Level Network Transformation — Cai et al., ICML 2018 | [1806.02639](https://arxiv.org/abs/1806.02639) | 把網路變換從 width/depth 推廣到路徑層級（單層 → 多分支），銜接到 ProxylessNAS |
| 12 ⬜ | Simple And Efficient Architecture Search for CNNs — Elsken et al., 2017 | [1711.04528](https://arxiv.org/abs/1711.04528) | network morphism + 爬山法，最精簡的「便宜 NAS」baseline |
| 13 ✅ | SMASH: One-Shot Model Architecture Search through HyperNetworks — Brock et al., ICLR 2018 | [1708.05344](https://arxiv.org/abs/1708.05344) | 用 hypernetwork 直接預測目標網路的權重，架構換了但 hypernetwork 留著 |
| 05 ⬜ | ENAS: Efficient NAS via Parameter Sharing — Pham et al., ICML 2018 | [1802.03268](https://arxiv.org/abs/1802.03268) | 權重共享的奠基作，把 NAS 成本壓了 1000×。**強烈建議補讀**，讀完再看 OFA 動機會清楚很多 |
| 07 ⬜ | Single Path One-Shot NAS with Uniform Sampling — Guo et al., ECCV 2020 | [1904.00420](https://arxiv.org/abs/1904.00420) | 訓練 supernet 時每步只採樣一條路徑、均勻取樣避免偏差。**OFA 的主要對照組** |

---

## Stage 3 — 硬體感知 NAS 與部署（本課主線）

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 20 ✅ | MnasNet: Platform-Aware NAS for Mobile — Tan et al., CVPR 2019 | [1807.11626](https://arxiv.org/abs/1807.11626) | 把手機實測 latency 直接放進 reward；缺點是每個架構都要真的量一次，很慢 |
| 21 ✅ | ProxylessNAS — Cai et al., ICLR 2019 | [1812.00332](https://arxiv.org/abs/1812.00332) | 拋棄所有 proxy，二值化 architecture parameter 讓記憶體從 O(N) 降到 O(1)；latency 用可微的預測模型 |
| 22 ⬜ | FBNet: Hardware-Aware Efficient ConvNet Design — Wu et al., CVPR 2019 | [1812.03443](https://arxiv.org/abs/1812.03443) | 與 ProxylessNAS 同期、獨立提出的可微硬體感知 NAS，用 Gumbel-Softmax。兩篇對照讀 |
| 24 ✅ | Once-for-All — Cai et al., ICLR 2020 | [1908.09791](https://arxiv.org/abs/1908.09791) | Progressive shrinking（resolution/kernel/depth/width 四維彈性），訓一次得 10¹⁹ 個子網路。**Lab 3 的核心** |
| 25 ✅ | MCUNet: Tiny Deep Learning on IoT Devices — Lin et al., NeurIPS 2020 | [2007.10319](https://arxiv.org/abs/2007.10319) | TinyNAS 用 FLOPs 累積分布挑搜尋空間（不用訓練），配 TinyEngine。Lec10 會再回來 |
| 27 ✅ | HAT: Hardware-Aware Transformers — Wang et al., ACL 2020 | [2005.14187](https://arxiv.org/abs/2005.14187) | OFA 思路搬到 Transformer；Lec08 p25–26「FLOPs ≠ latency」的兩張圖出自這裡 |
| 26 ✅ | NAAS: Neural Accelerator Architecture Search — Lin et al., DAC 2021 | [2105.13258](https://arxiv.org/abs/2105.13258) | 神經網路 × 編譯器 mapping × 加速器三層協同搜尋；importance-based encoding 解決 loop order 這類非數值參數 |

---

## Stage 4 — 搜尋空間本身的設計

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 08 ✅ | EfficientNet — Tan & Le, ICML 2019 | [1905.11946](https://arxiv.org/abs/1905.11946) | Compound scaling：放大模型時 depth/width/resolution 要一起動，用 grid search 找 α/β/γ |
| 23 ⬜ | Designing Network Design Spaces (RegNet) — Radosavovic et al., CVPR 2020 | [2003.13678](https://arxiv.org/abs/2003.13678) | 不搜單一模型而是逐步收斂「設計空間」。Lec07 p61 那題討論（RegNet vs MCUNet 的優缺點）的另一半 |
| 33 ✅ | Auto-DeepLab — Liu et al., CVPR 2019 | [1901.02985](https://arxiv.org/abs/1901.02985) | Network-level 的下採樣／上採樣拓撲搜尋；Lec07 p54 那張網格圖出自這裡（U-Net、Stacked Hourglass 都是其中一條路徑） |

---

## Stage 5 — Zero-shot NAS 與 pruning-at-init

這兩條線理論同源：pruning-at-init 的 saliency 指標本質上就是不訓練的架構評分函式，
所以 Song Han 把三篇 pruning 論文放進 Lec08 的書目。建議先讀 pruning 三篇再看 NAS 兩篇。

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 16 ⬜ | SNIP: Single-shot Network Pruning — Lee et al., ICLR 2019 | [1810.02340](https://arxiv.org/abs/1810.02340) | connection sensitivity：初始化後看一批資料的梯度就決定剪誰 |
| 17 ⬜ | GraSP: Picking Winning Tickets Before Training — Wang et al., ICLR 2020 | [2002.07376](https://arxiv.org/abs/2002.07376) | 改成保留「梯度流」而非單純的敏感度 |
| 18 ⬜ | SynFlow: Pruning without Any Data — Tanaka et al., NeurIPS 2020 | [2006.05467](https://arxiv.org/abs/2006.05467) | 完全不用資料，只靠 synaptic flow 守恆，避免 layer collapse |
| 14 ✅ | Zen-NAS — Lin et al., ICCV 2021 | [2102.01063](https://arxiv.org/abs/2102.01063) | Zen score = 對輸入擾動的敏感度（log 差值）+ BN 方差項做穩定性補償 |
| 15 ✅ | GradSign — Zhang & Jia, ICLR 2022 | [2110.08616](https://arxiv.org/abs/2110.08616) | 好模型的 sample-wise local minima 更密集 → 不同樣本梯度符號更一致 |
| 19 ⬜ | MAE-DET — Sun et al., ICML 2022 | [2111.13336](https://arxiv.org/abs/2111.13336) | 最大熵原則做零樣本偵測模型搜尋，把 zero-shot 從分類推到 detection |

---

## Stage 6 — 應用

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 28 ✅ | SPVNAS: Sparse Point-Voxel Convolution — Tang et al., ECCV 2020 | [2007.16100](https://arxiv.org/abs/2007.16100) | 3D point cloud 上的 OFA + 演化搜尋，3.4 → 9.1 FPS |
| 29 ✅ | PVNAS — Liu et al., TPAMI 2021 | [2204.11797](https://arxiv.org/abs/2204.11797) | SPVNAS 的期刊擴充版，Lec07 p71 的 fitness function 圖出自這裡 |
| 30 ✅ | Anycost GANs — Lin et al., CVPR 2021 | [2103.03243](https://arxiv.org/abs/2103.03243) | 小子網路即時預覽、大子網路最終算圖；生成模型的 OFA |
| 31 ✅ | Lite Pose — Wang et al., CVPR 2022 | [2205.01271](https://arxiv.org/abs/2205.01271) | 姿態估計：先重新設計搜尋空間，再做硬體感知 NAS |
| 34 ⬜ | NAS-FPN — Ghiasi et al., CVPR 2019 | [1904.07392](https://arxiv.org/abs/1904.07392) | 搜尋物件偵測的 feature pyramid 拓撲，而非 backbone |
| 32 ✅ | QuantumNAS — Wang et al., HPCA 2022 | [2107.10845](https://arxiv.org/abs/2107.10845) | Super-circuit → 搜抗噪 sub-circuit + 剪小幅度量子閘；真機 MNIST-4 準確率 47% → 85% |

---

## Stage 7 — 反思

| # | 論文 | arXiv | 重點 |
|---|---|---|---|
| 35 ⬜ | Exploring Randomly Wired Neural Networks — Xie et al., ICCV 2019 | [1904.01569](https://arxiv.org/abs/1904.01569) | 隨機圖生成的網路就能打平許多 NAS 結果 —— 對「搜尋策略有多重要 vs 搜尋空間有多重要」的直接質問。讀完整份清單後再讀，衝擊最大 |

---

## 若只讀三篇

1. **05 ENAS** — 權重共享的起點，理解為何 NAS 從「幾萬 GPU-hours」變成可負擔
2. **07 Single Path One-Shot** — OFA 的主要對照組，看清 supernet 訓練的取樣偏差問題
3. **35 Randomly Wired** — 對整個 NAS 範式的批判性觀點

## 與課程進度的對應

- Lec07（搜尋空間、搜尋策略）→ Stage 0–1、Stage 4
- Lec08（效能估計、zero-shot、硬體感知、協同搜尋、應用）→ Stage 2–3、Stage 5–6
- Lab 3（演化搜尋抽 subnet）→ 04 AmoebaNet、24 Once-for-All、28 SPVNAS
- Lec10 MCUNet / Lec11 TinyEngine → 25 MCUNet

---

*所有 PDF 由 arXiv API 查證標題後下載（比對相似度均為 1.00）。中繼資料見 `.metadata.json`。*
