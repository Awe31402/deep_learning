# EfficientML.ai 第十一講完整筆記：TinyEngine 與平行運算

> **課程**：MIT 6.5940 TinyML and Efficient Deep Learning Computing, Fall 2023 — Lecture 11
> **講者**：Song Han
> **來源**：
>   - 課堂逐字稿：[Lecture 11 錄影](https://youtu.be/HGsvWHqU29Y)（1:16:05）
>   - 官方投影片：`Lec11-TinyEngine.pdf`（79 頁）
>
> **標記說明**（同 [Lecture 9 筆記](lecture09_kd_note.md) 的慣例）
> - `【口頭】` = 教授課堂口頭補充、投影片沒有的內容
> - `【Q&A】` = 課堂上教授反問全班／學生提問
> - `【投影片新增】` = 投影片為後續學期更新版本，2023 課堂逐字稿沒有這段
> - 未標記者 = 投影片本身的內容

---

## 目錄

- [0. 這一講在整個課程的位置](#0-這一講在整個課程的位置)
- [1. Edge AI：資源到底有多緊](#1-edge-ai資源到底有多緊)
- [2. ⭐⭐ 迴圈最佳化](#2--迴圈最佳化)
- [3. ⭐ SIMD：一道指令、多筆資料](#3--simd一道指令多筆資料)
- [4. Multithreading：多執行緒](#4-multithreading多執行緒)
- [5. ⭐ CUDA 與 Tensor Core](#5--cuda-與-tensor-core)
- [6. ⭐⭐ 推論最佳化](#6--推論最佳化)
- [7. TinyEngine、TinyChatEngine 與 Lab 4/5](#7-tinyenginetinychatengine-與-lab-45)
- [8. 一頁速查表](#8-一頁速查表)
- [9. 與其他課程的連結](#9-與其他課程的連結)

---

## 0. 這一講在整個課程的位置

### 0.1 ⭐⭐ 本講與前十講的根本差別

【口頭】教授開場就把這件事講得非常清楚，這是理解整章的鑰匙：

> **前面學的（剪枝、量化、NAS、蒸餾）都在「改演算法」——跑的神經網路本身變了。**
> **這一講學的技巧是 functional-preserving（功能保持）、數學上完全等價的。**
> **從系統的角度看它們是最佳化；從演算法的角度看，網路一點都沒變。**

| | 前十講 | **本講** |
|---|---|---|
| **動到什麼** | 網路本身（權重被剪掉／被量化／架構被換掉／被蒸餾） | **只動執行方式** |
| **數學上** | 近似，會掉一點精度 | **完全等價，不掉精度** |
| **代價** | 精度 | 程式碼複雜度、binary size、記憶體 |

所以這兩類技巧是**正交的**，可以疊在一起用。

【口頭】教授對做研究的人的一句提醒：

> **「演算法每天都在變，但這些系統實作的基本原理，不管演算法怎麼演進都會一直有用。」**

### 0.2 這一講的三段結構

| 段落 | 內容 |
|---|---|
| **1. Edge AI** | 邊緣裝置有哪些硬體特性、為什麼難部署 |
| **2. 平行運算** | 迴圈最佳化、SIMD、多執行緒、CUDA |
| **3. 推論最佳化** | im2col、in-place depthwise、資料排列、Winograd |

【口頭】而且這些技巧**不只用在 TinyML**——Lab 5 要用同一套技巧把 **Llama 2 7B 跑在筆電上**。

---

## 1. Edge AI：資源到底有多緊

### 1.1 ⭐ 三個層級的硬體對照

| | **Cloud AI**<br>Nvidia H100 | **Mobile AI**<br>Apple M2 Ultra | **Mobile AI**<br>Qualcomm S8Gen2 | **Tiny AI**<br>STM32F746NG |
|---|---|---|---|---|
| **記憶體** | 80 GB | 64–192 GB | 8–24 GB | **320 kB** |
| **儲存** | ~TB/PB | ~GB/TB | ~GB | **1 MB** |
| **算力** | **1,979 TOPS** | 31.6 TOPS | 36 TOPS | **462 MOPS** |

【口頭】幾個現場補充：

- **Apple M2 Ultra 最大 192 GB**，比單張 GPU 的記憶體還大。
- **筆電的 30 TOPS 其實算力很足**，不要小看。
- **MCU 的 1 MB Flash 是唯讀的**——所以它**只能存權重，存不了 activation**。
  （↔ 這正是 [Lecture 10 §1.1](lecture10_mcunet_note.md) 講的 Flash／SRAM 分工。）
- 算力差距是 **1,979 TOPS vs 462 MOPS**，好幾個數量級。

【口頭】關於雲端的耗電，教授又講了一次實驗室的插座故事（[Lecture 7-8 §0](lecture07_08_nas_note.md) 也提過）：

> **一條 208 V 三相 50 A 的粗電纜，只能餵兩個 A100 節點；如果是 H100，一條只能餵一個節點。**

實驗室的伺服器有 **24 TB 儲存**。

### 1.2 ⭐ MCU vs MacBook Pro 的逐項拆解

Arm Cortex-M7 的 **STM32F746** vs **Apple MacBook Pro (M1 Ultra)**：

| | **CPU 核心數** | **最高時脈** | **GPU 核心** | **Neural Engine** | **L1** | **L2** | **L3** | **記憶體** | **儲存** |
|---|---|---|---|---|---|---|---|---|---|
| **STM32F746 MCU** | 1 | 216 MHz | 無 | 無 | 8 KB | **無** | **無** | 320 KB | 1 MB |
| **MacBook Pro** | 20 | 3200 MHz | 64 | 32 | 320 KB | 48 MB | 96 MB | 64 GB | 8 TB |
| **倍數差** | 20× | 15× | — | — | 40× | — | — | **210,000×** | **8,400,000×** |

【口頭】教授唸到最後一欄時說：**「8 TB 對 1 MB，這後面零太多我唸不出來。」**

**最該記住的三件事**：
1. MCU **沒有 GPU、沒有 Neural Engine**。
2. MCU **只有 L1，沒有 L2/L3**，而且 L1 只有 **8 KB**。
3. 記憶體差 **21 萬倍**。

### 1.3 記憶體階層與延遲

以筆電為例（Latency Numbers Every Programmer Should Know）：

| **大小** | **延遲** | **相對 L1** |
|---|---|---|
| **320 KB**（L1） | 0.5 ns | 1× |
| **48 MB**（L2） | 7 ns | 14× |
| **64 GB**（DRAM） | 100 ns | 200× |
| **8 TB**（SSD） | 1 ms | 極慢 |

> **越大 → 越慢 → 越便宜。**
> **所以整章平行運算的核心動作只有一個：想辦法讓資料待在越上層越好。**

---

## 2. ⭐⭐ 迴圈最佳化

四大平行運算技巧：

| 技巧 | 解決什麼 |
|---|---|
| **Loop optimization** | 改善 locality、減少分支開銷 |
| ├ **Loop reordering** | 換迴圈順序 → 改善 cache locality |
| ├ **Loop tiling** | 切分迭代空間 → 減少 cache miss |
| └ **Loop unrolling** | 展開迴圈 → 減少分支開銷（代價：binary 變大） |
| **SIMD** | 一道指令同時處理多筆資料 |
| **Multithreading** | 單一程序內多執行緒同時跑 |
| **CUDA** | 用 GPU 加速 |

以下全部用**同一個例子**：`N×N` 矩陣乘法，基準是 **Intel Xeon 4114 上的 naive 版 24,000 ms**。

---

### 2.1 ⭐ Loop Reordering（迴圈重排）

**naive 版是 i-j-k 三層迴圈**：

```c
for (i = 0; i < C->row; i++)
  for (j = 0; j < C->column; j++) {
    float acc = 0;
    for (k = 0; k < A->column; k++)
      acc += A[i][k] * B[k][j];
    C[i][j] = acc;
  }
```

**問題出在 B**。假設資料是 row-major（一列一列連續存）：

| 矩陣 | 存取式 | k 是內層迴圈時 |
|---|---|---|
| **A** | `A[i][k]` | k 變 → 沿著**列**走 → **連續**，很好 ✅ |
| **B** | `B[k][j]` | k 變 → 沿著**行**走 → `B[0][0]`、`B[1][0]`… **不連續** ❌ |

**【Q&A】教授問全班：如果把 i-j-k 換成 i-k-j 會怎樣？**

答：**B 就變成沿著列走了**（k 在外、j 在內），locality 變好。

**【Q&A】那 C 呢？變好還是變壞？**

答：**C 變差**。原本 `C[i][j]` 在 k 迴圈裡是固定的一格（locality 最好），現在 j 變成內層，C 就得一直換位置。

> **【口頭】沒有三全其美。A、B、C 一定要有取捨。**

**【Q&A】學生追問：B 變好、C 變壞，為什麼整體還是變好？**

> 【口頭】教授的答案：**「原本 B 的相鄰元素差了一整列（完全不連續）；而 C 我們至少還能一整列一整列地做規約（reduction），不是逐元素亂跳。這就是那個 trade-off。」**

**結果**：

| | 時間 |
|---|---|
| naive (i-j-k) | 24,000 ms |
| **reordering (i-k-j)** | **~2,000 ms → 12× 加速** |

**硬體完全沒換，只換了迴圈順序。**

【口頭】注意重排版還多了一行：

```c
float Aik = A[i][k];   // 提到內層迴圈外面
for (j = 0; j < C->column; j++)
  acc += Aik * B[k][j];
```

因為 `A[i][k]` 跟 j 無關，**先存成區域變數，避免內層迴圈裡反覆存取記憶體**。

---

### 2.2 ⭐⭐ Loop Tiling（迴圈分塊）

**問題**：cache 太小。MCU 的 L1 只有 **8 KB**，但矩陣可以很大。

B 矩陣的記憶體足跡是 **N²**。如果 N² 遠大於 cache，**資料還沒被重用完就被踢出去了**（cache miss）。

**解法**：把迭代空間切成小塊，**每一塊都能塞進 cache**，確保資料在被完全重用前不會被驅逐。

**一層一層加上去**：

| 版本 | 迴圈數 | B 的足跡 | A 的足跡 |
|---|---|---|---|
| **原始 (i-j-k)** | 3 | N² | N² |
| **tile j** | 4 | **N × T** | N² |
| **tile j, k** | 5 | **T × T** | N × T |
| **tile j, k, i** | **6** | **T × T** | **T × T** |

**六層迴圈版**（`BLK_SIZE = 32`）：

```c
for (ti = 0; ti < C->row;    ti += BLK_SIZE)
 for (tk = 0; tk < A->column; tk += BLK_SIZE)
  for (tj = 0; tj < C->column; tj += BLK_SIZE)
   for (i = ti; i < ti + BLK_SIZE; i++)
    for (k = tk; k < tk + BLK_SIZE; k++) {
      Aik = data_A[i * A->column + k];
      for (j = tj; j < tj + BLK_SIZE; j++)
        data_C[i * C->column + j] += Aik * data_B[k * B->column + j];
    }
```

**外層三個迴圈的步長是 T（tile size），內層三個步長是 1。**

> **T 怎麼選？讓 tile 塞得進 cache 就對了。**

**多層 tiling 對應多層 cache**：

【口頭】如果有 L1／L2 兩層 cache，就做**兩層 tiling**：第二層 tile 的工作集塞進 L2，第一層 tile 的工作集塞進 L1。

> **有幾層 cache，就做幾層 tiling。**

**結果**：

| | 時間 |
|---|---|
| naive | 24,000 ms |
| **tiling** | **~1,200 ms → 19× 加速** |

**【Q&A】學生問：怎麼知道 cache 有多大／有幾層？**

> 【口頭】教授：**這是 David Patterson 和 John Hennessy 課本裡的經典練習。畫「延遲 vs 區塊大小」的圖，你會看到延遲一段一段跳（bumpy）——每當區塊大小超過某一層 cache，延遲就突然跳一階。跳幾次，就代表有幾層 cache。**

---

### 2.3 ⭐ Loop Unrolling（迴圈展開）

**問題**：迴圈控制本身有開銷。

| 開銷 | 內容 |
|---|---|
| **指標運算** | 每輪都要動 i、j、k |
| **迴圈測試 + 分支** | 每輪都要測 `k < N`，成立就跳回去 |

**解法**：把迴圈本體複製幾份，步長跟著放大。

```c
// unroll k by 4
for (k = 0; k < A->column; k += 4) {
  acc[0] += A[i][k]   * B[k][j];
  acc[0] += A[i][k+1] * B[k+1][j];
  acc[0] += A[i][k+2] * B[k+2][j];
  acc[0] += A[i][k+3] * B[k+3][j];
  ...
}
```

| 項目 | 變化 |
|---|---|
| **指標算術** | **÷4** |
| **迴圈測試次數** | **÷4** |
| **迴圈本體 code size** | **×4** ← 代價 |

投影片的版本**同時展開 j（by 8）和 k（by 4）**。

**結果**：

| | 時間 |
|---|---|
| naive | 24,000 ms |
| **unrolling** | **~8,000 ms → 2.85× 加速** |

---

### 2.4 三者可以疊加

**【Q&A】教授問：這三個技巧互斥嗎？**

答：**不互斥**。

| 技巧 | 解決的面向 |
|---|---|
| reordering、tiling | **cache locality** |
| unrolling | **分支開銷** |

**【Q&A】學生問：那能不能乾脆把迴圈全部展開成一層？**

> 【口頭】教授：**不行。它們是正交的技巧——你還是需要 locality。你不會想沿著單一維度一路跑到底，那會毀掉另一個維度的 locality。**

【口頭】而且這是個**很大的設計空間**：unroll by 2 / 4 / 8？展開 j、k 還是 i-j-k 全展開？tile size 多少？

> **【口頭】而且最佳解**取決於硬體**——不同硬體偏好不同策略。這也是作業的加分題。**

---

## 3. ⭐ SIMD：一道指令、多筆資料

### 3.1 先複習 ISA

【口頭】教授先幫大家複習 6.191／6.004 的內容當暖身。

**ISA（Instruction Set Architecture，指令集架構）** = **軟體與硬體之間的介面**、也是使用者操控硬體的唯一管道。

常見指令：`ADD`、`COMPARE`、`JUMP`、`JUMP IF`、`LOAD`、`STORE`、`IN/OUT`。

**兩大類**：

| | **CISC**（複雜指令集） | **RISC**（精簡指令集） |
|---|---|---|
| **指令** | 多、專用、有些很少用到 | 少、規整、只實作常用的 |
| **每道指令做的事** | 多 | 少（輕量） |
| **同一程式的指令數** | 少 | 多 |
| **代表** | Intel x86 | **Arm、RISC-V** |
| **常見場景** | 桌機 CPU | **低功耗處理器** |

**例子：`C = A + B`**

| CISC | RISC |
|---|---|
| `add a, b, c`（1 道） | `load a, reg1`<br>`load b, reg2`<br>`add reg1 + reg2 = reg3`<br>`store reg3, c`（4 道） |

### 3.2 ⭐ SIMD 是什麼

> **一道指令，同時作用在多筆資料上。**

**張量天生就適合平行化**——一堆迴圈、一堆像素、一堆 token，套的是同一組權重。

| | **SISD**（單指令單資料） | **SIMD**（單指令多資料） |
|---|---|---|
| **暫存器寬度** | 32 bit | **128 bit** |
| **一次處理** | 1 個 FP32 | **4 個 FP32** |
| **算術運算數** | N³ | **N³ / 4** |

**兩個關鍵元件**：
1. **向量暫存器**（vector register）：一個暫存器裝得下多筆資料。
2. **向量運算**（vector operation）：算術與邏輯運算直接作用在整個向量上。

**好處**：吞吐量變高 + **能源效率變好**（指令開銷被攤分到更多資料上）。

【口頭】**連微控制器都有 SIMD**——可能不是 4 個，但至少能一次做 2 個。

### 3.3 SSE（Intel）vs NEON（ARM）

| | **SSE**（Intel） | **NEON**（ARM） |
|---|---|---|
| **指令** | `_mm_load_ps` / `_mm_mul_ps` / `_mm_add_ps` | `vld1q_f32` / `vmulq_f32` / `vaddq_f32` |
| **命名拆解** | `mm` = multimedia<br>`load/mul/add`<br>`ps` = packed single-precision | `v` = vector<br>`ld/mul/add`<br>`1` = 向量數<br>`q` = quadword（4 個 word） |

**點積的三種寫法**：

```c
// SISD
for k in range(0, N):
    C += A[k] * B[k]

// SSE
for k in range(0, N/4):
    C += _mm_mul_ps(_mm_load_ps(A[k*4]), _mm_load_ps(B[k*4]))

// NEON
for k in range(0, N/4):
    C += vmulq_f32(vld1q_f32(A[k*4]), vld1q_f32(B[k*4]))
```

**注意 `k*4`**：每次載入 4 個元素，所以**迴圈次數少 4 倍**。

### 3.4 SIMD 版矩陣乘法

```c
preprocessing();   // 初始化 A、B、C，並把 B 轉置成 transpose_tmp

for (i = 0; i < C->row; i++)
  for (j = 0; j < C->column; j++) {
    float accumulators[4] = {0, 0, 0, 0};
    __m128 *acc = (__m128*)accumulators;        // 四個 32-bit 累加器
    for (k = 0; k < A->column; k += 4) {
      __m128 val = _mm_mul_ps(_mm_load_ps(&A[i][k]),
                              _mm_load_ps(&transpose_tmp[j][k]));
      *acc = _mm_add_ps(*acc, val);
    }
    C[i][j] = accumulators[0] + accumulators[1]
            + accumulators[2] + accumulators[3];   // 最後把 4 格加總成 1 格
  }
```

ARM NEON 版本幾乎一模一樣，只是換成 `float32x4_t` / `vld1q_f32` / `vmulq_f32` / `vaddq_f32`。

**兩個容易漏掉的細節**：

1. **【口頭】為什麼要先轉置 B？**
   > **轉置 B 和做 loop reordering 是同一個效果**——都是為了讓 B 的存取連續。

2. **累加器是 128 bit（4 格）**，所以**最後一定要把 4 格規約成 1 格**才是 `C[i][j]`。

---

## 4. Multithreading：多執行緒

### 4.1 基本概念

> **Multithreading = 單一程序內同時執行多條執行緒。**
> **Thread（執行緒）是程式中最小的執行單位。**

| | **單執行緒** | **多執行緒** |
|---|---|---|
| **code、data、開啟的檔案** | 各自一份 | **共享** |
| **暫存器、堆疊、PC** | 一份 | **各自獨立** |

**四個好處**：

| 好處 | 說明 |
|---|---|
| **效能** | 多條執行緒同時做事 |
| **反應性** | 程式不會被單一操作卡住 |
| **資源利用率** | 共享資源，比開多個 process 便宜 |
| **程式結構** | 把複雜問題拆成小任務 |

【口頭】而且**一條執行緒被 I/O 或磁碟存取擋住時，同一顆 CPU 上的其他執行緒可以繼續跑**。教授用網頁伺服器輪流服務三個 client 的例子說明：client 2 閒著的時候，就去服務 client 1 或 3。

### 4.2 ⭐ 矩陣乘法怎麼切

**按 A 的列（row）切給不同執行緒**：

```
        thread 0 →  [ A 的前幾列 ]  ×  [ 整個 B ]  =  [ C 的前幾列 ]
        thread 1 →  [ A 的次幾列 ]  ×  [ 整個 B ]  =  [ C 的次幾列 ]
```

**【Q&A】教授問：執行緒之間有互動嗎？**

答：**沒有**。這正是我們要的——**執行緒之間的通訊越少、工作越獨立越好**。

### 4.3 Pthreads 寫法

```c
int main() {
  pthread_t threads[NUM_THREADS];
  ThreadData thread_data[NUM_THREADS];

  // 建立執行緒並分派工作
  for (int i = 0; i < NUM_THREADS; ++i) {
    thread_data[i].thread_id = i;
    pthread_create(&threads[i], nullptr, mat_mul_multithreading, &thread_data[i]);
  }
  // join：等所有執行緒完成
  for (int i = 0; i < NUM_THREADS; ++i)
    pthread_join(threads[i], nullptr);
  return 0;
}

void* mat_mul_multithreading(void* arg) {
  int thread_id = ((ThreadData*)arg)->thread_id;
  int rows_per_thread = SIZE_MATRIX / NUM_THREADS;
  int start_row = thread_id * rows_per_thread;          // 用 thread_id 算出自己的區段
  int end_row   = (thread_id + 1) * rows_per_thread;

  for (int i = start_row; i < end_row; ++i)
    for (int j = 0; j < SIZE_MATRIX; ++j)
      for (int k = 0; k < SIZE_MATRIX; ++k)
        C[i][j] += A[i][k] * B[k][j];
  return nullptr;
}
```

**兩個關鍵**：
1. **`thread_id` 唯一決定這條執行緒該做哪一塊**。
2. **`pthread_join` 一定要等全部完成**——【口頭】只要有一條沒跑完就去讀 C，**資料就是垃圾**。

**結果**：

| | 時間 |
|---|---|
| naive（單執行緒） | 24,000 ms |
| **4 條執行緒** | **4.1× 加速** |

**【口頭】為什麼 4 條執行緒能超過 4×？** 教授：**「可能是記憶體 locality 也順便變好了。」**

### 4.4 OpenMP：更簡單的寫法

**OpenMP**（Open Multi-Processing）= C/C++/Fortran 的共享記憶體平行程式 API。

| 特點 | 說明 |
|---|---|
| **可攜性高** | 同一份程式碼，不用分 ARM 版／Intel 版 |
| **易整合** | 加兩行就能把既有程式平行化 |
| **常用指示詞** | `#pragma omp parallel`（平行區）<br>`#pragma omp for`（平行化迴圈）<br>`sections` / `single` / `critical` / `barrier` |

```c
omp_set_num_threads(4);          // ① 指定執行緒數

#pragma omp parallel for         // ② 指定要平行化哪個迴圈
for (int i = 0; i < N; ++i)
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < N; ++k)
      C[i][j] += A[i][k] * B[k][j];
```

**只加兩行**，就把序列程式變成平行程式。

> **注意 pragma 放在 `i` 迴圈上面 → 平行化的是 i 維度 → 也就是按列切，跟 Pthreads 版一模一樣。**

**OpenMP 的程式碼比 Pthreads 乾淨很多。**

---

## 5. ⭐ CUDA 與 Tensor Core

### 5.1 CUDA 是什麼

| 項目 | 說明 |
|---|---|
| **為什麼用 GPU** | **指令吞吐量與記憶體頻寬都高得多** |
| **CUDA 是什麼** | Nvidia 在 **2006 年**推出的通用平行運算平台與程式模型 |
| **語法** | **類 C 語言**，用來寫跑在 GPU 上的程式 |

【口頭】教授的兩點補充：
- CUDA 已經是**十幾年、快二十年前**的東西了。
- **它的向後相容性非常好**——GPU 換了好幾代，程式模型還是同一套。這是它成為 AI 運算主力的關鍵。

【口頭】另外：**大家在 Lab 1–3 的 Colab 上，應該都感受過開 GPU 跟不開 GPU 的速度差。**

### 5.2 ⭐ CUDA 執行緒的兩層階層

**Grid（網格） → Block（區塊） → Thread（執行緒）**，thread ID 最多可以是三維。

```c
const int Nx = 12;
const int Ny = 6;
dim3 threadsPerBlock(4, 3);                                     // 每個 block 4×3 = 12 條
dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y);     // 3×2 = 6 個 block

matrixAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);             // 共 72 條 CUDA thread
```

**6 個 block × 每個 12 條 = 72 條 CUDA 執行緒。**

**兩段程式碼跑在不同地方**：

| | 跑在哪 | 做什麼 |
|---|---|---|
| **Host code** | **CPU** | 序列執行：配置記憶體、批次啟動 CUDA 執行緒 |
| **Kernel code** | **GPU（device）** | 平行執行：每條執行緒做自己那一份 |

```c
__global__ void matrixAdd(float A[Ny][Nx], float B[Ny][Nx], float C[Ny][Nx])
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // 自己在 grid 裡的絕對位置
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  C[j][i] = A[j][i] + B[j][i];
}
```

> **每條執行緒由「自己 block 的位置（blockIdx）」＋「自己在 block 裡的位置（threadIdx）」算出全域 index，然後只做自己那一格。**
> 【口頭】就像多維陣列攤平算 index 一樣。

**【Q&A】學生問：為什麼要分 block 和 thread 兩層？**

> 【口頭】教授：**「同一個 block 裡的執行緒必須 lockstep（步調一致）——這個例子裡 12 條執行緒得做完全一樣的事。如果 block 開太大，而你的工作沒那麼多平行度，就會浪費一堆執行緒。分層就是給你一點彈性。」**
> 而且**這也順便回答了為什麼需要記憶體階層**（見下一節）。

### 5.3 記憶體模型

**(a) Host 與 Device 是兩個獨立的位址空間**，中間走 PCIe：

```c
float* A = new float[N];                   // 在 host（CPU）記憶體配置
for (int i = 0; i < N; i++) A[i] = (float)i;

float* deviceA;
cudaMalloc(&deviceA, bytes);               // 在 device（GPU）記憶體配置
cudaMemcpy(deviceA, A, bytes, cudaMemcpyHostToDevice);   // 搬資料
```

**【Q&A】教授問：能不能從 host 直接讀 `deviceA[i]`？**

答：**不行。** `deviceA` 是 GPU 的位址，在 CPU 的程式碼裡是無效的。

> **【口頭】而且我們要盡量減少這種資料搬移。**

**(b) Kernel 看得到三種位址空間**：

| 位址空間 | 誰能存取 | 特性 |
|---|---|---|
| **Private memory** | 單一 thread | **最小、最快** |
| **Shared memory** | 同一 block 內全部 thread | 中等 |
| **Global memory** | 全部 thread | **最大、最慢** |

> **不同位址空間 → 不同 locality → 不同的 load/store 開銷。**

### 5.4 ⭐ CUDA 版矩陣乘法（含 shared memory tiling）

```c
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int A_row, int A_column, int B_column) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float As[TILE_SIZE][TILE_SIZE];   // 每個 block 一份，用來 tile A
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];   // 用來 tile B

  float value = 0;
  for (int i = 0; i < A_column / TILE_SIZE; i++) {
    As[threadIdx.y][threadIdx.x] = A[...];
    Bs[threadIdx.y][threadIdx.x] = B[...];
    __syncthreads();                           // 等所有執行緒載入完記憶體

    for (int k = 0; k < TILE_SIZE; k++)
      value += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();                           // 等所有執行緒乘加完
  }
  C[row * B_column + col] = value;
}
```

> **這裡把 §2.2 的 loop tiling 直接搬到 GPU 上——只是 cache 換成了 shared memory，而且要手動 `__syncthreads()`。**

**結果**（2080 Ti）：

| | 時間 |
|---|---|
| CPU naive | ~25,000 ms |
| **CUDA（端到端）** | **~258 ms → 94× 加速** |

**⭐【口頭】教授特別強調的一個數字**：

> **那 258 ms 裡，真正在 GPU 上算的 kernel 只佔 6.7 ms。**
> **剩下的全是 kernel 啟動開銷 + 資料搬移開銷。**

這句話的意思是：**GPU 加速常常不是卡在算力，而是卡在搬資料。**

### 5.5 ⭐ Tensor Core

| | **CUDA Core** | **Tensor Core** |
|---|---|---|
| **每個 cycle 做什麼** | 1 個 FP32 或 2 個 FP16 的乘加（MAC） | **一整個小矩陣乘法**（Turing 4×4×4；Ampere 8×4×8）FP16 |

**H100 的 TFLOPS**：

| 精度 | TFLOPS | 在哪 |
|---|---|---|
| **FP32** | 67 | CUDA Core |
| **TF32** | 989 | Tensor Core（**只有 Tensor Core 有**） |
| **FP16** | 1,979 | Tensor Core |
| **INT8 / FP8 / Sparse FP16** | **3,958** | Tensor Core |

> **從 FP32 到 INT8/FP8：吞吐量差 60 倍。**
> **8-bit 算術比 CUDA Core 的 FP32 快得多。**（↔ 這就是 [Lecture 5-6 量化](lecture05_06_quantization_note.md) 為什麼值得做的硬體理由。）

**實測**（A6000，N×N×N 矩陣乘法）：

> **Tensor Core 最快比 CUDA Core 快 3.8×，而且 N 越大優勢越明顯。**

### 5.6 MMA：用小的乘法拼出大的

**Tensor Core 提供的 intrinsic 是固定尺寸的**（例如 `16×8×16`）。要算更大的矩陣怎麼辦？**用 §2.2 的 tiling。**

**例：用 `16×8×16` intrinsic 算 `16×16×32` 的 MMA**

| 運算元 | 切成 |
|---|---|
| **A** | 2 個 16×16 tile |
| **B** | 4 個 16×8 tile |
| **C** | 2 個 16×8 tile |

**做四次 MMA、累加起來就好。**

> **【投影片新增】而且這四次 MMA 的順序無所謂**（因為只是累加）。
> （2023 逐字稿只簡略帶過「用 tiling 拼出更大的矩陣乘法」，投影片 p47–p52 才有完整分解圖。）

---

## 6. ⭐⭐ 推論最佳化

前面四個是**通用**的平行運算技巧；接下來四個是**針對神經網路推論**的最佳化。

| 技巧 | 目的 |
|---|---|
| **Im2col convolution** | 用高度最佳化的 GEMM 來做卷積 |
| **In-place depth-wise convolution** | **降低 peak memory** |
| **選對 data layout** | 改善 locality |
| **Winograd convolution** | **減少乘法次數** |

---

### 6.1 Im2col（Image to Column）

**想法**：矩陣乘法（GEMM）已經被最佳化到極致了，**那能不能把卷積變成矩陣乘法？**

**做法**：**攤平**。

| 攤平什麼 | 變成 |
|---|---|
| **每個 sliding window 內的 activation**（K²·C 個數） | 矩陣的**一列** |
| **每個 kernel**（K²·C 個數） | 矩陣的**一行** |

```
[ HW × K²C ]  ×  [ K²C × N ]  =  [ HW × N ]
   攤平的輸入        攤平的權重       輸出
```

**規約維度是 K²·C**（例如 3×3 的話就是 9C），輸出維度是 **HW × N**。

| | 說明 |
|---|---|
| **✅ Pro** | **直接用現成的高效 GEMM kernel** |
| **❌ Con** | **額外的記憶體** |

**❌ 額外記憶體有多嚴重？**

【口頭】教授用 3×3 舉例：窗口右移一格時，`b c d g h i l m n` 跟原本的 `a b c f g h k l m` **重疊了兩行**。

> **也就是說，攤平後大約有 2/3 的資料是重複的。**

**怎麼解？**

> **Implicit GEMM**：direct convolution 的變體，**直接在原本的權重和 activation 張量上運算**，不真的把矩陣物化（materialize）出來——邊算邊生。

---

### 6.2 ⭐⭐ In-place Depth-wise Convolution

**先回顧問題**（[Lecture 10 §1.3](lecture10_mcunet_note.md) 講過）：

> **MobileNetV2 這類 inverted residual block 雖然縮小了模型大小和 FLOPs，卻讓 peak memory 暴增 3–6 倍。**

**一般的 depthwise convolution**：

```
[ 輸入 activation：C 個 channel ]  →  [ 輸出 activation：C 個 channel ]
             同時存在 → peak memory = 2 × C × H × W
```

**關鍵觀察**：**depthwise 的每個 channel 是獨立算的**。

**所以可以 in-place（原地）更新**，只留一個暫存 buffer：

```
step 1：算 channel 1 → 結果放進 temp buffer → 寫回原本 channel 1 的位置
step 2：算 channel 2 → 結果放進同一個 temp buffer → 寫回 channel 2
...
step N：算 channel N → 同一個 buffer → 寫回 channel N
```

**同一個 temp buffer 被 N 個 channel 重複使用。**

| | **一般 depthwise** | **In-place depthwise** |
|---|---|---|
| **Peak Memory** | **2 × C × H × W** | **(1 + C) × H × W** |

> **C 很大時，相當於直接省掉將近一半的 activation 記憶體。**
> **而且完全不改變數學結果。**

---

### 6.3 ⭐ 選對 Data Layout

**兩種擺法**：

| Layout | 記憶體裡誰排在最內層 |
|---|---|
| **NCHW** | **W**（同一 channel 的 H×W 連續擺） |
| **NHWC** | **C**（同一像素的所有 channel 連續擺） |

**答案是：兩種都要，看是哪種卷積。**

#### (a) Point-wise（1×1）卷積 → 用 **NHWC**

1×1 卷積是**沿著 channel 方向做規約**。

| Layout | 權重存取順序 | 連續嗎 |
|---|---|---|
| **NCHW** | `000 001 002 … 009 010 011 … 018 019 020 …` | ❌ 要跨 H×W 跳 |
| **NHWC** | `000 009 018 … 001 010 019 … 007 016 025 …` | ✅ **channel 連續** |

> **TinyEngine 對 point-wise convolution 採用 NHWC。**

#### (b) Depth-wise 卷積 → 用 **NCHW**

depthwise 是**每個 channel 各自做 3×3**，channel 之間沒有互動。

| Layout | 權重存取順序 | 連續嗎 |
|---|---|---|
| **NHWC** | `000 009 … 072 081 … 001 010 …` | ❌ 同一個 kernel 的 9 格散在各處 |
| **NCHW** | `000 001 002 … 009 010 011 … 072 073 074 …` | ✅ **同一 channel 全部黏在一起** |

> **TinyEngine 對 depth-wise convolution 採用 NCHW**——這也正好配合 §6.2 的 in-place 更新（按 channel 一個一個處理）。

**⭐ 這一段的重點**：

> **沒有「最好的 layout」，只有「對這個算子最好的 layout」。**
> **推論引擎的價值，就在於它知道要在哪裡切換。**

---

### 6.4 ⭐ Winograd Convolution

**先算 baseline**：用 3×3 kernel 算出 **2×2 = 4 個輸出**要幾次乘加？

> **每個輸出要 9 × C 次 MAC，4 個輸出 → 36 × C 次 MAC。**

**Winograd 的做法**：

```
輸入 tensor  ──[ Data Transform（線上）]──┐
                                          ├─→ 逐點相乘 ─→ 沿 C 累加 ─→[ Output Transform ]─→ 輸出
Filter      ──[ Filter Transform（離線）]─┘
                                              16 × C 次 MAC
```

| | **MAC 次數（4 個輸出）** |
|---|---|
| **Direct convolution** | **36 × C** |
| **Winograd convolution** | **16 × C** |
| | **→ 少 2.25×** |

**公式**（3×3 卷積）：

```
Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A
      └ Filter Transform ┘ └ Data Transform ┘
          （離線）              （線上）
```

**⭐ 為什麼這招划算？看那三個轉換矩陣長什麼樣：**

| 矩陣 | 元素 |
|---|---|
| **B / Bᵀ**（資料轉換） | 只有 **+1、−1、0** |
| **Aᵀ**（輸出轉換） | 只有 **+1、−1、0** |
| **G / Gᵀ**（濾波器轉換） | 只有 **1、0、±1/2** |

**【口頭】教授的重點**：

> **1. Filter transform 可以離線做**——因為**推論時權重是固定的常數**。
> **2. Data transform 雖然要線上做，但矩陣只有 ±1 和 0，用位移（shift）就能實作——完全不需要乘法。**

所以「多出來的兩次轉換」幾乎是免費的，換來乘法次數少 2.25 倍。

> **這是推論階段非常常用的一招。**

---

## 7. TinyEngine、TinyChatEngine 與 Lab 4/5

【口頭】教授在最後把整章收束到作業上：

| | 內容 |
|---|---|
| **開源** | **MCUNet + TinyEngine** 都已開源，可以直接在微控制器上試 |
| **去年的作業** | 在 MCU 上跑 MCUNet |
| **今年（2023）的作業** | **在筆電上跑大型語言模型** |

**Lab 4 / Lab 5**：

| Lab | 做什麼 |
|---|---|
| **Lab 4** | 用 **AWQ** 把 LLM 量化到 **4 bit**（週四釋出） |
| **Lab 5** | 用 **TinyLLMEngine** 部署，**在自己電腦上跑本地 chatbot** |
| | 實作 **loop unrolling / reordering、SIMD、multithreading** |
| | 量測不同最佳化技巧帶來的**實機延遲改善** |

> **【口頭】會給你一份 baseline starter code，你的任務就是讓它在你的筆電上跑得更快。**

【口頭】另外，**每一頁投影片的右上角都有連結**，可以直接抓下對應的範例程式碼自己編譯來玩。教授說：

> **「很有可能你調出來的會比我們的還快。」**

---

## 8. 一頁速查表

### 8.1 該記住的核心觀念

| 觀念 | 一句話 |
|---|---|
| **本講技巧的本質** | **Functional-preserving——數學完全等價，只改執行方式** |
| **平行運算的目的** | **把資料留在越上層的記憶體越好** |
| **Loop reordering** | 換順序 → 改 locality，但一定有取捨 |
| **Loop tiling** | 切塊塞進 cache；**有幾層 cache 就做幾層 tiling** |
| **Loop unrolling** | 少分支，代價是 binary 變大 |
| **三者關係** | **正交，可以疊加**，而且最佳組合取決於硬體 |
| **SIMD** | 一道指令 4 個 FP32；**MCU 也有** |
| **Multithreading** | 按列切 → **執行緒之間零互動** |
| **CUDA 兩層階層** | Block／Thread 分層是為了**平行度不足時不浪費執行緒** |
| **GPU 的真實瓶頸** | **常常不是算力，是 kernel 啟動與資料搬移** |
| **Tensor Core** | 一個 cycle 做一整個小矩陣乘法，不是一個 MAC |
| **Im2col** | 換來 GEMM，代價是約 2/3 重複資料；用 implicit GEMM 解 |
| **In-place depthwise** | **2CHW → (1+C)HW**，因為 channel 彼此獨立 |
| **Data layout** | **point-wise 用 NHWC，depth-wise 用 NCHW** |
| **Winograd** | 轉換矩陣只有 ±1/0 → **用位移實作，不需乘法** |

### 8.2 該記住的數字

（基準：Intel Xeon 4114，naive 矩陣乘法 **24,000 ms**）

| 技巧 | 加速 |
|---|---|
| **Loop reordering**（i-j-k → i-k-j） | **12×** |
| **Loop tiling**（BLK_SIZE = 32，六層迴圈） | **19×** |
| **Loop unrolling**（j by 8、k by 4） | **2.85×** |
| **Multithreading**（Pthreads，4 threads） | **4.1×** |
| **CUDA**（2080 Ti，端到端） | **94×**（其中 kernel 只佔 6.7 ms） |
| **Tensor Core vs CUDA Core**（A6000） | **最高 3.8×** |
| **Winograd vs direct conv** | **MAC 少 2.25×** |
| **In-place depthwise** | **2CHW → (1+C)HW** |

**硬體數字**：

| 數字 | 是什麼 |
|---|---|
| **320 kB / 1 MB / 462 MOPS** | STM32F746NG 的記憶體／儲存／算力 |
| **8 KB** | MCU 的 L1 cache（**且沒有 L2/L3**） |
| **210,000× / 8,400,000×** | MacBook Pro 與 MCU 的記憶體／儲存差距 |
| **0.5 ns / 7 ns / 100 ns / 1 ms** | L1 / L2 / DRAM / SSD 的延遲 |
| **128 bit = 4 × FP32** | SIMD 向量暫存器 |
| **67 / 989 / 1,979 / 3,958 TFLOPS** | H100 的 FP32 / TF32 / FP16 / INT8-FP8（**差 60×**） |
| **72 = 6 blocks × 12 threads** | 投影片的 CUDA 例子 |

### 8.3 四大平行技巧速記

| | 平行的粒度 | 誰負責 |
|---|---|---|
| **Loop optimization** | 單執行緒內的記憶體存取模式 | 編譯器 + 你 |
| **SIMD** | **資料層級**（一道指令多筆資料） | intrinsics |
| **Multithreading** | **執行緒層級**（多核心） | Pthreads / OpenMP |
| **CUDA** | **大規模執行緒層級**（GPU） | CUDA kernel |

---

## 9. 與其他課程的連結

| 本講觀念 | 連到哪裡 |
|---|---|
| **§0.1 功能保持 vs 改演算法** | **[Lecture 3-4 剪枝](lecture03_04_pruning_note.md)、[Lecture 5-6 量化](lecture05_06_quantization_note.md)、[Lecture 7-8 NAS](lecture07_08_nas_note.md)、[Lecture 9 KD](lecture09_kd_note.md)** —— 那些都改演算法，本講不改 |
| **§1 MCU 的 320 kB / 1 MB** | **[Lecture 10 §1](lecture10_mcunet_note.md)** —— 同一組硬體約束 |
| **§5.5 INT8/FP8 比 FP32 快 60×** | **[Lecture 5-6 量化](lecture05_06_quantization_note.md)** —— 量化為什麼真的會變快的硬體理由 |
| **§6.2 in-place depthwise** | **[Lecture 10 §1.3 / §4](lecture10_mcunet_note.md)** —— peak activation 是 MCU 的牆 |
| **§6.3 NHWC / NCHW** | **[Lecture 10 §2](lecture10_mcunet_note.md)** —— TinyEngine 就是靠這些決策撐起 MCUNet |
| **§2.2 tiling → §5.4 shared memory** | 同一個想法在 CPU cache 與 GPU shared memory 上的兩種化身 |
| **§7 Lab 4 AWQ 量化** | **Lecture 12 Transformer/LLM**（下一講） |

---

## 附：這一講的一句話總結

> **前十講都在問「網路能不能更小」；這一講第一次問「同一個網路，能不能跑得更快」。**
> **答案全部指向同一件事：算力通常不是瓶頸，記憶體才是。**
> **迴圈重排、分塊、SIMD、多執行緒、CUDA、im2col、in-place、layout、Winograd ——**
> **九個技巧，九種讓資料待在更靠近運算單元的地方的方法。**
> **而且它們一行精度都不會掉。**

---

*筆記依據 MIT 6.5940 Fall 2023 Lecture 11（[錄影](https://youtu.be/HGsvWHqU29Y)，1:16:05）的課堂逐字稿與官方投影片 `Lec11-TinyEngine.pdf`（79 頁）整理。標記【口頭】處為投影片沒有、僅課堂講述的內容；【Q&A】為教授反問全班或學生提問；【投影片新增】為投影片版本較新、2023 逐字稿沒有的段落。*
