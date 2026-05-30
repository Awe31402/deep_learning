# Session Transcript: PyTorch Mixed Precision Q&A (2026-05-30)

## Context
Discussion on the implementation details and weight precision of PyTorch Automatic Mixed Precision (AMP) using `mixed-precision.py`.

---

### Round 1: Mixed Precision Script Overview
**User:**
/home/awe/disk/deep_learning/model-lightweighting/mixed-precision.py 是一個混和精度訓練的簡易範例

**Gemini:**
是的，這份程式碼 `mixed-precision.py` 是一個非常經典且標準的 **PyTorch 自動混合精度（AMP, Automatic Mixed Precision）** 訓練範例。

以下為您簡析該範例的核心運作機制，以及我們可以如何進一步協助您：

### 核心技術與工作原理

1. **自動混合精度 (`torch.cuda.amp.autocast`)**
   * **作用**：在 `with autocast():` 區塊中，PyTorch 會自動為不同的運算選擇合適的精度。例如：矩陣乘法（GEMM）和卷積（Convolution）等計算密集型操作會自動轉換成高效率的 **FP16**；而數值較敏感的操作（如 `Softmax`、`Loss` 計算）則保留在 **FP32**，以兼顧**顯存節省、計算加速**與**數值穩定度**。
   * **程式碼片段**：
     ```python
     with autocast():
         outputs = model(inputs)
         loss = criterion(outputs, targets)
     ```

2. **梯度縮放器 (`torch.cuda.amp.GradScaler`)**
   * **作用**：FP16 的數值表示範圍較窄，微小的梯度在反向傳播時很容易發生**數值下溢（Underflow）**變成零。`GradScaler` 通過在反向傳播前將 Loss 乘以一個縮放因子，將梯度放大到 FP16 能表示的範圍，並在更新參數前再縮放回去（Unscale）。
   * **程式碼片段**：
     ```python
     scaler.scale(loss).backward() # 放大梯度
     scaler.step(optimizer)        # 還原梯度並更新參數
     scaler.update()               # 動態更新縮放因子
     ```

---

### 我們可以為您做些什麼？

如果您想基於這個範例進行深入研究或修改，我隨時可以協助您：

* 🚀 **基準效能測試 (Benchmarking)**：修改程式碼以對比 **混合精度 (AMP)** 與 **單精度 (FP32)** 在訓練速度（每秒處理的影像數）與顯存佔用（VRAM 使用量）上的具體效能差距。
* 🔄 **API 現代化更新**：將程式碼中舊版的 `torch.cuda.amp.autocast` 更新為 PyTorch 2.x 推薦的現代 `torch.amp.autocast(device_type="cuda")` API，並可嘗試引入 `bfloat16`（如果您的 GPU 支援 Ampere 架載或更新版本）。
* 🛠️ **程式碼調優與除錯**：優化程式碼結構，例如解決 `start_time` 全域變數的隱式依賴，或調整 `DataLoader` 的 `pin_memory=True` 以獲得更好的 GPU 傳輸效能。
* 📉 **整合其他模型輕量化技術**：配合模型輕量化主題，加入**模型剪枝 (Pruning)**、**知識蒸餾 (Knowledge Distillation)** 或 **訓練後量化 (PTQ)**。

請告訴我您接下來想針對這個範例進行什麼樣的實作或探討！

---

### Round 2: Output Model Precision
**User:**
最後輸出的模型會的精度也會是混合精度嗎

**Gemini:**
**不會，最後輸出的模型權重（Weights）依然是標準的單精度（FP32）。**

這是一個非常關鍵且常見的問題。在 PyTorch 的自動混合精度（AMP）機制中，模型的權重儲存與更新與我們直覺想像的有所不同。以下是詳細的運作原理與說明：

---

### 1. 為什麼輸出的模型仍然是 FP32？

在 AMP 訓練期間，PyTorch 採用了 **「主權重（Master Weights）」** 的策略：

* **記憶體中的真實權重是 FP32**：模型的所有參數（Weights 和 Biases）在記憶體中始終以 **FP32** 格式保存。
* **臨時轉換（On-the-fly Copy）**：只有在進入 `with autocast():` 區間進行前向傳播（Forward）與反向傳播（Backward）時，PyTorch 才會在後台**臨時**將 FP32 權重複製並轉換為 **FP16** 來進行快速的矩陣運算。
* **更新時使用 FP32**：反向傳播計算出梯度後，優化器（Optimizer）會使用還原後的梯度去更新那套 **FP32 的主權重**。這樣可以確保微小的權重更新不會因為 FP16 的精度不足而直接被捨入（Round）為零。
* **儲存結果**：當您執行 `torch.save(model.state_dict(), ...)` 時，儲存的是這套 FP32 的主權重。因此，匯出的 `.pth` 檔案大小和精度與一般的 FP32 模型完全一樣。

---

### 2. 如果我想在部署（Inference）時使用純半精度（FP16）模型呢？

如果您希望在推論或部署時，讓模型檔案大小減半（例如 ResNet50 從約 90MB 降到 45MB）並加速推論，您可以在載入模型後，**手動將模型轉為純 FP16**。

操作方式如下：

```python
# 1. 載入儲存的 FP32 權重
checkpoint = torch.load('./checkpoint/best_model.pth')
model.load_state_dict(checkpoint['model'])

# 2. 將整個模型轉換成純 FP16
model = model.half()  # 這會把所有參數轉為 float16
model.to(device)
model.eval()

# 3. 推論時，輸入的 Tensor 也必須手動轉為 FP16
with torch.no_grad():
    for inputs, targets in testloader:
        inputs = inputs.to(device).half()  # 注意：輸入必須轉成 .half()
        targets = targets.to(device)
        
        outputs = model(inputs)
        # 後續計算...
```

### 💡 總結
* **訓練與儲存時**：模型本體依然是 **FP32**，混合精度只發生在運算的中間過程。
* **部署推論時**：如果需要極致的輕量化與速度，可以手動使用 `.half()` 將其轉為**純 FP16 模型**。

---
