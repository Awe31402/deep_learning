# 線性層反向傳播比對分析 (Linear Layer Backpropagation Comparison)

本文件比對了以下兩者在線性層反向傳播（Linear Layer Backpropagation）上的數學推導與代碼實現細節：
1. **代碼實現**：`/home/awe/disk/deep_learning/mit-deep-learning/hw1.py` 中的 `Linear.backward`。
2. **教科書理論**：`/home/awe/disk/deep_learning/mit-deep-learning/backpropagation.md` 第 14.6.1 節「線性層的反向傳播」。

---

## 1. 核心差異概述：批次處理 (Batched) vs. 單一資料點 (Single-Sample)

* **教科書理論 (`backpropagation.md`)**：
  * 理論推導主要針對**單一數據點**進行分析。
  * 輸入 $\mathbf{x}_{\texttt{in}}$ 被視為列向量（維度 $[N \times 1]$），傳回的梯度 $\mathbf{g}_{\texttt{out}}$ 為行向量（維度 $[1 \times M]$）。
  * 權重矩陣 $\mathbf{W}$ 的維度為 $[M \times N]$。
* **程式碼實現 (`hw1.py`)**：
  * 直接採用深度學習實務上的**批次處理（Batched Data）**。
  * 輸入 `x`（即 `self.inputs[0]`）形狀為 `[batch_size, in_features]`。
  * 傳回梯度 `dLdout` 形狀為 `[batch_size, out_features]`。
  * 權重 `self.weight` 形狀為 `[out_features, in_features]`。

---

## 2. 前向傳播 (Forward Pass) 對比

* **理論公式**：
  $$\mathbf{x}_{\texttt{out}} = \mathbf{W} \mathbf{x}_{\texttt{in}} + \mathbf{b}$$
  * 其中 $\mathbf{W}$ 乘在輸入的前面（左乘）。
* **程式碼實現**：
  ```python
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return x @ self.weight.t() + self.bias
  ```
  * **維度分析**：`[batch_size, in_features] @ [in_features, out_features] + [out_features]` $\rightarrow$ 輸出 `[batch_size, out_features]`。
  * **說明**：為符合 PyTorch 的張量慣例，程式碼將批次輸入 `x` 置於前面（右乘轉置權重），數學上等同於 $\mathbf{X}_{\texttt{out}} = \mathbf{X}_{\texttt{in}}\mathbf{W}^\top + \mathbf{b}^\top$。

---

## 3. 反向傳播 (Backward Pass) 各梯度對比

### A. 輸入梯度計算（傳回前一層的梯度：$\mathbf{g}_{\texttt{in}}$ 或 $dL/dx$）

* **教科書理論**：
  $$\mathbf{g}_{\texttt{in}} = \mathbf{g}_{\texttt{out}} \mathbf{W}$$
  * 在書中虛擬碼表示為：`J_in = matmul(J_out, W)`。
  * 維度：$[1 \times M] \times [M \times N] \rightarrow [1 \times N]$。
* **程式碼實現 (`hw1.py`)**：
  ```python
  grad_input = dLdout @ self.weight
  ```
  * **維度分析**：`[batch_size, out_features] @ [out_features, in_features]` $\rightarrow$ `[batch_size, in_features]`。
  * **對比結論**：兩者在數學上完全等價。程式碼藉由矩陣乘法，同時對批次（batch）中的所有樣本進行了輸入梯度的並行計算。

---

### B. 權重梯度計算（對參數 $\mathbf{W}$ 的梯度：$\frac{\partial J}{\partial \mathbf{W}}$）

* **教科書理論**：
  $$\frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}_{\texttt{in}} \mathbf{g}_{\texttt{out}}$$
  * 這是輸入列向量 $\mathbf{x}_{\texttt{in}}$（$[N \times 1]$）與梯度行向量 $\mathbf{g}_{\texttt{out}}$（$[1 \times M]$）的**外積（Outer Product）**，維度為 $[N \times M]$。
  * 由於權重矩陣 $\mathbf{W}$ 的形狀為 $[M \times N]$，因此虛擬碼在更新權重時需要對其轉置：
    ```python
    self.W -= self.lr * dJdW.transpose()
    ```
* **程式碼實現 (`hw1.py`)**：
  ```python
  self.weight.grad = dLdout.t() @ self.inputs[0]
  ```
  * **維度分析**：`dLdout.t()` 的形狀為 `[out_features, batch_size]`，`self.inputs[0]`（即 `x`）形狀為 `[batch_size, in_features]`。相乘後得到的形狀為 `[out_features, in_features]`。
  * **對比結論**：
    1. 程式碼採用 `dLdout.t() @ x`，在數學上直接等於 $\mathbf{g}_{\texttt{out}}^\top \mathbf{x}_{\texttt{in}}$（即轉置後的外積）。這與書中轉置後的結果 $(\mathbf{x}_{\texttt{in}}\mathbf{g}_{\texttt{out}})^\top = \mathbf{g}_{\texttt{out}}^\top\mathbf{x}_{\texttt{in}}$ 完美契合。
    2. 由於在計算時直接將維度調整為 `[out_features, in_features]`，使得梯度形狀與權重形狀完全一致，在進行梯度更新時**不需要再進行額外的轉置操作**。
    3. 同時，該矩陣相乘也隱式地在批次（Batch）維度上完成了梯度的累加（Sum over Batch）。

---

### C. 偏置梯度計算（對參數 $\mathbf{b}$ 的梯度：$\frac{\partial J}{\partial \mathbf{b}}$）

* **教科書理論**：
  $$\frac{\partial J}{\partial \mathbf{b}} = \mathbf{g}_{\texttt{out}}$$
  * 維度：$[1 \times M]$（與偏置向量 $\mathbf{b}$ 形狀一致）。
* **程式碼實現 (`hw1.py`)**：
  ```python
  self.bias.grad = dLdout.sum(dim=0)
  ```
  * **維度分析**：`dLdout` 形狀為 `[batch_size, out_features]`，在第 0 維（Batch 維度）上求和後得到形狀 `[out_features]`。
  * **對比結論**：數學上完全一致。程式碼通過 `.sum(dim=0)` 累加了整個批次中所有數據點對偏置的梯度貢獻。

---

## 4. 總結對照表

| 比對項目 | 教科書理論 (`backpropagation.md`) | 程式碼實現 (`hw1.py`) | 數學與實現說明 |
| :--- | :--- | :--- | :--- |
| **處理模式** | 單一數據點 (Single Sample) | 批次處理 (Batched Data) | 程式碼利用矩陣並行處理加速運算。 |
| **前向傳播公式** | $\mathbf{x}_{\texttt{out}} = \mathbf{W}\mathbf{x}_{\texttt{in}} + \mathbf{b}$ | `x @ weight.t() + bias` | 右乘轉置權重，符合 PyTorch 的 API 設計習慣。 |
| **輸入梯度 ($dL/dx$)** | $\mathbf{g}_{\texttt{in}} = \mathbf{g}_{\texttt{out}}\mathbf{W}$ | `dLdout @ weight` | 兩者完全等價。 |
| **權重梯度 ($dL/dW$)** | $\frac{\partial J}{\partial \mathbf{W}} = \mathbf{x}_{\texttt{in}}\mathbf{g}_{\texttt{out}}$ (外積，$[N \times M]$) | `dLdout.t() @ x` (維度 $[M \times N]$) | 程式碼直接計算符合權重形狀的梯度，免去了更新時的轉置步驟，並完成批次累加。 |
| **偏置梯度 ($dL/db$)** | $\frac{\partial J}{\partial \mathbf{b}} = \mathbf{g}_{\texttt{out}}$ | `dLdout.sum(dim=0)` | 程式碼在 Batch 維度上對偏置梯度求和。 |
