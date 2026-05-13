# 2D Classification with a Single Neuron: Theory vs. Implementation

本文件深入整理了 Volpe 《Deep Learning Crash Course》中 "Classifying 2D Data" 一節的理論觀點，並與 `dl_crash_course_01/two_dim_prediction.py` 的實作進行對照。

---

## 1. 理論觀點：2D 數據分類

當數據從 1D 擴展到 2D 時，神經元的處理邏輯也隨之升級。

### A. 數據結構
- **輸入向量**：$x = (x_0, x_1)$，代表平面上的座標。
- **目標值**：$y \in \{0, 1\}$，代表該點所屬的類別。

### B. 2D 神經元數學模型
書中定義的神經元計算公式為：
$$y = H(w \cdot x) = H(w_0x_0 + w_1x_1)$$
其中 $H$ 為 Heaviside Step Function（階梯函數）。

### C. 決策邊界 (Decision Boundary)
- **幾何意義**：在 2D 空間中，決策邊界是一條直線。
- **通過原點**：由於目前的模型不包含偏置項 (Bias)，該直線方程式為 $w_0x_0 + w_1x_1 = 0$，因此**必定通過原點 $(0,0)$**。
- **斜率**：邊界的斜率由權重比值決定（斜率為 $-w_0/w_1$）。

---

## 2. 程式碼實作對照：`two_dim_prediction.py`

### A. 神經元函數實作
書中建議使用 NumPy 的矩陣乘法運算子 `@`：
```python
# 書中建議方式
return (x @ w > 0).astype(int)
```
而 `two_dim_prediction.py` 則採用了更顯式的點積計算法：
```python
# 腳本中的實作
def neuron_class_2d(w, x):
    return ((w[0] * x[:, 0] + w[1] * x[:, 1]) > 0).astype(int)
```
這兩種方式在數學上是完全等價的。

### B. 權重更新邏輯 (Training Loop)
書中提供的更新公式（基於 $error = y_p - y_{gt}$）：
$$w = w - \eta \cdot error \cdot x$$

`two_dim_prediction.py` 中的實作（基於 $error = y_{gt} - y_p$）：
```python
error = y_i - y_pred_i
w += learning_rate * error * x_i
```
這兩者在本質上是一致的（方向修正相同）。

### C. 數據視覺化
- **散點圖 (Scatter Plot)**：使用 `plt.scatter` 並配合顏色映射 (`cmap='bwr'`) 來區分不同的類別。
- **顏色條 (Colorbar)**：幫助直觀理解數值（0 或 1）與顏色的對應關係。

---

## 3. 關鍵發現與總結

| 特性 | 書中描述 | `two_dim_prediction.py` 實作 |
| :--- | :--- | :--- |
| **數據來源** | `data_class_2d_clean.csv` | 支持 `clean` 與 `noisy` |
| **矩陣計算** | 使用 `@` 運算子 | 使用 `w[0]*x[0] + w[1]*x[1]` |
| **訓練次數** | `num_train_iterations` | 10,000 Epochs |
| **偏置項** | **未包含** | **未包含** |
| **預測評估** | 觀察 Plot 圖形 | 計算 Test Accuracy + 繪圖 |

### 結論：
`two_dim_prediction.py` 忠實地呈現了書中關於 2D 單神經元分類的核心概念。它展示了如何透過簡單的線性組合與階梯函數，在二維空間中畫出一條「分類線」。雖然目前沒有偏置項導致靈活性受限（必須過原點），但這為後續引入 **Bias** 與 **多層網路 (MLP)** 奠定了直觀的基礎。
