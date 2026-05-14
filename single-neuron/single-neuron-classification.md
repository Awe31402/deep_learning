# Classifying Data with a Single Neuron

本章節介紹了深度學習中最基本的構建單元：**人工神經元 (Artificial Neuron)**，並展示了如何使用它來處理二元分類問題。

## 1. 核心觀念：生物學啟發
人工神經元模仿生物神經元的運作：
- **輸入 (Input)**：對應生物神經元的樹突接收的刺激信號。
- **活化電位 (Activation Potential)**：神經元內部對信號的加權處理。
- **活化函數 (Activation Function)**：決定神經元是否發射信號（輸出）。

## 2. 數學模型
神經元的運算過程分為兩個階段：

### A. 計算加權總和
$$p = w \cdot x + b$$
- $x = (x_0, ..., x_{N-1})$：輸入向量。
- $w = (w_0, ..., w_{N-1})$：權重向量，代表輸入的重要性。
- $b$：偏置 (Bias)，增加分類邊界的靈活性。

### B. 活化函數 (Heaviside Step Function)
$$y = H(p)$$
- 如果 $p > 0$，則 $y = 1$。
- 如果 $p \le 0$，則 $y = 0$。

## 3. 數據處理與視覺化 (1D 範例)
在 1D 數據分類中，神經元的目標是在數線上找到一個點，將數據分為兩類。

### 加載數據 (`loader.py`)
```python
import csv
from numpy import asarray

def load_data_1d(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = asarray(data).astype(float)
    x = data[:, 0]  # 輸入特徵
    y = data[:, 1]  # 地面真值 (Ground Truth)
    return (x, y)
```

## 4. 神經元的訓練演算法
訓練的本質是透過迭代調整 **權重 $w$** 和 **偏置 $b$**，使預測值 $y_p$ 接近地面真值 $y_{gt}$。

### 訓練步驟：
1. **初始化**：隨機設定權重與偏置。
2. **選擇樣本**：從數據集中隨機挑選一個點 $x$。
3. **預測**：計算 $y_p = H(w \cdot x + b)$。
4. **評估與更新**：
   - 如果預測正確 ($y_p = y_{gt}$)，則不變。
   - 如果預測錯誤 ($y_p \neq y_{gt}$)，按以下公式更新：
     $$w = w - \eta(y_p - y_{gt})x$$
     其中 $\eta$ 為 **學習率 (Learning Rate)**。

## 5. 重點摘要與限制
- **線性可分性 (Linear Separability)**：單一神經元只能處理可以用直線（或超平面）切開的數據。對於「非凸」(Non-convex) 或交錯的數據，需要多層神經網路。
- **學習率 $\eta$**：
    - 太大：可能跳過最小值（震盪）。
    - 太小：收斂過慢。
- **決策邊界**：神經元的權重決定了決策邊界的斜率，偏置則決定了位置。
