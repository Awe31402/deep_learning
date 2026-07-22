# 誤差反向傳播 (Backpropagation) 數學推導總結

根據《Volpe Deep Learning Crash Course》第一章的內容，反向傳播演算法的核心在於利用 **連鎖律 (Chain Rule)** 來計算梯度。

## 1. 設定與損失函數
考慮一個兩層神經網路（隱藏層 + 輸出層）：
- **輸入**: $x_n$
- **隱藏層輸出**: $y_{1,q}$ (活化電位為 $p_{1,q}$)
- **輸出層輸出**: $y_{2,m}$ (活化電位為 $p_{2,m}$)
- **目標值**: $\tilde{y}_m$

損失函數使用 **均方誤差 (MSE)**:
$$E = \frac{1}{2} \sum_{m=0}^{M-1} (y_{2,m} - \tilde{y}_m)^2$$

---

## 2. 輸出層權重更新 ($w_{2,qm}$)
目標是計算 $\frac{\partial E}{\partial w_{2,qm}}$。透過連鎖律：
$$\frac{\partial E}{\partial w_{2,qm}} = \frac{\partial E}{\partial y_{2,m}} \cdot \frac{\partial y_{2,m}}{\partial p_{2,m}} \cdot \frac{\partial p_{2,m}}{\partial w_{2,qm}}$$

### 各分量推導：
1. **誤差**: $\frac{\partial E}{\partial y_{2,m}} = (y_{2,m} - \tilde{y}_m)$
2. **活化函數導數**: $\frac{\partial y_{2,m}}{\partial p_{2,m}} = f'(p_{2,m})$
3. **輸入信號**: $\frac{\partial p_{2,m}}{\partial w_{2,qm}} = y_{1,q}$

定義 **誤差信號 (Error Signal)** $\delta_{2,m}$:
$$\delta_{2,m} = (y_{2,m} - \tilde{y}_m) f'(p_{2,m})$$

權重更新公式：
$$\Delta w_{2,qm} = -\eta \cdot \delta_{2,m} \cdot y_{1,q}$$

---

## 3. 隱藏層權重更新 ($w_{1,nq}$)
這是「反向傳播」名稱的由來。隱藏層的輸出影響了所有輸出層神經元，因此誤差必須從輸出層傳回：
$$\frac{\partial E}{\partial w_{1,nq}} = \frac{\partial E}{\partial y_{1,q}} \cdot \frac{\partial y_{1,q}}{\partial p_{1,q}} \cdot \frac{\partial p_{1,q}}{\partial w_{1,nq}}$$

### 反向傳導誤差 $\frac{\partial E}{\partial y_{1,q}}$:
$$\frac{\partial E}{\partial y_{1,q}} = \sum_{m} \frac{\partial E}{\partial p_{2,m}} \cdot \frac{\partial p_{2,m}}{\partial y_{1,q}} = \sum_{m} \delta_{2,m} \cdot w_{2,qm}$$

定義隱藏層誤差信號 $\delta_{1,q}$:
$$\delta_{1,q} = f'(p_{1,q}) \cdot \sum_{m} (\delta_{2,m} \cdot w_{2,qm})$$

權重更新公式：
$$\Delta w_{1,nq} = -\eta \cdot \delta_{1,q} \cdot x_n$$

---

## 4. 核心直覺總結
1. **反向流動**: 誤差從輸出層開始計算 ($\delta_{2,m}$)，然後根據連接權重 ($w_{2,qm}$) 按比例分配回隱藏層，形成 $\delta_{1,q}$。
2. **統一形式**: 每一層權重的梯度 = **(該層的誤差信號 $\delta$) × (該層的輸入信號)**。
3. **必要條件**: 活化函數 $f(p)$ 必須是可微的，因為推導中每一層都需要用到 $f'(p)$。
