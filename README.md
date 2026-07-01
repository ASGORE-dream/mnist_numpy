# MNIST 手写数字识别 — 纯 Python + NumPy

从零实现一个完整的手写数字识别神经网络，**不使用任何深度学习框架**（PyTorch、TensorFlow 等），仅用 Python 和 NumPy。

## 网络结构

```
输入层 (784) → 隐藏层 (128, ReLU) → 输出层 (10, Softmax)
```

## 训练结果

| 指标 | 结果 |
|:----|:----:|
| 训练集准确率 | ~95.9% |
| 测试集准确率 | ~96.0% |
| 交叉熵损失 | ~0.15 |

## 使用步骤

### 1. 环境准备

安装 Python 3.8+ 和依赖库：

```bash
pip install numpy matplotlib pillow
```

### 2. 下载数据集

从 [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) 下载 `mnist_train.csv` 和 `mnist_test.csv`，放到项目目录下。

**CSV 格式说明：** 每行第一列为标签（0-9），后面 784 列为 28×28 像素值（0-255）。

### 3. 训练模型

```bash
python tslin.py
```

训练会运行 2000 个 epoch，结束后自动保存模型权重（`w1.npy`、`b1.npy`、`w2.npy`、`b2.npy`）。

### 4. 运行手写识别程序

```bash
python digit_recognizer.py
```

会弹出一个手写板窗口：
- 按住鼠标左键在黑色画板上写数字
- 点击 **"识别"** 查看预测结果和置信度
- 点击 **"清空"** 重新书写

## 项目文件说明

| 文件 | 说明 |
|:----|:----:|
| `tslin.py` | 训练脚本（数据加载 → 前向传播 → 反向传播 → 参数更新） |
| `digit_recognizer.py` | 手写数字识别 GUI（基于 tkinter） |
| `w1.npy` ~ `b2.npy` | 训练好的模型权重 |
| `mnist_train.csv` | 训练数据集（需自行下载） |
| `mnist_test.csv`  | 测试数据集（需自行下载） |

## 技术要点

- **前向传播：** 矩阵运算实现批量推理
- **激活函数：** ReLU（隐藏层）、Softmax（输出层）
- **损失函数：** 交叉熵损失（Cross-Entropy Loss）
- **反向传播：** 手动推导梯度，链式法则
- **优化方法：** 批量梯度下降（Batch Gradient Descent）

  如果您有什么问题的话，请直接私信联系我，我会尽量给出解答
  

## 许可证

MIT
