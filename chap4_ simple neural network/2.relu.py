import numpy as np
import random
import matplotlib.pyplot as plt


class Add:
    """加法运算层（支持广播）"""

    def __init__(self):
        self.mem = {}  # 用于存储前向传播的输入，供反向传播使用

    def forward(self, x, b):
        '''
        前向传播：计算 x + b（支持广播）
        参数：
            x: 形状(N, d) 输入数据（一批向量）
            b: 形状(d,) 或 (1, d) 偏置项
        返回：
            x + b 的结果
        '''
        h = x + b  # 元素级加法（自动广播）
        self.mem = {'x': x, 'b': b}  # 保存输入供反向传播使用
        return h

    def backward(self, grad_y):
        '''
        反向传播：计算加法层的梯度
        参数：
            grad_y: 形状(N, d)，上游传来的梯度
        返回：
            grad_x: 对x的梯度
            grad_b: 对b的梯度
        '''
        x = self.mem['x']
        b = self.mem['b']

        # x的梯度直接等于上游梯度（因为∂(x+b)/∂x = 1）
        grad_x = grad_y.copy()

        # b的梯度需要沿batch维度求和（因为b被广播到所有样本）
        grad_b = np.sum(grad_y, axis=0)

        return grad_x, grad_b


class Matmul:
    """矩阵乘法层"""

    def __init__(self):
        self.mem = {}  # 存储前向传播的输入

    def forward(self, x, W):
        '''
        前向传播：计算矩阵乘法 x @ W
        参数：
            x: 形状(N, d) 输入数据
            W: 形状(d, d') 权重矩阵
        返回：
            矩阵乘积结果
        '''
        h = np.matmul(x, W)
        self.mem = {'x': x, 'W': W}  # 保存输入和权重
        return h

    def backward(self, grad_y):
        '''
        反向传播：计算矩阵乘法的梯度
        参数：
            grad_y: 形状(N, d')，上游传来的梯度
        返回：
            grad_x: 对x的梯度
            grad_W: 对W的梯度
        '''
        x = self.mem['x']
        W = self.mem['W']

        # 根据链式法则，x的梯度 = 上游梯度 @ W的转置
        grad_x = grad_y @ W.T

        # 根据链式法则，W的梯度 = x的转置 @ 上游梯度
        grad_W = x.T @ grad_y

        return grad_x, grad_W


class Relu:
    """ReLU激活层"""

    def __init__(self):
        self.mem = {}  # 存储前向传播的输入

    def forward(self, x):
        '''
        前向传播：计算ReLU激活函数 max(0, x)
        参数：
            x: 任意形状的输入
        返回：
            逐元素应用ReLU的结果
        '''
        self.mem['x'] = x  # 保存输入用于反向传播
        return np.where(x > 0, x, np.zeros_like(x))  # 逐元素ReLU

    def backward(self, grad_y):
        '''
        反向传播：计算ReLU的梯度
        参数：
            grad_y: 与x同形，上游传来的梯度
        返回：
            grad_x: 对x的梯度
        '''
        x = self.mem['x']

        # 梯度规则：如果x>0则传递上游梯度，否则为0
        mask = (x > 0).astype(np.float32)
        grad_x = grad_y * mask
        return grad_x


class myModel:
    """两层ReLU神经网络模型"""

    def __init__(self):
        # 初始化权重和偏置（随机正态分布）
        self.W1 = np.random.normal(size=[1, 60])  # 输入层到隐藏层的权重
        self.W2 = np.random.normal(size=[60, 1])  # 隐藏层到输出层的权重
        self.b1 = np.random.normal(size=[60])  # 隐藏层偏置
        self.b2 = np.random.normal(size=[1])  # 输出层偏置

        # 初始化各层运算
        self.mul_h1 = Matmul()  # 第一个矩阵乘法层
        self.mul_h2 = Matmul()  # 第二个矩阵乘法层
        self.add_h1 = Add()  # 第一个加法层（隐藏层）
        self.add_h2 = Add()  # 第二个加法层（输出层）
        self.relu = Relu()  # ReLU激活层

    def forward(self, x):
        '''前向传播'''
        # 第一层：x @ W1 + b1
        self.h1 = self.mul_h1.forward(x, self.W1)
        self.h1_add = self.add_h1.forward(self.h1, self.b1)

        # ReLU激活
        self.h1_relu = self.relu.forward(self.h1_add)

        # 第二层：h1_relu @ W2 + b2
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_add = self.add_h2.forward(self.h2, self.b2)

    def backward(self, label):
        '''反向传播（自动微分）'''
        # 输出层加法反向传播
        self.h2_add_grad, self.b2_grad = self.add_h2.backward(label)

        # 输出层矩阵乘法反向传播
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_add_grad)

        # ReLU反向传播
        self.h1_relu_grad = self.relu.backward(self.h2_grad)

        # 隐藏层加法反向传播
        self.h1_add_grad, self.b1_grad = self.add_h1.backward(self.h1_relu_grad)

        # 隐藏层矩阵乘法反向传播
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_add_grad)


def mse_loss(y_true, y_pred):
    """计算均方误差损失"""
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true, y_pred):
    """计算均方误差对预测值的梯度"""
    return 2 * (y_pred - y_true) / len(y_true)


def train_one_step(model, x, y):
    '''单步训练（前向+反向+参数更新）'''
    # 前向传播
    model.forward(x)

    # 反向传播（计算梯度）
    model.backward(mse_gradient(y, model.h2))

    # 梯度下降更新参数
    learning_rate = 5e-4  # 学习率
    model.W1 -= learning_rate * model.W1_grad
    model.W2 -= learning_rate * model.W2_grad
    model.b1 -= learning_rate * model.b1_grad
    model.b2 -= learning_rate * model.b2_grad

    # 计算并返回当前损失
    loss = mse_loss(model.h2, y)
    return loss


def test(model, x, y):
    '''在测试数据上评估模型'''
    model.forward(x)
    loss = mse_loss(model.h2, y)
    return loss


def my_func(x):
    '''要逼近的目标函数'''
    if x < -3:
        return x ** 2 - 20  # x<-3时为二次函数
    elif x < 3:
        return x + 6  # -3≤x<3时为线性函数
    else:
        return -x**2 + 5  # x≥3时为二次函数


def get_data(num, batch_size):
    '''生成训练数据'''
    data_x = []
    data_y = []
    for _ in range(num):
        # 生成-10到10之间的随机x值
        x = np.array([random.uniform(-10, 10) for _ in range(batch_size)])
        # 计算对应的y值（使用目标函数）
        y = np.array([my_func(item) for item in x])
        # 调整形状为(1, batch_size)
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)
        data_y.append(y)
        data_x.append(x)
    return data_x, data_y


def draw_graph(x_test, y_true, y_pred):
    '''绘制真实函数与预测结果的对比图'''
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_true, 'b-', label='True function')
    plt.plot(x_test, y_pred, 'r--', label='Predicted')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Approximation with Two-Layer ReLU Network')
    plt.grid(True)
    plt.show()


def main():
    # 初始化模型
    model = myModel()

    # 生成训练数据（8000批，每批10个样本）
    train_data, train_label = get_data(8000, 10)

    # 训练循环
    for epoch in range(8000):
        # 在单批数据上训练
        loss = train_one_step(model, train_data[epoch].T, train_label[epoch].T)
        print('第', epoch, '轮训练，损失值:', loss)

    # 生成测试数据用于可视化（-10到10的密集采样）
    arr = np.arange(-10, 10 + 0.01, 0.01)
    matrix = arr.reshape(1, -1)

    # 获取模型预测结果
    model.forward(matrix.T)

    # 计算真实值和预测值
    y_true = np.array([my_func(item) for item in arr])  # 真实函数值
    y_pre = model.h2_add.T.flatten()  # 模型预测值

    # 绘制结果对比图
    draw_graph(arr, y_true, y_pre)


if __name__ == "__main__":
    main()