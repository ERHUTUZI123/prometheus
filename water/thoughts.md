读Growing Isotropic Neural Celluar Atomata:


1

Isotropic使用了一个50%的随机异步更新机制来强行打破对称性
$$
M_{i,t}\sim\operatorname{Bernoulli}(0.5)
$$
使没有固定方向偏好的局部规则也能生成不对称图案。但我仍然怀疑，这种随机性的引入是否具有充分的生物学依据。为什么随机性应当表现为每个细胞独立地、以固定概率跳过更新？这种假设与真实细胞活动之间有什么对应关系？

2
分裂与分化是否可由共同的随机动力学解释？ 已有研究表明，随机波动可参与触发分化，分化前的细胞状态存在波动，细胞周期与分化也共享部分调控机制。这支持从统一动力学角度研究二者，(但尚不能证明它们只是同一随机过程的两种结果)

3
NCA：一次状态更新的数学过程与设计直觉

以下数字仅作演示，不是训练后的真实权重。设计直觉解释的是架构选择，不代表训练后每个隐藏单元都有明确含义。

## 1. 读取自己和周围 8 个邻居

每个细胞保存 16 个状态值。中心细胞读取的邻域形状为：

$$
3\times3\times16.
$$

按通道分别看，例如：

$$
R=
\begin{bmatrix}
0.1&0.2&0.3\\
0.1&\boxed{0.2}&0.3\\
0.1&0.2&0.3
\end{bmatrix},
\qquad
G=
\begin{bmatrix}
0.4&0.4&0.4\\
0.5&\boxed{0.5}&0.5\\
0.6&0.6&0.6
\end{bmatrix}.
$$

B、Alpha 和 12 个隐藏通道，也各有一个这样的局部矩阵。

**设计直觉：** 只看自己，无法与其他细胞协调；读取邻居，就可以根据周围情况改变自己。多个通道提供保存不同信息的空间。

## 2. 提取自身值、水平变化和竖直变化

使用两个 Sobel 核，对每个通道分别计算：

$$
K_x=\frac18
\begin{bmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{bmatrix},
\qquad
K_y=\frac18
\begin{bmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{bmatrix}.
$$

对于 R 通道：

$$
R_{\text{自身}}=0.2,
$$

$$
g_{x,R}
=
\frac{(0.3-0.1)+2(0.3-0.1)+(0.3-0.1)}8
=0.1,
$$

$$
g_{y,R}=0.
$$

对于 G 通道，同理得到：

$$
G_{\text{自身}}=0.5,
\qquad
g_{x,G}=0,
\qquad
g_{y,G}=0.1.
$$

| 通道 | 自身值 | 水平变化 | 竖直变化 |
|---|---|---|---|
| R | 0.2 | 0.1 | 0 |
| G | 0.5 | 0 | 0.1 |

**设计直觉：** 告诉细胞“我现在是什么状态”和“附近的状态朝哪个方向增加或减少”。用这三个数概括每个通道的局部情况，会舍弃部分邻域细节。

## 3. 拼成 48 维感知向量

16 个通道，每个贡献三个数。按“自身值、水平变化、竖直变化”分组拼接：

$$
p_i^t=
\begin{bmatrix}
x_i^t\\
(K_x*X^t)_i\\
(K_y*X^t)_i
\end{bmatrix}
\in\mathbb R^{48}.
$$

展开为：

$$
p_i^t=
\begin{bmatrix}
0.2\\
0.5\\
\vdots\\
\text{通道16自身值}\\[4pt]
0.1\\
0\\
\vdots\\
\text{通道16水平变化}\\[4pt]
0\\
0.1\\
\vdots\\
\text{通道16竖直变化}
\end{bmatrix}.
$$

三组各有 16 个数，共 48 个。

**设计直觉：** 把不同来源的信息放在一起，让后面的网络能够联合考虑它们。拼接本身只是排列数字，还没有作出更新决定。

## 4. 第一层：加权组合，再经过 ReLU

第一层计算：

$$
z_i^t=W_1p_i^t+b_1,
\qquad
h_i^t=\operatorname{ReLU}(z_i^t),
$$

其中：

$$
W_1\in\mathbb R^{128\times48},
\qquad
b_1\in\mathbb R^{128},
\qquad
h_i^t\in\mathbb R^{128}.
$$

为方便手算，假设前两个隐藏单元只使用 R、G 自身值，其余输入权重暂设为零：

$$
z_i^t=
\underbrace{
\begin{bmatrix}
1&2&0&\cdots&0\\
-1&0.2&0&\cdots&0\\
\vdots&\vdots&\vdots&\ddots&\vdots
\end{bmatrix}
}_{128\times48}
\underbrace{
\begin{bmatrix}
0.2\\
0.5\\
\vdots
\end{bmatrix}
}_{48\times1}
+
\underbrace{
\begin{bmatrix}
-0.3\\
0\\
\vdots
\end{bmatrix}
}_{128\times1}.
$$

前两个结果：

$$
z_{i,1}^t
=1\times0.2+2\times0.5-0.3
=0.9,
$$

$$
z_{i,2}^t
=-1\times0.2+0.2\times0.5
=-0.1.
$$

经过 ReLU：

$$
h_i^t=
\begin{bmatrix}
\max(0,0.9)\\
\max(0,-0.1)\\
\vdots
\end{bmatrix}
=
\begin{bmatrix}
0.9\\
0\\
\vdots
\end{bmatrix}.
$$

实际网络中，每个隐藏单元都可以结合全部 48 个输入，包括水平、竖直变化。

**设计直觉：** 用不同权重组合信息，形成对不同局部情况的响应。偏置调整门槛，ReLU 让响应具有条件性。

128 个中间值提供多种组合信息的机会；128 是容量选择，不是必须的数字。这里的“响应”仍只是中间数值，不是细胞已经执行了某个动作。

## 5. 第二层：把响应转换成 16 个状态增量

第二层计算：

$$
\Delta x_i^t=W_2h_i^t+b_2,
$$

其中：

$$
W_2\in\mathbb R^{16\times128},
\qquad
b_2\in\mathbb R^{16},
\qquad
\Delta x_i^t\in\mathbb R^{16}.
$$

继续使用简化权重：

$$
\Delta x_i^t=
\underbrace{
\begin{bmatrix}
0.1&-0.2&0&\cdots&0\\
-0.1&0.3&0&\cdots&0\\
\vdots&\vdots&\vdots&\ddots&\vdots
\end{bmatrix}
}_{16\times128}
\underbrace{
\begin{bmatrix}
0.9\\
0\\
\vdots
\end{bmatrix}
}_{128\times1}
+
\underbrace{
\begin{bmatrix}
0.01\\
0\\
\vdots
\end{bmatrix}
}_{16\times1}.
$$

前两个输出为：

$$
\Delta R
=0.1\times0.9-0.2\times0+0.01
=0.10,
$$

$$
\Delta G
=-0.1\times0.9+0.3\times0
=-0.09.
$$

因此：

$$
\Delta x_i^t=
\begin{bmatrix}
+0.10\\
-0.09\\
\vdots
\end{bmatrix}.
$$

**设计直觉：** 同一种中间响应，对不同状态通道可能需要产生不同影响。第二层权重决定各通道增加、减少多少。

最后一层不加 ReLU，让增量可以为负。128 个隐藏单元与 16 个状态通道没有一一对应关系。

## 6. 随机决定本步是否执行增量更新

每个细胞独立抽取一个随机更新掩码：

$$
m_i^t=
\begin{cases}
1,&\text{概率 }0.5,\\
0,&\text{概率 }0.5.
\end{cases}
$$

将增量乘以这个掩码：

$$
m_i^t\Delta x_i^t.
$$

同一个掩码控制该细胞全部 16 个通道的增量：

- 掩码为 1：执行本次增量更新。
- 掩码为 0：本次增量为零。

**设计直觉：** 让细胞不必每一步都同时更新，促使规则适应邻居更新节奏的不一致。

## 7. 将增量加回原状态

计算暂时的新状态：

$$
\widetilde x_i^{t+1}
=
x_i^t+m_i^t\Delta x_i^t.
$$

假设本步掩码为 1：

$$
\widetilde x_i^{t+1}
=
\underbrace{
\begin{bmatrix}
0.2\\
0.5\\
\vdots
\end{bmatrix}
}_{\text{原状态}}
+
\underbrace{
\begin{bmatrix}
0.10\\
-0.09\\
\vdots
\end{bmatrix}
}_{\text{增量}}
=
\begin{bmatrix}
0.30\\
0.41\\
\vdots
\end{bmatrix}.
$$

**设计直觉：** 保留已有状态，在它的基础上调整，而不是每一步都重新生成全部状态。网络也可以输出零增量来保持原状。

## 8. 应用存活掩码

记一个位置的存活判断为：

$$
A_i(X)
=
\mathbf 1
\left[
\max_{j\in\mathcal N(i)}\alpha_j>0.1
\right],
$$

其中邻域包含自身及周围 8 个位置。

检查更新前、后的邻域是否都满足条件：

$$
\ell_i^t
=
A_i(X^t)\,
A_i(\widetilde X^{t+1}).
$$

最终状态为：

$$
x_i^{t+1}
=
\ell_i^t\widetilde x_i^{t+1}.
$$

也就是：

$$
x_i^{t+1}
=
\begin{cases}
\widetilde x_i^{t+1},&\text{更新前后都满足存活条件},\\
\boldsymbol 0,&\text{否则，16 个通道全部清零}.
\end{cases}
$$

**设计直觉：** 让生长依托已有活细胞附近发生，并清除空白区域中的隐藏状态。

即使本步没有执行增量更新，也仍然要经过存活掩码。

## 9. 所有细胞反复执行同一个过程

完整的一步更新可以概括为：

$$
\underbrace{3\times3\times16}_{\text{读取邻域}}
\rightarrow
\underbrace{48}_{\text{感知向量}}
\rightarrow
\underbrace{128}_{\text{中间响应}}
\rightarrow
\underbrace{16}_{\text{状态增量}}
\rightarrow
\text{随机更新}
\rightarrow
\text{存活掩码}.
$$

所有位置、所有时间步共享同一套参数：

$$
\theta=\{W_1,b_1,W_2,b_2\}.
$$

但每个细胞的状态与邻域不同，因此可以产生不同的更新。更新后的状态又成为下一步邻居感知的输入。

**设计直觉：** 让整体结构来自局部交互的反复传播，而不是由中央控制器逐个指定细胞行为。

## 10. 通过训练学习共享规则

从初始状态运行多步后，将最终 RGBA 与目标图案比较，计算损失，再通过展开的计算过程反向传播，调整共享参数。

$$
X^0
\xrightarrow{F_\theta}
X^1
\xrightarrow{F_\theta}
\cdots
\xrightarrow{F_\theta}
X^T
\xrightarrow{\text{与目标比较}}
\mathcal L.
$$

**设计直觉：** 不手工规定每种局部情况应该怎样更新，而是通过整体结果的误差，寻找有用的局部规则。

需要区分：

- **运行时：** 细胞状态不断变化，共享参数通常保持固定。
- **训练时：** 根据多步运行的误差更新共享参数。

架构提供局部感知与调整的形式，训练寻找能产生目标行为的具体规则。