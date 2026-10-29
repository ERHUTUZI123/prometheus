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
NCA 的核心是：**用一个共享的小神经网络定义细胞的局部更新规则，再通过多步展开训练，让局部交互产生指定的整体图案。** 数学上，它是一个带随机更新的、空间局部耦合的递归动力系统。下面针对 Mordvintsev 等人的 [2020 年原论文](https://distill.pub/2020/growing-ca/)解释。

**1．每个细胞保存一个状态向量**

在二维网格上，位置 \(i\) 的细胞状态为

\[
x_i^t\in\mathbb R^{16}.
\]

前四维是 RGBA，其余十二维是可学习用途的隐藏状态，可以承担记忆、信号传递等功能。所有细胞共享参数，但状态可以不同。

**2．感知邻居，再决定自己的变化**

细胞读取自身状态，以及用固定 Sobel 核计算的两个方向梯度：

\[
p_i^t=
\begin{bmatrix}
x_i^t\\
(K_x*X^t)_i\\
(K_y*X^t)_i
\end{bmatrix}
\in\mathbb R^{48}.
\]

其中卷积只覆盖 \(3\times3\) 邻域。随后用同一个 MLP 计算增量：

\[
\Delta x_i^t
=
f_\theta(p_i^t)
=
W_2\operatorname{ReLU}(W_1p_i^t+b_1)+b_2,
\]

网络维度为 \(48\rightarrow128\rightarrow16\)。

每个细胞以概率 \(0.5\) 更新：

\[
\widetilde x_i^{t+1}
=
x_i^t+m_i^t\Delta x_i^t,
\qquad
m_i^t\sim\operatorname{Bernoulli}(0.5).
\]

这里一个随机变量控制该细胞全部通道。实现还使用存活掩码：只有更新前后邻域中都存在 \(\alpha>0.1\) 的细胞，才保留其状态，否则清零。这些细节可在[官方实现的 `CAModel`](https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)核对；此处链接是可变的 `master` 版本，未锁定提交。

**3．为什么局部规则能够生成整体结构？**

把整个网格写成一个大状态 \(X^t\)，上述操作就是

\[
X^{t+1}=F_\theta(X^t;m^t).
\]

迭代展开后：

\[
X^T=
F_\theta\bigl(
F_\theta(\cdots F_\theta(X^0;m^0)\cdots;m^{T-2});
m^{T-1}
\bigr).
\]

虽然每一步只发生局部交互，经过多步，信息可以逐渐传播到更远的位置。**空间上的整体组织，来自时间上的反复通信。**

从神经网络角度，可以把它理解为一个在时间上反复使用相同权重的残差卷积网络。

**4．训练的究竟是什么？**

从单个种子出发，运行随机步数 \(T\)，让最终 RGBA 接近目标图像 \(Y\)：

\[
\mathcal L(\theta)
=
\mathbb E_{T,m}
\left[
\frac{1}{4HW}
\left\|
P_{\mathrm{RGBA}}X^T-Y
\right\|_F^2
\right].
\]

梯度通过展开轨迹反向传播。忽略符号中的随机掩码，链式法则可写为

\[
\frac{\partial X^{t+1}}{\partial\theta}
=
\frac{\partial F_\theta}{\partial X^t}
\frac{\partial X^t}{\partial\theta}
+
\frac{\partial F_\theta}{\partial\theta}.
\]

所以，最终图像的误差会调整每一步共享的局部规则。原实现采用随机展开长度及逐参数张量的梯度归一化；硬阈值存活判断本身不提供可微梯度。[训练代码](https://github.com/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb)

**5．“长出来”与“自我修复”的区别**

只约束某个时刻接近目标，并不能保证之后保持稳定。

原论文因此引入状态池：把以前生成的状态继续作为训练起点；进一步对其中部分状态施加破坏，再训练其恢复。这使模型同时学习生长、维持和修复。[原论文实验 2、3](https://distill.pub/2020/growing-ca/#experiment-2)

用动力系统语言解释，这相当于希望目标外观附近形成一个吸引区域。但这是训练意图和实验现象，**不是收敛保证**。此外，外观稳定不要求全部隐藏状态停止变化；内部仍可能持续活动。

最关键的理解是：**NCA 学到的是一套随当前局部状态作出反应的演化规则；目标图案是这些规则反复运行后产生的结果。**