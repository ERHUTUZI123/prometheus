# Review 050 — value 约束成立，READ 的次概率接口仍待实测

**判断：后验投影确实把低 NLL 与真实递归 value 坐标相连；这项性质没有推出完整 carry 的运算闭包，也没有保证新 MIX 获取。** 已执行配对中，投影臂四阶段末的旧 long 较好，新任务 IID 却为24.61%，同后验记忆的 raw 臂为62.11%。不能把前者写成普遍保持优势，也不能用后者否定所有可学习 value 接口。当前有一个可判别的具体缺口：`h=pE` 使用概率质量1，`READ=(aV)E` 使用质量可小于1的次概率。它是源码推出的接口差异，**尚不是已观察到的失败原因或已选定修复**。

## 核查边界与实际证据

本 reviewer 读取当前 Method §3.7、实际 `continuous_primitive_block_posterior_value.py`、冻结训练 driver、原 stage3 数据语义及两臂完整 JSONL；独立重加每臂4×600条更新的时间与 MAC，检查 stage2／3 VALID 曲线和实际 TEST／资源记录。没有 SSH、模型／Torch执行、checkpoint反序列化或新实验。初始化、schedule、梯度数组与 checkpoint 的独立复核由另一 reviewer 完成，避免重复。

实际 source SHA 为 `d25a0ddda846ab0858873eca37e1ccb8269a083b312c2af2dc9ad6923b61af80`。配对证据见 [独立 raw 核查](../../probes/topology-architecture/actual-posterior-raw-fourstage-independent-raw-checks-v2.json)，SHA `715be4f7fd2e86564c3ab098b02866511e128640496395d53c341e5ac9a2bbb5`；完整成本／获取综述见 [review049](review049-posterior-value-acquisition-cost.md)。本次读取的 posterior／raw JSONL SHA 分别为 `71629e4de9b80aed5a0076aa9e50f4e04631e58fb7b4fde31692584e628ed59a`／`56fb5234b67a726841643059982c4141d756c38ccc0336bdfbe9e4b3e27909a2`。终端 driver 没有逐例 TEST 向量，准确率为来源、流程与计数已核查的 producer 结果，不称独立重算。

## SOURCE／RECONSTRUCTION：两个不同的值界限

源码先计算候选 \(\chi\)，再令 \(p=\exp\operatorname{logsoftmax}(W\chi+b)\)，缓存该 logp；仅投影臂取 \(h^+=pE\)，raw 臂取 \(h^+=\chi\)。两臂都将 p 写入后验槽，都将读到的后验乘 learned E。缓存输出没有重新解码投影 h；下一步 `_transition` 直接读取 h、controller、keys、values，不读取缓存 logp 来重建 h。

令 \(R=\sqrt H\)，固定参数的实数表达式中 \(\|E_s\|\le R\)。对投影后的 active transition，

\[
h-E_y=\sum_{s\ne y}p_s(E_s-E_y),\qquad
\boxed{\|h-E_y\|\le2R(1-p_y)}.
\]

这是 Method 的正确三角不等式推导，前提是该步的真实 y 与较大 p_y；正确 argmax 单独不提供小误差。零步初值仍是普通 encode(initial)／readout 缓存，没有这个后验 h 恒等式。它也是实数级约束，不是对全部 FP32 执行的逐位证明。参数更新后 E 与转移同时改变，不能从一次执行的界限推出能力保持。

源码的槽更新为 \(V_j^+=(1-\alpha_j)V_j+\alpha_jp\)，其中 \(0\le\alpha_j\le1\)，初始 \(V_j=0\)。因此每行均为次概率；读注意力为概率权重，故

\[
u=\sum_j a_jV_j\ge0,\quad m=\sum_su_s\in[0,1],\quad r=uE.
\]

READ 对目标值的正确界限是

\[
r-E_y=\sum_{s\ne y}u_s(E_s-E_y)-(1-m)E_y,
\]
\[
\boxed{\|r-E_y\|\le 2R(m-u_y)+R(1-m)=R(1+m-2u_y)}.
\]

若 \(m>0\)、\(q=u/m\)，该界等于 \(R[(1-m)+2m(1-q_y)]\)。所以 q 的类别正确、甚至 \(q_y=1\)，仍不能直接套用 h 的界。例如 \(u=.1e_y\) 时，argmax 完全正确，但 \(r=.1E_y\)、误差为 \(.9\|E_y\|\)；空读 m=0 则 r=0。**读取的类别、读取的质量、真正进入候选算子的向量幅度必须分开。** 同样，E 的范数界不保证不同原型的仿射可区分性；raw 已保存的 E／Gram 可以检查这一点，不能从范数归一化直接推定值编码唯一。

## 为什么这还不是完整 carry 闭包

carry 还含 controller、keys、其余槽与缓存输出。即使 h 相同，READ、CALL和后继候选仍可能因这些变量不同而不同。这有时是正确的上下文依赖，不能要求“同 h 必须在所有上下文同输出”来否定架构；真正需要的是：在**匹配逻辑控制与记忆条件**下，不同合法历史形成的同一值能否继续执行同一运算。当前范数界没有证明这个行为等价关系。

投影还把 h 限在最多7维的8原型凸包中，却保留16维 controller、连续 keys 与概率槽；这不是8个离散机器状态，也不能据此证明二元计算不可能。反过来，缓存 p 与端点分类正确也不保证 WRITE 保存了正确对象、READ 取回了正确对象，或低秩候选已学会组合两个值。只监督程序终点，更没有逐步保证每个中间 p 都校准。

## VERIFIED：收益与失败同时存在，不能混成保持结论

| 四阶段末 soft TEST，开发 seed0 | 后验投影 | raw h／同后验记忆 |
|---|---:|---:|
| 新 MIX IID／long／unseen |24.61%／17.38%／24.61%|62.11%／62.50%／62.50%|
| 旧0／1／2 long |38.87%／77.34%／73.63%|21.88%／52.15%／54.10%|
| 旧0／1／2 IID |85.16%／90.63%／84.96%|87.30%／89.45%／83.79%|

原始曲线显示：投影臂在 stage2末旧 VALID 为99.22%／100%／97.66%，终端 NLL 为.04565／.00871／.11650；不是前面所有任务都没学会。新 stage3 第600步 minibatch NLL 仍1.73382，最低也仅1.26607；新 VALID 准确率26.37%、NLL1.86341，raw 则62.50%、NLL.53162。新获取失败已在 TRAIN／VALID 出现，不只是 TEST 分布变化。

但两臂自 stage0 就执行不同投影，stage2权重和能力已不同，且 stage2未开 TEST。因此四阶段末 old-long 差异是有意义的已观察结果，**不是隔离了 stage3遗忘量的纯 retention contrast**；可能包含早期学习／长程归纳偏置差异。单个配对 seed 也不足以建立可复现广泛优势。低分保留，不设准确率门槛，不因为 raw 较好或投影较差继续调到满意数字。

本 reviewer 重加的2400次更新 wall 为29.738029s／29.253677s，CPU30.591556s／30.171580s；padding前向51.000115G／50.056397G MAC，相差.9437184G，恰为固定 \(N S H\) 投影项。whole driver59.890329s／60.763883s、本地coordinator66.043601s／66.877108s，不能把小的 whole-wall 反向差异称为投影效率优势。两臂相同13,397P、NN53,588B、Adam107,284B、replay55,296B+元数据；各实际2400更新、两次capture、6次warmup、每阶段诊断、维护／reload／TEST／inference与8份 safety/final checkpoint保留。checkpoint总4,758,184B／4,757,992B。248float／992B只是 N8逻辑carry；实际allocator peak约68.32MB、reserved约103–105MB，不能从缩小槽维度推定总内存优势。完整FLOPs、GPU busy time、driver/context memory、能耗与provider-effective compute仍未知。

## OPEN：先用已有冻结 operand audit 分流，再考虑一次因果干预

四检查点 VALID audit 已实际完成并同步原始 u、mass、E、attention以及 WRITE／READ／pre-MIX carry；其指标分析正在进行。正确前缀值只作离线评分，绝不进入网络、路由或训练。在核对实际指标后，该证据可以区分：

1. WRITE 时 p 已错：缺口先在值形成；读归一化不会恢复正确对象。
2. WRITE 正确、后续槽或 READ 条件类别错：优先检查写入干扰／寻址与控制；放大错误读取不会修复它。
3. 当前值与 READ 的条件类别正确，但 m 小、r 明显衰减：这是上述**幅度接口假设**的直接支持条件。
4. 两个语义操作数都对、质量接近1而 MIX仍错：问题更接近二元运算／控制选择的形成，不应继续归咎于 value 范数或仅调投影强度。

stage2只有15-token词表，截断 inactive MIX15 可以合法观察旧 WRITE／READ 与 pre-MIX carry，不能冒称已经执行不存在的 MIX-token-conditioned read／运算。E 的区分性、概率熵、mass和向量norm也应共同解释，不能只按 argmax 给接口“通过”。

**仅在基线诊断支持第3项后**，一个有后果而无新增参数的候选因果检查是：固定当前 NN 与同一 pre-MIX carry，只在最后一步把 READ 映射改成 \(r_{\rm norm}=[u/\max(m,\varepsilon)]E\)，空 u=0 仍为0；不改 θ、不注入真值、保留全部 VALID 行和原始负结果，固定 ε 并收费。只改最后一步可避免把上游写入轨迹变化混入 READ幅度效应。它会有放大错误／微小质量读的风险，故若真实 m≈1或 q 本来错误，不应选择这个修复。此处是待数据决定的诊断解释，**未实现、未执行、未证明成功，当前没有选择新的训练改动**。

`CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`低NLL约束递归value；READ幅度可能妨碍后继二元计算 | SOURCE/RECONSTRUCTION成立，失败原因OPEN | 本文源代码推导、真实配对raw/cost、已保存但指标待核查的冻结operand audit | h约束未到完整语义carry闭包；mass假设尚无观测支持 | 24.61%新获取失败与所有旧/新结果保留 | 先读冻结VALID操作数证据，再决定是否值得一次固定NN/零参数因果干预；不启动新训练或变更方向。`
