# Review 049 — VALUE 约束已有实际对照，旧长程序改善与新操作学习失败同时成立

**判断：本次 fresh seed0 的概率记忆配对实际完成了两臂各 2400 次更新。强制 `h=pE` 的 POSTERIOR 臂未改善新二元操作获取，显著弱于保留原候选值的 RAW 臂；它对三个旧任务的最终 long 准确率全部更高，这项正向局部结果必须保留。** 两者尚未建立广泛能力与完整资源优势。没有发现足以把新任务差距归为数据、采样、来源或 Graph 数值实现错误的具体证据；差距的内部原因仍需冻结检查点的 VALID 操作数／记忆诊断，不能靠追加训练、rank、seed 或新配置消除。

问题与 Fundamental Question 的连接是：把当前预测显式变成后续计算使用的 VALUE，能否帮助同一可训练 Block 保持、调用并组合已学操作。该连接是机制假设；“值处于某个有界集合”不等于已学会可继续使用的运算。本次不是对任意神经计算、概率反馈状态或长期研究方向的否定，也不是新颖性证据。

本 reviewer 读取冻结 core、Graph helper、engine、driver、protocol、coordinator receipts、逐更新 JSONL、原始 schedule／gradient NPZ 及 checkpoint 文件 SHA。独立核查只使用标准库，未导入 Torch／NumPy，未反序列化 NN，未执行模型、训练、SSH、编译或安装。主要可重算证据为 [配对 raw 检查](../../probes/topology-architecture/actual-posterior-raw-fourstage-independent-raw-checks-v2.json)，SHA `715be4f7fd2e86564c3ab098b02866511e128640496395d53c341e5ac9a2bbb5`，对应 [标准库 verifier](../../probes/topology-architecture/verify_posterior_value_pair_saved.py)，SHA `c584078a56efc9d23e98c6284303868545bb10c444d234ca3efbe4da2bf6642c`。[输入与更新有限性补充](../../probes/topology-architecture/actual-posterior-raw-input-update-supplement-v1.json)，SHA `71868255daba1369c6587ddfb0f347b828cd4f4d23df6bbbfbfba55662b6192a`，直接重算全部 20 个原始输入文件 SHA，并检查所有 4800 条实际更新的 NLL／preclip gradient 有限。

两次实际 namespace 为 `parameter-posterior-value-{posterior,raw}-n8-seed0-fourstage-graph-v2`。它们使用相同八份科学源码：core `d25a0ddd…61af80`、旧 fcba `fcba2e82…88158`、driver `ec75f714…881f1`、masked helper `7151938b…5ffb4`、engine `3c564c8d…f65e4` 及原 parent／replay／growth 源码；本地冻结 bytes 与两次记录的完整 SHA 映射吻合。两臂记录的初始化 θ 摘要均为 `213c1c0cda65eefd4b06de958d56c349e047cd319681e26e25b5e8a5756f5aad`；源码确为相同 seed0、相同旧 27 tensor 构造顺序、fresh NN、0 parent restore、0 shared historical updates。这里核实的是原始初始化摘要与构造路径，没有把未保存的初始 tensor payload 冒充本 reviewer 重新生成的权重。

所有四份 current index 数组（每份 600×64）与三份 replay draw 数组（每份 600×16）的原始 payload SHA 跨臂完全一致。实际 20 个 TRAIN／VALID／TEST 文件 SHA 也逐一相同，并与旧数据源文件及冻结 manifest 匹配。每阶段恰有 step1…600，stage0 为 64 current，后续为 48 current+16 Replay128；每臂总共 124,800 current 与 28,800 replay label reads。原 `SeedSequence([0,8,stage,9027])`、终端 mean NLL、Adam .003、clip1、600步预算没有因质量改变。每阶段 VALID 为 0／100…600；全部 12 次 TEST 文件打开均在最后一次更新之后，存在 2400-update freeze 记录，且 NN 在 TEST／inference 后未变。四阶段各自维护前 safety 与维护后 checkpoint 的实际 bytes／SHA、reload、Replay seen／draws／来源核查均吻合。15→16 token 的原始一次随机行增长保留旧权重与 Adam 状态，未提供新 token 语义。

实现对照是明确的：两臂都计算旧候选值 χ、`logp=logsoftmax(readout(χ))`、`p=exp(logp)`；两臂都把概率向量写入 8×8 的 Vp，读取 `u=attention@Vp` 后得到 `uE`，并从同一缓存输出 logp。区别是 POSTERIOR 令后续 h=pE，RAW 令 h=χ。RAW 按事前要求不计算丢弃的 pE，计算费用自然略低。公开 step、sequence、embedded 与 all-output 路径均返回缓存，未对投影后的 h 二次 decode；checkpoint 固定 mode 并拒绝跨 mode。新架构从 stage0 起训练，旧 Graph baseline 不可冒充这个单变量对照。

Graph 方面，每臂恰有 B64L16 与 B64L48 两次 capture；前者复用前三阶段，增长前销毁，再构建后者。图内原位清零已分配 grad，clip／Adam 在图外，27 项 grad storage 在各 capture 内稳定。独立解码每臂 28 对 reference／graph 原始文件，共 **389,144 对标量／臂，778,288 对标量总计**；output、loss 和全部 27 项 gradient 均通过原定 output atol／rtol=2e-5／2e-5、gradient=2e-5／2e-4。POSTERIOR 最大 output／gradient 绝对差为 3.337860e-6／3.516674e-6，RAW 为 5.722046e-6／7.152557e-7。这些是每阶段固定首个 TRAIN batch 在多个当前 NN 状态下的检查，不是全部 batch、权重轨迹相等或任意权重的浮点定理。所有已记录 update 的 loss 与梯度范数均有限。

| 最终 soft TEST 任务 | POSTERIOR IID／long／unseen | RAW IID／long／unseen |
|---|---:|---:|
| old0 | .851563／.388672／.671875 | .873047／.218750／.902344 |
| old1 | .906250／.773438／.378906 | .894531／.521484／.222656 |
| old2 | .849609／.736328／.257813 | .837891／.541016／.208984 |
| new binary | .246094／.173828／.246094 | .621094／.625000／.625000 |

这些准确率是来源、冻结与调用流程已核实的实际 producer aggregates。原 driver 未保存逐例 TEST 预测向量，所以本 review 不宣称独立逐例重算准确率。POSTERIOR 对旧 long 的改善分别为 16.99、25.20、19.53 个百分点，不能因新任务失败而删去；旧 IID／unseen 则有混合取舍。也不能直接命名为“保持优势”：两臂从 stage0 已有不同学习轨迹，stage2 TEST 正确保持未打开，最终 long 差异混合了先前获取、泛化和后续学习影响。

VALID 可看清差距形成的阶段。POSTERIOR stage2 最后 old0／1／2 已达 .992188／1／.976563，RAW 为 .9375／.972656／.9375。开始 stage3 时新 VALID 两臂均为 .125；POSTERIOR 在 step100／200／300／400／500／600 为 .128906／.125／.132813／.138672／.177734／.263672，RAW 为 .169922／.400391／.578125／.529297／.617188／.625。POSTERIOR 能学前三阶段，但本次预算下新二元组合获取较弱；不能解释成所有阶段都不可学，也不能把 RAW 的有限成功升级为完整执行闭包。

**数学界限与其缺口。** 固定某个 NN 权重时，E 的各行范数不超过 √H，POSTERIOR 每个 active transition 后严格有 h=Σp_j E_j。因此 `p_y≥1−ε ⇒ ||h−E_y||≤2√H ε`；空序列初始 h=E(initial)、普通初始 readout 缓存不要求这一等式。界限是给定正确高置信度预测后的 VALUE 接口性质，不使预测正确、不保证 context／key／memory 等价，也不保证随后任意操作保持结果。当前新任务低置信／错误预测没有违反这个条件界限。

还有一个可区分的具体接口：零初始化加凸写使每行 Vp 质量至多 1，读取 u=attention@Vp 仍是 subprobability。令 m=Σu_j≤1，则即使归一化条件分布 u/m 的 argmax 正确，读值仍可能被质量 m 衰减。对目标 y，`||uE−E_y||≤√H[(1−m)+2(m−u_y)]`。这表明“h 的预测正确”或“READ argmax 正确”均不足以推出后续 CALL 获得完整幅度的操作数。它是独立源码推导，尚不是实际失败原因；必须查看四个冻结 NN 在同一 VALID 程序上的写入、读取质量、操作数与 MIX 前后输出。

另一个竞争解释是表示／可塑性瓶颈：h 被限制在八个 prototype 的凸包（仿射维度至多 7），而 RAW 保留 32 维候选值；相同参数数量并不意味着相同有效状态容量。概率映射也改变 recurrent Jacobian，可能舍弃对当前 label 无用、对随后操作有用的信息。controller16、keys128 与 Vp64 仍可携带历史，因此这不是整个 carry 只能表达八种状态的证明，也不是二元运算不可能性定理。记忆绑定、覆盖／擦除及候选 χ 的组合形成仍是竞争机制。此前实际两 NN 的初始 E→decoder 均为 8/8 identity，故不能再把这次结果归于未经观察的初始 codec 错配。

| 每臂实际费用 | POSTERIOR | RAW |
|---|---:|---:|
| stage0／1／2／3 更新 wall s | 5.347298／5.243464／5.191222／13.956044 | 5.270732／5.167188／5.161692／13.654065 |
| 2400 更新 wall／CPU s | 29.738029／30.591556 | 29.253677／30.171580 |
| science wall／CPU s | 59.890329／59.867619 | 60.763883／60.739235 |
| remote coordinator／local SSH wall s | 64.806760／66.043601 | 65.717504／66.877108 |
| padded／active forward MAC proxy | 51.000115G／24.996867G | 50.056397G／24.534690G |
| CUDA peak allocated／reserved B | 68,319,744／104,857,600 | 68,303,360／102,760,448 |
| CPU peak RSS B | 1,657,249,792 | 1,643,720,704 |
| 8 safety+final checkpoint B | 4,758,184 | 4,757,992 |

每臂实际训练 3,686,400 个 padded transitions，对应 1,805,378 个 active transitions，padding 比为 2.042×；不能用 active MAC 代替已执行算术。两臂差额 943,718,400 padded MAC 恰为 transitions×8×32 的 pE 成本，POSTERIOR 更新时间本次高 1.66%，单次 host 测量不构成稳定吞吐定律。两臂合计 4800 SGD、58.991706s 更新、120.654212s science、132.920709s local SSH wall，分别收费；共同 CPU admission 的实际 local10.647296s 也保留，它只有丢弃的 synthetic optimizer 检查、0 research SGD。冻结 v1 未执行，v2 为事前 metadata 名称修正，不能算成失败训练或已训练额外 arm。

setup 是 0 SGD，不能从成本中删除。POSTERIOR L16／L48 setup 为 1.143481／1.179944s，RAW 为 1.053941／1.397066s；其中 warmup 合计 .763717／.842261s、capture 描述 .582304／.607358s，instantiate 也在实际 ledger 中。每臂有 2400 training、6 warmup、28 reference+28 graph diagnostic，即 **2462 次实际 F/B**；两次 capture 描述本身 0 次 GPU F/B。额外 62 次 F/B 对应 3968 个终端 label-example exposures，均无 optimizer update。27 项诊断、输入 copy、NN／Adam snapshot、维护、增长、reload、序列化都在 whole science 中；嵌套时间不能再次相加。VALID 合计 6.955532／7.442462s，TEST 8.688139／9.211707s，另付 inference 2.288427／2.337528s。

最终两臂 P 均为 **13,397**，NN53,588B、Adam107,284B、registered buffer0、Replay128 原始数组55,296B加 metadata1739B；persistent grad 最后53,588B，L16阶段53,524B。carry 是 248 FP32／例（992B），旧 Vh32 core 为432（1728B）；这项公共减少属于两臂共同的概率记忆改动，不能用于解释 POSTERIOR 对 RAW 的效果。它仍高于 GRU32 的64 FP32 recurrent state。两幅图 static inputs 为9728／26112B，private pool 的精确 byte 数未知；表中峰值已取每臂 capture 前后／阶段／final 共11处已记录 allocator peak 的最大值，包含实际 audit／reload开销，不能用裸 NN bytes 替代。

每臂另有约157,016B冻结科学 source、共享原始20输入文件290,357B；28份梯度 NPZ 分别3,310,144／3,301,471B，七份 sampler NPZ 各395,407B，learn目录实际 payload 合计9,877,421／9,868,979B。这是逻辑文件 payload，未假装是物理磁盘去重、带宽或 provider 账单。最终 POSTERIOR NN575,085B／SHA `758b2cd42205ab2cabe05f1405c4e29553231d7d8779a0d22d564c26889a06ef`；RAW575,021B／SHA `1b865db347952acbaae4234542689661a616ed10cf0ce580a773955d954ec9d2`。原有旧版本训练、失败与审计费用仍属 campaign 历史，不能由这次 fresh Graph pair 冲销。

两臂使用未修改的旧 GPU runner `85e8ae4f…13d0` 和原900s外限／850s内限、UID65534、文件／网络隔离与 .3 GPU allocator 配额，无 compiler exception 或新权限。这里的 forward MAC 只覆盖声明的线性／点积／blend proxy，不含完整 nonlinear、backward、optimizer FLOPs；GPU busy、完整 driver/context memory、private pool exact bytes、energy 与 provider-effective compute 仍未知。相较实际 fresh GRU32 的14,024P与17.281469s更新，两臂 P 略低但更新时间更高；任务质量与完整生命周期仍有取舍，不支持共同资源胜利，更不支持 Primary 冻结或 campaign 完成。

`CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`显式预测→VALUE耦合帮助同一Block后续计算 | PARTIAL/ADVERSE：本次POSTERIOR旧long更好、新二元获取更弱；条件VALUE界限成立但全carry闭包未建立 | 两个fresh2400 actual/source/raw/cost链 | 没有逐例TEST向量；内部失败位置待冻结VALID操作数/记忆诊断；单seed不外推 | 旧桥负alpha、初始codec错配反证、所有旧版本/失败均保留，不追加强度/seed/rank训练 | 读取已冻结POST/RAW stage2+stage3四NN在相同VALID程序的操作数、read/write质量与MIX形成证据，再决定一个机制修复；此review不启动新执行、不改Method/NOW。`
