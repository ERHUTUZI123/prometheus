# Review 043 — continuous core：已获取的能力、实际成本与尚未识别的机制

**判断：首阶段能力获取成立；动态 CALL 的独占收益、广泛强基线优势和长期保持收益均未成立。** 连续版本完成了此前 categorical 版本未完成的任务，但同参数 static 也完成了。字符 LM 的长度稳定性是真实观测；匹配参数的 GRU 更准、更快。真实特征任务中 dynamic 的 TRAIN 明显不足；相同16维前端的 static Speech 已显著改善，因此既不能只解释成 VALID 过拟合，也不能把 dynamic/static 差距单独归因于前端维数。上述结果约束当前版本的主张，不关闭 campaign，也不要求 Monodromy 留在 Primary。Primary 尚未冻结。

## 实际审阅范围与独立性

本次逐条读取四个 context、四个字符 LM、四个原始真实特征训练及新增六个真实控制 run 的 `records.jsonl`、最终报告和冻结源码；核对了四组 600 步、十四组 300 步的连续实际 update 序列，共 6,600 次历史 update。18 个最终 checkpoint 的实际文件大小/SHA 与回执吻合，每个 configuration 引用的冻结源码 SHA 也吻合。原始真实数据四个 checkpoint 另经受限 inert ZIP/pickle 描述符读取，确认 `completed_updates=300`、实际参数量、adapter 形状和 core 配置；未导入 Torch/NumPy 或执行保存的模型代码。新增六个控制的同模态 feature metadata SHA、batch SHA 与原始两臂一致。

四个独立字符推理审计的冻结源码及审计 driver SHA 吻合；其 checkpoint SHA 均对应原训练的最终文件。重新读取、校验了 18 个 per-target NPZ，重算 NLL，并确认跨模型、执行模式和三种长度的 512 个 target/position 完全相同；重算了全部 432 个记录的 latency draw 的 median。四个审计的 before/after model-state hash 相同，记录 `no_training=true`。这里没有重新运行模型推理、训练或 SSH；冷重载和实际 forward 一致性属于已执行审计的证据，而非本审阅重新实现的一次复制。

作者关系披露：我参与了旧 categorical 模块、driver 集成和接续检查器；连续 core 由另一 worker 实现。独立核对原始字节、算术和反例边界可以发现错误，不等于独立代码作者或跨实现复现。

主要证据目录：

- [dynamic context](../../probes/topology-architecture/runs/parameter-context-n8-seed0-dynamic-stage1-v1-soft_only-continuous/learn/)、[static context](../../probes/topology-architecture/runs/parameter-context-n8-seed0-static-stage1-v1-soft_only-continuous/learn/)、[GRU44](../../probes/topology-architecture/runs/parameter-context-n8-seed0-gru-stage1-v1-width44/learn/)、[TF32](../../probes/topology-architecture/runs/parameter-context-n8-seed0-transformer-stage1-v1-width32/learn/)；对应 `versions/code-同名run/`。
- 字符训练：`runs/parameter-char-{block-300-v1-continuous,block-300-v1-continuous-static,transformer-300-v1-width36,gru-300-v1-width44}/learn/`；推理审计为各自后接 `-inference-audit-v1/audit/`。
- 真实训练：`runs/parameter-real-{cifar,speech}-{block_dynamic,block_static,transformer,gru,cnn}-300-v1/learn/`；对应 `versions/code-同名run/`。

context/字符的连续源码是 `b8a51a6c…`；真实任务增加 `forward_embedded` API 的冻结源码是 `8deca750…`，真实 driver 是 `857969e7…`。不能以当前 shared 文件替换这些证据的实际来源。

## 1. 首阶段获取：成功，但还不是持续学习证据

全部 seed 0、N=8、600 次更新、无路由/操作/中间状态教师。Block 是 soft-only 训练，native 只硬化 CALL，E/R、状态、primitive 函数和内容记忆仍连续。下表为 soft/native；GRU/TF 两列相同只是同一普通神经接口，不能算两种独立执行验证。

| 模型 | P | TRAIN | VALID | TEST IID | TEST unseen | TEST long |
|---|---:|---:|---:|---:|---:|---:|
| continuous dynamic | 28,365 | .99683/.96924 | 1/.97656 | .99414/.95117 | .85547/.80273 | .60938/.39453 |
| continuous static | 28,365 | .99829/.93091 | 1/.93359 | .99805/.92383 | .96484/.71875 | .73438/.45313 |
| GRU44 | 25,572 | 1/1 | 1/1 | 1/1 | .93555/.93555 | .48438/.48438 |
| Transformer32 | 26,728 | .74805/.74805 | .68359/.68359 | .65039/.65039 | .29492/.29492 | .13477/.13477 |

**VERIFIED：** 连续 Block 在这个有限训练预算下取得有用执行能力，且 soft 长序列结果有相对这些匹配尺寸控制的优势点。**未建立：** 跨 seed、跨任务的普遍参数/计算优势；native 的同等长序列优势；后续学习后的保持。动态 native IID 比 static 高，但 static soft 的 IID/unseen/long 均不差，不能选择某一模式/指标来宣布动态 CALL 独占增益。

源码中的 static 只把 CALL 的 `h/read` 输入置零；primitive 本身仍读 `[h,x,read]`，内容记忆仍执行同样的读写。因此它是一个强的“去掉显式 contextual CALL 输入”的对照，不是“整个系统不依赖上下文”。其成功不证明状态/记忆无用；它确实阻止把整个连续架构的成绩归因于 dynamic caller。

## 2. 字符 LM：匹配参数与 hard-CALL 的实际结果

四组均是同一 300×64 minibatch 预算、LR .003、TRAIN 长度 32、soft 训练、512 个配对 TEST 目标。不是全语料 PPL 或 all-position causal GPT 训练；prefix 内双向 attention 的输入均先于终端目标，使用它本身不构成泄漏。

以下 PPL 取冻结推理审计重新计算的向量：

| 模型/执行 | P | ctx32 | ctx64 | ctx128 |
|---|---:|---:|---:|---:|
| dynamic soft | 41,593 | 12.45828 | 12.46752 | 12.49622 |
| dynamic hard CALL | 41,593 | 12.98156 | 12.94637 | 12.95013 |
| static soft | 41,593 | 12.38315 | 12.45243 | 12.43575 |
| static hard CALL | 41,593 | 13.99553 | 14.01967 | 13.97012 |
| Transformer36 | 43,300 | 12.80292 | 49.83730 | 48.40862 |
| GRU44 | 41,548 | 10.08964 | 10.08974 | 10.08975 |

hard-CALL PPL 在此有效：输出仍是归一化的连续 readout，审计最大 log-normalizer 误差不超过 `3.53e-7`。它不等于旧 categorical one-hot native 接口，也不是完全离散状态程序。

重新计算 ctx32 配对 NLL，dynamic soft 减 static soft 为 `+0.00605` nat/target，逐行描述性 SE `.01762`；减 TF36 为 `−0.02729`，SE `.02989`；减 GRU44 为 `+0.21088`，SE `.03938`。这些 SE 只描述这 512 行的离散程度；语料位置相关、单 seed 和事后观察不允许将其写成跨任务显著性。不能把相对 TF36 的微小 IID 点估计提升写成稳健优势。

长度稳定也不证明利用了较远的字符：ctx128 相对 ctx32，dynamic 只有 11/512 个预测改变，逐目标 NLL 平均绝对变化 `.01262`；GRU44 为 0 个改变、`.000198`。TF36 则改变 451 个，NLL 增加 1.3300 nat。这里直接证明某些配置在更长输入下没有恶化，尚未证明额外远距信息产生收益；TF 的固定长度外退化也不能推广成整个 Transformer 家族的限制。

## 3. hard CALL 没有兑现墙钟加速

以下是同一审计内 12 次原始 draw 的 p50，单位 ms，resident input、同步 no-grad forward；不含数据/H2D/加载，不是带 KV cache 的自回归服务。记录说明 Root 在各串行审计前检查过 GPU 无其他计算进程；本审阅没有重新观测当时 occupancy。

| 执行 | B1,L32 | B64,L32 | B1,L128 | B64,L128 |
|---|---:|---:|---:|---:|
| dynamic soft | 24.245 | 26.566 | 94.487 | 105.353 |
| dynamic hard CALL | 25.042 | 26.823 | 100.316 | 107.582 |
| static soft | 24.850 | 26.941 | 99.953 | 108.727 |
| static hard CALL | 25.849 | 27.651 | 103.622 | 110.579 |
| TF36 | .982 | .983 | .974 | .975 |
| GRU44 | 1.165 | 1.212 | 2.643 | 2.928 |

**VERIFIED：** 当前实现的 hard CALL 少算 expert 的 MAC proxy，却没有更快。B64,L128 dynamic 的 soft/hard proxy 为 `.23752/.19453` G MAC，而墙钟反向变化；GRU44 为 `.19060` G。不能由 MAC 降低推出硬件速度或能耗优势。每步 Python 调度、读写记忆、小 kernel 和 gather 都仍存在；它们是否是主要时间来源尚无 profiler 分解。

资源优势必须分项陈述。该 case 记录的 CUDA allocated peak 约为 dynamic soft 10.43 MB、GRU44 17.74 MB、TF36 50.01 MB，包含模型/输入/框架 resident allocation；不是完整进程 RSS。Block 每个 example 的持续执行 carry 是 h64+keys(8×16)+values(8×64)=704 个浮点数，不是“8个标量”；GRU44 两层 hidden 是88个。persistent carry、序列 workspace 和 allocator peak 是不同量。live inference optimizer 为 0，但训练 optimizer 和 checkpoint 不能省略：dynamic/static 最终文件约 .726 MB，TF36 .609 MB、GRU44 .534 MB。参数相近不等于 checkpoint 或时间相同。历史训练时间包含并发条件；不得把它们当孤立吞吐。既有串行推理审计已足够否定“本实现现在全面更省资源”的表述。

## 4. 原始四个真实训练与新增六个控制：须区分两个缺口

这些是官方数据 split 中前 10 类的单阶段小样本开发，保留完整 Speech35/CIFAR100 输出头。Speech 固定 log-mel 为 98×40；CIFAR 是 64 个 4×4×RGB patch。TRAIN-only mean/std，共享 feature cache 和同模态完全相同的 batch SHA，300×64 更新。不是完整类别 benchmark 或持续学习实验。

| 数据/模型 | P | TRAIN soft | VALID soft | TEST soft | VALID hard CALL |
|---|---:|---:|---:|---:|---:|
| CIFAR dynamic Block | 40,793 | .48672 | .39200 | .42100 | .23400 |
| CIFAR static Block | 40,793 | .49141 | .39000 | .40100 | .25800 |
| CIFAR TF64 | 110,116 | .85234 | .50600 | .49900 | — |
| CIFAR GRU64 | 59,556 | .65547 | .39200 | .44600 | — |
| CIFAR CNN | 38,852 | .49297 | .39800 | .44300 | — |
| Speech dynamic Block | 32,280 | .34219 | .30469 | .29766 | .22188 |
| Speech static Block | 32,280 | .75938 | .71094 | .71172 | .28281 |
| Speech TF64 | 105,379 | .93438 | .76406 | .76484 | — |
| Speech GRU64 | 54,819 | .83828 | .77344 | .76250 | — |
| Speech CNN | 35,683 | .65781 | .60938 | .59766 | — |

这不是只有 TEST/VALID 才失败：Speech dynamic Block 的最后 50 个 batch 平均 NLL 1.73395，对照 TF .21452；CIFAR 为 1.54086 对 .51828。TRAIN 随训练仍上升，支持优化/表示/预算不足；“只需解释为过拟合”与这些曲线不符。TF 在同一特征任务能学习，反对“反馈/数据根本不够”的普遍解释。但 TF 大约多三倍 P，且前端不同，不能据此定位某一个 core 部件的因果责任。

**新增 static Speech 是更有辨别力的对照：** 同32,280P、同16维前端、同初始化程序/seed、同300批次，TRAIN/VALID 从 dynamic 的 `.34219/.30469` 变成 `.75938/.71094`。相同总体容量和前端约束并没有迫使所有 Block 失败。这支持“当前 dynamic CALL 条件化的整体训练/执行方式存在有害作用”的 seed0 配置级结论；但 ablation 同时改变了 forward 路由和反向的 h/read→CALL 耦合，尚不能归因于其中一个。CIFAR static 没有类似改善，因而也不能写成“static 普遍优于 dynamic”。Root 报告 detached-CALL CPU gradient contract 已通过；它只检查路径正确性，该新训练变体在本审阅截止时未计入任何已测训练收益。

全部已记录 loss/gradient 有限。Speech Block 的 preclip 最大 norm 为 609.71，median 2.176，94.7% update 被 clip；CIFAR Block median 2.504、99.3% 被 clip。TF 也几乎全部被 clip。源码确实在 optimizer 前全局 clip 至 1；不能把 preclip 峰值直接解释成巨大 Adam 状态，更不能在没有逐参数梯度/状态敏感度的情况下宣布 vanishing 或 CALL-starvation 已被识别。hard CALL 的额外损失是真实的，但 soft TRAIN 不足先于它，因而 hardening 不能独自解释实数据落后。

实际训练墙钟 CIFAR Block/TF 为 66.28/7.85 秒，Speech 为 97.51/7.92 秒；Block 的 forward-MAC proxy 反而更低。这些任务并发条件下的时间要完整收费，但不能替代 profiler 或孤立吞吐。feature cache 也分别有 34.21 MB/50.23 MB，加上归一化数组；不是无成本的 raw 输入。

## 5. 一项独立修正轴：去掉真实输入的硬 rank 瓶颈，不能冒充 CALL 失败主因

**SOURCE + 独立 checkpoint 元数据核对：** Block 实际 `projection.weight` 为 CIFAR `[16,48]`、Speech `[16,40]`，TF 分别为 `[64,48]`、`[64,40]`。Block 在线性投影后才把输入交给 primitive/caller/memory。因此每个 patch/frame 至少有 32/24 维的固定 kernel；这些方向一旦被投影丢弃，下游再复杂也不能恢复。TF 的前端可以满列秩。这个不可逆瓶颈是已确认的表示差别，**还不是其丢弃方向含任务信息的证明**。同16维 static Speech 的成功进一步排除了“它单独解释 dynamic/static 差距”，故本建议只保留为独立前端轴，不替代正在检验的 CALL 梯度假说。

建议仅作为一个未执行的新版本改变：真实连续输入 adapter 用 `Linear(D,64)`，core 的 `token_dim=64`，保持 H64/K4/rank8/memory8/slot1 和原300步数据/优化协议；不提供类别对应操作、正确 CALL 或中间目标。它移除的是架构强制的局部信息丢弃，不是给出正确计算。是否值得实施与保留由预先指定的 TRAIN/VALID、完整成本决定，不能用既有 TEST 分数调选。该理由不自动适用于离散字符 embedding；16 维实向量本身可以区分有限字符集合。

按当前冻结代码解析计数，改变将增加 CIFAR 10,944P（总51,737），Speech 10,560P（总42,840）；soft forward 额外约每样本 .6912M/1.020768M MAC，尚不含 backward 和调度。以上是公式计数，**未执行、未证明改善**，也不会修复已经观测到的 Python 串行时间问题。必须重报实际 P/optimizer/cache/内存/时间并保留原 strong baselines；不能把旧 P、旧 checkpoint 或旧收益移植给它。

最小额外实际测量是从已冻结 projection 与现有 TRAIN feature cache 计算实际 rank 和中心化输入 `z=x−mean_TRAIN` 的 `||z−P†P z||²/||z||²` 能量比例；不需要新标签、optimizer step 或 TEST 选择。已准备 [audit_frozen_frontend_rank.py](../../probes/topology-architecture/audit_frozen_frontend_rank.py)，仅静态解析/数学检查，**没有执行**；Root 可在既有有界 CPU runner 内执行。它只能判断被强制丢弃的输入变动是否实质存在，仍不能证明 label relevance。即便该修正之后改善，也还需区分前端信息、额外参数和优化条件；若不改善，不能继续声称这个瓶颈解释了主要失败。未区分的其他解释包括低 rank expert 容量、反复 LayerNorm/读写动力学及训练预算；本审阅没有把它们排成新的实验清单。

## 6. 接续失败是工具兼容失败，新增学习证据仍为零

四个 `parameter-context-n8-seed0-*-stage3-v1-*-resume` 的实际失败回执均 returncode 1、`stage=-1`、`completed_updates=0`，raw 中没有 `actual_update`，也没有完成的新阶段报告。原因是 `unrecognized legacy resume driver behavior`，不是模型失去旧能力。原 phase0 checkpoint 保留。

[context-resume-python-ast-diagnosis-v1.json](../../probes/topology-architecture/context-resume-python-ast-diagnosis-v1.json) 记录同一 remote Python 解释器中 old/new 执行 AST 签名相同：block `b9ac3660…`、GRU `08e6135e…`、TF `4afecb6d…`。我此前从本地不同 Python AST 序列化得到的魔术常量没有跨解释器稳定性，造成了假拒绝；本审阅明确承担这项实现错误。Root 选择仅替换为实测 remote3.12 的对应常量，并保留 whitelist、源码、NN、Adam、RNG 检查。这是兼容修复，不是训练算法调整；不能记作完成1200新步或持续保持改善。修复运行中的结果不在本次证据截止范围内。

新 continuous core 的 LayerNorm、连续残差和可覆盖内容记忆不满足旧固定覆盖/marked fiber 的 operation-class 合同。旧 monodromy 证明、旧 loop admission 与 teacher-free confidence maintenance 不会自动保护这些参数的后续语义。首阶段获取已测，后续 acquisition/retention 必须读真实接续结果后再判断。

**范围结论：** 当前证据允许报告有用的首阶段连续能力、一个相对固定 TF 配置的长度稳定点和部分推理内存优势；不允许报告 dynamic CALL 的一般因果优势、全面资源优势、长期保持、广泛强架构胜利或完成 Primary。审阅否定的是过度表述，不是本 campaign 的继续资格。
