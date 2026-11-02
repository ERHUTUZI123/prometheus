# Review 044 — 实际接续、因果字符 LM 与记忆依赖

**判断：三阶段接续和 proper all-token AR 训练确已完成；冻结 VALID 计算确实依赖自身键值记忆。动态 CALL、该记忆机制的持续获取收益及广泛完整资源优势仍未建立。** Replay 改善 dynamic 的部分保持，但同预算 GRU 的质量与参数结果更好；原第三阶段未见组合不构成两地址记忆的充分测试。上述边界约束当前实现和主张，不关闭 campaign，不要求 Monodromy 留在 Primary，也不把任何低分基线排除为无效。Primary 尚未冻结。

## 实际读取范围、时间与验证边界

本审阅在 2026-11-02 约 00:11 EST 完成主要字节核查，写入时的证据截止为本文件列出的 runs。先读最新 `AGENTS.md`、`ALL.md` TODO5、`water/method.md` §3.2 和已经实际完成的 review043；没有把 review043 的旧接续失败状态延用到现已完成的新运行。

本次实际读取：

- 四个 `parameter-context-n8-seed0-*-stage3-v2-*-resume` 及四个 `*-stage3-v1-*-resume-replay128x16` 的完整 raw、result、冻结源码；每臂准确出现阶段1/2各600个连续 `actual_update`，共9,600个新增更新；96个中间／最终 checkpoint 实际大小和 SHA 均吻合。四个原 phase0 最终 checkpoint 的实际 SHA 与两种接续的父谱系逐一吻合，未把共享父节点当作独立 seed。
- 两个 `parameter-real-{speech,cifar}-block_detached-300-v2`、六个 `parameter-real-{speech,cifar}-{block_dynamic,block_static,block_detached}-300-v1-token64`、五个 `parameter-char-alltoken-{dynamic,static,detached,gru,transformer}-300-v1` 的完整 raw、summary、冻结源码。每臂都有1至300的连续真实更新。连同上项，本次新增范围共13,500次已记录更新、174个实际 checkpoint SHA；这不是独立实验数或科学进展指标。
- context 的15份原始 NPZ、12份 Replay128 snapshot、字符 all-token 的5份数据 NPZ及endpoint、24份逐目标输出向量、memory-audit-v2 的36份逐行输出 NPZ。使用标准库解析 NPY 数值，未导入 Torch、执行 checkpoint pickle、重新推理、训练、SSH 或改科学代码。
- memory-audit-v1 的真实失败和 memory-audit-v2 的36条 raw／result／exit0／冻结源码。v1→v2源码唯一变化是全局日志字段 `wall_seconds` 改名 `session_wall_seconds`；旧失败保留。

所有上述 training result 所引冻结 source SHA 与本地实际文件吻合。实际恢复时 NN、Adam、RNG 和输出相等，是原执行记录与其源码共同支持的证据；本审阅重新核对字节和算术，**没有独立重跑这些恢复／forward 操作**。参数数目另从实际冻结构造式重算；参数、checkpoint 和成本报告不等同于跨实现复制。

以下 `SOURCE` 指已读源码／协议，`ACTUAL` 指实际原始记录或数值文件，`RECONSTRUCTION` 指本审阅推导，`OPEN` 指尚未测定。没有由摘要或另一模型的同意批准结论。

主证据根目录为 `../../probes/topology-architecture/`，以下 run 名均相对于其 `runs/`，实际代码在 `versions/code-<run>/`。

## 1. 接续：获取与保持都有真实结果，不能写成无遗忘

**ACTUAL，DIRECT：** 八个接续均恢复同一已付费 phase0 NN／Adam／RNG，新增两个阶段，未重跑600个旧更新；每阶段64×600次标签暴露。无 replay 时都是当前阶段64行；Replay128 每批48行当前＋16行旧样本，更新数和总标签暴露不变，因此当前新任务暴露从38,400降至28,800，旧样本为9,600。Replay不是免费保持，也不是相同当前训练暴露。

最终三任务 TEST IID soft 如下；百分数按原始512行结果，P包含全部参数。

| 模型 | P | 无回放：阶段0／1／2 | Replay128：阶段0／1／2 |
|---|---:|---|---|
| dynamic Block | 28,365 | 82.8125／90.2344／91.9922 | 94.1406／91.4063／89.0625 |
| static Block | 28,365 | 81.4453／94.9219／94.1406 | 86.3281／92.1875／90.8203 |
| GRU44 | 25,572 | 93.9453／95.8984／93.3594 | 96.6797／96.4844／94.7266 |
| Transformer32 | 26,728 | 48.6328／79.2969／83.0078 | 56.8359／78.1250／81.6406 |

dynamic 的三任务均值从88.3464%变为91.5365%；GRU从94.4010%变为95.9635%。dynamic 的旧任务0改善伴随新任务2降低；static 的平均数没有因同一回放规则而提升。不能把一次回放效果概括为架构独有收益。

训练中保持也没有被最终分数抹去：无回放 dynamic 的任务0 VALID从phase0的1.0，在阶段1第100步降至.6484375；回放对应最低已观测值为.91015625。这里是每100步采样的最低值，不是所有更新间的真实最小值。无回放最终任务0 soft/native IID为.828125/.787109375，Replay为.94140625/.88671875；硬 CALL 没有自动继承 soft 保持。

**ACTUAL资源：** 无回放 dynamic 两个新阶段训练 forward proxy合计15.348845184G MAC，GRU44为12.315274752G；Replay分别15.210695472G与12.204388416G。它们不含反向、optimizer或所有非线性成本。原进程训练秒数全部收费，但并发条件使其不能直接作为独占吞吐排名。参数增长为0不是能力保护证书。

Replay实际payload为128×(48个token＋6个标量)×8=55,296B，另有metadata／RNG、索引证据、压缩snapshot、checkpoint与维护。12份snapshot内 raw state/token/label与实际源TRAIN逐项相等，来源阶段／索引留存；这是普通有容量约束的回放对照。配置与训练raw均没有借入VALID/TEST作为回放样本。正式完整成本仍须保留bootstrap旧标签重访问、来源审计、失败／维护及父阶段成本，不能只记这个55,296B。

## 2. 任务偏斜和代数简化：限制的是测量覆盖

**ACTUAL，DIRECT：** 重新解析原15份NPZ，目标频数与 `parameter-context-frequency-controls-v1/analysis/summary.json` 完全相符。阶段2 TRAIN的多数类为0（2,588/4,096）；固定预测这个TRAIN多数类，在TEST long为357/512=.697265625，恰等于无回放dynamic soft；TEST unseen为132/512=.2578125，dynamic为133/512=.259765625。相同总准确率不证明相同预测，但这些分数不足以建立复杂计算优势。阶段1/2 IID仍明显超过常量，不应把偏斜反向推广成全部能力虚假。

**SOURCE＋RECONSTRUCTION，经实际512行分别核对：** `contextual_task_data.py` 的阶段2 unseen骨架为

`KEY0 WRITE STEP KEY1 WRITE CLEAR KEY0 READ ADDRFLIP READ STEP2`。

初态q、mode=0时，key0保存q，key1保存(q+1) mod8；第一次READ读出的q紧接着被第二次READ覆盖，最后输出严格等于

\[
\left\lfloor\frac{(q+1)\bmod8}{2}\right\rfloor.
\]

随机插入NOP不改变该式；实际512个target全部满足它。一个模型可以从保留的初态和骨架识别直接计算目标，不需要保留第一格的独立贡献。阶段1 unseen的SAVE／LOAD骨架也严格简化为“原SET后执行一次STEP”，实际512行全部吻合。因此这两个结构化集合不是独立地址记忆或复杂SAVE／LOAD组合的充分证明，即使以后分数很高也是如此。该判断不要求修改旧数据、旧分数或retroactively重命名它们。

**覆盖：** 总分对原有限任务表现是DIRECT；对一般内容寻址／双地址组合能力仅PARTIAL。它不证明模型的记忆闲置；下一节的真实干预已经反驳这一过强解释。

## 3. 冻结记忆干预：自身历史和键值配对确实影响计算

证据为 `parameter-context-memory-audit-v2/audit/`；audit源码SHA为 `17bb5fdf6ce77d7b0ffdb9bfc6904988ef2c813b77ebf21b8cba690b80f963f4`，读取的实际checkpoint为无回放dynamic阶段2最终文件，SHA `ed89805b23241051540403be5297b7cd8e656ffa57349b47200b16b246f96714`。原core严格为 `b8a51a6c…`。

**ACTUAL，DIRECT：** 六种干预×soft/native×三阶段共36项，raw和result一一对应；36份NPZ的SHA、每行argmax、confusion、准确率、balanced accuracy和配对预测变化全部重算吻合。float64重算NLL与float32摘要最大差2.82e−7 nat，符合求和精度差异。before/after权重hash一致，原文件哈希未变；0更新、0 TEST标签，9,216次VALID标签评估。

阶段2 VALID、256行、soft结果：

| 干预 | 准确率 | 八类balanced accuracy | NLL | 对full改变预测数 |
|---|---:|---:|---:|---:|
| full | .953125 | .902806 | .226426 | 0 |
| zero read | .828125 | .626332 | 1.095773 | 44 |
| uniform read | .859375 | .775449 | .510308 | 32 |
| 联合roll键和值 | .953125 | .902806 | .226426 | 0 |
| 只roll值 | .878906 | .740263 | .570813 | 28 |
| 另一active样本的键值历史 | .800781 | .638275 | 1.337586 | 50 |

联合键值置换在六组stage/interface中均不改变预测，最大概率浮点差不超过2.33e−6；它是针对slot标签重排的有辨别力sham。只置换值则破坏键值对应，确有损伤；uniform与zero、跨样本历史也有损伤。阶段0同样受损，说明记忆可参与普通状态计算，不能仅按SAVE/READ符号判断是否使用。

**允许结论：** 这个实际冻结模型的VALID计算依赖自身键值历史，且键值对应／内容读出有实际作用。**不允许结论：** 已经证明记忆是后来获取或保持的原因、学会精确符号地址语义、优于可重训无记忆架构，或存在拓扑保护。干预每一步的read会进一步影响后续状态／write，是整条执行路径的依赖测量；冻结损伤并不等于移除后重新训练仍不能学会。

v1为真实logging collision失败，已保存一个full-soft-stage0输出；该输出全部数组与v2对应项逐字节相同。v1已付费约.7933秒及256次VALID输出须保留，不能把重试当零成本。v2整次约3.5034秒，含hook统计和额外baseline read，不是正常推理延迟。

## 4. 动态、static与detached：信用路径有局部证据，通用修复没有

**SOURCE：** static只把CALL的h/read输入置零；expert、读写记忆仍有上下文。detached前向仍传真实h/read，仅这两条CALL输入使用stop-gradient，x与caller参数以及其他expert/read/write路径继续训练。dynamic为原普通反向传播，detached是有意省略部分链式项的半梯度更新，不能叫作相同前向目标的完整普通梯度，也不是单纯前向路由消融。CPU检查的实际隔离梯度与同前向记录支持这项实现合同，不支持训练优势本身。

**ACTUAL：** 原16维接口Speech，同32,280P、300×64更新和完全相同batch，dynamic TRAIN/VALID=.3421875/.3046875，detached=.73203125/.7015625，static=.759375/.7109375。CIFAR同40,793P，dynamic=.48671875/.392，detached=.459375/.338。因此前者是一个配置中信用路径确实改变优化轨迹的有力对照，不能写成跨模态通用修复。保留双方结果，无额外budget择优。

六个token64运行确认了实际adapter和core token_dim都为64；参数从Speech32,280→42,840、CIFAR40,793→51,737，与冻结源码公式一致。改变同时扩大adapter、CALL、expert和read/write输入容量，是输入接口＋容量的开发轴，不是纯粹“移除前端rank瓶颈”的单因素因果试验。

| 任务／64维版本 | P | TRAIN／VALID soft | TEST soft／hard CALL | 训练forward G MAC proxy | 实际训练秒 |
|---|---:|---|---|---:|---:|
| Speech dynamic | 42,840 | .780469／.735938 | .722656／.488281 | 75.3070 | 106.147 |
| Speech static | 42,840 | .814844／.753125 | .717188／.379688 | 75.3070 | 110.673 |
| Speech detached | 42,840 | .764063／.710938 | .687500／.556250 | 75.3070 | 122.607 |
| CIFAR dynamic | 51,737 | .503906／.392000 | .422000／.306000 | 49.9040 | 78.311 |
| CIFAR static | 51,737 | .488281／.404000 | .405000／.287000 | 49.9040 | 72.166 |
| CIFAR detached | 51,737 | .514844／.400000 | .416000／.302000 | 49.9040 | 72.216 |

这些实际秒数收费，不能从并发训练直接推出硬件速度排名。较宽接口显著改善原dynamic Speech训练，CIFAR soft VALID未类似改善。不能以TEST最高者选择Primary，也不能把额外10,560／10,944P收益全部归因于信息rank。原强基线仍保留；此开发没有证明同能力完整资源优势。参数／Adam／cache／峰值和checkpoint在每个原summary中分别计价，不能用旧16维P报告64维结果。

## 5. Proper all-token AR：无标签对齐错误，额外历史收益很小

**SOURCE＋ACTUAL，DIRECT：** 新数据采用原TRAIN-only词表和90/5/5连续分割，各window输入为 `ids[p−L:p]`，目标为 `ids[p−L+1:p+1]`。实际5份数据均满足逐位shift、全长、初态0及metadata SHA。Transformer36实际使用严格上三角bool mask，三个causal encoder layer，无CLS；GRU和Block逐步输出。TRAIN输入的未来后缀扰动实测五臂过去输出最大差都为0；源码没有把未来label放进当前输入。

每臂固定300×32×64=614,400个训练label观察、9,600个window，同batch SHA `538c1157…`，学习率.003、Adam、clip1。重叠window和重复采样仍逐次收费；这不是614,400个独立新语料字符。旧terminal-prefix是不同反馈协议，不能与本项直接合并成学习效率增益。

TEST的三个长度共享512个endpoint，最后64位target逐字节一致，共32,768个配对label。24份向量的hash／大小、全部NLL、last64 NLL、PPL和准确率均重算一致；每个run训练结束冻结在读取TEST之前，实际冷重载输出相同且推理后权重未变。正式跨seed复现和精确训练恢复仍未检验。

下表P、forward代理和墙钟均来自本项新运行；GRU／TF的P不同于旧terminal-prefix包装，不能移植旧值。

| 模型 | P | 配对last64 PPL：L64／128／256 | 训练forward G MAC proxy | 实际训练秒 | Adam tensor B | CUDA累计peak allocated B |
|---|---:|---|---:|---:|---:|---:|
| dynamic soft | 41,593 | 10.35835／10.25148／10.25285 | 21.7154 | 75.962 | 332,844 | 45,642,240 |
| static soft | 41,593 | 10.30601／10.18034／10.18297 | 21.7154 | 75.241 | 332,844 | 45,642,240 |
| detached soft | 41,593 | 10.24277／10.11376／10.11377 | 21.7154 | 74.069 | 332,844 | 45,642,240 |
| GRU44 | 32,660 | 8.66407／8.55153／8.55153 | 16.9771 | 2.495 | 261,324 | 51,620,864 |
| causal Transformer36 | 39,664 | 10.53295／15.18484／14.73574 | 29.8156 | 6.747 | 317,476 | 104,680,448 |

全window PPL与这个配对指标分开保存，L128/256不是同一target集合；本表不以其替代last64。hard CALL PPL在L64分别dynamic18.56334、static18.98933、detached21.50122；soft训练取得的质量不能自动转给硬调用。

新增64位历史使dynamic平均NLL只降低.010371 nat/target（L128对L64），static .012268、detached .012675、GRU .013075；从L128再到256后三个Block几乎不再改善，GRU只差约5.1e−9 nat。dynamic／static／detached／GRU的L64→128预测变化分别为1,560／927／771／656个，共32,768位。它们支持有限warm-up／额外历史效应，不足以建立远距离依赖能力或大幅长上下文优势。

TF更长前缀的退化是真实本配置结果，但last64的绝对sinusoidal位置也由0–63移到64–127或192–255；额外历史和位置分布改变一起发生。不能据此宣布Transformer家族无法长序列，也不能因为分数低追加无限调参。该固定运行有效且完整保留。

**成本判断：** Block在此有较低的累计CUDA allocated峰值点，但GRU拥有更少P、更低forward proxy、更低PPL；Block仍携带704个float32，即2,816B/example的h/K/V，GRU44两层carry为88个float，即352B。上述峰值包含框架／输入／审计，不是完整RSS或单独activation；字符cache数组另12,746,752B，Transformer还有102,400B位置与mask buffer。forward_all对每个token输出并保留BPTT中间量，不能由固定推理carry推出常量训练内存。当前没有缓存生成延迟、完整训练FLOPs、能耗或统一完整成本优势。

## 6. 当前最薄弱接口与下一单个决定

**最薄弱的机制结论是“自身记忆参与计算”与“可复用的地址操作帮助获取并保持新能力”之间仍缺一条可观察的因果连接。** v2干预已修复了“是否完全闲置”的疑问；旧unseen骨架对两地址组合没有足够覆盖。继续只提高这些聚合分数会绕开这个缺口。

**AGENT PROPOSAL：** 下一项应是一个预先冻结的新阶段终端任务，让两个独立写入值都对最终查询有不可消去的贡献，设置反事实配对只改变其中一个写值／地址，保持另一项及末尾token相同；同一core、同一终端信息、匹配P／更新／标签／replay容量的dynamic/static/detached/GRU/Transformer共同进入，并报告旧三阶段保持与全部新增成本。先用oracle仅检查新TRAIN/VALID任务的依赖和目标频数，oracle不进入learner，不以旧TEST筛选。它是新prospective发展测试，旧数据和旧结论保留；不提前指定胜者，不把该建议当作执行新资源权限。

该建议针对测量与新能力依赖的可识别性，**不是已经验证的架构修复**。如果新阶段失败，只能收紧该版本／预算下的获取与保持结论；如果成功，还必须区分普通replay、上下文CALL、内容记忆和资源差异，不能直接宣布范式优势。

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

- 三阶段真实接续／普通回放改善部分保持 | VERIFIED in stated seed0 scope | 8完整raw、96 checkpoint、父SHA | 单seed、硬化损失和完整成本 | 原resume-v1兼容失败保留 | 新可识别阶段能否获得并保持。
- 冻结计算依赖自身内容记忆 | VERIFIED for frozen VALID dependence | 36项实际向量及联合置换sham | 不等于获取因果或精确符号地址 | memory-audit-v1日志失败保留 | 让两个独立记忆值共同决定新终端标签。
- dynamic／detached通用能力或完整资源优势 | OPEN／NOT ESTABLISHED | 真实模态及proper AR五臂 | 跨任务质量／成本优势和正式复现缺失 | 所有低分与额外开发成本保留 | 不以该摘要冻结Primary。

本审阅只新增此review，没有修改Method、NOW、策略或科学代码；Root负责合并已验证的证据边界与下一行动。
