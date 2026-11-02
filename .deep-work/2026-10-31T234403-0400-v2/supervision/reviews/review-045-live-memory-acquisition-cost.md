# Review 045 — 新双值阶段、冻结记忆依赖与四阶段已付费训练成本

**判断：新阶段发生了部分获取，同时旧任务保持明显受损；冻结计算确实依赖记忆读出及键值对应。写入地址弥散支持一次已授权、参数不变的 delta-write 控制，但尚不证明它是获取瓶颈或修复有效。没有广泛能力／完整资源优势证据，Primary 仍未冻结。** 本次是实际结果后的独立审查，不是重建历史缺失 review，也不是对 Root 已选定控制所增设的审批。

## 实际检查及边界

读取四臂 stage4 的完整 raw、result、protocol、实际最终 NN ZIP、冻结源文件及 current-only 联合成本账本。每臂都有连续 1–600 次 `actual_update`；逐项重加训练 wall／CPU／CUDA-stream／forward MAC proxy，与 result 相等。四个最终 NN 的实际 SHA／bytes／literal schema、stage3／step600／complete 均吻合；父 phase2 和 phase0 NN 的实际 SHA 也与谱系吻合。NN／Adam／Replay 的相等恢复及 CUDA RNG 恢复由真实执行记录和已读源码支持，本次没有重新执行这些操作。

用标准库直接解析 NPZ/NPY，重算四臂新阶段冻结预测的准确率；重算 VALID v2 四份 raw 的 argmax、NLL、配对变化，以及全部 65,536 个直接 R 系数。R 最大重算差为 0，NLL 与 float32 摘要最大差小于 6e−8 nat。新旧 source、数据及原失败均未修改；无 Torch 导入、checkpoint 反序列化、SSH、模型推理或实验重跑。另检查五个独立字符 LM graph audit 的原 summary：共48个case，原／tensor／graph全部输出合同通过，权重不变、0更新／0标签；它们不并入此处学习能力结果。

## 获取是部分的，保持不是自动成立

新任务确有独立 write-order/query-order 的完整256格覆盖，目标均衡；改变单一写值的反事实全部改变终端目标。只把“两个写入值都影响标签”的测量缺口修复了，不能据此宣称任意 key 的通用内容寻址。原 overwritten-READ 数据和分数保留。

| 模型 | 当前 P | 新 VALID soft | 新 IID soft | 新 long soft | 当前阶段训练秒 |
|---|---:|---:|---:|---:|---:|
| dynamic | 28,381 | .582031 | .566406 | .539063 | 91.424 |
| static | 28,381 | .597656 | .599609 | .597656 | 85.897 |
| GRU44 | 25,616 | .625000 | .628906 | .623047 | 5.718 |
| Transformer32 | 26,760 | .330078 | .310547 | .197266 | 5.577 |

所有新 VALID 从约 .125 上升；普通 TRAIN 多数类／末4 token lookup 保持 .125。它们支持本预算下的部分获取，没有任何预设基线分数门槛。dynamic 的新 VALID 七点为 .125、.138672、.193359、.25、.318359、.482422、.582031，不能把尚在上升的曲线改写成已达到能力极限，也不自动授权追加训练。

旧三任务 IID 的匹配比较显示遗忘：dynamic 从 .941406/.914063/.890625 降为 .792969/.820313/.789063；其余三臂也都有旧任务下降。原曲线的 OLD VALID 下降同样存在。新能力的**以后学习后保持**尚未测试，不将它与已经测到的旧能力遗忘混为一项。GRU在这组结果中拥有更少P、较高新任务表现；dynamic也没有优于同P static。墙钟受实际执行与并发条件影响，不能作为普适硬件速度结论。

## 冻结 VALID：记忆依赖成立，写入瓶颈仍是机制假设

| 干预 | 准确率 | NLL | 相对 full 改变预测行 |
|---|---:|---:|---:|
| full | .582031 | .695427 | 0 |
| zero read | .125000 | 7.904994 | 454 |
| 联合 roll 键和值 | .582031 | .695427 | 0 |
| 只 roll 值 | .380859 | 1.813057 | 257 |

联合置换最大概率差1.133e−6、无预测改变，提供了有效 sham；仅错配值显著损伤。结论限定为当前冻结 VALID 路径的计算依赖，不是学习／获取因果，也不等于从头训练无记忆模型不可能成功。

从 raw 独立重算，1,024个 WRITE token 的平均地址熵为1.801160（uniform8为2.079442），平均最大地址概率 .304416、gate .888017；其余16,128个 active token 分别为 .460311、.845015、.246684。WRITE确实比其他token更弥散。这是描述性候选瓶颈，不是“地址熵造成当前准确率”的识别结果。

以源写入质量加权，首个 WRITE 到 READ／MIXREAD 的直接 R 均值为 .733261/.729336，第二个为 .858000/.852680。它们没有显示所有旧内容被普遍抹为零；弥散与相互污染比“记忆完全闲置／完全泄漏”更符合这些局部证据。R仅是已实现特征固定时的直接线性路径系数：源token结束后至查询token开始前的乘积，不含门、地址、候选、状态及后续反馈的总 Jacobian。不能从 R 单独推出总信息保持。

因此，已授权的同参数 delta-write 对照是合理的一个区分动作；结果仍需实测，新增读出／维护／训练成本也要收费。现有证据没有决定修复必然胜出，也未排除表示、终端信用分配、CALL耦合等竞争解释。修复依据上述 VALID 证据；TEST只在此如实复核和报告。

## 四阶段实际已付费训练，不重复计入继承副本

新派生文件为 `../../probes/topology-architecture/memory-live-stage-total-paid-training-cost-evidence-v1.json`，SHA `2b8c452a1d2f11416ecb11e1806d50c995036f13ba7c57d37d161c78e012682b`。阶段0只取 protocol 的 `parent_lineage.historical_resume_lineage.historical_checkpoint_costs`，与真实 phase0 result 对应项交叉核对；阶段1／2只各取 `historical_training_reports` 的一次实际训练；阶段3只取本次 result.costs。继承的相同历史副本不再相加。

| 模型 | 最终 P | 四阶段训练 wall 秒 | CPU 秒 | CUDA-stream 秒 | 训练 forward G MAC proxy |
|---|---:|---:|---:|---:|---:|
| dynamic | 28,381 | 171.859 | 171.818 | 171.475 | 52.333504 |
| static | 28,381 | 167.490 | 167.456 | 167.104 | 52.333504 |
| GRU44 | 25,616 | 15.022 | 15.030 | 14.736 | 41.996609 |
| Transformer32 | 26,760 | 19.812 | 19.790 | 19.553 | 104.365339 |

每臂恰好2,400个不同已执行更新、153,600次训练标签暴露＝124,800当前阶段＋28,800旧回放。P／optimizer／checkpoint／峰值存储是快照，不跨阶段相加。该表是**训练循环合计**，不是完整campaign账单：评估、冷恢复／增长、编译、replay维护、检查点、诊断／失败及推理仍由各自实际证据另计；原 current-only 联合账本保持正确且未改。完整训练 FLOPs、能量等未知项保留 null，forward MAC不替代它们。历史性能没有与新版本性能合并。

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

- 双值阶段部分获取与旧任务遗忘 | VERIFIED in seed0/budget scope | 四臂600 raw、真实NN receipts、冻结预测及原曲线 | 跨seed与后来保持未测 | 旧测量缺口原样保留 | 完成已授权单一delta-write控制。
- 冻结计算依赖记忆及配对 | VERIFIED for present VALID dependence | 四份raw、joint sham、R重算 | 非获取因果／非总Jacobian | VALID audit v1自测fixture失败、0干预测量；v2修正后真实完成，成本分开保留 | 将写入弥散假设交给匹配对照。
- 广泛能力／完整资源优势 | NOT ESTABLISHED | 更强同P／更小P对照与完整未知成本边界 | 多任务可复现优势缺失 | 所有实际低分与失败保留 | 不以该开发结果冻结Primary。

本次只新增此 review 与一个成本派生artifact；没有更改Method、NOW、科学代码、原账本或Root的已授权行动。
