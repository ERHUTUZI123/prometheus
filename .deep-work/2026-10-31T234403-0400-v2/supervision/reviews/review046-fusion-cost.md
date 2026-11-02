# Review 046 — 融合成本、控制／值分离与尚未关闭的训练瓶颈

**判断：OWN fusion 是已执行、数值合约通过的本模型推理实现改进，尚不构成对 GRU／Transformer 的资源优势。新 factorized Block 用更少 NN／Adam 存储取得了一次有意义的获取与保持结果，但同一预算下 anchor1 的失败、旧 long／unseen 弱项，以及约 13 倍 GRU 的训练墙钟，仍阻止广泛能力／完整成本优势或 Primary 冻结结论。** 本次区分旧 58fe 字符 LM kernel、28.6k G8 继续学习、13.4k factorized 从头四阶段三条不同谱系，不合并性能。

## 实际核查范围

核对冻结 kernel、prologue、runner、audit source 与实际 SHA；对 factorized v2/v3 的逐条 `actual_update` 重加次数／时间／MAC，逐 step 比较丢弃阶段与恢复阶段 NLL，检查训练冻结后才打开 TEST 的源码和事件顺序。独立辅助核查直接用标准库 ZIP/NPY 解码六份 fusion 原始向量、重算 logp／argmax／归一化与 JSONL 分位数，并逐元素核对旧 dynamic 三 backend 的向量。模型共识没有被用作数值证据。

核查没有 SSH、安装、checkpoint 反序列化、模型／kernel 执行或训练。factorized 的 TEST 数字是被核对来源、流程与版本的实际 producer 结果；该 driver 没保存逐例 TEST 向量，本次没有独立重算其准确率。派生账本为 [`review046-fusion-factorized-raw-checks.json`](../../probes/topology-architecture/review046-fusion-factorized-raw-checks.json)。

## 旧 58fe：数学规则未替换，实数等价不等于浮点逐位等价

`fused_continuous_block_fp32.py` SHA `5cf6929bb0da78d8a7dc65cbdfe7d9dcaf0ba9d182845564f3f36c06e6f88d29` 仍执行 learned encoder(initial0)、learned keys／zero values、slotted READ/CALL、低秩 exact-erf GELU 混合、residual LayerNorm，再用 **new h、RAW token、last read** 做原 learned write-value projection 与当前 convex write，最后原 readout/logsoftmax。slot 没被误加到 WRITE；没有 task feature、参数裁剪、标签或替代输出模型。25 项原参数完整保留，总 P=41,593，另收费完整 FP32 packing。

六个原始 NPZ 共 1,478,400 个 FP32 logp／14,784 个预测位置，最大绝对差 `9.250640869140625e-5`，全部满足固定 atol=2e-4／rtol=2e-5，0 个 argmax 差异；float64 稳定重算最大 |logsumexp|≤2.780e-7。最终 h／keys／values 均有限且 allclose，最大差分别约 1.407e-5／6.522e-5／1.075e-4。新 raw reference/pure 与旧 dynamic `original_eager / tensor_eager / cuda_graph` 六 case 数组逐元素相等；原模型 state hash 均为 `214704c5…106e0`。覆盖固定 TRAIN-only tiled windows，不是任意输入的误差定理，也不是新的 TEST 学习结果。

下面各格统一为 **CUDA-event p50 / synchronized wall p50，ms**，直接由逐 draw 重算；不能把 fused .885040 event 与 TF .183890 wall 混比。

| B,L | 同 Block tensor graph | fused graph | Transformer graph | GRU graph |
|---|---:|---:|---:|---:|
| 1,64 | 3.380464 / 3.406612 | .885040 / .906471 | .168672 / .183890 | .061296 / .077500 |
| 1,128 | 6.739536 / 6.766524 | 1.758464 / 1.782516 | .189504 / .205407 | .096688 / .113374 |
| 1,256 | 13.465376 / 13.500227 | 3.504464 / 3.531704 | .220240 / .235828 | .166688 / .183008 |
| 32,64 | 6.319088 / 6.345608 | .909120 / .933151 | .196864 / .211857 | .070352 / .087396 |
| 32,128 | 12.611008 / 12.643656 | 1.844704 / 1.869602 | .291392 / .307515 | .110688 / .127019 |
| 32,256 | 25.204272 / 25.248254 | 3.609440 / 3.650201 | .627680 / .645244 | .195904 / .214240 |

同模型 graph-event 改善 B1 为 3.82–3.84×、B32 为 6.84–6.98×，但仍慢于 TF 4.62–15.91×、GRU 12.92–21.02×。288 个 timing draw 完整，repeat0–11 与四 backend 旋转顺序吻合。源码在计时前检查 fused graph 输出，但 NPZ 保存的是 fused **eager** 输出，未保存独立 graph raw／计时后输出；不能将本次 raw 核查扩大为每次 graph replay 的数值证明。

## Resident 改善必须支付 cold、packing 与 graph 生命周期

| 实际账项 | Fusion | GRU graph audit | TF graph audit |
|---|---:|---:|---:|
| 本地 SSH 总 wall s | 33.360779 | 7.321995 | 7.669167 |
| remote coordinator wall s | 32.270263 | 6.257100 | 6.605814 |
| scientific driver wall s | 18.874081 | 1.752048 | 2.260109 |
| 原训练循环 wall s（历史，勿再加到单次延迟） | 75.962459 | 2.494903 | 6.747454 |
| learned P | 41,593 | 32,660 | 39,664 |
| weights＋fixed buffers＋packing 字节下界 | 332,744 | 130,640 | 261,056 |

Fusion compiler stage 含 Torch/CUDA 为 12.044640s，其中固定 prologue 8.335268s，实际 pack 2.054ms。原 weights166,372B **仍在**，新增166,372B不是替换 P；另有 B×L×100×4 输出和每例2,816B最终 carry。新 Triton cache5,914,228B／26files，C probe15,128B，三份另存 PTX1,498,805B均保留。每 L 的 B1/B32 compiled hash、对象 ID、PTX SHA 相同，science 使用 prologue 原 canonical module/JIT；无 post-strict fallback。原严格16检查、transition与进程组清理成功记录保留。

整场 fusion 有四 backend／12 capture／288 draw；旧 GRU/TF 是三 backend／6 capture／216 draw。因此 whole-audit 账单是真实支出，**不是匹配工作量的部署 cold-start 比率**。两张 graph 同时存活，capture 间没有 reset memory peak，不能从相同峰值字段推断 fused graph 免费或独立更省内存。raw driver/context/executable memory、能量、GPU busy-time、完整 FLOPs 和 provider-effective compute 未知。metadata 的 n_regs=255／n_spills=130 值得后续定位，但不是已识别瓶颈的证明，也没有将 spills 擅自换成字节。

## 新 fcba：控制历史与可执行值分离是具体机制候选，仍非保护定理

冻结 source `fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158` 定义：controller c 先用 GRUCell(raw token,h,c) 更新；READ 用(c,raw+slot)寻址；CALL 用(h,c,raw+slot,read)选择 K4 低秩编辑；G8 按 `h'=h+g*(LN(h+delta)-h)` 更新；WRITE 用(c,RAW token)地址并把 **h' 本身** 写为 value。终端状态接口是 sqrt(H)·normalize(E(state))，value H32、controller16、每例 carry432 scalar；没有 operator/route/intermediate-state oracle。它删除 learned write-value projection、缩小状态宽度并加入控制历史，不能把收益单独归因于 identity-write，也不能把旧 H64 fusion 速度移植给这个新 core。

G8 的直接 identity 系数是 1-g，不是 g。bias4 初始 g≈.982，直接 identity 仅≈.018；额外门导数、LN 与共享参数仍会改变总 Jacobian。grouped-G8 warm-start source也明确承认启用 gate 改变原函数，NN／Adam prefix相等不等于旧行为受保护。该 28,645-P 谱系新 IID .625、old0 IID .753906，不能据获取改善抹去遗忘。

fcba 的实际 CPU rule check 检查了 ragged／zero-length、future causality、identity value-write、非零梯度路径和 exact reload；包含 **1次 synthetic Adam update**，不伪称0更新，也不算研究任务训练。两条四阶段线路均只用 raw初始／token／terminal y；anchor1 只是把同一终端标签映射为 stopgrad(E(y)) 的辅助 MSE，不是免费中间真值。anchor0也实际执行该 lookup/MSE 后乘0，费用没免掉。

| factorized 结果，seed0 | P | 新 IID soft | old0/1/2 IID soft | 新 long / unseen soft |
|---|---:|---:|---|---|
| anchor0 | 13,397 | .621094 | .841797 / .896484 / .855469 | .607422 / .585938 |
| anchor1 | 13,397 | .179688 | .941406 / .912109 / .841797 | .164063 / .160156 |

anchor0 是有意义的部分获取与保持开发结果；anchor1 清楚反驳“这个权重1终端值锚定必然有益”。anchor0 的 old0 long=.234375、old1/2 unseen=.089844/.162109 仍弱；一个 seed、一个预算、联合架构变化不建立稳定广泛优势。新能力在以后的学习后保持仍未由这些末次评估决定。

## 真实已付费计算与下一具体瓶颈

逐条 raw 重算：

| 线路 | 有效四阶段更新 | 有效训练 wall s | 实际已付费更新（含丢弃） | 实际训练 wall s | 有效 forward G MAC proxy |
|---|---:|---:|---:|---:|---:|
| factorized anchor0 | 2,400 | 202.871446 | 2,408 | 204.290782 | 24.650234 |
| factorized anchor1 | 2,400 | 189.035957 | 3,000 | 287.658491 | 24.650234 |
| 原四阶段 GRU44，对照历史 | 2,400 | 15.022 | 2,400（该账本范围） | 15.022 | 41.996609 |

v2→v3 的 stage3 NLL 对 anchor0 的8条、anchor1 的600条逐 step 最大重复差均为0；v3从真实stage2 NN／Adam／RNG／Replay恢复，旧三阶段未重训。padding-width维修只在训练结束的 replay storage 上将全零128列视图裁为48，检查没有丢active token；此前失败／中止训练依然计入，不把复跑伪称从不存在的最新checkpoint继续。

factorized 最终 NN53,588B、Adam107,284B、Replay55,296B＋metadata1,739B。每线四checkpoint合1,326,796B；anchor0另有VALID7.142s／TEST8.175s／reload.436s／maintenance.264s，anchor1为6.356／7.244／.382／.237s。它们不与训练循环混称完整campaign账单；失败冷恢复、评估、存储、CPU/GPU峰值、后续推理等继续各自收费。

较少 forward MAC 与约13倍GRU墙钟同时成立，说明**不能用 proxy 或小P宣布计算优势**。源码逐 token nonzero/gather和大量小Torch算子提供可检验的实现瓶颈候选；CUDA-stream elapsed近wall也不等于GPU计算单元持续繁忙。下一已授权 NO-RUN 工程契约是：复用原 `_transition` 的全行masked前后向、固定B64 buckets8/16/48，严格比较原gather输出及每项参数梯度；图内 grad.zero_，Adam留在图外，0 SGD；显式付费padding、capture/pool、warmup、NN/grad/storage。未做此实际诊断前，不推算它会把203s变成多少，也不干扰Root的continuation-learning维修。

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

- 同一旧Block resident推理融合改善 | VERIFIED in six fixed TRAIN-input cases | NPZ/288 raw draws/PTX/source pins | graph逐draw raw未保存、非任意输入定理 | 无数值失败；spill/寄存器压力仍在 | 保留实际完整成本，不宣称胜过TF/GRU。
- 新factorized机制取得小P部分获取 | VERIFIED recorded seed0 development | fcba source、CPU实际rule check、四阶段raw/冻结后TEST | 因果归属、跨seed、later retention与旧long/unseen | anchor1负结果，v2 padding失败及复跑全收费 | Root继续已授权learning repair；Graph F/B仅先交NO-RUN契约/代码。
- 广泛能力／完整资源优势、Primary冻结 | NOT ESTABLISHED | 以上强基线与全账本边界 | 完整实际P+训练/推理/维护优势未形成 | 保留低分与未知成本 | 不用kernel工程改进替代科学验收。

本次新增review与只读派生证据；没有改Method/NOW/core、原模型、原raw或任何已运行版本，没有启动实验。
