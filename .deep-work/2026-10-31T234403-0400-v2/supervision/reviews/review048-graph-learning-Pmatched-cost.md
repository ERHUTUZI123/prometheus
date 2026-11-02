# Review 048 — Graph 已进入实际学习，近参数基线仍限定完整成本结论

**判断：同一 fcba Block 的 Graph600 与 AR300 都已实际完成学习；前后向实现改进显著缩短了更新时间，未取得广泛能力、参数与完整成本的共同优势。** AR 的近参数 GRU34 质量更好且更快；Transformer24 质量较弱但更快、参数更少。当前应检验可继续使用、组合与保持的内部计算能力，不继续用更多 kernel 变体代替这个科学缺口。没有准确率门槛，也没有因基线低分追加调参。

## 核查范围

本 reviewer 读取原／新 driver 与 fcba 源码，独立确认 AR `AllTokenLM` AST 与冻结 parent `3e6447a2…84960` 完全一致，实际四文件 SHA 匹配；直接用标准库解码 admission／step100／200／300 四份原始梯度 NPZ，共 **902,088** 对 output、loss、27 项参数梯度标量，并复核四臂各 300 条更新记录与计时累计。没有模型、Torch、SSH、编译或训练执行，没有 checkpoint 反序列化。

以下既有独立验证产物保留原始证据链；本 review 没把读到的 summary 当成额外独立实验：

- [AR 四臂原始 TEST 向量核查](../../probes/topology-architecture/actual-Pmatched-AR300-independent-raw-checks-v2-graph.json)，SHA `1144f27f7a767fed420aea94dbecb0dfd19a520af5ef8ed2869876d19627cfe8`；覆盖 917,504 个 target positions，核对 common data／sampler／paired targets，并重算 NLL、PPL、accuracy。
- [AR Graph 梯度与轨迹核查](../../probes/topology-architecture/actual-ARGraph300-gradients-trajectory-raw-checks-v1.json)，SHA `2acec98cf799b8203c98d8151197e997e97fa9853b4aca254dbab3d4b61e83e3`。
- [Graph600／eager／GRU32 成本证据](../../probes/topology-architecture/actual-graph-eager-gru32-cost-evidence-v1.json)，SHA `3ae0281d51fef9357ae1615eee7722e4d8079e9543855439d7a74239abc108f5`；[Graph600 原始 schedule／轨迹核查](../../probes/topology-architecture/factorized-graph600-v1-independent-raw-checks.json)，SHA `f3845cc14b1abde0ecafdf5d2b05bd2361a8cf3d69ec59e70e757877d270c428`。

## Graph600：真实继续学习，不能改写为四阶段全 Graph 重训

Graph600 从 **同一 anchor0 stage2 NN／Adam／RNG／Replay128** 恢复，按原 sampler 做 600 次 48 current+16 replay、Adam .003／clip1；保持 NLL+0×binding 的原计算。固定 B64L48 的全行 masked `_transition` 进入唯一 Graph，grad.zero_ 在图内，clip／Adam 在图外，指针固定；warmup、capture 和梯度诊断均为 0 SGD。安全 checkpoint 在维护前保存，128→48 只取已证明全零的维护尾部视图；final NN／Adam／RNG／replay 与 forward reload 已检查。使用原有隔离 runner，无新增 compiler 权限。

| 实际谱系 | 最终 P | stage3 更新 wall s | retained 2400 更新 wall s | new IID |
|---|---:|---:|---:|---:|
| fcba eager v3 | 13,397 | 111.355712 | 202.871446 | .621094 |
| fcba Graph600 | 13,397 | 11.549469 | 103.065203 | .623047 |
| fresh GRU32 | 14,024 | 5.715203 | 17.281469 | .615234 |

Graph 的 stage3 更新时间改善 **9.64×**，但 retained 前 1800 步仍是共享的 eager 历史；103.065203s 是该混合实现的真实谱系账，不是四阶段全 Graph 重训结果。它仍比 GRU32 retained 更新时间高 5.96×，单独 stage3 高 2.02×。Graph 的 old IID0/1/2 为 .876953/.916016/.853516，优于本次 GRU32 的 .814453/.806641/.822266，但 long／unseen 呈混合取舍。终端 driver 未存逐例 TEST 向量，这些准确率是已核对来源／流程的实际 producer 结果，不宣称独立逐例重算。

同一 600 个 current／replay 数组与 active 计数全部吻合；Graph 实算 **1,843,200** transitions，对应原 active **1,016,659**，收费前向 MAC 为 25.136G，对照 active proxy13.869G。更快来自执行实现，不是更少的实际算术。Graph 的 F/B CUDA-event 总计10.628333s，含 VALID／diagnostic／batch 的 loop16.985986s，整个 driver31.668939s、本地 coordinator37.347421s；restore+growth3.811608s、setup1.157519s、诊断1.229422s、TEST7.352887s、额外 inference1.944449s 都保留，嵌套时间不可相加。NN53,588B、Adam107,284B、replay55,296B及其元数据、grad53,588B、private pool、原始向量与 checkpoint 均须付费；final574,701B与维护前 safety574,573B 是实际磁盘产物，不能只报参数。

**轨迹并不相同。** Graph600 与 eager 的 NLL 差从初期浮点量级放大，最大绝对差1.507546、RMSE .247799、最后一步差−.080986；同一当前 NN 的七次固定 TRAIN 梯度合约仍通过，最大 output／gradient 差为5.722046e-6／4.768372e-7。不同 FP32 运算形状与 Adam 放大可解释这种现象，但本次没排他识别其全部原因；不能宣称逐位权重轨迹等价。

## AR300：同 NN／反馈／预算，更新快 10.40×，整个 driver 快 2.466×

固定 fcba `fcba2e82…88158`、20,721P；完整100×32 encoder仍分配计数。新训练路径逐 token 调用原 `_transition` 并做每个 post-token readout／logsoftmax，目标仍是原 all-token mean NLL，**没有 binding、teacher 或新参数**。相同 seed0、300×32×64、614,400 个 SGD label positions、原 Adam .003／clip1、VALID50、checkpoint50…250与 freeze-before-TEST。eval／native／reload 保持原 wrapper。训练全64位置均 active，没有额外 token padding。

四份原始 current-NN 对照的 output 最大差 **0**、0 argmax 差，全部27项 gradient 最大差 **9.313226e-9**，满足预定 output atol/rtol=2e-5/2e-5、gradient=2e-5/2e-4。它们使用同一固定首个 TRAIN batch，覆盖初始及100／200／300步的 NN 状态，**不是对全部300个 batch 或任意权重的普遍等价证明**。原／新实际训练 NLL 最大差2.861023e-6、RMSE4.766384e-7、177/300步相等；不要求或声称 bitwise weights。

下列 PPL 统一为共同最后64目标，每个 L 有32,768个 paired target；full-window PPL另存原证据，不混列。

| 实际 arm | P | paired PPL64／128／256 | train wall s | 整个 driver s | NN／Adam／buffer B |
|---|---:|---:|---:|---:|---:|
| fcba eager |20,721|10.5930／10.4980／10.5602|79.288931|117.671864|82,884／165,876／0|
| fcba Graph |20,721|10.5930／10.4980／10.5602|7.623333|47.721731|82,884／165,876／0|
| GRU34×2 |21,180|9.4485／9.3490／9.3490|1.090724|5.476958|84,720／169,484／0|
| Transformer24×3 |19,564|11.4140／13.3578／15.8349|3.118619|7.775041|78,256／156,676／90,112|

这些是预先按 P 选择的**近参数**控制，不是完全相等 P；Graph 比 GRU34少2.17%参数，比 Transformer24多5.91%。GRU34质量更好，训练快6.99×；Transformer24质量较弱，训练仍快2.44×。它们的低分不触发准确率门槛或配置重选。相对 old GRU44／官方 Mamba2 的既有质量差距也没有被这次成本实现关闭。

whole-driver 比例不能冒充同工作量吞吐：Block仍评估 soft+native，共10,240 windows／917,504 prediction positions；两基线只评估 soft，共5,120 windows／458,752 positions。Graph Block 的 eager eval32.158895s已占整个 driver **67.4%**。不得删掉这项费用，也不得把10.40×更新改善写成10.40×完整训练管线改善。

## 全生命周期与仍未关闭的科学缺口

AR Graph setup1.576804s（含首批 validation／clone／grad allocation），warmup .663849s、capture .340314s、instantiate .086033s，三次后续 diagnostic1.166827s。300个真实训练 F/B、3 warmup、4 reference与4 graph diagnostic共 **311次实际 F/B**，额外22,528个 TRAIN label positions；另有一次 capture 描述，0次 GPU F/B执行，不算训练更新。static／固定诊断输入各33,280B、grad82,884B，诊断的 NN／Adam deepcopy、原始 NPZ、copy与序列化均计入实际时间／峰值。copy .042710s嵌套在训练时间内，不可再加一次。checkpoint六份共约1.85MB，final309,155B；原 eager 六份约1.81MB，final301,219B。

合并 capture前后峰值后，Graph allocated peak55,024,128B／reserved102,760,448B，原 eager37,181,952B／62,914,560B，分别增加约48%／63%；private pool精确大小未测。persistent grad字段不等于只给Graph增加一次原 eager从来没有的梯度成本；两者均需反向梯度，实际 allocator峰值是已测比较。CPU peak RSS、CUDA启动、setup、冷缓存、source／data／raw存储及原有失败均保留。完整 training FLOPs、GPU busy time、driver/context memory、energy、provider-effective compute未知，forward-MAC不能填这些空白；本次也不支持排他同负载吞吐或新颖性主张。

这两次真实学习缩小了一个实现成本瓶颈，未证明 Block 能以更优能力—参数—成本取舍学习可持续执行、通用变换、内容寻址和组合机制。主缺口仍是有用 closure／后续计算／保持的实际能力，而非再造一组 kernel 变体。ROOT已选定一次 **+1P soft value bridge** 的下一项受限检验；它尚未在本证据中验证，计划代码不算能力、成本或因果证据。本 review 不启动训练，不据已见 TEST 调参，不改 Method／NOW。

`CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`同一Block的实际学习执行成本降低 | VERIFIED，限本次seed0／协议／host | Graph600及AR300 raw/source/cost链 | sampled current-NN gates，非普遍浮点定理；complete优势未成立 | 原eager／failed／profile及低baseline结果全部保留 | 以既定+1P bridge真实检验后续计算接口；不增加kernel变体来替代能力证据。`
