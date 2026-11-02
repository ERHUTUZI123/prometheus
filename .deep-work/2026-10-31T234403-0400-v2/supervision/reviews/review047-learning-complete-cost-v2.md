# Review 047 v2 — 后继行为反馈、完整学习成本与实际 F/B Graph 审计

本版保留 v1 原文及其 SHA，修正 teacher cache 位置／批次计数，并加入随后同步的实际 CPU／GPU 图审计；不改写旧训练或 raw。

**判断：profile 反馈对当前 Block 的后继可执行行为有实际、局部的学习收益；同算法 GRU 达到相近或更高水平，尚无独有的组合能力或完整资源优势。新 factorized 字符 LM 已按原协议完成，参数／前向 MAC 更少，但训练墙钟没有改善，质量仍落后官方 Mamba2／GRU。这里支持继续定位学习机制与执行瓶颈，不支持 Primary 冻结或预定胜利。** 无准确率门槛；Transformer 的低分是保留的实际结果，不因此追加调参。

## 核查与证据界限

本 reviewer 直接用标准库 ZIP/NPY 解码七个 closure 模型的 25,088 行 logp／预测，重算每行 argmax、294 个 full／within48／above48 子集的正确数、consistency、joint 与 conditional 分母；七臂 targets／source coordinates 完全相同。另解码四个 AR 模型十二份 TEST 向量，共 917,504 个 target position，重算 NLL／PPL／accuracy，核对所有共同 sampler、metadata、endpoints 与 paired-last64 targets。重加 profile 各 600 条、AR 各 300 条更新记录的时间／MAC，并核对 source／checkpoint／raw SHA。

没有 SSH、安装、训练、模型执行、Torch／NumPy import 或 checkpoint 反序列化。主证据为 [`review047-learning-cost-raw-checks.json`](../../probes/topology-architecture/review047-learning-cost-raw-checks.json)，SHA `6191bd7739c80635b22aab4f784a764cb946a85ec9f9e37ea8e0ab7c0f801603`；标准库复核程序为 [`verify_review047_saved_arrays.py`](../../probes/topology-architecture/verify_review047_saved_arrays.py)。profile 原 TEST driver 未保存逐例向量，其 old/new IID 数字只核对实际 producer、来源与冻结流程，不虚称独立逐例重算。

## 后继反馈到底学习了什么

fcba core 将 value h32、GRU controller c16 与内容寻址 keys／values 分离：controller 看 raw token 与 h；READ／CALL 看 c 与加 slot 的 token；低秩 GELU 算候选 value；八组 sigmoid gate 混合 h 与 LN(h+delta)；WRITE 地址由 c 与 RAW token 决定，写入新 h 本身。它减少 learned write-value projection 并使后续计算可显式读取 value，但参数共享／Adam／replay 仍会改变旧运算，没有拓扑保护定理。

profile source 从完全相同的各自 stage2 NN／Adam／RNG 出发，运行固定 600 步、48 current+16 replay、Adam .003／clip1。在 terminal NLL 外加入

`KL(T_old(word, y) || p_new(prefix followed by word))`，系数 1。

teacher 是 OWN 旧 NN 从已观察 terminal y 出发的输出分布；四个 word 只按原 phase0 TRAIN 首四个不同长度1／2词选择：`[1]、[1,3]、[2]、[2,3]`。32×8 FP32 cache=1,024 B，未引入正确中间状态／调用标签或新的外部标签；当前48行每步额外使用 derived continuation targets。它约束“到达此 terminal 后还能怎样执行”，比仅把 h 拉向 encoder(y) 更贴近可继续计算的接口。teacher 错误也会传递；有限四词约束不能唯一识别通用算子。该具体收益同样可由普通 GRU 获得。

以下是 MIX15 的同一 256 个 base cases；closure 是新增的 postdesign DEVELOPMENT 数据，基于既有 TRAIN source256–511 的后继，不冒充独立 held-out TEST。SET0/1_STEP 是明确训练重叠的 cache词；NOP插入／四词／CLEAR 不在四词 cache 中，但其 primitives 已属旧任务，并非全新运算的发现。

| 模型 | empty 正确 | SET0_STEP | SET0_NOP_STEP | SET1_NOP_STEP | 四词组合 | CLEAR |
|---|---:|---:|---:|---:|---:|---:|
| Block noaux | 160/256 | 6/256 | 5/256 | 70/256 | 2/256 | 204/256 |
| Block profile | 156/256 | 156/256 | 156/256 | 156/256 | 50/256 | 164/256 |
| GRU profile | 160/256 | 160/256 | 159/256 | 160/256 | 64/256 | 135/256 |
| Transformer profile | 43/256 | 42/256 | 43/256 | 31/256 | 30/256 | 4/256 |

Block SET0_NOP_STEP 从1.953%到60.938%是检查过 raw 的局部改善；四词19.531%仍弱，CLEAR从79.688%降至64.063%，不是全方位提升。GRU 的相应 NOP62.109/62.5%、四词25%否定“此反馈只在本 Block 有效”的主张。正确输出不能替代 executable consistency：例如 Block profile SET0_NOP_STEP accuracy60.938%，相对于其 OWN base预测的功能 consistency只有49.219%；四词 consistency14.063%。各 NN own-base-correct 集合不同，conditional准确率不能直接充当匹配的因果比较。长度>48 的少数行与固定<=48子集均保存，主弱项不由删除长行解决。READ10干预条件下 Block profile empty仅11.328%，其他非CLEAR后继约11–13%，未显示完整控制历史鲁棒性。

原任务实际 frozen TEST：profile Block old IID0/1/2=92.383/91.992/82.617%，new IID58.398%；GRU87.109/83.594/81.25%，new63.086%；Transformer49.023/74.414/73.047%，new19.336%。保持、获取、closure与组合是不同指标，不能挑一个合并成总胜利。Transformer低分无已发现的预测输出泄漏或模型置换；v1 GRU失败在0更新的 carry等价检查，v2把 suffix改为相同 packed路径，保留原失败，没有因准确率修配置。一个次要文本错误保留在 Transformer closure 的 `parent_paid_lineage_scope`：误写“GRU44”，但 restore.parent／P=26,760／source／checkpoint 均明确是 Transformer，不能把这个标签当作执行谱系。

## 参数更少，但完整学习账仍贵

| profile 谱系 | P / NN bytes | Adam bytes | replay bytes (+JSON) | stage3 train s | retained四阶段 train s | retained前向 MAC |
|---|---:|---:|---:|---:|---:|---:|
| Block | 13,397 / 53,588 | 107,284 | 55,296 (+1,739) | 118.242640 | 209.758375 | 25.250196G |
| GRU | 25,616 / 102,464 | 204,984 | 55,296 (+1,739) | 5.563319 | 14.867801 | 43.010369G |
| Transformer | 26,760 / 107,040 | 214,252 | 55,296 (+1,739) | 7.836477 | 22.071594 | 151.497972G |

Block retained训练约14.11×GRU墙钟，尽管P约52.3%、forward MAC约58.7%。stage3实际CPU118.636917s、CUDA-stream118.188242s；GRU5.609900／5.527155s，TF7.858093／7.801060s。前向MAC排除反向、非线性、辅助loss、索引与审计，不能当训练FLOPs。cache形成各.042445/.031226/.010184s，包含4个B8 teacher generation与4个B8 comparison forward，即各32行，不能写作32次独立forward调用。报告的teacher forward MAC只计generation；carry拼接预检的1个prefix+4个suffix+4个完整forward在cache timer之前，仅计入整个driver wall，相关MAC没有完整计入。不能从小cache推出免费反馈。

profile三臂 stage3整个driver138.448293/10.621918/12.794221s包含评估、维护、reload等嵌套成本，不与训练项直接相加。对应 final checkpoint332,057/399,157/465,157 B，另保存 maintenance前 safety checkpoint331,786/398,965/464,848 B。Block/GRU/TF reload.103668/.010496/.019203s，maintenance本次.052593/.049268/.048184s；本次额外closure forward3.435707/.404832/.383029s，3584行、147456 active transitions，另付数据／raw与source存储。Transformer还有16,640 B注册buffer。live eval optimizer/cache=0，不会抹去checkpoint内历史Adam／replay或单独历史teacher NPZ的存储与形成成本（checkpoint仅保留teacher元数据／hash，不内嵌cache数值）；replay NumPy数组：`saved_replay_tensor_bytes=0`只计Tensor，actual55,296 B replay在parent资源中。

这些 retained四阶段共有早期训练，必须按谱系共享一次，不能把 noaux／profile 当独立 seed相加。先前 noaux stage3与failed maintenance、原factorized8/600次丢弃更新、GRU pre-SGD失败等开发支出继续留在原证据及review046；本表不是整个campaign总账，不能用retained配方隐藏试错。provider-effective compute、全程GPU busy time／能耗、driver/context分配和完整FLOPs仍未知。

## 新 factorized 字符 LM：协议忠实，质量与速度边界清楚

固定 fcba source `fcba2e82…88158`，driver `3e6447a2…84960`。对 parent567e的diff仅改架构为H32/c16、资源适配与单臂限制，main数据／采样／300×32×64／Adam .003／clip1／VALID节奏／freeze-before-TEST流程保留。原全100×32 encoder仍分配计数；无profile／anchor辅助loss，没有输出替换。27项parameter_counts求和20,721。所有四臂实用同614,400训练label、同TRAIN/VALID/TEST与batch索引；paired-last64每L均同32768个目标。L128/256的full-window PPL另报，不与paired列混用。

| 实际 arm | P | paired PPL64 /128 /256 | train wall s / CPU s | train前向 MAC | NN /Adam /buffer B |
|---|---:|---:|---:|---:|---:|
| factorizedH32/c16 |20,721|10.5930 /10.4980 /10.5602|79.288931 /79.284373|10.341581G|82,884 /165,876 /0|
| 原Transformer36×3 |39,664|10.5329 /15.1848 /14.7357|6.747454 /6.716997|29.815603G|158,656 /317,476 /102,400|
| 原GRU44×2 |32,660|8.6641 /8.5515 /8.5515|2.494903 /2.467893|16.977101G|130,640 /261,324 /0|
| 官方Mamba2×1 |41,116|7.1728 /7.0476 /7.0492|3.557807 /3.556065|未知|164,464 /328,980 /0|

factorized比原Transformer在L128/256的固定最后64目标上好，L64相近；Mamba2与GRU均更好。它的完整train loop仍比原Transformer约11.75×、GRU31.78×慢，不能将约半参数／低MAC写成完整资源优势。factorized soft/native评估共10,240 windows、917,504 positions、33.277600s；另reload.309824s、checkpoint.051320s，整个driver117.671864s、bounded runner122.533959s。各baseline仅soft评估，不能把原driver全时比当严格同工作量吞吐比；固定训练循环和按backend评估成本已分开。

Mamba2不是免费warm baseline：本次全新cache prologue82.666404s（训练形状forward29.949115s/backward49.879118s等），0SGD无任务数据，Tritoncache55,189,171 B/964files；ROOT coordinator92.226986s、本地93.306731s均保留。其3.557807s训练loop不包含prologue，比较resident训练效率可以单列，比较完整生命周期必须付cold费用。既有smoke／失败是另付历史，不能重复算作这次训练，也不能消失。Mamba2 fullFLOPs未测，不能以0补齐。源码源真性为现有安装及固定五文件边界，不扩称全传递依赖认证。

## 随后完成的固定 NN 前后向 Graph 检查

ROOT 后来实际执行并同步了 `parameter-factorized-graph-fb-cpu-admission-v1` 与 `parameter-factorized-graph-fb-gpu-audit-v1`，均 exit0完成。CPU实际admission gate了GPU相同fcba NN、checkpoint `4a9e…fd35a`、同cases SHA `22a4d88eafc49ef8f3556ac24e428e8a1b7e16e34bf09f4d37cd4a7c692d7871`；ROOT-run采用原c9 CPU／85e GPU runner，未用compiler例外，source audit SHA `5d097de134f103af21095190dad05cfa2411e0ddc3a6368d4059b71af879c312`。本review仍未执行NN／SSH。

本review直接解码CPU三份与GPU十五份输出／loss／全部27项梯度的 raw pair，重算250,380个标量的固定atol/rtol，核对所有同case reference逐位一致及计时前／后全部梯度。最大output绝对差1.907349e-6、全参数gradient最大差7.450581e-8，通过原定输出2e-5/2e-5、梯度2e-5/2e-4容差。fixed NN unchanged、SGD=Adam=clip=0；原gather的None-vs-zero／all-present语义有显式gate，graph内清零且grad地址稳定。原始证据派生为 [`review047-factorized-graph-raw-checks.json`](../../probes/topology-architecture/review047-factorized-graph-raw-checks.json)，SHA `1bc2e58b55058be9df030fb20b580bc2de504927e1988fdb89f7c2f2f38b3fc4`。

| B64 bucket | active / padded transitions | gather eager p50 wall ms | masked eager ms | masked Graph ms | Graph vs gather |
|---|---:|---:|---:|---:|---:|
|8|144 /512|13.954762|33.211462|2.961889|4.71×|
|16|144 /1024|14.306277|57.872964|5.904659|2.42×|
|48|2643 /3072|164.809889|154.485054|17.750958|9.28×|

同一case各backend三次warmup、12draw（graph只做三次sidewarmup，计时前一次正确性replay），分位数从raw draws重算。对应CUDA-event Graph p50为2.938512/5.878112/17.705760ms；没有混用wall/event。全行padding额外算3.56×/7.11×/1.16×active transitions，forward MAC分别6,995,968/13,975,552/41,893,888，原gather分别1,979,392/1,979,392/36,045,760。L8/16是固定TRAIN前64个eligible短词，actual max length仅2；不是各阶段完整batch长度分布。masked eager的慢项说明padding会增加算术成本，Graph收益仍是执行实现而非学习机制变优。

冷启动／pool不可省略：三case sidewarmup .110313/.162802/.469545s，capture .087636/.098224/.220130s，instantiate .011137/.020188/.070333s；独立private pool、内存峰值在capture前reset。capture记录的CPU autograd工作也计时，不能把capture描述的一次F/B当已经执行的一次GPU更新。GPU science10.290965s/CPU10.287052s、remote coordinator14.485539s／reaped CPU14.479808s、本地15.455200s；CPU admission science.787754s／CPU.788175s、remote4.509397s、本地5.505653s。共147次实际GPU F/B、另3次capture描述，9次实际CPU F/B；0训练更新。snapshot/raw写出／import／setup都在whole费用内，nested费用不可相加。

仍保留全部13,397P／53,588B NN及53,588B gradient buffers，历史Adam107,284B只作已付checkpoint负担，audit没有构造live Adam。GPU各capture后peak allocated38,734,336/59,046,400/89,175,040B，peak reserved52,428,800/75,497,472/111,149,056B；这些是allocator快照／整个审计的当时工作集，不是精确独立pool大小或纯NN内存，driver分配未知，不能以graph reset推定所有运行时存储归零。

这个结果缩小了前后向launch瓶颈，尚未训练。未来ordinary原地weight／Adam数值更新本身不必导致recapture；参数／gradient storage地址、shape或growth改变则须重新warmup／capture并收费。eagerAdam／clip、每批静态input copy／batch、validation／growth／maintenance与各阶段recapture成本尚未在新Graph训练loop实测，也未证实任意weights／inputs上的完整优化器轨迹等价，不能把固定checkpoint 4.71×等直接推广为整段203秒训练的已实现加速。

`CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`广泛能力与完整成本优势 | UNRESOLVED，当前不成立为已证结果 | 本review raw/source账；review046融合账；本次实际固定NN Graph F/B | 单seed／局部closure，组合仍弱；完整Graph学习轨迹／生命周期尚未实测，P匹配控制未执行 | anchor1、低TF分数、GRU carry v1、maintenance失败全部保留 | ROOT在既有授权范围内判断是否把通过的精确masked Graph接入有界训练；本review不启动训练，不追加准确率调参。`
