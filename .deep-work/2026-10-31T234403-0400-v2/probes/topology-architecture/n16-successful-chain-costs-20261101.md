# N16 成功链成本证据（非运行计划）

M0 未通过。记录当前成功链与开发额外消耗，主要基线为 Transformer。

成功链：原阶段0的800步 → 原阶段1的800步 → 数据约束形成后200步 → 修订父检查点上阶段2的800步 → 历史形成后200步。合计2800次梯度更新。

| 片段 | 参数 | 同步训练秒 | 评估秒 | 保护秒 | 已记录 elapsed 秒及范围 |
|---|---:|---:|---:|---:|---|
| task0-source800 | 2642 | 8.0368 | 12.8103 | 0.8455 | 22.3993; stage wall, not entire source invocation |
| task1-source800 | 2642 | 10.5215 | 11.9606 | 1.0434 | 24.3309; stage wall, not entire source invocation |
| task1-formation-extra200 | 3254 | 2.9464 | 6.2504 | 0.4525 | 14.0563; standalone repair internal wall including initialization/search |
| task2-repaired-parent800 | 10984 | 16.6344 | 13.7650 | 2.1335 | 34.3910; stage wall, not entire source invocation |
| task2-history-extra200 | 16354 | 3.2525 | 7.4459 | 0.8675 | 16.3500; standalone repair internal wall including initialization/search |

上述同步训练片段之和 41.3917 秒；评估 52.2322 秒。它们不是单次端到端成功运行的总耗时。

形成在追加SGD前的step0已经达到100%；后续200步检查能力保持与持续更新。额外搜索、重复标签访问、神经安装与连续执行全部在JSON中单列；嵌套计时不可重复相加。

三个强基线亦合计2800次更新，包括原0/1阶段成本；新增1200步的实际样本索引已与对应修订逐字节核对。Replay使用一半current、一半old，标签重复访问与普通学习方式不同。

原失败阶段2、OFF消融、缓存重训、形成/执行归因和数值探针属于额外开发消耗，已分列，未从成本记录删除。原共享调用的初始化/收尾开销未武断分摊。

尚缺重复端到端计时、能耗、完整RAM/VRAM、FLOPs及长期扩展成本；当前不能宣称完整资源优势。详见同名JSON及 step3-contribution-results.json / step3-learning-curves.csv / step3-costs.csv。
