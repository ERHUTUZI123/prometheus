# 六尾实验完整已测结果

同一N8四阶段任务；每个模型各自相同stage2 NN/Adam/RNG/Replay恢复，新执行600步。三模型×配对/同池打乱。开发seed0。所有TEST均在该尾段训练冻结后打开。普通基线native是同一个NN接口别名。

## 新任务与旧IID（soft，百分比）

| 模型/采样 | P | 新VALID | 新IID | 新long | 新unseen | 旧0 IID | 旧1 IID | 旧2 IID | 新native IID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prometheus typed/paired | 10,991 | 26.17 | 27.93 | 27.15 | 24.61 | 84.77 | 86.33 | 87.30 | 20.90 |
| Prometheus typed/shuffle | 10,991 | 62.50 | 58.01 | 62.50 | 62.50 | 88.87 | 86.13 | 87.11 | 45.90 |
| GRU32/paired | 14,024 | 61.52 | 62.89 | 61.72 | 62.50 | 82.42 | 83.59 | 81.25 | 62.89 |
| GRU32/shuffle | 14,024 | 62.70 | 62.50 | 62.30 | 62.50 | 85.35 | 85.74 | 83.20 | 62.50 |
| Transformer32/paired | 26,760 | 28.52 | 25.39 | 22.46 | 25.00 | 49.41 | 68.75 | 71.48 | 25.39 |
| Transformer32/shuffle | 26,760 | 49.02 | 46.68 | 33.59 | 46.29 | 47.07 | 69.14 | 70.12 | 46.68 |

## 全部任务的IID/long/unseen（soft，百分比）

| 模型/采样 | 任务 | IID | long | unseen |
|---|---|---:|---:|---:|
| Prometheus typed/paired | 旧0 | 84.77 | 19.14 | 57.81 |
| Prometheus typed/paired | 旧1 | 86.33 | 58.98 | 19.14 |
| Prometheus typed/paired | 旧2 | 87.30 | 69.34 | 28.52 |
| Prometheus typed/paired | 新3 | 27.93 | 27.15 | 24.61 |
| Prometheus typed/shuffle | 旧0 | 88.87 | 27.54 | 89.45 |
| Prometheus typed/shuffle | 旧1 | 86.13 | 57.81 | 14.84 |
| Prometheus typed/shuffle | 旧2 | 87.11 | 67.97 | 22.07 |
| Prometheus typed/shuffle | 新3 | 58.01 | 62.50 | 62.50 |
| GRU32/paired | 旧0 | 82.42 | 16.41 | 43.36 |
| GRU32/paired | 旧1 | 83.59 | 46.48 | 13.48 |
| GRU32/paired | 旧2 | 81.25 | 69.14 | 33.01 |
| GRU32/paired | 新3 | 62.89 | 61.72 | 62.50 |
| GRU32/shuffle | 旧0 | 85.35 | 21.68 | 47.66 |
| GRU32/shuffle | 旧1 | 85.74 | 46.68 | 16.41 |
| GRU32/shuffle | 旧2 | 83.20 | 67.77 | 41.41 |
| GRU32/shuffle | 新3 | 62.50 | 62.30 | 62.50 |
| Transformer32/paired | 旧0 | 49.41 | 11.13 | 20.12 |
| Transformer32/paired | 旧1 | 68.75 | 12.30 | 22.46 |
| Transformer32/paired | 旧2 | 71.48 | 41.41 | 25.78 |
| Transformer32/paired | 新3 | 25.39 | 22.46 | 25.00 |
| Transformer32/shuffle | 旧0 | 47.07 | 10.74 | 23.63 |
| Transformer32/shuffle | 旧1 | 69.14 | 22.66 | 22.66 |
| Transformer32/shuffle | 旧2 | 70.12 | 32.62 | 25.20 |
| Transformer32/shuffle | 新3 | 46.68 | 33.59 | 46.29 |

## 成本（新600步；时间嵌套，不相加）

| 模型/采样 | train s | driver s | SSH s | soft推理p50 ms | soft推理p95 ms | 权重B | Adam B | Replay B | buffer B | CUDA峰值B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prometheus typed/paired | 11.740 | 31.713 | 37.909 | 42.041 | 44.995 | 43964 | 88008 | 55296 | 0 | 51525120 |
| Prometheus typed/shuffle | 11.708 | 32.333 | 39.769 | 42.572 | 46.648 | 43964 | 88008 | 55296 | 0 | 51528192 |
| GRU32/paired | 4.843 | 10.324 | 16.261 | 1.649 | 1.693 | 56096 | 112248 | 55296 | 0 | 22766080 |
| GRU32/shuffle | 5.176 | 10.387 | 16.170 | 1.637 | 1.659 | 56096 | 112248 | 55296 | 0 | 22944768 |
| Transformer32/paired | 6.014 | 11.049 | 16.887 | 1.080 | 1.186 | 107040 | 214252 | 55296 | 16640 | 48002048 |
| Transformer32/shuffle | 6.951 | 11.916 | 18.042 | 1.060 | 1.113 | 107040 | 214252 | 55296 | 16640 | 48002560 |

全部144个TEST接口/任务/划分的准确率、NLL与成本字段见同名CSV。全部学习曲线见six-paired-tails-curves-v1.csv。

共享索引+两项CPU恢复检查一次计费后，六尾新增协调端总168.149337s；未包含整个campaign历史开发费。完整训练FLOPs、硬件能耗与整轮付费账单unknown。typed两臂使用训练Graph，但这里推理是eager；不同模型阶段2的初始能力不同。

## 原始来源SHA256

- `parameter-typed-paired-tail-paired-n8-seed0-600-v1/learn/result.json` SHA `aebfb41c41946aee611ece3e7e92bee0cf2673978b4182412f718f289462b028`；final checkpoint SHA `5bb678b87a688cb373b001b5054a1cacbc36b5cd2f3c49012bc97658345a6bc8`。
- `parameter-typed-paired-tail-same-pool-shuffle-n8-seed0-600-v1/learn/result.json` SHA `c669043dd45a68ef53e20f1cd76db08012140745e28e06e95da737723311a976`；final checkpoint SHA `cbaf9828486a17458e45e561a7ff2e47c17da049cafd84e809f9563b27280abb`。
- `parameter-paired-tail-gru32-paired-n8-seed0-600-v1/learn/result.json` SHA `d2b9ce9211d009836364a750880357974a9c95ec2e36ab1d6cbd6108ee61e88e`；final checkpoint SHA `8af64b0d49e1ffe15a9373e2e53505801705b766e8b51893daab5df2768bb479`。
- `parameter-paired-tail-gru32-same-pool-shuffle-n8-seed0-600-v1/learn/result.json` SHA `76e7cbaf32365091dfc70ed1fcf28e4dcced6aa8f6d2a42c1768f64a8f447983`；final checkpoint SHA `e3bfe53b260dff38c2b7386fd97ef1e2b085d10f026eb7b29bb1a43c48700cbb`。
- `parameter-paired-tail-transformer32-paired-n8-seed0-600-v1/learn/result.json` SHA `836e7b0597068c34600a3393b288322c72b77589446587fdaec7aeaca99b356e`；final checkpoint SHA `06a6c2e00d06d84fd46e92f5379ca317633e762c9db61779bdf4e4a3e3ad686e`。
- `parameter-paired-tail-transformer32-same-pool-shuffle-n8-seed0-600-v1/learn/result.json` SHA `2f8f92e3a33dddc4b0d73c7f10cd580eca6d7c377bfe1a9dadaadc70035a9e68`；final checkpoint SHA `5b6ab57c440cde3f75b1e3614b0657edc0069f51782b71ef7cc1a30351de240a`。
