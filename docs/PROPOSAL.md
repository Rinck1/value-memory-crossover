# 条件价值查表记忆的交叉临界规律、现象学模型与跨层多粒度容量分配（Research Proposal v5）

> **核心定位与实证突破：**  
> 本方案聚焦离散 N-gram 查找表（Value Embeddings）在**等激活计算量（Equal Active FLOPs）**条件下突破密集主干容量瓶颈的机制与工程设计。在 $V=50{,}257$ 全词表 `30M`-token 真实预训练桥接实验中，完整方案 **`U123`**（`4.8234`）不仅较同计算量基线 `plain_d256`（`4.9855`）降低 **`0.1621` nats**，而且连续超越 $2.25\times$ 激活算力的加宽密集主干 `plain_d384`（`4.8570`），并在 `30M` 处完成对未剪枝哈希对照 `CTRL_hs`（`4.8324`）的交叉反转。  
> 完整黑白学术排版 PDF 见 [ICML_Research_Proposal_v5.pdf](./ICML_Research_Proposal_v5.pdf)。

---

## 1. 核心正面实证与机制解释

1. **`<UNK>` 混杂消除，全频次交叉律全面打通**（[unk_filtered_eval.json](../results/unk_filtered_eval.json)）：  
   在 $V=1{,}500\text{--}6{,}000$ 上，1-gram 表在更新次数 $c<100$ 的低频区稳健领先，2-gram 表在 `100–999` 区间反超。此前 `1000+` 桶混杂源于截断词表中 `51.94%` 词对类型（`67.83%–71.52%` 验证位置）含 `<UNK>`；剔除含 `<UNK>` 输入对后，`pc_1000` 全线翻转为 2-gram 净胜（`Engram - Word`: `-0.0380 / -0.0501 / -0.0241`；`MP-VE - Word`: `-0.0506 / -0.0580 / -0.0456` nats）。
2. **零先验优化方向印证与多 Seed 高显著性胜出**（[p0_results.json](../results/p0_results.json) & [upgrade_search_results.json](../results/upgrade_search_results.json)）：  
   - 零权重全 `0.5` 初始化的 5 桶直通门控在反传下自发将 `1–9` 欠训练桶压至 `0.12–0.16`、将 `100+` 成熟桶拉至 `0.89–1.00`，从第一手梯度动力学证实最优方向。
   - **5-seed 主线**（$L=4, w=32, V=6{,}000$）：`mp_ve_last2_dense` 参数较 `hard_stack` 砍半（`982K` vs `1.92M`）仍反超 `+0.0136±0.0096`（`t=+3.19`）；`mp_ve_all4_dense` 领先 `hard_stack` **`+0.0228±0.0050`（`t=+10.29`）**、领先 `word` **`+0.0366±0.0083`（`t=+9.89`）**；10% 前缀路由（`5.3496`）与切换 `AdamW` 骨干（`5.3745` vs `word 5.4123`）均保持稳健优势。
   - **Round 30 跨词表/尺寸持出泛化**（$V=4{,}500, d=48, L=4$）：5/5 分桶预测符号全中，全局对比绝对误差仅 `0.0016 / 0.0029 / 0.0088` nats（全低于 `0.015` 阈值）。
   - **匹配控制臂胜出**：`U123` 达到 **`5.3190 ± 0.0068`**，以少 **35%** 表参显著击败结构完全对齐的 `CTRL_hs`（`+0.0159±0.0055, t=+4.96`）与 `CTRL_word`（`+0.0481±0.0042, t=+19.74`）。
3. **三个可检验现象学模型闭环组织规律**：  
   - **KKT 三档分配**解释 `0 / r / d` 路由及条件性 `3–12×` 带宽比；$d=24, 48$ 持出检验在 `b_500_1000`（`+0.0175 / +0.0346` nats）拒绝朴素线性 $\alpha=1.0$，且 `MP-VE` 全局净增益在最宽处 $d=128$ 增至 `+0.0179` nats。
   - **Zipf–Heaps `T_cross` 三项分解**（剪枝去污染 $-\Delta_{\text{prune}}$、精确去碰撞 $+\Delta_{\text{exact}}$、预热税 $-\Delta_{\text{warmup}}$）在全词表 `30M` Bridge 获定量实证。
   - **跨层共享方差缩减假设**（`SNR_eff ∝ L/(1+(L-1)ρ_noise)`）解释 `U1` 带来的协同增益放大。

---

## 2. 全词表 `V=50,257` Bridge 与工程账本

![Fig 7 Bridge](../figs_v5/fig7_bridge.png)

| Arm (`V=50,257, 30M tokens`, Seed 0) | Val Loss | `pc_0` | `pc_100` | Table Params | B1 (ms) | B16 Infer (tok/s) |
|---|---|---|---|---|---|---|
| `plain_d256_eqFLOP` | 4.9855 | 5.0715 | 4.7376 | 0 | 15.68 | 287.3K |
| `word_d256_last2` | 4.8961 | 4.9432 | 4.6631 | 25.7M | 12.74 | 327.4K |
| `mp_ve_d256_prefix10` | 4.8697 | 4.9314 | 4.6254 | 69.4M | 14.00 | 286.4K |
| `U1_shared_d512_all6` | 4.8658 | 4.9305 | 4.6271 | 69.4M | 15.29 | 276.9K |
| `plain_d384_eqWeights` (`2.25x` FLOPs) | 4.8570 | 4.9012 | 4.6261 | 0 | 13.51 | 250.2K |
| `CTRL_hs_d256_prefix10` | 4.8324 | 4.9028 | 4.6003 | 76.9M | 18.32 | 240.9K |
| `CTRL_hs_pruned_d256_prefix10` | 4.8263 | 4.8597 | 4.6064 | 76.9M | 18.30 | 242.8K |
| **`U123_d256_prefix10`** | **4.8234** | **4.8537** | **4.5833** | 76.5M | 18.93 | 224.3K |

- **全词表突破与三项归因**：`U123` 全程 4 个节点持续胜过 $2.25\times$ 算力的 `plain_d384`（`+0.0600 -> +0.0720 -> +0.0557 -> +0.0336`）；同时如期实现相对 `CTRL_hs` 的先输后赢交叉反转（`7.5M: -0.0183 -> 15M: -0.0001 -> 22.5M: +0.0041 -> 30M: +0.0090`），其中 `CTRL_hs_pruned` 剥离 `pc_0` 剪枝去污染 `+0.0431` nats，精确 `2g/3g` 银行在 `pc_100` 净胜 `+0.0231` nats。
- **Zipf 压缩可行性与推理优势**：100M tokens 中仅 `0.954%`（`109,774`）词对覆盖 `55.76%` 验证位置，使全词表容量从 `8.8B` 压缩至 `76.5M`；推理期保持 `224K–327K tok/s` 高吞吐（`U1_shared` B1 延迟 `15.29 ms` 持平基线）。同时坦诚汇报两项约束边界：短中程小规模下 Equal Total Weights 密集主干仍更优，且 `SparseRowAdam` 劣于 Dense Adam（`-0.0184±0.0023, t=-13.60`）。

---

## 3. `125M × 2.5B` 规模化实验计划

基于 [pairstat_v50257.json](../results/pairstat_v50257.json)，`2.5B` tokens 下成熟词对覆盖率下界超越 `82.7%`。云端执行脚本为 [train_125m.py](../train_125m.py) 与 [run_125m_experiments.sh](../run_125m_experiments.sh)（详见 [CLOUD_AGENT_HANDOFF.md](../CLOUD_AGENT_HANDOFF.md)）：
- **实验波次**：Wave 1（`plain`, `u123`, `ctrl_hs_pruned`, `word`）$\to$ Wave 2（`ctrl_hs`, `ctrl_word`, `plain_eq_weights` $d=1024$）$\to$ Wave 3（`mp_ve`, `u1` 与核心 4 臂多 Seed 复现）。
- **机制测量与下游评估**：运行 $M \in \{6\text{k}, 24\text{k}, 96\text{k}\}$ 与 $L \in \{2,4,6\}$ 检验（[run_theory_tests_L_M_rho.py](../experiments/run_theory_tests_L_M_rho.py)）及 `lm-eval` 零样本下游基准测试。
