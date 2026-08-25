# OF-base 与 OF-ApexTarget：HM3Dv1/v2 四组 full 配对审计

日期：2026-08-25 UTC
状态：冻结结果、日志与代码只读审计；未修改 navigation、frontier、target pipeline、模型或阈值

## 1. 范围与口径

本报告只比较四个 full，不引入 T1、H1、H2、random-100 或其他实验：

| 名称 | 冻结结果 |
| --- | --- |
| OF-base v1 | `results/openfrontier_base_sam3_full_hm3dv1_2000_seed20260727` |
| OF-ApexTarget v1 | `results/of_apex_full_hm3dv1_2000_seed20260727` |
| OF-base v2 | `results/openfrontier_base_sam3_full_hm3dv2_1000_seed20260727` |
| OF-ApexTarget v2 | `results/of_apex__full_hm3dv2_1000_seed20260727` |

- v1 两侧各 2000 个 episode JSON，index 连续，manifest SHA-256 均为
  `8f57bfb1a1a72a012d2fdb7634a5290304b6797c5058e24e778e277b3855d731`，是严格 paired。
- v2 两侧各 1000 个 episode JSON。manifest 文件因运行元数据不同而哈希不同，但忽略
  `selection_mode/source_manifest` 后，`(index, scene, episode_id, target)` 逐项一致；已有 V2
  审计也验证了 episode JSON 与各自 manifest 1000/1000 对齐。
- 同一 dataset version 内，环境、episode、frontier、planner、Qwen frontier selector 相同；主要
  自变量是 target recognition/approach。目标模块会提前改变轨迹，因此 paired difference 是目标
  模块的完整下游效应，不是单帧 detector accuracy。

HM3Dv1 冻结 runner 以 0.1 m 为原始协议，只保存了 OF-base 的 0.1 m summary。为与 ApexTarget
统一，本报告从同一冻结轨迹离线重建 1 m 指标：显式 STOP 且最终 geodesic
`distance_to_goal < 1.0 m` 记为成功；新纳入成功的路径效率利用数据集 episode geodesic distance
与 Habitat soft-SPL 恒等式恢复。该方法在 OF-base 1000 个已有严格成功和 ApexTarget 1086 个已
保存的 1 m 成功上交叉验证，最大 SPL 绝对误差分别为 `2.39e-4` 和 `6.84e-5`。

## 2. 四组主结果

| Dataset | Method | SR@1m | SPL@1m | SR@0.1m | SPL@0.1m | Exceptions | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HM3Dv1, N=2000 | **OF-base** | 54.00% | **0.2747** | 50.00% | **0.2503** | 4 | 301,471 s |
| HM3Dv1, N=2000 | OF-ApexTarget | **54.30%** | 0.2472 | **50.45%** | 0.2300 | 3 | 416,913 s |
| HM3Dv2, N=1000 | **OF-base** | **70.80%** | **0.3299** | **39.50%** | **0.1821** | 0 | 157,137 s |
| HM3Dv2, N=1000 | OF-ApexTarget | 65.20% | 0.2790 | 37.60% | 0.1580 | 2 | 217,487 s |

关键修正是：ApexTarget 在 v1 的 1 m SR 正提升只有 **+0.30 pp（6/2000）**，不是稳定优势；
paired exact McNemar/binomial `p=0.799`。同时 SPL 下降 `0.0274`，运行时间增加 38.3%。v2 则是
OF-base 明确领先：SR `+5.60 pp`、SPL `+0.0509`，paired `p=0.000169`，运行时间少 27.7%。

## 3. Paired flips 与类别异质性

| Dataset | Both success | OF-base only | Apex only | Both fail | 净 OF-base flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 890 | 190 | 196 | 724 | -6 |
| v2 | 572 | 136 | 80 | 212 | +56 |

### 3.1 HM3Dv1

| Category | N | Base SR/SPL@1m | Apex SR/SPL@1m | Apex ΔSR | Base-only / Apex-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| bed | 433 | 51.96% / 0.2487 | 51.96% / 0.2423 | 0.00 pp | 33 / 33 |
| chair | 428 | 61.92% / 0.3412 | 65.65% / 0.3081 | +3.74 pp | 43 / 59 |
| plant | 84 | 32.14% / 0.2087 | 36.90% / 0.1617 | +4.76 pp | 9 / 13 |
| sofa | 376 | 60.64% / 0.3317 | 63.56% / 0.3229 | +2.93 pp | 24 / 35 |
| toilet | 398 | 54.02% / 0.2152 | 48.74% / 0.1785 | -5.28 pp | 51 / 30 |
| tv_monitor | 281 | 42.70% / 0.2410 | 41.28% / 0.1839 | -1.42 pp | 30 / 26 |

v1 的总结果是类别收益互相抵消：Apex 的 chair/sofa 精度倾向被 toilet/TV 的召回和闭环损失
抵消。plant 只有 84 个 episode，净增仅 4 个，不能与 v2 的 plant 大幅退化相抵消。

### 3.2 HM3Dv2

| Category | N | Base SR/SPL@1m | Apex SR/SPL@1m | Apex ΔSR | Base-only / Apex-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| bed | 165 | 80.61% / 0.3711 | 81.82% / 0.3527 | +1.21 pp | 10 / 12 |
| chair | 195 | 82.05% / 0.4154 | 85.64% / 0.3851 | +3.59 pp | 13 / 20 |
| plant | 152 | 59.87% / 0.2708 | 32.89% / 0.0936 | -26.97 pp | 51 / 10 |
| sofa | 187 | 71.12% / 0.3466 | 74.87% / 0.3763 | +3.74 pp | 13 / 20 |
| toilet | 166 | 66.27% / 0.2730 | 52.41% / 0.2011 | -13.86 pp | 30 / 7 |
| tv_monitor | 135 | 60.00% / 0.2695 | 54.07% / 0.2053 | -5.93 pp | 19 / 11 |

跨 v1/v2 稳定的倾向不是“Apex 总体更强”，而是：Apex 对 chair/sofa 更保守、更精确；对
plant/toilet/TV 更容易遭遇 detector recall、几何过滤、两次正观测 gate 或 approach 损失。

## 4. 共同成功时的效率

| Dataset | Metric | OF-base mean / median | Apex mean / median |
| --- | --- | ---: | ---: |
| v1, N=890 | SPL@1m | **0.5245 / 0.5247** | 0.4612 / 0.4187 |
| v1, N=890 | navigation steps | **131.7 / 97** | 173.2 / 133 |
| v2, N=572 | SPL@1m | **0.4725 / 0.4669** | 0.4296 / 0.3911 |
| v2, N=572 | navigation steps | **128.1 / 104** | 159.2 / 122 |

v1 共同成功中，首次 target visible 基本相同（Base 72.5/37，Apex 71.1/37）；但首次可用候选
为 Base 91.0/59、Apex 118.9/84，首次接管为 Base 127.2/92、Apex 138.3/99.5。Apex 的多帧
fusion 和显式 approach 并未换来总体 SR，只稳定带来了额外步数与 SPL 损失。

## 5. 目标模块到底改变了什么

### 5.1 OF-base

OF-base 使用 SAM3 在最近六帧 composition 上产生 target masks，将 mask+depth 变成 object
frontier；到候选视点后，Qwen 判断整张 composition 是否存在目标。Qwen 没有明确验证当前被选中
的 mask，所以“画面里有正确类别”与“锁定的是正确 3D centroid”之间存在关联缺口。接受后，以
`path exhausted OR distance-to-centroid < 1m` STOP。

### 5.2 OF-ApexTarget

ApexTarget 每帧运行 YOLOv7（COCO 类别；非 COCO 时 DINO）和 bbox-MobileSAM，再以 0.1 m
voxel 做 3D cluster fusion。可靠目标要求 confidence `>=0.65`、positive volume `>=8`、至少
两次正观测，并加入 confusable-label competition 与弱负证据；之后执行独立 approach，水平停止
阈值为 0.9 m。它绕过 OF-base 的 Qwen target verification。

因此 ApexTarget 同时做了两件事：

1. 用多帧、候选绑定的几何证据减少快速 false commitment；
2. 引入 detector label recall、SAM/geometry acceptance、cluster association、two-positive gate
   和更长 approach 五个新的失败点。

### 5.3 v1 discordant pairs

190 个 **OF-base-only** 中，ApexTarget 的首个未解决阶段为：

| Apex stage | N |
| --- | ---: |
| GT visible，但同帧无 target-label YOLO box | 49 |
| 有几何候选，但始终未 reliable | 47 |
| target 从未达到 50 pixels visible | 29 |
| reliable 后无显式 STOP | 25 |
| target YOLO box 被 SAM/geometry 拒绝 | 20 |
| 显式 STOP 但仍在 1 m 外 | 19 |
| exception | 1 |

196 个 **Apex-only** 中，OF-base 的首个未解决阶段为：

| OF-base stage | N |
| --- | ---: |
| Qwen accepted 并 STOP，但仍在 1 m 外 | **140** |
| 有 candidate，但 Qwen 未接受 | 32 |
| Qwen accepted，但无显式 STOP | 16 |
| never visible | 5 |
| visible，但无 SAM3 candidate | 3 |

140 个错误承诺中，55 个接受窗口没有足够 GT target pixels，85 个虽看见目标但锁定位置/终点
仍不满足 1 m。Apex 的 v1 正面价值是真实的 precision/grounding 改善，但它被 190 个召回、可靠
性和 approach 损失几乎完全抵消。

### 5.4 v2 discordant pairs

已有逐事件审计显示，136 个 **OF-base-only** 中 Apex 停在：同帧 YOLO miss 60、candidate 后
未 reliable 28、reliable 后 stuck/max-steps 17、never visible 12、SAM/geometry rejection 10、
STOP 失败 7、exception 2。60 个 YOLO miss 中 41 个是 plant；28 个未 reliable 全部没有满足
第二次正观测。

80 个 **Apex-only** 中 OF-base 停在：错误 STOP 34、never visible 19、candidate 后 Qwen 未接
受 16、接管后 stuck/max-steps 7、visible 但无 candidate 4。错误 STOP 主要是 chair 11、sofa
13。这再次表明 Apex 的价值集中于 precision，而总损失集中于 recall 和 downstream closure。

## 6. 两种方案共同具备的系统问题

以下“上限”是日志责任桶占全部 episode 的比例，不是可相加的反事实收益；“预期”是针对 v2
主线、保持其他模块不变的保守工程判断。

| 共同问题 | 冻结证据与诊断上限 | 保守预期 | 破坏风险 | 优先级与建议 |
| --- | --- | ---: | --- | --- |
| visual-only proposal 不保证覆盖 | OF-base v2 有 59 个 never-visible（5.9 pp）；v1 有 348 个（17.4 pp）。Apex 也分别有 91/533 个 | **+1–3 pp** | 低—中；无约束扩大候选池时为高 | **P0 research**：visual 保持主来源，只把 accumulated geometric frontier 作为有来源标记的补充 |
| selector 无可观测的最优性诊断 | 当前只保存 Qwen 概率，无法判断“正确候选已存在但选错” | 先不报提点 | 极低 | **P0 instrumentation**：同一 candidate set 上记录 navmesh-to-GT shadow oracle、rank/regret，不控制轨迹 |
| target candidate 与验证/3D endpoint 不一致 | OF-base v2 107 个 accepted-STOP-fail；v1 373 个。v1 Apex-only 中 140 个直接来自此阶段 | **+2–4 pp** | 中；hard reject 会损失真阳性 | **P1**：候选绑定验证、多候选选择、歧义候选保留，而不是全局加严类别阈值 |
| target centroid/STOP 与 evaluator success region 不一致 | OF-base v2 19 个 accepted-no-STOP，另有 17 个 1–1.5 m near miss；Apex 同样出现 reliable 后无 STOP | **+0.5–1.5 pp** | 中—高；全局提前 STOP 会制造失败 | **P1**：单独审计 endpoint、可达 success region 与显式 STOP 闭环，不与识别阈值一起改 |
| 目标 proposal recall/观察视角不足 | OF-base v2 57 个 visible-no-SAM3-candidate（5.7 pp 上限）；Apex v2 有 158 个 visible-frame YOLO miss | **+1–2 pp** | 中 | **P2**：先由 frontier 提供第二观察视角，再评估 SAM3/YOLO；不先换模型或堆类别阈值 |
| FrontierNet 下游实现无 completeness 保证且有上游细节不一致 | 漏检只有未来重见才能恢复；聚类实际不使用 gain；depth-gradient helper 存在 degree/radian 接口不一致 | 未知，不应伪造 | 高；改变全部轨迹 | **P2 isolated ablation**：在 frontier 主 idea 后分别修，不能捆绑成系统补丁 |

即使后续各项都有效，收益也不能相加。对 v2，完成 frontier coverage、candidate-grounded target
和 STOP 闭环后的合理总预期是 **+3–6 pp SR**，而不是按失败桶宣称十几个点；其中当前 frontier
idea 自身先以 **+1–3 pp** 作为成功预期更稳健。

## 7. ApexTarget 的定位

结论：**保留为实验模块和 precision 机制参考，不作为 v1 或 v2 默认 target module。**

- v2 有显著 SR/SPL 净损失，不能进入主线。
- v1 的 +0.30 pp SR 不显著，且 SPL、步数、运行时间全面退化；不能称为有效总体提升。
- 值得复用的是“候选绑定、多帧正证据、confusable competition”的思想，不是整套 YOLO→SAM→
  voxel fusion→two-observation→approach 管线。
- v1 说明 OF-base 的快速错误承诺非常真实；v2 说明用 Apex 整体替换它会付出更大的 recall 代价。

## 8. v2 主线与 v1 的参考价值

以 v2 作为主要实验基准是合理的：它使用明确的官方 1 m 口径、类别更均衡，且 Apex 的失败原因
在 1000 个严格 paired episode 上有更清楚的逐阶段证据。

v1 仍应保留为**二级泛化检查**，理由不是跨楼层，而是：

1. 2000 episodes 提供更大样本；
2. 它揭示了不同类别分布下 precision/recall trade-off 会改变总 SR；
3. v1 的 140 个 Apex-only/OF-base 错误承诺，是 candidate-grounded refinement 的强独立证据；
4. 一个只在 v2 提升、却在 v1 大量破坏成功 episode 的补丁，不应轻易称为系统性改进。

因此开发和选型看 v2，最终外部有效性检查再看 v1；不围绕 v1 单独调参。

## 9. 接下来先验证 frontier idea 是否可行

**可行，而且优于现在打系统性补丁。** 推荐顺序：

1. 冻结 OF-base，不改 target pipeline；加入 selector shadow oracle 和 `source=visual|geometric|object`
   诊断，行为保持不变。
2. 在独立分支实现 accumulated geometric frontier 的最小闭环；不替换 visual frontier，不同时修
   degree/radian、gain、SAM3 或 STOP。
3. 建立 40–60 episode paired 集：探索型失败、OF-base 成功保护集、若干 Apex 成功/Base 失败
   case。检查能否救回失败以及是否破坏原成功轨迹。
4. 最多一个 episode 确认可运行，随后直接跑冻结小集；达到正向信号后再跑 v2 full。
5. frontier 结论冻结后，依次做 candidate-grounded target 与 STOP closure；不要一次打系统补丁，
   否则无法归因研究贡献。

## 10. Git 回退与分支纪律

- OF-base 代码与轻量 full 日志位于远端 `main`；应以 annotated tag
  `of-base-full-v1-v2-20260727` 冻结本报告对应提交。
- OF-ApexTarget 完整代码及两组轻量 full 日志位于远端 `apextarget-experimental`，commit
  `f99514f2ce35b72fce719f67d59c5572ab11bda4`；应以 tag
  `of-apextarget-full-v1-v2-20260727` 冻结。
- 后续实现从 OF-base tag 新建 `research/accumulated-geometric-frontier`，不直接在 `main` 开发。
- 每个正式实验记录 base commit、branch、manifest SHA-256 和 source tag；完成 paired full 后再
  决定是否合并。

这样可以随时精确回退到两套 target pipeline，也避免 frontier 研究污染当前复现基线。
