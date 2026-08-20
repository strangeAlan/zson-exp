# OF-ApexTarget v1 目标识别中期分析（已归档）

状态：**已归档；不得再用作最终结论**<br>
快照日期：2026-08-13 UTC<br>
对应结果目录：`results/openfrontier_apextarget_v1_deterministic_full_hm3dv1_2000_seed20260727`<br>
快照进度：799/2000 episodes<br>
代码基线：`OF-ApexTarget v1 deterministic`

> Full HM3Dv1 已于 2026-08-16 确认完成。最终 2000-episode 数据、跨楼层分层和失败漏斗见
> [`HM3DV1_FULL_AUDIT.md`](HM3DV1_FULL_AUDIT.md)。本文仅保留 799-episode 历史快照，所有
> 数字均已被最终审计替代。
>
> 重新阅读或使用本文前，先查看结果目录的 `summary.json`、`progress.log` 和
> `episodes/*.json`。如果已经达到 2000/2000，必须重新计算本文全部类别表格、failure
> counts 和 detector funnel，并把标题状态改为最终分析。不能把这里的 799-episode
> 数字当作论文结果或最终优化依据。

本文只分析当前目标感知链，不修改正在运行的 full 实验，也不把 exploration、planner
或 evaluator failure 全部归因于 detector。

## 1. 当前结论摘要

1. 工程中已经实现 GroundingDINO 服务与调用路径，但当前 HM3Dv1 六类目标实际上全部走
   YOLOv7；截至本快照，191,445 次逐帧 detector trace 中 DINO 调用为 0。
2. 当前没有任何类别呈现严格的「零成功」或「零检测」，但类别差异很大。
3. `chair`、`bed` 当前识别链总体较强；`sofa` 的召回较高，但 false-positive termination
   较多。
4. `toilet` 同时存在可见目标漏检、未达到 reliable 和大量 max-steps，值得在 full 结束后
   优先做 detector funnel 审查。
5. `plant` 当前 SR 极低且 false positive 很多。它不像单纯的 detector miss，更像类别定义、
   未标注实例、易混对象和可靠性接管共同造成的问题。
6. `tv_monitor` 当前失败不能简单归因于识别。明显可见时目标检测率尚可，而
   robot-stuck/max-steps 数量很高；需要把 perception 与 execution/exploration 分开。
7. 在 full 完成前，不应修改 threshold、backend 或 fusion。当前实验必须保持 frozen，最终再
   用同一批 episode traces 做离线类别诊断。

## 2. 当前目标识别策略

### 2.1 类别标准化

Habitat HM3D episode 的目标名称先经过静态映射：

| HM3D 名称 | 内部 detector 名称 |
| --- | --- |
| `plant` / `potted_plant` | `potted plant` |
| `tv_monitor` / `television_screen` | `tv` |
| `sofa` / `loveseat` | `couch` |
| `chair` | `chair` |
| `bed` | `bed` |
| `toilet` | `toilet` |

代码位置：`zson3/target/pipeline.py:HM3D_TO_T1_TARGET`。

### 2.2 target aliases 与易混类别

每个目标使用一组静态 target aliases 和 confusable labels：

| Target | Target label | Confusable labels |
| --- | --- | --- |
| chair | chair | couch, bed, dining table, bench |
| bed | bed | couch, chair, dining table, bench |
| potted plant | potted plant | vase, cup, bowl, chair |
| toilet | toilet | sink, chair, bowl, bench |
| tv | tv | laptop, microwave, oven, book |
| couch | couch | chair, bed, bench, dining table |

这些易混类别并不是 target positive。它们会进入同一 detector observation，然后在
ApexFusion cluster 中作为竞争标签，防止一个几何 cluster 只因出现过 target label 就永远
被当作目标。

当前 TV 的 confusable table **不包含 mirror**。这与此前出现的 TV/镜面混淆有关，但不能在
当前版本中直接加 `mirror`：YOLOv7 COCO 没有 mirror；按当前 backend resolver，只要加入一个
非 COCO label，整个目标及其所有 confusables 都会从 YOLOv7 切换到 DINO。这不是局部增加一个
负类，而是完整 detector backend 变化，必须作为独立实验。

代码位置：`zson3/target/fusion.py:STATIC_GOAL_LABELS`。

### 2.3 detector backend 的真实选择规则

当前规则是：

```text
target aliases + confusable labels 全部属于 COCO
    -> YOLOv7
否则
    -> GroundingDINO
```

这不是 YOLO+DINO ensemble，也没有「YOLO 没检出时再调用 DINO」的 fallback。

HM3D 当前六类及其全部 confusable labels 都属于 COCO，因此全部选择 YOLOv7。DINO 的以下
工程部分虽然存在，但当前 full HM3Dv1 没有消费：

- `zson3/services/apex_target.py` 中的 `/gdino` HTTP adapter；
- `scripts/start_apex_target_services.sh` 中的 GroundingDINO server；
- 配置项 `apex_target_dino_threshold: 0.4`。

服务被启动或 health check 通过，不等于评测实际调用了它。

### 2.4 每帧感知与融合链

当前 T1-fidelity 路径每个 RGB-D observation 都执行：

```text
RGB
 -> YOLOv7（当前 HM3D 六类）
 -> 过滤到 target + confusable labels
 -> confidence >= 0.8
 -> 每个 bbox 调 MobileSAM
 -> mask + depth 投影到 world-space voxel cloud
 -> geometry acceptance
 -> 跨帧 cluster association
 -> positive / competing-label / weak negative evidence
 -> confidence + volume + observation count reliability gate
 -> stable target medoid
 -> OpenFrontier approach / STOP integration
```

主要阈值：

- YOLOv7 detection threshold：0.8；
- DINO threshold：0.4，但当前未使用；
- reliable confidence：0.65；
- reliable minimum voxel volume：8；
- reliable minimum positive observations：2；
- usable depth：0.5–3.5 m；
- voxel size：0.10 m。

检测框不会直接创建高 utility object frontier。只有 ApexFusion 产生 reliable target 后，target
才获得导航控制权。Qwen 不参与 target acceptance；它只用于原有 OpenFrontier frontier
scoring。

## 3. 799/2000 中期运行快照

总体：

| Metric | 中期值 |
| --- | ---: |
| Episodes | 799/2000 |
| SR | 56.32%（450/799） |
| SPL | 0.2580 |
| SR@1m | 59.32%（474/799） |
| SPL@1m | 0.2703 |
| Exceptions | 0 |
| Detector frame traces | 191,445 |
| YOLOv7 traces | 191,445 |
| GroundingDINO traces | 0 |

### 3.1 类别级漏斗

这里的 `GT visible >=50 px` 表示 episode 中至少一帧 Habitat semantic GT 的目标像素数达到
50。它只是中期诊断门槛，不是论文 metric，也不等价于 detector 理应检出的清晰视图。

| Category | N | SR | SR@1m | 有 target detection | 曾 reliable | GT visible >=50 px | detection / visible | reliable / visible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bed | 177 | 64.4% | 66.1% | 166 | 138 | 137 | 133/137 | 122/137 |
| chair | 176 | 78.4% | 79.0% | 169 | 165 | 171 | 165/171 | 161/171 |
| plant | 38 | 7.9% | 7.9% | 33 | 33 | 7 | 3/7 | 3/7 |
| sofa | 164 | 64.0% | 68.9% | 156 | 154 | 127 | 124/127 | 122/127 |
| toilet | 146 | 32.9% | 39.0% | 85 | 75 | 95 | 72/95 | 64/95 |
| tv_monitor | 98 | 42.9% | 45.9% | 78 | 70 | 67 | 59/67 | 52/67 |

注意：`有 target detection` 表示整个 episode 任意帧曾产生 target label；它不说明 bbox 正确，
也不说明对应 HM3D GT instance。`曾 reliable` 同样可能包含真实未标注实例或 false positive。

### 3.2 termination reason 按类别

| Category | object_found | 1m only | false_positive | max_steps | robot_stuck | no_frontiers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bed | 111 | 2 | 18 | 37 | 8 | 1 |
| chair | 137 | 1 | 20 | 12 | 5 | 1 |
| plant | 3 | 0 | 20 | 6 | 9 | 0 |
| sofa | 104 | 8 | 39 | 8 | 5 | 0 |
| toilet | 47 | 5 | 13 | 70 | 9 | 2 |
| tv_monitor | 31 | 3 | 11 | 31 | 22 | 0 |

`false_positive` 是 evaluator 与 HM3D semantic GT 的不匹配结果，不能自动等价为肉眼可见的模型
幻觉。HM3D 可能有未标注实例，最终仍需要对代表案例分成：明确误检、易混类别、真实但未标注、
无法确定。

## 4. 当前暴露的问题

### 4.1 DINO 是未被实际覆盖的代码路径

虽然 DINO service、client 和 threshold 齐全，当前主评测没有实际执行它。因此不能声称当前
baseline 已验证了可替换 detector，也不能根据 full 结果评价 DINO 的收益或稳定性。

更隐蔽的问题是 backend 选择粒度过粗：一个非 COCO confusable label 会让整个 detector set
切换到 DINO，而不是只让 DINO补充该 label。

### 4.2 单 backend、无条件 fallback 会放大类别偏差

YOLOv7 对 COCO 类别的表现并不均匀。当前 `toilet` 在 GT clearly-visible proxy 中只有 72/95
episodes 出现 target detection；TV 是 59/67。由于没有 DINO fallback，一旦 YOLO 在关键视角
漏检，这一帧不会进入 segmentation 和 fusion。

但不能据此立即双模型每帧并行：它会显著增加延迟，也可能扩大 false positives，并改变已经
冻结的 T1 perception condition。

### 4.3 Plant 的主要风险不是简单漏检

Plant 38 个 episodes 中：

- 33 个曾产生 target detection；
- 33 个曾达到 reliable；
- 20 个以 evaluator `false_positive` 结束；
- 只有 3 个成功；
- 只有 7 个达到了当前 `GT visible >=50 px` proxy。

这更像以下问题的组合：

- YOLO `potted plant` 对装饰、花瓶或植物纹理产生误检；
- HM3D semantic annotation 与 RGB 中真实植物实例不完整对应；
- 当前 confusable competition 无法压制某些植物样外观；
- 两帧可靠性和小体积门槛可能让稳定重复误检接管。

因此，直接降低 plant threshold 或增加召回很可能进一步恶化 false positives。

### 4.4 Toilet 同时有 perception miss 与 exploration timeout

Toilet 的 visible proxy 中 23/95 没有 target detection，31/95 没有达到 reliable；同时全类有
70 个 max-steps。后者可能包含：

- 根本没探索到目标区域；
- 看见目标但 detector miss；
- detector 有框但 MobileSAM/深度几何失败；
- cluster 没达到 reliable；
- 目标位于不可达或跨楼层区域。

只有逐阶段 funnel 和 camera trajectory 能区分，不能只调 YOLO threshold。

### 4.5 TV failure 不全是 detector failure

TV 在明显可见 proxy 下已有 59/67 target detection、52/67 reliable；但全类同时出现 31 个
max-steps 和 22 个 robot-stuck。TV 确有漏检和镜面易混风险，但当前数据也指向明显的
exploration/execution failure。把所有失败归因于 detector 会导致错误优化。

### 4.6 Sofa 的主要警报是 false-positive termination

Sofa 的 visible detection/reliability 很高，却出现 39 个 `false_positive`，为当前最多。应优先
人工区分 couch/chair/bed 易混、未标注 couch、cluster association 错误和 STOP/距离问题，
而不是继续提高 recall。

### 4.7 “检测到”到“成功”之间仍有多层损失

完整因果链至少分为：

```text
GT visible
 -> detector target box
 -> MobileSAM mask
 -> valid depth geometry
 -> correct cross-frame association
 -> reliable target
 -> reachable approach endpoint
 -> correct STOP
 -> evaluator success
```

当前 episode summary 足以定位类别风险，但不足以把每一处损失归因到唯一组件。后续分析必须
保留这个漏斗，不能只比较 SR 与 detector count。

## 5. Full 完成后的必做更新

确认 `summary.json["episodes"] == 2000` 且运行正常结束后：

1. 把本文状态从「中期」改为「full final」，记录最终 commit/tag、服务模型与 checkpoint；
2. 重算总体 SR/SPL/SR@1m/SPL@1m；
3. 重算六类的 episode 数、SR、SPL、SR@1m 和 SPL@1m；
4. 重算每类的完整 perception funnel：
   `GT visible -> target bbox -> mask -> geometry -> reliable -> approach -> STOP -> success`；
5. 给 GT visibility 使用多个门槛（例如 50/500/2000 pixels）和连续帧条件，避免单像素噪声；
6. 按 same-floor 与 cross-floor-required 分层，再看类别结果；
7. 人工审查 plant/sofa false positives、toilet misses、TV mirror/stuck 的固定代表案例；
8. 确认全部 trace 的 detector backend count，不能只根据配置推断；
9. 将目标从未可见的 episodes 与可见但未检测的 episodes 分开；
10. 更新本文所有表格和结论，并注明哪些判断在 799 快照后发生变化。

建议把最终统计实现成独立的只读脚本，输入 episode JSON，输出版本化 JSON/Markdown；不要把
诊断统计插进 agent 控制闭环。

## 6. 可优化方向与优先级

以下都是 full 完成后的候选实验，不是当前基线应立即实施的修改。

### P0：先完成类别级离线审计

从已保存 trace/关键帧中建立小型固定 fixture：

- true visible + YOLO hit；
- true visible + YOLO miss；
- YOLO target hit + MobileSAM/geometry reject；
- target detection + never reliable；
- reliable + GT unmatched；
- reliable + approach/STOP failure。

每类优先覆盖 plant、toilet、TV、sofa。这个 fixture 是后续 threshold/backend 变化的共同输入，
避免重复跑 simulator 和只看 random-100 flips。

### P1：离线比较 YOLOv7 与 GroundingDINO

在同一批固定 RGB frames 上分别运行：

- 当前 YOLOv7；
- 当前 GroundingDINO；
- 不同 target aliases/prompt；
- 必要时 detector score sweep。

记录 target recall、confusable recall、bbox-GT overlap、false boxes、延迟。必须先证明 DINO 对
某个类别有互补召回，而不是假设 open-vocabulary detector 必然更强。

### P2：把 backend selection 改成显式、可配置策略

如果离线证据成立，可考虑：

```text
detector_policy:
  default: yolo
  per_category:
    tv: yolo_then_dino_on_miss
    toilet: yolo_then_dino_on_miss
    potted plant: yolo_only
```

这比当前「任一 label 非 COCO就整组切 DINO」更清晰。每条策略必须写入 episode trace，服务
失败时 fail closed，并允许完整回退 frozen YOLO baseline。

### P3：只对有证据的类别做条件 fallback

候选触发方式包括：

- 目标达到一定 GT-free visibility proxy 不可用，运行时不能使用 GT；
- 在新的、信息量较高的视角中 YOLO 无 target hit；
- exploration 到达新区域或 frontier replan boundary 时；
- YOLO 只产生某个高风险 confusable label 时调用 DINO 复核。

不建议第一版每步同时运行 YOLO+DINO。它成本高、改变时序，而且多 detector union 会增加
MobileSAM 和 fusion 的 false-positive 输入。

### P4：类别级 calibration，而不是统一降阈值

当前 YOLO threshold 统一为 0.8。后续可以从离线 PR curve 评估 per-category threshold，但必须
同时观测：

- visible recall；
- false boxes/frame；
- geometry accepted false boxes；
- false reliable clusters；
- first reliable latency。

Plant/sofa 已经有明显 false-positive 风险，不应因为 toilet/TV 漏检而全局降低 threshold。

### P5：改进 aliases/confusables，但与 backend 变更解耦

可以评估 TV 的 `television`, `monitor`, `television screen`，以及 mirror/reflection negative
evidence；但应让 label vocabulary 和 detector backend 成为两个独立配置。否则添加一个 DINO
专属 confusable 会无意切换整个目标 detector。

### P6：分离 segmentation、geometry 与 fusion correctness

对 detector 已命中的帧，继续测：

- bbox 是否覆盖正确实例；
- MobileSAM mask precision/recall；
- depth range rejection；
- world projection 与 voxel volume；
- 连续帧是否融合为同一 cluster；
- confusable 是否正确竞争；
- negative evidence 是否错误抹除 true target。

只有定位到明确 correctness bug 才修改相关模块。不要用 threshold 调整掩盖坐标、mask 或
association 错误。

### P7：保持 detector adapter 可替换

长期接口应继续保持：

```text
detect(image, labels/prompt) -> boxes, phrases, confidences
segment(image, box)          -> mask
fusion(observations)         -> reliable target evidence
```

YOLO、DINO 或未来 detector 不应持有 ObjectMemory、导航状态或 STOP 权限。这样才能进行公平的
perception ablation，并防止 detector 更换扩大为系统重写。

## 7. 当前明确不建议做的修改

full 结束前不要：

- 把所有 HM3D 类别强制切到 DINO；
- 同时每步运行 YOLO+DINO；
- 全局降低 YOLO threshold；
- 因 TV 镜面问题直接在 confusable table 中加入 `mirror`；
- 修改 ApexFusion 主公式、reliability threshold 或 observation count；
- 把 Qwen verifier 重新放回 target acceptance；
- 根据当前按场景顺序产生的 799 episodes 调参；
- 把 max-steps、robot-stuck 或 target-never-visible 都算作 detector miss。

## 8. 暂定研究判断

当前目标链不是「DINO+YOLO 双 detector baseline」，而是带有未覆盖 DINO 路径的 YOLOv7 +
MobileSAM + ApexFusion baseline。

它没有发生整类零召回，但 plant、toilet、TV、sofa 分别暴露了不同性质的问题：

```text
plant      -> false-positive / annotation / category ambiguity 优先
toilet     -> detector recall + reliability + exploration timeout 混合
tv_monitor -> 部分 detector miss，但 stuck/max-steps 同样突出
sofa       -> recall 较好，false-positive termination 优先
chair/bed  -> 当前相对健康，作为回归保护类
```

最有性价比的下一步不是立即接入 DINO，而是在 full 完成后用固定可见帧完成 YOLO-vs-DINO
离线互补性审计。只有证明 DINO 能救回特定类别、且不会显著增加 false reliable target，才实现
按类别、按条件触发的 bounded fallback。
