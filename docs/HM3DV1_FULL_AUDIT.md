# OF-ApexTarget v1：HM3Dv1 full-2000 审计

状态：**最终只读审计（2000/2000）**<br>
审计日期：2026-08-16 UTC<br>
代码基线：`dd894ff725e1c855b1d6406849db5b02e2be4a35`，branch `zson3-runtime-0.3.3`<br>
结果目录：`results/openfrontier_apextarget_v1_deterministic_full_hm3dv1_2000_seed20260727`

本文只分析冻结实验产生的 episode trace，不改变 agent、模型、planner 或 evaluator。下文的
`GT visible >= 50 px` 是诊断 proxy，不是 Habitat 指标；`cross-floor-required` 也是离线静态
分组，不对 policy 可见。

## 1. 总结果

| Metric | Result |
| --- | ---: |
| Episodes | 2000 |
| SR（冻结 0.1 m protocol） | **50.45%**（1009/2000） |
| SPL | **0.2300** |
| SR@1m | **54.30%**（1086/2000） |
| SPL@1m | **0.2472** |
| Exceptions | 3（0.15%） |
| 总 wall time | 416,913 s（约 115.8 h） |
| 每 episode 平均/中位时间 | 208.5 / 170.3 s |
| 峰值进程 RSS | 6316 MiB |

`reason` 不是 success 的同义词。例如 34 个 `robot_stuck`、13 个 `max_steps_reached` 和 1 个
`no_frontiers` episode 在 Habitat 0.1 m metric 下仍成功。因此不能直接把 reason counts 当作
成功/失败计数。

| Termination reason | N | SR successes | SR@1m successes |
| --- | ---: | ---: | ---: |
| object_found | 961 | 961 | 961 |
| max_steps_reached | 594 | 13 | 22 |
| false_positive | 241 | 0 | 0 |
| robot_stuck | 129 | 34 | 51 |
| object_found_at_1m_only | 51 | 0 | 51 |
| no_frontiers | 21 | 1 | 1 |
| exception | 3 | 0 | 0 |

## 2. 主要结论：跨楼层缺失是最大的结构性瓶颈

离线定义为：若 episode 所有标注 success viewpoint 与起点的垂直距离都大于 1.5 m，则标为
`cross_floor_required`；否则标为 `same_floor_available`。这是基于标注 goal viewpoints 的静态
可达性 proxy，不证明某条具体最短路径必定经过楼梯。

| Group | N / 占比 | SR | SPL | SR@1m | SPL@1m |
| --- | ---: | ---: | ---: | ---: | ---: |
| same-floor available | 1659 / 82.95% | **60.28%** | 0.2746 | 64.92% | 0.2954 |
| cross-floor required | 341 / 17.05% | **2.64%** | 0.0131 | 2.64% | 0.0131 |

跨楼层组只成功 9/341。若该组仅达到当前同层 SR，整体 SR 的理论差距约为
`(60.28%-2.64%)*17.05% = 9.83` 个百分点。它不是 detector 小调参问题，而是当前系统没有
显式楼层表示、楼梯状态机和跨层地图生命周期所造成的能力缺口。

跨楼层失败中，293/332（88.3%）从未产生一帧 `GT target >= 50 px`；同层失败中这个数是
240/659（36.4%）。因此 full 结果中大量 `target never visible` 不能归罪于 YOLO。

## 3. 类别结果

| Category | N | SR | SPL | SR@1m | SPL@1m | Same-floor SR | Cross-floor SR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bed | 433 | 48.96% | 0.2254 | 51.96% | 0.2423 | 71.92% | 1.42% |
| chair | 428 | 63.32% | 0.2938 | 65.65% | 0.3081 | 66.18% | 5.00% |
| plant | 84 | 33.33% | 0.1475 | 36.90% | 0.1617 | 47.46% | 0.00% |
| sofa | 376 | 59.31% | 0.3026 | 63.56% | 0.3229 | 69.43% | 8.06% |
| toilet | 398 | 42.21% | 0.1549 | 48.74% | 0.1785 | 47.31% | 2.22% |
| tv_monitor | 281 | 38.08% | 0.1736 | 41.28% | 0.1839 | 45.92% | 0.00% |

全量类别排序不能直接解释成 detector 排序，因为 bed 的 32.6% episode 被判为跨楼层，而 chair
只有 4.7%。同层分层后，bed/sofa/chair 是相对健康的回归保护类；plant、toilet、TV 仍有明显
感知或接近问题。

## 4. 失败漏斗

按 episode 中任一 Habitat semantic target mask 是否达到 50 pixels，并将 detector、MobileSAM
几何接收、ApexFusion reliable 状态与最终 success 串成一个互斥漏斗。991 个 0.1 m 失败为：

| First unresolved stage | N | 失败占比 | 全集占比 |
| --- | ---: | ---: | ---: |
| target not meaningfully visible | 533 | 53.78% | 26.65% |
| visible-frame detector miss | 182 | 18.37% | 9.10% |
| reliable target but navigation/termination failed | 165 | 16.65% | 8.25% |
| geometry seen but never reliable | 65 | 6.56% | 3.25% |
| visible-frame geometry rejection | 43 | 4.34% | 2.15% |
| runtime exception | 3 | 0.30% | 0.15% |

这里的 `visible-frame detector miss` 是：在所有 `>=50 px` GT-visible frames 上都没有 target
label box；它不排除 episode 其他帧出现 GT-unmatched box。`reliable target` 也只表示融合器认为
可靠，不证明它对应正确标注实例。

在 1454 个曾有 `>=50 px` GT target 的 episodes 中：

```text
GT-visible episode                         1454
  -> 同一 GT-visible frame 有 target box  1267 / 1454 = 87.14%
  -> 同一 frame target geometry accepted  1224 / 1454 = 84.18%
  -> episode 曾有 reliable target          1203 / 1454 = 82.74%
  -> 最终 0.1 m success                     998 / 1454 = 68.64%
```

在 546 个从未达到 50 pixels 的 episodes 中仍有 338 个 target detection、267 个 target geometry
和 215 个 reliable target，但只有 11 个成功。这是需要人工复查的 **GT-unmatched reliable
target 候选池**，不能未经可视化就称为 hallucination：HM3D 可能漏标真实实例，目标也可能只以
小于阈值的像素出现。

## 5. 同层失败才适合用于近期 bounded fixes

同层 659 个失败的拆分：

| Stage | N | 同层全集占比 |
| --- | ---: | ---: |
| target not meaningfully visible | 240 | 14.47% |
| reliable but navigation failed | 161 | 9.70% |
| visible-frame detector miss | 152 | 9.16% |
| geometry seen but never reliable | 64 | 3.86% |
| visible-frame geometry rejection | 40 | 2.41% |
| exception | 2 | 0.12% |

161 个 `reliable but navigation failed` 中：53 个 `false_positive`、49 个只满足 1 m、35 个
max-steps、24 个 robot-stuck。这是 target approach/STOP 与可靠目标人工真伪审查的主要池。

当前六类全部实际使用 YOLOv7；GroundingDINO adapter 虽存在，但 full trace 没有走 DINO。
因此可以得出“YOLO 路径存在类别差异”，不能声称 DINO 已被验证，也不能把 DINO 直接接入主
闭环。下一步合理做法是用固定 visible-miss/GT-unmatched frames 离线比较 YOLO 与 DINO。

## 6. 可恢复的 runtime correctness bug

三个 exception 完全相同：Habitat episode 已结束后，旋转恢复路径仍调用
`habitat_env.step("turn_right")`，触发：

```text
AssertionError: Episode over, call reset before calling step
```

| Index | Scene / episode | Target | Floor group |
| ---: | --- | --- | --- |
| 1185 | mL8ThkuaVTM / 96 | toilet | cross-floor-required |
| 1299 | p53SfW6mjZe / 12 | bed | same-floor-available |
| 1578 | qyAac8rV8Zk / 93 | tv_monitor | same-floor-available |

这是局部、可复现、可验证的 episode-over guard 缺失，应该在独立 runtime-fix 分支处理；它只占
0.15%，不是 SR 主瓶颈。

## 7. 下一阶段测试集设计

不要继续在 random-100 上循环修补。基于 full trace 建立三个互不重叠的集合：

### FailurePack（建议 48 episodes，可查看）

- 12 个 same-floor target-never-visible：探索覆盖/Frontier scoring；
- 10 个 same-floor visible detector miss：按六类分层，toilet/TV 加权；
- 6 个 geometry rejection：MobileSAM、depth、world projection；
- 6 个 geometry-seen-but-never-reliable：association/fusion；
- 10 个 reliable-but-failed：FP、1m-only、max-steps、stuck 各覆盖；
- 3 个 runtime exceptions；
- 1 个 no-frontier 代表例。

FailurePack 允许保存视频、GT overlay 和完整 trace，用来定位分支修复。它不能作为最终报告集。

### RegressionPack（建议 48 episodes，可查看）

- 只选 same-floor 成功 episode；
- 六类各 8 个；
- 每类覆盖短/中/长轨迹、不同 scene，以及可靠目标出现早/晚的情况；
- 排除 GT-unmatched success 和异常 termination，保证它保护的是已知正确行为。

任何 perception、fusion、planner 或 recovery 分支都必须同时跑 FailurePack 和 RegressionPack，
并报告 paired flips、first-reliable step、steps、SR/SPL 与 SR@1m/SPL@1m。

### SealedHoldout（建议 200 episodes，不查看）

- 从未进入前两包的 same-floor episodes 中按 scene、category、原始 success/failure 分层抽取；
- manifest 与 hash 冻结；开发期间不得看视频或按 episode 调参；
- 只有 FailurePack 改善且 RegressionPack 不退化后才能运行一次。

跨楼层 341 episodes 应另建 `MultiFloorPack`，不混入近期 detector/approach 修复评价。其 9 个
现有成功应保留作未来跨楼层回归，但需要先确认是否真的发生了跨层行为。

## 8. 建议的开发顺序

1. 冻结当前 full 结果，不再根据 random-100 选择修复。
2. 从 full trace 生成上述 manifests，并人工确认 FailurePack 的 GT overlay 分类。
3. 先修 3 个 episode-over exception；这是最明确的 bounded correctness fix。
4. 在 shadow/offline 模式审查 YOLO visible misses 与 GT-unmatched reliable candidates；不要先改阈值。
5. 分开研究 reliable-target 后的 approach/STOP 与探索覆盖。
6. 将显式 multi-floor 作为独立能力分支；它是最大潜在收益项，但不能与 detector 修复混测。
7. 每个分支通过 FailurePack + RegressionPack 后只运行一次 sealed holdout；最后才重跑 full。

## 9. 最终判断

当前 OpenFrontier+ApexTarget 仍有继续作为研究基座的价值：同层 SR 60.28%、整体 SPL 0.2300，
说明 exploration、目标融合与几何 executor 已形成可用闭环；它不是一个“整体失效后等待大修”的
系统。但 full 数据证明其首要缺口是 multi-floor，其次才是同层的探索覆盖、detector 漏失和
reliable-target 后的接近/终止。

下一步不应直接重跑 HM3Dv1 full，也不应全局替换 detector。应先固定分层失败集与回归集，按
独立分支验证 bounded fixes；跨楼层能力单独立项。
