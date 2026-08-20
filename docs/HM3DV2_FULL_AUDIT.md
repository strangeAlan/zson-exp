# OF-ApexTarget v1：HM3Dv2 full-1000 审计

状态：**最终只读审计（1000/1000，exit code 0）**<br>
审计日期：2026-08-20 UTC<br>
代码基线：`dd894ff725e1c855b1d6406849db5b02e2be4a35`，branch `zson3-runtime-0.3.3`<br>
结果目录：`results/openfrontier_apextarget_v1_deterministic_full_hm3dv2_1000_seed20260727`

本文沿用 `HM3DV1_FULL_AUDIT.md` 的 trace 审计方法，但按 HM3Dv2 官方口径，以 **1 m
Success/SPL 为主指标**。0.1 m 仅作为轨迹终点精度的附加诊断，不用于判断 V2 方法优劣，也不对
policy 决策产生影响。`GT visible >= 50 px` 是 evaluator 诊断 proxy，不对 policy 可见。

HM3Dv2 的任务设定保证每个目标均可在起始楼层找到，本报告不做跨楼层分组。V1 已经单独确认当前
方法缺少跨楼层能力，那个结论不在本次 V2 审计中重复展开。

## 1. 完整性与官方主结果

- `summary.json`、`summary.txt`、`progress.log` 均记录 1000/1000；runner `exit_code=0`。
- `episodes/` 有且只有 1000 个 JSON；2 个异常都被记录，没有静默缺失或 `failures/` 残片。
- 完成时原始冻结文件 SHA-256：`manifest.json`
  `606ebb74a19c970e92f319f3c2be2600000c6931d7ed038906d08250215c8b62`，`summary.json`
  `88c5326e55b752ff14c1b48ec7e5b410be8a429fbafeb81364fe168b6501e0bb`。

| Metric | Result |
| --- | ---: |
| Episodes | 1000 |
| **SR@1m（HM3Dv2 官方主指标）** | **65.20%**（652/1000） |
| **SPL@1m（HM3Dv2 官方主指标）** | **0.2790** |
| SR@0.1m（附加参考） | 37.60%（376/1000） |
| SPL@0.1m（附加参考） | 0.1580 |
| Exceptions | 2（0.20%） |
| 总 wall time | 217,487 s（约 60.4 h） |
| 每 episode 平均/中位时间 | 217.5 / 145.2 s |
| 峰值进程 RSS | 5840 MiB |

因此，本次 V2 的核心结果是 **SR 65.20%、SPL 0.2790**。37.60% 不是 V2 官方 SR，不能据此
判断结果“很差”。两种半径间的差距只说明有大量轨迹在 STOP 时已进入官方成功区，但没有贴近到
0.1 m；是否继续优化这最后 0.9 m 应由机器人安全、效率或下游需求决定，不应压过官方指标。

| Termination reason | N | 1 m successes | 0.1 m reference successes |
| --- | ---: | ---: | ---: |
| object_found | 359 | 359 | 359 |
| object_found_at_1m_only | 247 | 247 | 0 |
| robot_stuck | 63 | 40 | 13 |
| max_steps_reached | 264 | 6 | 4 |
| false_positive | 59 | 0 | 0 |
| no_frontiers | 6 | 0 | 0 |
| exception | 2 | 0 | 0 |

`reason` 不是 Habitat success 的同义词，不能用 reason count 代替 metric。

## 2. 类别结果

| Category | N | **SR@1m** | **SPL@1m** | SR@0.1m ref | SPL@0.1m ref |
| --- | ---: | ---: | ---: | ---: | ---: |
| bed | 165 | **81.82%** | 0.3527 | 46.06% | 0.1904 |
| chair | 195 | **85.64%** | 0.3851 | 58.46% | 0.2593 |
| plant | 152 | **32.89%** | 0.0936 | 25.66% | 0.0697 |
| sofa | 187 | **74.87%** | 0.3763 | 36.36% | 0.1946 |
| toilet | 166 | **52.41%** | 0.2011 | 30.12% | 0.1087 |
| tv_monitor | 135 | **54.07%** | 0.2053 | 21.48% | 0.0817 |

官方口径下，bed、chair、sofa 已形成较强闭环；plant 是最明显的类别瓶颈，toilet 和 TV 次之。
bed/sofa/TV 的 0.1 m 参考值与 1 m 差距较大，可以作为近距离行为研究样本，但不应计为官方失败。

## 3. 官方 1 m 失败漏斗

漏斗判据与 V1 审计相同，只把最终 success 换成 V2 官方 1 m success。作为统计实现的一致性检查，
同一代码在 V1/0.1 m 上能逐项重现旧报告的 `533/182/165/65/43/3`。V2 的 348 个官方失败为：

| First unresolved stage | N | 官方失败占比 | 全集占比 |
| --- | ---: | ---: | ---: |
| visible-frame detector miss | 158 | 45.40% | 15.80% |
| target not meaningfully visible | 91 | 26.15% | 9.10% |
| reliable target but navigation/termination failed | 46 | 13.22% | 4.60% |
| geometry seen but never reliable | 36 | 10.34% | 3.60% |
| visible-frame geometry rejection | 15 | 4.31% | 1.50% |
| runtime exception | 2 | 0.57% | 0.20% |

在 909 个曾有 `>=50 px` GT target 的 episodes 中：

```text
GT-visible episode                          909
  -> 同一 GT-visible frame 有 target box   748 / 909 = 82.29%
  -> 同一 frame target geometry accepted   732 / 909 = 80.53%
  -> episode 曾有 reliable target           710 / 909 = 78.11%
  -> 最终官方 1 m success                    652 / 909 = 71.73%
```

从未达到 50 pixels 的 91 个 episodes 中，仍分别有 45 个 target detection、42 个 geometry、
32 个 reliable target，但没有一个成功。这 32 个只能称为 GT-unmatched candidates；未经 overlay
人工确认，不能直接叫 hallucination 或 false positive。

46 个 `reliable but official-failed` 包含 23 个 `false_positive`、14 个 `max_steps_reached` 和
9 个 `robot_stuck`。官方失败的最大可操作池是 158 个 visible-frame detector misses，其次是 91 个
never-visible 探索失败；“最后 0.9 m”不再是官方失败主因。

当前六类在 full trace 中全部走 YOLOv7（252,524 个 detection events），没有 DINO 闭环样本。

## 4. Runtime correctness bug

两个 exception 与 V1 的三个 exception 完全同源：episode 已结束后，旋转恢复路径仍调用
`habitat_env.step("turn_right")`，触发 `AssertionError: Episode over, call reset before calling step`。

| Index | Scene / episode | Target |
| ---: | --- | --- |
| 144 | CrMo8WxCyVb / 4 | toilet |
| 961 | ziup5kvtCCR / 11 | plant |

这个 guard 缺失已跨 V1/V2 重现，应单独修复并回归；它只影响 0.2%，不是主指标瓶颈。

## 5. 下一步实验

1. 用冻结的同一 1000-episode manifest 运行 pre-Apex OpenFrontier + SAM3；环境运行时和 Qwen
   本地化之外保持原移植路径不变，以 SR@1m/SPL@1m 做主对照。
2. 官方失败开发集优先覆盖 VisibleMiss（158）、NeverVisible（91）、ReliableButFailed（46）和
   Fusion（51：15 geometry rejection + 36 never reliable）；改动必须同时观察成功回归样本。
3. 247 个 `object_found_at_1m_only` 已是官方成功，只在确实需要更近终点时建立独立 ApproachPack，
   不与官方失败混合调参。
4. 单独修复 episode-over guard。

## 6. 最终判断

HM3Dv2 官方结果是 **SR 65.20%、SPL 0.2790**。主要失败来自 visible-frame detector miss、目标未
进入有效视野，以及少量融合后导航失败；不是跨楼层问题，也不应由 0.1 m 的附加结果主导判断。

pre-Apex OpenFrontier+SAM3 full-1000 对照将复用完全相同的 episode 身份与顺序，并以官方 1 m
指标比较。只有这一 paired-manifest 对照完成后，才能判断 ApexTarget 相对原目标路径的净收益。
