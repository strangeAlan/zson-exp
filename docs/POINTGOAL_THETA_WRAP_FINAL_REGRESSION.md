# PointGoal theta wrap 单变量修复与最终回归

日期：2026-08-28

冻结基线：`of-base-full-v1-v2-20260727`（`3a79d975`）

实验分支：`fix/pointgoal-theta-wrap`

实现 commit：`09a108d`

最终结论：**SHELVE；不合并到 OF-base，不继续 PointNav / target-approach 支线。**

## 结论先行

`rho_theta()` 的输出现在按 Habitat 官方约定规范到 `[-pi, pi]`，静态正确性成立，且保护组
零回归。但这不是当前 PointNav 的行为 bug：当前 wrapper 使用的 observation key 是
`pointgoal_with_gps_compass`，对应 Habitat 的
`IntegratedPointGoalGPSAndCompassSensor`。Habitat-Baselines 在输入网络前将二维极坐标转换为：

```text
(rho, theta) -> (rho, cos(-theta), sin(-theta))
```

因此 `theta` 与 `theta +/- 2pi` 对当前 checkpoint 严格等价。完整 paired replay 证实：

- 5,028 个 planner rows 中，旧版 552 个 theta 越界，新版降为 0；
- Oracle A 的 240 个 pursuit rows 中，动作差异为 0；
- Oracle B 的 156 个 pursuit rows 中，动作差异为 0；
- 对应 rho、位置、最终距离也逐行一致；
- Oracle B 仍为 **0/4 reached**；
- 保护组仍为 **10/10 成功**，成功、steps、SPL 全部逐 episode 完全一致。

所以此前 executor 审计中“越界角度作为连续数值污染神经策略”的解释需要修正。角度范围不符合
sensor contract 是代码整洁性问题，但 active policy 的周期编码已经将它消解，不能提供 rescue
headroom。

## 1. 严格单变量范围

默认 policy 只有一行变化：

```python
theta = np.arctan2(np.sin(theta), np.cos(theta))
```

没有修改 endpoint、centroid、approach viewpoint、target pipeline、STOP、target lock、
`close_enough`、`too_hot`、heat reset、checkpoint、frontier、planner 其他逻辑或 evaluator。

Oracle 诊断基础设施从冻结基线单独 cherry-pick；普通 OF-base protection replay 不实例化 Oracle。

## 2. 静态正确性

`tests/test_pointgoal_theta_wrap.py` 直接调用 Habitat
`PointGoalSensor._compute_pointgoal()` 作为参考，覆盖多组 robot yaw / world goal，以及左右方向、
`+/-pi`、branch cut 和稠密角度网格。

| 检查 | 结果 |
|---|---:|
| pytest | 74 passed / 0 failed |
| rho 对齐 Habitat | 通过 |
| theta circular equality | 通过 |
| 所有 theta 位于 `[-pi, pi]` | 通过 |
| 左右符号与 `+/-pi` 边界 | 通过 |
| branch-cut 两侧 | 通过 |

运行命令：

```bash
.local/envs/zson3/bin/python -m pytest -q tests/test_pointgoal_theta_wrap.py
```

## 3. theta 越界归零

完全相同的 protection / Oracle A / Oracle B replay 产生相同数量的 planner rows：

| Cohort | Rows | 旧版越界 | wrap 后越界 |
|---|---:|---:|---:|
| Protection | 2,374 | 232 | 0 |
| Oracle A | 1,797 | 184 | 0 |
| Oracle B | 857 | 136 | 0 |
| **Total** | **5,028** | **552** | **0** |

wrap 后日志范围为 `[-3.14, 3.13]`（日志保留两位小数），满足 contract。

## 4. Oracle A 与 accepted-no-STOP

候选分类、endpoint 和 Oracle A 结果与旧版完全相同：

- 正确 candidate 的 accepted-no-STOP：`5`；
- Oracle A rescue：`1/5 = 20%`；
- 全部 8 个 accepted-no-STOP：`1/8 = 12.5%`；
- 全部正确 candidate failure（含 Probe 12）：`2/6 = 33.3%`；
- 新增 rescue：`0`。

五个正确 candidate 的 accepted-no-STOP pursuit 如下。`Move` 是接管后首末 position 的欧氏
位移；所有数字与旧版一致。

| Probe / full | Target | rho start/min/final | TURN | Move | Final GT | A result |
|---|---|---|---:|---:|---:|---|
| 25 / 950 | plant | 2.63 / .11 / .11 | 38.9% | 2.53 m | .009 m | reached |
| 26 / 835 | bed | 1.56 / 1.54 / 1.60 | 59.0% | .35 m | 1.182 m | fail |
| 27 / 995 | tv | 5.40 / 5.40 / 5.40 | 45.2% | 0 m | 5.109 m | fail |
| 30 / 827 | bed | 1.67 / 1.67 / 1.70 | 61.9% | .07 m | 1.376 m | fail |
| 31 / 745 | tv | 3.22 / 3.16 / 3.84 | 60.0% | .85 m | 9.505 m | fail |

上述 5 例以及 Probe 12/32/35 控制的旧 theta 中确实存在 `2pi` 等价改写，但 240 个 A pursuit rows 的
action、rho 和 position 差异全部为零。

## 5. Oracle B 决定性结果

Oracle A 失败后机械生成的 B 集仍是相同四例：835、995、827、745。四个 endpoint 都是固定、
navmesh reachable 的合法 GT success viewpoint。

| Probe / full | rho start/min/final | F / TURN | TURN | Move | 旧/新 reached | Final GT |
|---|---|---:|---:|---:|---:|---:|
| 26 / 835 | 1.20 / 1.16 / 1.16 | 16 / 20 | 55.6% | .38 m | 0 / 0 | 1.163 m |
| 27 / 995 | 5.11 / 5.11 / 5.11 | 17 / 14 | 45.2% | 0 m | 0 / 0 | 5.109 m |
| 30 / 827 | .66 / .66 / .66 | 16 / 26 | 61.9% | 0 m | 0 / 0 | 1.365 m |
| 31 / 745 | 2.88 / 2.81 / 2.81 | 16 / 30 | 65.2% | .08 m | 0 / 0 | 9.918 m |

在 B 的 156 个 pursuit rows 中：

- 旧版 theta 越界 66 rows，wrap 后为 0；
- action difference `0/156`；
- 最大 rho difference `0`；
- 最大 position difference `0`；
- Oracle B reached 仍为 **0/4**。

这不是偶然的模型鲁棒性，而是 active integrated PointGoal encoder 的明确 `sin/cos` 周期不变性。
`vlfm/policy/utils/pointnav_policy.py` 构造的 observation key 是
`pointgoal_with_gps_compass`；Habitat-Baselines 的
`rl/ddppo/policy/resnet_policy.py` 对该 key 在网络前执行上述周期转换。只有未来更换为直接线性嵌入
`pointgoal` 的 policy 时，未 wrap theta 才可能成为实际行为错误。

## 6. 正常成功保护组

| Metric | Baseline | Wrapped | Paired delta |
|---|---:|---:|---:|
| Episodes / success | 10 / 10 | 10 / 10 | 0 loss |
| SR@1m | 100% | 100% | 0 |
| SPL@1m | .4633 | .4633 | 0 |
| Exact success | — | 10/10 | — |
| Exact steps | — | 10/10 | — |
| Exact SPL | — | 10/10 | — |

steps 为 `30, 29, 27, 231, 381, 444, 488, 453, 39, 410`，与冻结结果逐项一致。

## 7. Gate、heat reset 与最终决策

预设扩展条件要求 Oracle B 至少 `3/4` reached 且保护集基本无回归。实际为：

```text
Oracle B reached:       0/4
Protection regressions: 0/10
Expand to Probe64:      NO
```

因此没有启动 64-episode Regression/Failure Probe，也没有启动 full。

index 706 型 heat-reset 问题从代码与旧 full trace 看仍存在：它本来就没有 theta 越界，wrap 不可能
修复它。但它是与 theta 独立的单例 evidence，保守 headroom 仅约 `1/1000`；在 Oracle B 0/4 且
PointNav 总支线已达停止门槛后，不值得再开 heat-reset commit。

最终决定：

1. theta wrap 是 contract-correct、低风险的 cleanup，但对当前 checkpoint 是行为 no-op；
2. 按本轮预设性能门槛，**不合并到 OF-base**，保留 `09a108d` 作为可回退实验 commit；
3. 不做 heat-reset 修复；
4. 正式结束 target grounding / approach / PointNav patch 支线；
5. 后续研究回到 frontier / exploration 主线。

## 8. 产物

- `results/target_approach_oracle_theta_wrap_seed20260727/pointgoal_theta_wrap_audit_v1.json`
- `results/target_approach_oracle_theta_wrap_seed20260727/oracle_summary_v1.json`
- `scripts/audit_pointgoal_theta_wrap.py`
- `tests/test_pointgoal_theta_wrap.py`

结果 JSON、manifest、progress 与 compact summary 可版本化；raw logs、videos 和 overlays 保留本地，
不纳入 Git。
