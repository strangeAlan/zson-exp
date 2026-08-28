# 研究支线封存与 OF-base 恢复记录（2026-08-28）

## 状态

当前默认开发线已恢复到冻结的 OF-base：

- 基线 commit：`3a79d97556f25be9a291aac6b32e09cd76445f50`
- 基线 tag：`of-base-full-v1-v2-20260727`
- 默认分支：`main`
- 默认系统：OpenFrontier visual frontier + SAM3 + Qwen；不包含 ApexTarget、geometry-frontier 在线控制、target-closure、stable-approach 或 PointNav patch。

本次归档提交只增加文档和紧凑实验记录，不改变 OF-base policy、配置、模型接口或 evaluator。后续新 idea 必须从 `main` 新建独立分支，不在已封存支线上继续叠加补丁。

## 可恢复版本

以下 annotated tags 固定了各支线最后的代码、文档和审计状态：

| 支线 | 归档 tag |
| --- | --- |
| Geometric Frontier Completion v1 | `archive/20260828/geometry-completion-v1` |
| Grounded Unified Frontier v2 | `archive/20260828/grounded-unified-frontier-v2` |
| Target Closure v0 | `archive/20260828/target-closure-v0` |
| Target Closure safe-v1 | `archive/20260828/target-closure-safe-v1` |
| Stable Target Approach v0 | `archive/20260828/stable-target-approach-v0` |
| Target Approach Oracle | `archive/20260828/target-approach-oracle` |
| PointGoal theta wrap | `archive/20260828/pointgoal-theta-wrap` |

查看某一支线的审计文档无需切换工作区，例如：

```bash
git show archive/20260828/target-approach-oracle:docs/TARGET_APPROACH_ORACLE_CEILING.md
git show archive/20260828/target-approach-oracle:docs/POINTNAV_EXECUTOR_FINAL_AUDIT.md
git show archive/20260828/pointgoal-theta-wrap:docs/POINTGOAL_THETA_WRAP_FINAL_REGRESSION.md
```

如确需复现，应从 tag 新建分支，而不是移动 `main`：

```bash
git switch -c reproduce/<name> archive/20260828/<tag-name>
```

## 四次 full 基准结论

核心口径为 SR@1m / SPL@1m。详细 paired audit 见 [OF_BASE_APEXTARGET_V1_V2_FULL_AUDIT.md](OF_BASE_APEXTARGET_V1_V2_FULL_AUDIT.md)。

| 数据集 | OF-base | OF-ApexTarget | 结论 |
| --- | --- | --- | --- |
| HM3Dv1，2000 episodes | 54.00% / 0.2747 | 54.30% / 0.2472 | ApexTarget 的 SR 仅 +0.30pp，且 SPL 明显下降；不足以替代 base |
| HM3Dv2，1000 episodes | 70.80% / 0.3299 | 65.20% / 0.2790 | ApexTarget 分别下降 5.60pp / 0.0509；明确负收益 |

因此 OF-base 是唯一默认主线；ApexTarget 只保留为目标识别实验模块和 HM3Dv1/HM3Dv2 差异证据。

## 已封存 ProbeSet 结果

### Frontier completion（V2 Probe56）

冻结 OF-base 对照为 28/56，SR 0.5000，SPL 0.2300。

| 方案 | 成功 | SR | SPL | rescue / loss | common-success 平均步数变化 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Geometry v0 coverage override | 26 | 0.4643 | 0.1742 | 8 / 10 | +34.33 | geometry 过度接管，退化 |
| Geometry v1 semantic-first fallback | 26 | 0.4643 | 0.1921 | 5 / 7 | +37.86 | 降低但未消除回归 |
| Grounded unified frontier v2 | 22 | 0.3929 | 0.1506 | 6 / 12 | +55.88 | 统一 VLM 排序仍显著损害 SR/SPL |

三版均出现“救回少量失败、破坏更多原成功”的稳定模式。当前证据否定的是这些 geometry frontier 的在线控制实现，不是否定 frontier proposal coverage 这一研究问题。后续若重启该方向，应提出新的、可证伪的候选生成假设，不继续调 override、cooldown 或 gain 尺度。

### Target closure / approach（V2 Probe64）

冻结 OF-base 对照为 32/64，SR 0.5000，SPL 0.2640。

| 方案 | 成功 | SR | SPL | rescue / loss | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| Target Closure v0 | 35 | 0.5469 | 0.2624 | 7 / 4 | 有净 rescue，但破坏保护轨迹且无 SPL 收益 |
| Target Closure safe-v1 | 32 | 0.5000 | 0.2608 | 0 / 0 | 保持原行为，但没有任何 rescue |
| Stable Target Approach v0 | 35 | 0.5469 | 0.2511 | 5 / 2 | 未达到 regression ≤1，SPL 继续下降 |

这说明 late-stage patch 可以改变个别 outcome，但现有失败不是一个可由小型 endpoint/closure 修复稳定解决的集中机制。

## Oracle、PointNav 与 theta wrap 终局结论

- 在 candidate 正确的 accepted-no-STOP failure 中，Candidate-derived Oracle A 仅救回 1/5（20%），未达到约一半的继续开发门槛。
- 对 Oracle A 失败的四例，GT success-viewpoint Oracle B 使用当前 PointNav 为 0/4 reached；同 endpoint 的 Habitat official greedy follower 为 4/4 reached。执行能力确有问题，但 full 中可可靠归因的 headroom 不足以支持替换执行器。
- `rho_theta()` 的 theta wrap 静态正确性测试为 74/74；越界记录从 552 降为 0。
- 然而当前 PointNav policy 通过 `sin(theta)` / `cos(theta)` 编码角度；Oracle A 240 行、Oracle B 156 行动作比较均为 0 差异，10/10 保护样例 success、steps、SPL 完全一致。它是表示层 correctness cleanup，但对当前 checkpoint 是行为 no-op，不合并到 OF-base。
- heat-reset 证据只集中于极少数案例，不满足独立修补门槛。

最终决策：正式结束 target grounding / target approach / PointNav patch 支线。除非未来出现新的大规模、机制一致证据，否则不再围绕相同失败继续堆叠 recovery。

## 数据保留与清理规则

Git 中保留：

- OF-base 与 OF-ApexTarget 四次 full 的 manifest、summary、progress 等紧凑关键日志；
- 六次核心 ProbeSet 的 manifest、summary 和必要 progress；
- `results/archive/20260828_retired_research/retired_research_summary_v1.json`，作为机器可读统一索引；
- 本文档、既有 full audit，以及各 archive tag 内的专项审计。

本机另保留、但不提交 Git：

- `results/archive/20260828_retired_research/structured_episode_evidence.tar.gz`
- 6430 个归档条目；SHA-256：`e9619669dccd9b8f0badbe6073e7e3f69ed32525e158ea3713a28cb735ff8ec4`
- 内容包括四次 full、三次 geometry Probe、三次 target Probe 和 Oracle 的逐 episode 结构化 JSON。需要深度 paired audit 时再解压。

已清理：smoke、random100、中止/重复 full、展开的 `raw.log`、`episode_logs`、逐 episode JSON 和视觉临时产物。它们要么没有决策价值，要么已经被上述压缩证据和 archive tag 覆盖。

## 下一阶段原则

1. 新 idea 一律从冻结 OF-base 的 `main` 开新分支。
2. 首先使用固定 FailureProbe / RegressionProbe 做 paired 验证，保持 episode identity、manifest 和 seed。
3. 以 SR@1m 为核心，SPL 和保护集回归为硬约束；0.1m 仅作诊断。
4. 不把已否定的 geometry override、ApexTarget、closure 或 approach patch 隐式带入新实验。
5. 小集只有在出现机制一致的 rescue 且保护轨迹基本无损时，才进入更大评测。
