# PointNav Executor 最终只读审计

日期：2026-08-28

冻结基线：`of-base-full-v1-v2-20260727`（`3a79d975`）

结论：**A. Bounded integration fix**；不支持整体替换执行器

## 结论先行

当前执行链存在两个明确、局部、可复现的 correctness 问题，其中第一个是主要问题：

1. `rho_theta()` 计算出的 `theta` 没有 wrap 到 Habitat PointGoal 的
   `[-pi, pi]`。当前实现会把正确的 `+2.0 rad` 送成约 `-4.283 rad`；二者在几何上等价，
   但对神经网络的连续数值输入并不等价。冻结 full-1000 中有 **888/1000** 个 episode
   至少一次收到越界 theta，共 17,554 个 planner row；708 个成功 episode 中也有 613 个暴露，
   所以“暴露”不能直接当作可恢复 headroom。
2. object lock 后，goal rotation 每轮被设成 current rotation；`update_start_goal()` 又在完整
   goal pose 任何变化时清空 `forward_failure_heat/rotation_heat`。高 TURN 轨迹因此可循环数百步，
   heat 仍长期只有 0--1，`too_hot` 失去原本的熔断作用。

严格按 frozen trace 计算，full-1000 中 **14/19 accepted-no-STOP** 在接管后直接出现越界
theta；另有 1 例不越界但呈现长循环和 heat-reset。真正可归因的 late-stage bounded headroom
上界因此是 **15/1000（1.5 pp）**，全部 accepted-no-STOP 的绝对上界是 19/1000。它不达到
“另开执行器研究主线”的 20--30/1000 门槛，但满足“明确局部 correctness bug”这一 A 类门槛。

最终决策是：

- **GO：**只允许一次 bounded adapter/state fix，优先 theta wrap；
- **NO-GO：**不替换 PointNav，不开发 map-based local planner，不继续 target surface/endpoint/radius；
- target grounding / approach 主线正式结束；executor research 主线搁置。若 bounded fix 在冻结
  paired ProbeSet 上没有明确 rescue，则直接转为 C（Shelve），不再扩张方案。

## 1. 完整执行链

```text
FrontierManager world-space goal pose
  -> Wavemap free/occupied snap；近于 1.5 m 时沿 agent-target ray 改写 goal
  -> rho_theta(world camera pose, effective goal)
  -> WrappedPointNavResNetPolicy(depth, [rho, theta]) raw action
  -> close_enough / too_hot / raw STOP mapping
  -> FrontierManager path/non-path
  -> PointnavAgent 直接 env.step(action)
  -> Habitat collision filter、pose/sensor update
```

代码事实如下。

- object lock 的位置由 `get_object_free_point()` 每轮计算，orientation 复制当前 pose；visual
  frontier 则使用 frontier pose。Oracle-B 的 analysis hook 只把前者替换为固定 pose，默认路径不变。
- planner 先用 Wavemap 的 `snap_to_free()`；若 start-goal 欧氏距离小于 1.5 m，再调用
  `get_closest_navigable_point()`。这不是 Habitat navmesh snap。
- 正常模式下，wrapper 对网络 raw action **不做改写**：`1/2/3` 原样传递，raw `0` 映射成
  `None`。因此冻结日志在正常模式记录的高频 TURN 就是神经策略 raw TURN。
- `close_enough` 在 `rho < .2`，或 `rho < .5` 且 forward heat 至少 3 时接管，只执行目标
  orientation 对齐。
- `too_hot` 在 forward/rotation heat 任一达到 15 时接管；但同一轮 `solve()` 返回 false，
  manager 丢弃内部刚算出的 orientation action。因此该动作不会到达 Habitat。
- 数字 action 与 Habitat 完全一致：`0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT`；
  `PointnavAgent` 对非空 action 直接 `env.step(action)`，没有第二层左右互换。

相关实现位置：

- [`planner/pointnav_planner.py`](../planner/pointnav_planner.py)：goal snap、rho/theta、wrapper；
- [`frontier/manager.py`](../frontier/manager.py)：world goal 与 planner 调用；
- [`vlfm/policy/utils/pointnav_policy.py`](../vlfm/policy/utils/pointnav_policy.py)：checkpoint、RNN、raw action；
- [`nav/pointnav_agent.py`](../nav/pointnav_agent.py)：最终 Habitat action；
- [`utils/transform.py`](../utils/transform.py)：OpenFrontier/Habitat 坐标变换。

## 2. 坐标与动作正确性

### 2.1 轴、符号和左右动作

OpenFrontier 当前 pose 实际是 CV camera convention：`+X right, +Y down, +Z forward`；
代码注释所写的 “X forward, Y left” 不准确。`atan2(R[1,0], R[0,0])` 取的是 camera
right-axis yaw，后面的 `-pi/2` 正好把它换成 forward-axis bearing。

我用 Habitat `PointGoalSensor._compute_pointgoal()` 对多个 yaw 和四个世界方向逐项对照：

```text
wrap(current theta) == Habitat official theta
```

误差为数值零。因此轴交换、角度正负和 LEFT/RIGHT 定义本身正确；**唯一的 coordinate
correctness bug 是最后没有 wrap**：

```text
current: theta = atan2(...) - pi/2        # range can be [-3pi/2, pi/2]
official: theta = wrap_to_pi(theta)       # [-pi, pi]
```

这也解释了失败 trace 中同一固定目标附近频繁出现 `+1.2 -> -4.5 -> +1.2` 的约 `2pi`
不连续跳变。

### 2.2 raw 与 final action

full 日志没有单列 raw/final 两栏，但代码可以无歧义地恢复：

- normal branch：`final == raw`，仅 raw `STOP(0)` 变成内部 `None`；
- close-enough：final 来自 wrapper orientation alignment；
- too-hot：内部 action 不执行，manager 返回 no-path；
- Oracle-B 的失败前 actions 全处于 normal branch；最后一次 `path_active=false` 才是 heat
  failure。因此其高 TURN 不是 wrapper 把 FORWARD 改成 TURN。

## 3. Oracle-B 四例

四个 requested endpoint 全程固定、位于同一 navmesh island、可达，且本身就是合法 GT success
viewpoint。`F/T` 为执行链送出的 FORWARD / LEFT+RIGHT；`OOR/J` 为 theta 越界次数 / 约 `2pi`
branch-cut jump 次数。碰撞数来自完整 episode，不能当作接管后碰撞数。

| Probe / full | goal path | effective XY drift | rho start/min/final | theta OOR/J | F/T | net move | final GT | 主要归因 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| 26 / 835 | 1.376 m | 0 | 1.20/1.16/1.16 | 11/3 | 16/20 | .37 m | 1.163 m | invalid theta + raw oscillation |
| 27 / 995 | 5.109 m | 0 | 5.11/5.11/5.11 | 3/2 | 17/14 | 0 m | 5.109 m | invalid theta + no motion |
| 30 / 827 | 1.365 m | .50 m | .66/.66/.66 | 42/0 | 16/26 | 0 m | 1.365 m | theta 全程越界；另有 near-goal ray rewrite |
| 31 / 745 | 9.841 m | 0 | 2.88/2.81/2.81 | 10/5 | 16/30 | .08 m | 9.918 m | invalid theta + local point-goal 无全局路径 |

四例最终都由 forward heat=15 熔断；TURN 比例为 45.2%--65.2%。Probe 30 的 .50 m
effective-goal 改写是现有 `<1.5 m` ray heuristic，仅存在于这一例，不能解释其他三例。

### Action-level GT follower control

我从记录的接受位置恢复 agent state，让 Habitat-Sim 官方 `GreedyGeodesicFollower` 使用同样的
`.25 m FORWARD / 30° TURN` 离散动作去完全相同 endpoint：

| Probe | initial/final geodesic | actions F/L/R | control collisions | 结果 |
|---|---:|---:|---:|---|
| 26 | 1.376/.172 m | 5/0/8 | 0 | reached |
| 27 | 5.109/.160 m | 21/6/5 | 0 | reached |
| 30 | 1.365/.162 m | 5/1/7 | 0 | reached |
| 31 | 9.841/.083 m | 44/14/27 | 4 | reached |

结果是 **4/4 reached**。所以这些 endpoint、离散动作、门口和 navmesh 并没有结构性不可执行；
当前链失败位于 learned PointNav / adapter 一侧。由于 4/4 同时受到 invalid theta 污染，现有证据
不能把它进一步归成“PointNav 模型本身失败”，也不足以支持替换模型。

## 4. accepted-no-STOP 与正常对照

19 个 accepted-no-STOP 中：

- 14 个在接管后直接收到越界 theta；
- 15 个接管后循环至少 55--414 planner rows，TURN 比例约 45%--89%；其中 14 个也是 theta
  越界组，另 1 个是 full index 706；
- 这些长循环中，forward heat 多数最高只有 0--2、rotation heat 多数最高只有 1。原因不是
  真正有进展，而是 object goal orientation 随当前 pose 变化，触发整套 heat reset；
- 剩余四例接管后只有 3、20、25、27 个 planner rows，混合了极晚接管、已有成功距离但无 STOP、
  以及不足以确定的短 pursuit，不归因给 PointNav。

正常保护组提供了反例：full index 258 在 11 rows 内把 rho 从 .42 降到 .02，597 在 24 rows
内从 .30 降到 .09；两者接管段 theta 均在官方范围内。432/586/690 等在接受时已接近目标，
只产生两个内部 no-action row 后按 legacy path-exhausted 正常结束。另一方面，613 个成功 episode
也曾在探索某处暴露越界 theta，说明 bug exposure 具有广度，但不能据此声称同等数量可被救回。

## 5. full-1000 责任桶

下面是对 292 个官方失败的互斥、保守归因。百分比以 full-1000 为分母；“headroom”均不是已验证
SR 增益。

| 唯一主要桶 | N | full 占比 | 证据强度 |
|---|---:|---:|---|
| 1. goal-coordinate / adapter bug | **14** | **1.4%** | 接管后 theta 越界并进入长 pursuit；明确 defect，rescue 未反事实验证 |
| 2. wrapper state/heat bug（不与上行重复） | **1** | **0.1%** | index 706：无 theta 越界但长循环，goal-pose 变化持续清 heat |
| 3. PointNav model-only failure | **0** | 0 | Oracle-B 四例均被 adapter bug 混杂，不能单独归模型 |
| 4. collision / doorway / local geometry-only | **0** | 0 | 同 endpoint 官方 follower 4/4 reached；没有独立 collision-only 例 |
| 5. endpoint / upstream candidate or earlier stage | **273** | **27.3%** | 59 never-visible + 57 no-candidate + 50 no-accept + 107 accepted-STOP-fail |
| 6. 无法确定 / short late pursuit | **4** | **0.4%** | accepted-no-STOP，但不满足长时间执行责任条件 |
| **Total failures** | **292** | **29.2%** | — |

另有 86 个 failure 的最后 50 planner rows 呈现“高 TURN + rho 不降”的宽松 signature；其中
21 never-visible、32 visible-no-candidate、19 candidate-no-accept，只有 14 accepted-no-STOP。
前 72 个没有固定且已验证正确的 endpoint，不能从日志排除 frontier/candidate 改变，因此只是
**86/1000 的可疑上界**，绝不是 executor headroom。

可用于决策的三层数量是：

```text
固定合法 endpoint 下确认当前链失败：4/1000
明确 bounded defect 暴露的 late-stage failure：15/1000
全部 accepted-no-STOP 绝对上界：19/1000
```

## 6. 替换风险与最终 GO / NO-GO

整体替换执行器风险高：它会改变所有 frontier pursuit、正常成功轨迹、SPL 和 target closure，而目前
没有 20--30/1000 的独立 model/local-execution 责任证据。相反，theta wrap 是一行语义明确的
adapter correction；goal-state heat reset 也可限制在“translation goal 未改变时不要因 copied
orientation 清 progress state”的局部范围。

因此最终三选一为 **A. Bounded integration fix**，不是 B：

1. 若后续实施，只先做 theta wrap，并在固定 Oracle-B/accepted-no-STOP + 正常保护集上 paired；
2. 若仍有 index 706 型循环，再单独修 heat reset，不同时改 endpoint、STOP 或 target pipeline；
3. 不依据本审计启动新 full；只有 ProbeSet 证明 rescue 且保护无回归后，才值得一次正式评估；
4. 若 bounded fix 不能形成明确 rescue，永久搁置 PointNav executor 开发。

target grounding / approach 已由 Oracle-A ceiling 否决，本审计不推翻该结论。唯一保留的是一次
PointNav **adapter correctness cleanup**，不是新的 target approach 或 executor research idea。

## 审计产物

- `results/target_approach_oracle_ceiling_seed20260727/pointnav_executor_audit_v1.json`
- `results/target_approach_oracle_ceiling_seed20260727/pointnav_gt_follower_control_v1.json`
- [`scripts/audit_pointnav_executor.py`](../scripts/audit_pointnav_executor.py)
- [`scripts/replay_pointnav_gt_follower_control.py`](../scripts/replay_pointnav_gt_follower_control.py)

本轮没有修改默认 policy、PointNav weights、frontier、target pipeline 或 evaluator，也没有启动新的
full 实验。
