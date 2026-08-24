# OF-base 目标识别与 Frontier 生成只读审计

日期：2026-08-22

审计对象：OF-base、OpenFrontier 上游、冻结 HM3Dv2 full-1000 日志及本地
`references/` 参考仓库

范围：只读代码与结果审计；未修改 agent、target pipeline、frontier pipeline、配置或 evaluator

## 1. 结论摘要

OpenFrontier 确实包含针对易混类别乃至特定 benchmark scene 的专门规则，但它没有
ApexTarget 那种显式的 confusable-label fusion，也没有按类别设置数值检测阈值。它采用的主要
方式是：

1. 类别专用验证提示词；
2. 类别别名；
3. 场景级目标名称与排除规则。

现有 HM3Dv2 日志不支持直接“收紧 sofa 阈值”。OF-base 的 sofa 错误终止大部分不是单纯把
chair 误认成 sofa，而更接近候选关联、目标 centroid、approach 与官方 1 m success region
之间的不一致。直接增加全局或类别硬阈值会同时损失大量真阳性。

目标识别方面，最有依据的微调方向是把当前全图类别验证改成**候选绑定验证**：明确判断当前
SAM3 mask/box，而不是判断最近六帧拼图中任意位置是否存在该类别。BeliefMapNav 的 contour/
bbox refinement 和多候选选择为这一方向提供了直接参考。

更值得作为主研究方向的是 **visual frontier + geometry frontier 的双来源候选生成**：visual
frontier 提供语义质量，geometry frontier 提供覆盖率，解决视觉候选集合本身没有正确方向的
情况。跨楼层应保留为次级、协议相关能力边界，不应成为当前主 idea。

对 OF-base 自身 292 个 V2 失败进一步拆解后，系统级的第一瓶颈是**探索覆盖与候选召回**：
59 个 episode 从未形成足够目标可见性，57 个已经看见目标却始终没有 SAM3 mask/candidate，合计
116/292（39.73%）。目标管线内部的第一瓶颈则是**候选绑定与停止几何不一致**：107 个 episode
在 Qwen 接受后显式 STOP，但不满足官方 1 m；其中 59 个验证窗口根本没有足够目标 pixels，另有
48 个虽看到目标，却停在错误的 3D centroid 或 1 m success region 之外。原始日志中的
`max_steps_reached` 多数只是这些上游问题的最终表现，不是可行动的根因。

## 2. OpenFrontier 已有的 benchmark 定向目标策略

OF-base 的以下逻辑与 OpenFrontier 上游提交
`a3f8b83da6135a88247651534061df2ea05850f6` 一致。

### 2.1 类别专用验证提示词

`vlm/utils.py::detect_target_object` 为六个 HM3D ObjectNav 类别加入了不同定义和反例：

- sofa：只接受可清楚容纳两人以上的 sofa、loveseat 或 sectional，排除单人 chair 和
  armchair；
- bed：要求完整床垫或寝具，排除 sofa/couch；
- chair：要求单人 chair 或 stool，排除 sofa、armchair、长凳和 exercise chair；
- TV：接受电视和电脑显示器，排除平板、厨房电器等显示屏；
- plant：接受盆栽、花瓶中的花和人工花，排除画作、照片和无叶装饰枝；
- toilet：只接受成人马桶，排除儿童或便携式马桶。

代码位置：[`vlm/utils.py`](../vlm/utils.py#L228-L254)；
[OpenFrontier 上游对应实现](https://github.com/cvg/OpenFrontier/blob/a3f8b83da6135a88247651534061df2ea05850f6/vlm/utils.py#L220-L246)。

### 2.2 类别别名与场景级补丁

常规类别别名包括：

- `tv_monitor -> television_screen`；
- `plant -> potted_plant`；
- `sofa -> loveseat`。

另有针对具体 HM3D scene ID 的规则：

- 指定场景的 bed 改成 `queen_bed`；
- TV 改成 `computer_monitor`；
- plant 改成 `flowers`；
- chair 改成 `brown chair`；
- 某场景固定观察高度，并使用 `plant_special_no_purple_flowers` 排除“金色桌子上的紫花”。

代码位置：[`nav/habitat_agent.py`](../nav/habitat_agent.py#L67-L91)；
[OpenFrontier 上游对应实现](https://github.com/cvg/OpenFrontier/blob/a3f8b83da6135a88247651534061df2ea05850f6/nav/habitat_agent.py#L62-L86)。

这些规则是 OpenFrontier 官方代码的一部分，不是 OF-base 迁移新增。它们也是非常明确的
benchmark 定向设计。后续不应继续按 scene ID 扩张这类补丁，因为其泛化性和研究解释性较弱。

在当前 V2 manifest 上，上述场景—类别规则涉及 34 个 episodes，OF-base 成功 7 个，
ApexTarget 成功 3 个。这个对照说明这些场景整体困难，但没有“移除规则”的反事实结果，因此
不能据此断言规则有害或有益。

### 2.3 没有类别专用数值阈值

OpenFrontier 的目标验证仍使用统一的 `termination_threshold=0.7`，没有 sofa、chair 等类别
各自的接受阈值。SAM3 的 `detection_score` 被写入诊断与对象轨迹，但当前 OF-base 接管条件并
未按类别使用该分数。

## 3. 当前目标识别链路的关键结构问题

OF-base 当前目标链路为：

```text
SAM3 在最近六帧拼图中产生 masks
  -> mask + depth 转成 3D object frontier
  -> 导航到所选候选视点
  -> Qwen 查看最近六帧拼图，判断画面里是否存在目标类别
  -> 接受后锁定此前所选 object frontier
```

SAM3 候选生成见 [`nav/agent.py`](../nav/agent.py#L350-L469)，目标验证见
[`nav/agent.py`](../nav/agent.py#L775-L875)。

这里存在一个重要的候选关联缺口：Qwen 验证的是整张六帧拼图中“有没有该类别”，并没有明确
验证当前 `goal_object` 对应的 SAM3 mask。画面中可能确实存在 sofa，但 Qwen 看到的 sofa 不
一定就是当前准备接管的 mask/3D centroid。

BeliefMapNav 的参考实现采用了更明确的候选绑定方式：

- 对 contour/bbox 中的对象做类别专用 refinement；
- 多个带颜色标记的候选同时存在时，让 VLM 选择最符合目标定义的一个；
- 为 couch/chair、TV/画框和黑色窗户、bed/场景环境提供显式反例。

参考位置：本地
`/home/hsy/references/BeliefMapNav/vlfm/vlm/openai_api.py:13`；
[BeliefMapNav 官方仓库](https://github.com/ZiboKNOW/BeliefMapNav/blob/c9800d09c2e0f0203a0efbee976d421afade7456/vlfm/vlm/openai_api.py#L13-L109)。

因此，目标识别微调的优先级应是：

1. 候选绑定验证：明确判断被框出或被 mask 标出的当前候选；
2. 候选选择：多个 mask 存在时，选择最符合 benchmark 类别定义的候选，而不是只做全图
   二分类；
3. 类别化观察距离：在验证之前为不同尺度类别取得更合适的完整视图；
4. 类别提示词的小幅补充；
5. 最后才考虑数值阈值。

## 4. HM3Dv2 full-1000 失败漏斗

数据源：
`results/openfrontier_base_sam3_full_hm3dv2_1000_seed20260727/episodes`。
官方 HM3Dv2 SR@1m 为 70.80%（708/1000），共有 292 个官方失败。

按 first unresolved stage 做互斥划分：

| 首个未解决阶段 | N | 失败占比 |
| --- | ---: | ---: |
| target 从未达到 50 pixels GT-visible | 59 | 20.21% |
| GT-visible，但始终没有 SAM3 geometry candidate | 57 | 19.52% |
| 有 candidate，但 Qwen 始终未接受 | 50 | 17.12% |
| Qwen 已接受，但没有显式 STOP | 19 | 6.51% |
| Qwen 已接受并 STOP，但官方失败 | 107 | 36.64% |
| **Total** | **292** | **100%** |

这五组不能全部归因于“识别”：

- 59 个 never-visible 更接近探索覆盖与 frontier proposal/selection 问题；
- 57 个 visible-no-candidate 更接近 SAM3 提议召回或观察视角问题；
- 50 个 candidate-no-accept 才是典型验证召回问题；
- 19 个 accepted-no-STOP 是接管后的 approach/recovery 问题；
- 107 个 accepted-STOP-official-fail 同时混合了语义误判、候选关联、centroid 和停止距离问题。

## 5. OF-base 自身失败 case 的细粒度归因

### 5.1 终止 reason 不是根因

runner 最终写出的 reason 为：

| Terminal reason | N | 失败占比 |
| --- | ---: | ---: |
| `max_steps_reached` | 172 | 58.90% |
| `false_positive` | 107 | 36.64% |
| `robot_stuck` | 8 | 2.74% |
| `no_frontiers` | 5 | 1.71% |

这里的 `false_positive` 只表示 agent 显式 STOP 而官方 `success=0`，并不等价于 107 次语义
误识别；`max_steps_reached` 也只表示预算耗尽，不说明此前是没看见目标、SAM3 没提议，还是 Qwen
拒绝。按 trace 中最后一个已通过的阶段继续拆分，才能得到下面的瓶颈。

### 5.2 探索覆盖：59 个 never-visible

59 个 episode 在整条轨迹中从未出现 `>=50 pixels` 的 evaluator target mask：

- 52 个以 max-steps 结束，4 个 robot-stuck，3 个 no-frontiers；
- navigation steps 的 median 为 496，45/59 跑到至少 490 steps；
- toilet 占 28/59，plant 与 TV 各 10，bed 7，chair 与 sofa 各 2；
- 8/59 曾产生并验证错误的 SAM3 候选，但 Qwen 均拒绝；它们仍然没有获得真正目标视角。

因此这组主要不是“接管太晚”，而是接近完整使用 500-step 预算后仍未覆盖到目标。它直接支持把
geometry frontier 用作 visual frontier 的召回补充。59 个 episode 对应 5.9 pp 的诊断上限，
但不是可直接相加的预期 SR 增益。

### 5.3 SAM3 candidate recall：57 个 visible-but-no-candidate

这 57 个 episode 的所有 segmentation event 都是 `mask_count=0`，不是 mask 生成后被 3D 几何
过滤：

- 56/57 至少有一个 SAM3 六帧 batch 与 `>=50 pixels` 的 GT-visible frame 重叠；
- 48/57 至少有一个 batch 达到 `>=500 pixels`，32/57 达到 `>=2000 pixels`；
- 单 episode 最大 target pixels 的 median 为 2558，GT-visible frames 的 median 为 25；
- 56/57 最终 max-steps，navigation steps 的 median 为 495；
- plant 占 22/57，sofa 13，TV 9，toilet 7，bed 4，chair 2。

所以这组不能主要解释成目标只在两次 SAM3 调用之间短暂闪现。多数 episode 给过多次、且面积不小
的目标视图，但 SAM3 text-prompt 分割没有形成任何 mask。直接瓶颈是观察视角下的 SAM3 proposal
recall；frontier 覆盖和类别化观察距离可能提供更好的第二视角，但不能替代这一事实。

### 5.4 Qwen validation：50 个 candidate-no-accept

对这 50 个 episode 继续按验证时的 GT evidence 拆分：

| 子类 | N | 可支持的解释 |
| --- | ---: | --- |
| 验证窗口有 `>=50 pixels` GT target，但 Qwen 拒绝 | 31 | 直接的验证召回缺口 |
| 所有被验证窗口都没有足够 GT target，Qwen 拒绝 | 17 | Qwen 拒绝与 trace 一致；上游给的是错误候选/视角 |
| 有 candidate，但从未抵达验证 | 2 | candidate navigation/可达性问题，均为 TV |

31 个 trace-supported Qwen false rejection 中，29 个验证窗口达到 `>=500 pixels`，23 个达到
`>=2000 pixels`；41 次带 GT 的拒绝事件概率全部只有 `0.0` 或 `0.1`。类别分布为 TV 13、
toilet 8、plant 6、chair 2、sofa 2。这里没有接近 0.7 阈值的连续概率，因此小幅调低统一阈值
不会解决这组问题。

同时，这 50 个 episode 的首次 candidate median step 为 110、首次 verification median step 为
150，最终 navigation steps median 为 494。失败既不是普遍“最后几十步才第一次看见候选”，也
不全是 Qwen：只有上述 31 个能由冻结 trace 直接归为验证假阴性。

### 5.5 接管后无 STOP：19 个 approach/termination failures

19 个 episode 已有 Qwen accepted event，却没有 `termination_event`：16 个 max-steps，3 个
robot-stuck。它们并非主要由晚接管构成：

- 首次接受的 median step 为 235，结束前剩余预算 median 为 265 steps；
- 13/19 在 step 400 以前接管，只有 6/19 属于最后 100 steps 才接管；
- 16/19 的验证窗口有足够 GT target；
- 全部记录的 `approach_path_steps` 都为 1；该字段是 planner 输出的 pose 数，不等于一个 Habitat
  action；
- 最终 `distance_to_goal` 有 4/19 已经 `<=1 m`、9/19 已经 `<=1.5 m`，但因为没有显式 STOP，
  官方 Success 仍为 0。

这说明接管后的 pointnav/replan、centroid 距离判定与 STOP 闭环存在独立问题。尤其 4 个终点已在
官方 1 m 区域却没有 STOP，是明确的 termination opportunity loss；其余 episode 还混有错误
centroid 与 approach/recovery 失败。

### 5.6 接受并 STOP、但官方失败：107 个失败的三分法

这 107 个 episode 是最大的单一失败阶段，而且 STOP 很早：termination step 的 median 为 92、
Q3 为 194.5。它们不是接近 500 steps 后偶然失败，而是错误承诺提前截断探索。

使用 Qwen 验证前六帧是否存在 `>=50 pixels` GT target，并用 `1.5 m` 作为非官方的 near/far
诊断边界，可以互斥拆成：

| 子类 | N | 失败占比 | 解释 |
| --- | ---: | ---: | --- |
| 验证窗口无足够 GT target，却接受并 STOP | 59 | 20.21% | 全图验证接受了错误上下文/错误候选，是最强的 false-commitment 证据 |
| 验证窗口有 GT，官方终距在 `(1.0, 1.5] m` | 17 | 5.82% | 1 m 边界附近的 centroid/STOP 几何失配；其中 sofa 13 |
| 验证窗口有 GT，官方终距 `>1.5 m` | 31 | 10.62% | 看到正确类别，但锁定的 SAM3 mask、3D centroid 或 approach endpoint 没有落到正确 success region |
| **Total** | **107** | **36.64%** | — |

59 个无 GT 的错误接受按类别为 chair 16、bed 13、plant 12、TV 8、sofa 6、toilet 4。这是当前
全图 Qwen 判断没有绑定到所选 mask/centroid 的直接风险：画面语义判断与实际接管对象之间没有可
审计的一一对应关系。

另 48 个验证窗口确有目标的失败中，17 个是 1–1.5 m near miss，31 个终距仍大于 1.5 m。后者
不能再叫做纯语义误判；更可能是 Qwen 看见了正确类别，但 agent 接近的是同一 composition 中另一个
SAM3 mask、物体局部 centroid，或不可对齐官方 geodesic success region 的 endpoint。这也解释了
为什么 sofa 的主要问题不是简单收紧类别阈值。

当前 STOP 条件为“规划路径耗尽 **或** 到锁定 centroid 的欧氏距离小于 1 m”。全部 810 个显式
STOP episode 中：

- path-exhausted 触发 318 个，其中 42 个官方失败，失败率 13.21%；
- centroid-distance 触发 492 个，其中 65 个官方失败，失败率同为 13.21%。

因此损失不是单独由 `path_exhausted` 这个 OR 分支造成；两种内部停止代理都没有与 evaluator 的
1 m success region 完全对齐。需要解决的是候选绑定、目标几何表示和 STOP 判据之间的整体一致性。

### 5.7 按类别看，瓶颈并不相同

| Category | 失败数 | 主要构成 | 首要瓶颈 |
| --- | ---: | --- | --- |
| bed | 32 | 17 accepted-STOP-fail，其中 13 个验证窗口无 GT | 错误候选接受/绑定 |
| chair | 35 | 27 accepted-STOP-fail，其中 16 个无 GT、10 个 GT-visible 但终距 >1.5 m | precision、candidate grounding 与 centroid |
| plant | 61 | 10 never-visible + 22 no-candidate；另 19 accepted-STOP-fail | SAM3/探索召回，其次错误接受 |
| sofa | 54 | 31 accepted-STOP-fail；其中 25 个验证窗口有 GT，13 个为 1–1.5 m near miss | 目标几何与官方 1 m 对齐，而非单纯类别阈值 |
| toilet | 56 | 28 never-visible + 7 no-candidate + 12 candidate-no-accept | frontier 覆盖与目标召回 |
| TV | 54 | 22 candidate-no-accept，其中 13 个为 GT-present Qwen rejection；另有 19 个 visibility/proposal failures | Qwen 验证召回与上游覆盖并存 |

### 5.8 主要瓶颈排序

从系统层面，当前 V2 OF-base 的瓶颈应按下面的口径表述：

1. **探索覆盖 + SAM3 proposal recall：116/292（39.73%），是最大的根因族。** 其中 59 个
   never-visible 直接指向 frontier coverage，57 个 visible-no-candidate 直接指向 SAM3/观察视角。
2. **候选绑定 + centroid/STOP 对齐：107/292（36.64%），是最大的单一失败阶段。** 59 个是
   无 GT 的错误承诺，48 个是看到目标后仍停错位置。
3. **Qwen validation recall：至少 31/292（10.62%）有直接 GT evidence。** 不能把完整的 50 个
   candidate-no-accept 都算成 Qwen 假阴性。
4. **接管后的 approach/termination：19/292（6.51%）。** 其中多数不是接管太晚；4 个已经进入
   官方 1 m 区域但没有 STOP。
5. **纯 stuck/no-frontier 不是主要瓶颈。** 原始终止 reason 中两者合计只有 13/292；172 个
   max-steps 的真正来源已经分散在上述上游阶段。

这些数量是冻结轨迹的责任分层，不是互相独立的可恢复 SR，也不能当作修改某个模块后的反事实
收益。它们给出的明确结论是：系统主线应优先提高 frontier/proposal coverage；目标子系统则应
优先做 candidate-grounded verification 和 evaluator-aligned stopping，而不是先做类别硬阈值。

## 6. 为什么不应直接收紧 sofa

OF-base 的 31 个 sofa“Qwen 接受并 STOP、官方失败”中：

- 25/31 在 Qwen 验证使用的最近六帧里存在至少 50 pixels 的 evaluator target mask；
- 只有 6/31 在验证窗口中没有足够 GT target pixels；
- 多个失败终点的官方 `distance_to_goal` 约为 1.0–1.6 m，更接近目标几何、approach 和
  official success region 的对齐问题。

因此，31 个 sofa 错误终止并不等于 31 个 sofa/chair 类别混淆。对 sofa 增加硬识别门槛不能
解决其中的大多数。

Qwen 接受概率也高度量化：

| Sofa outcome | p=0.9 | p=0.95 | p>0.95 | Total |
| --- | ---: | ---: | ---: | ---: |
| 官方成功 | 20 | 47 | 66 | 133 |
| 接受并 STOP、官方失败 | 8 | 21 | 2 | 31 |

如果把接受阈值提高到 `0.95`，按冻结日志中已发生的决定，会同时挡住 20 个成功和 8 个失败；
如果改成严格 `>0.95`，会同时挡住 67 个成功和 29 个失败。被拒后继续探索可能改变最终结果，
所以这不是 SR 反事实，但已足以说明现有概率缺少安全的阈值分离度。

在所有“接受并 STOP、官方失败”中，验证窗口没有足够 GT target pixels 的类别分布为：

| Category | N |
| --- | ---: |
| chair | 16 |
| bed | 13 |
| plant | 12 |
| tv_monitor | 8 |
| sofa | 6 |
| toilet | 4 |
| **Total** | **59** |

如果进行候选级 refinement，直接证据支持的优先级是 chair，其次 bed/plant/TV，而不是首先对
sofa 做全局收紧。

## 7. 参考代码支持的低风险识别策略

### 7.1 候选绑定 refinement

把当前候选的 mask/contour 和适量上下文明确展示给 Qwen，问题从“图中是否有 sofa”改成
“标出的候选是否是 sofa”。这直接对应 BeliefMapNav 的 `detection_refinement`，能减少“Qwen
看见另一个正确对象，却接管错误 centroid”的关联问题。

### 7.2 多候选选择而非硬拒绝

当同一帧或同一 composition 中存在多个候选时，让 Qwen 从带标记候选中选出最符合类别定义的
一个。BeliefMapNav 已对 couch 和 chair 写了候选选择提示词。它比提高全局阈值更保留召回。

### 7.3 类别化观察距离

BeliefMapNav 硬编码了以下 observation ranges：

- bed、couch：1–3 m；
- TV、chair：1–2 m；
- plant：1–1.8 m；
- toilet：0.7–1.5 m。

参考：
本地 `/home/hsy/references/BeliefMapNav/vlfm/policy/itm_policy.py:89`；
[BeliefMapNav 官方仓库](https://github.com/ZiboKNOW/BeliefMapNav/blob/c9800d09c2e0f0203a0efbee976d421afade7456/vlfm/policy/itm_policy.py#L89-L161)。

OF-base 当前对所有 object frontier 使用统一的约 0.7 m viewpoint separation。较大的 sofa/bed
在过近视角下可能缺少完整轮廓。类别化观察距离不会直接删除候选，因此比硬阈值更适合作为低风险
实验；但 BeliefMapNav 主要把这些范围用于 frontier observation belief，迁移到目标验证仍需
paired 实验，不能预设一定提升。

### 7.4 不把房间先验作为硬门槛

BeliefMapNav 为每个目标硬编码了房间、常见位置、邻近物和概率，例如 sofa 对应 living/media/
bedroom，bed 对应 bedroom，toilet 对应 bathroom 与 sink/toilet-paper-holder 等。它们适合作为
frontier ranking 的软先验，但不适合作为目标存在性的硬拒绝条件，因为真实布局与数据标注会有
例外。

### 7.5 “无损”只能作为设计目标

任何新增 hard gate 都可能丢失当前成功 episode，不能在 full paired 实验前保证不损失 SR/SPL。
更安全的行为是：歧义候选暂不永久删除，而是降级、保留或获取一个更合适的候选绑定视角。

## 8. Geometry frontier 应成为主要研究方向

### 8.1 当前 OF-base 的候选缺口

OpenFrontier 的探索候选来自当前 RGB-D 图像的 FrontierNet 预测：

```text
RGB-D
  -> FrontierNet distance field / class mask
  -> image-space frontier clusters
  -> depth anchoring into 3D
  -> persistent FrontierManager
```

实现位置：[`frontier/detector.py`](../frontier/detector.py#L96-L180)。

Wavemap 在当前系统中提供 free/occupied 几何信息，用于过滤、gain 更新与 planner，但不从累积
free/unknown boundary 产生补充 frontier。因此，一旦当前视角的 visual frontier proposal 没有
覆盖正确方向，后续 VLM 只能在一个缺少正确答案的候选集合上打分。

OpenFrontier 论文的失败分析也指出：机器人可能经过目标附近，却没有生成能引导其进一步接近
目标的 informative frontier，并把更可靠的 frontier detection 列为改进方向。
[OpenFrontier 论文](https://arxiv.org/abs/2603.05377)

### 8.2 VLFM/BeliefMapNav 的直接参考

VLFM 从深度构建 occupancy map，并在 navigable/explored boundary 上提取 geometry frontier；
BeliefMapNav 继承了同类实现：

```text
explored_area = dilate(explored_area)
frontiers = detect_frontier_waypoints(
    navigable_map,
    explored_area,
    area_threshold,
)
```

参考位置：
本地 `/home/hsy/references/BeliefMapNav/vlfm/mapping/obstacle_map.py:155`；
[BeliefMapNav 官方仓库](https://github.com/ZiboKNOW/BeliefMapNav/blob/c9800d09c2e0f0203a0efbee976d421afade7456/vlfm/mapping/obstacle_map.py#L155-L170)；
[VLFM 论文](https://arxiv.org/abs/2312.03275)。

### 8.3 推荐的问题定义

主研究问题可以定义为：

> visual frontier 提供语义质量，geometry frontier 提供候选覆盖率；两者联合解决视觉候选集合
> 本身缺少正确探索方向的问题。

建议保持以下边界：

- visual frontier 仍是主候选；
- geometry frontier 从当前楼层的 occupancy free/unknown boundary 产生；
- 两类 frontier 做空间去重；
- geometry frontier 只补充 visual frontier 未覆盖的方向或连通区域；
- 几何候选必须满足可达性、边界支持面积和空间新颖性；
- 不把当前图像中不可见的 geometry frontier 伪装成 visual mark 交给 Qwen；
- geometry frontier 先按探索价值导航，进入可视范围后再转化为 visual/semantic frontier；
- 记录 `source=visual|geometry|object`，使 paired audit 能独立归因。

最重要的设计原则是“补召回而不是替换”：如果直接用 geometry frontier 替换 visual frontier，
会丢失 OpenFrontier 的语义优势；如果无约束地合并所有候选，则可能扩大候选池、增加路径分叉并
损害 SPL。

V2 的 59 个 never-visible 失败是该方向最直接的理论 headroom，对应 5.9 pp SR；geometry
frontier 不可能保证恢复全部，但能直接作用于这类覆盖缺口。它也可能通过提供第二观察视角，间接
改善 57 个 GT-visible-but-no-SAM3-candidate episode。

## 9. 跨楼层的研究定位

当前 OF-base 没有显式 floor state、stair detector、楼层拓扑或 transition planner。代码只会
随 agent 高度更新 `nav_level`，并对 frontier 的 z 和可达性做过滤；这不构成可靠的跨楼层系统。

严格来说，BeliefMapNav 后发布仓库已加入 stair map、多楼层 obstacle map 和楼层切换代码；
但论文的验证结论仍是：3D voxel map 虽能表达不同高度，系统受楼梯识别和局部 planner 限制，
无法可靠到达跨楼层目标。论文报告，不同楼层目标占其失败案例的 36.78%。
[BeliefMapNav 论文](https://arxiv.org/abs/2506.06487)

这说明跨楼层是一个独立的楼梯感知、楼层拓扑、transition execution 和恢复系统，不应当被视为
geometry frontier 的自然附赠能力。

当前定位建议：

- HM3Dv2：明确作为 same-floor ObjectNav，跨楼层不进入主指标和主开发线；
- HM3Dv1：保留为能力边界和次级失败分析维度，等待 OF-base V1 与 T1/ApexTarget 对比；
- 主论文/主 idea：聚焦 dual-source frontier generation 与 frontier proposal recall；
- 未来扩展：在 geometry frontier 稳定后，把楼梯建模为独立的 `transition frontier`，放在单独
  分支或附加实验中。

## 10. 推荐研究顺序

```text
冻结 OF-base baseline
  -> 完成 HM3Dv1，与 T1/ApexTarget 做阶段和类别对比
  -> 候选绑定 target refinement 的离线 shadow audit
  -> visual + geometry frontier proposal/coverage audit
  -> 单独 paired full experiment
  -> 未来再研究 stair/transition frontier
```

目标识别实验应先在冻结成功/失败候选图像上做 shadow decision，检查 candidate-level precision
和 recall，不先改变导航轨迹。geometry frontier 实验则应重点记录：

- visual-only、geometry-only 和重叠候选数量；
- time-to-first-visible；
- never-visible failure 数量；
- visual/geometry frontier 的选择与成功来源；
- unique explored area、navigation steps、SR 和 SPL；
- no-frontier emergency rotation 与 stuck/max-steps。

正式实现后仍遵守当前实验纪律：最多用 1 个 episode 确认可运行，随后直接启动冻结 manifest 的
完整 paired 实验，不进行反复 smoke 或在线阈值搜索。

## 11. 最终判断

1. OpenFrontier 已经包含类别专用和 scene-specific benchmark 策略；继续堆同类提示词的边际收益
   预计有限。
2. 直接提高 sofa 或其他类别阈值没有“低损失”证据，当前概率分布也不支持安全切分。
3. 目标识别最值得保留的实验方向是 candidate-grounded refinement、候选选择和更合适的观察
   视角，而不是重新引入 ApexTarget 式重型 target fusion。
4. visual + geometry dual-source frontier 与现有失败漏斗、OpenFrontier 论文失败分析、VLFM/
   BeliefMapNav 参考实现三者一致，更适合作为主研究贡献。
5. 跨楼层应作为次级、协议相关扩展；不进入当前 HM3Dv2 主线，也不应分散 frontier generation
   的研究重点。
6. OF-base 自身 V2 失败的首要系统瓶颈是探索/候选覆盖（116/292）；目标管线的首要瓶颈是
   candidate grounding 与 1 m STOP 几何对齐（107/292），不是单独的 sofa 阈值或 max-steps。
