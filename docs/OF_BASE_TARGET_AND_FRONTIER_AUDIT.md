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

## 5. 为什么不应直接收紧 sofa

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

## 6. 参考代码支持的低风险识别策略

### 6.1 候选绑定 refinement

把当前候选的 mask/contour 和适量上下文明确展示给 Qwen，问题从“图中是否有 sofa”改成
“标出的候选是否是 sofa”。这直接对应 BeliefMapNav 的 `detection_refinement`，能减少“Qwen
看见另一个正确对象，却接管错误 centroid”的关联问题。

### 6.2 多候选选择而非硬拒绝

当同一帧或同一 composition 中存在多个候选时，让 Qwen 从带标记候选中选出最符合类别定义的
一个。BeliefMapNav 已对 couch 和 chair 写了候选选择提示词。它比提高全局阈值更保留召回。

### 6.3 类别化观察距离

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

### 6.4 不把房间先验作为硬门槛

BeliefMapNav 为每个目标硬编码了房间、常见位置、邻近物和概率，例如 sofa 对应 living/media/
bedroom，bed 对应 bedroom，toilet 对应 bathroom 与 sink/toilet-paper-holder 等。它们适合作为
frontier ranking 的软先验，但不适合作为目标存在性的硬拒绝条件，因为真实布局与数据标注会有
例外。

### 6.5 “无损”只能作为设计目标

任何新增 hard gate 都可能丢失当前成功 episode，不能在 full paired 实验前保证不损失 SR/SPL。
更安全的行为是：歧义候选暂不永久删除，而是降级、保留或获取一个更合适的候选绑定视角。

## 7. Geometry frontier 应成为主要研究方向

### 7.1 当前 OF-base 的候选缺口

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

### 7.2 VLFM/BeliefMapNav 的直接参考

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

### 7.3 推荐的问题定义

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

## 8. 跨楼层的研究定位

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

## 9. 推荐研究顺序

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

## 10. 最终判断

1. OpenFrontier 已经包含类别专用和 scene-specific benchmark 策略；继续堆同类提示词的边际收益
   预计有限。
2. 直接提高 sofa 或其他类别阈值没有“低损失”证据，当前概率分布也不支持安全切分。
3. 目标识别最值得保留的实验方向是 candidate-grounded refinement、候选选择和更合适的观察
   视角，而不是重新引入 ApexTarget 式重型 target fusion。
4. visual + geometry dual-source frontier 与现有失败漏斗、OpenFrontier 论文失败分析、VLFM/
   BeliefMapNav 参考实现三者一致，更适合作为主研究贡献。
5. 跨楼层应作为次级、协议相关扩展；不进入当前 HM3Dv2 主线，也不应分散 frontier generation
   的研究重点。
