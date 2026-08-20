# OpenFrontier 的 BEV 增强式 VLM Frontier Scoring

状态：设计评估，不是已批准实现<br>
日期：2026-08-11<br>
适用基线：`OF-ApexTarget v1 deterministic` 及其后续分支<br>
边界：本文不修改当前 full HM3Dv1 评测，不改变 target detection/fusion、FrontierNet、Wavemap、planner 或 executor。

## 1. 结论

这个方向可行，也值得做成下一阶段的受控研究假设，但不应直接移植 H2 的「VLM 覆盖几何 frontier」控制方式。

推荐的第一版是：给当前 OpenFrontier 的 RGB SoM frontier scorer 增加一张由系统内部地图生成的 BEV 上下文，让 VLM 对同一组候选分别输出：

- `semantic_probability`：从 RGB 场景语义判断该方向与目标类别的关联；
- `history_spatial_score`：从 BEV 判断该方向是否重复、回环，或是否通向独立且尚未覆盖的分支；
- `confidence` 与可审计原因。

VLM 第一版不返回最终 frontier，不直接 override planner。最终 utility 仍由确定性代码组合已有 information gain、distance、semantic probability 和经过边界约束的 history signal。

这保留了 H2 中已显示局部价值的「全局探索上下文」，同时规避 H2 在完整集上出现的高频 override 与明显 SPL 损失。

## 2. 已核实事实

### 2.1 当前 OpenFrontier scorer 看到了什么

`nav/agent.py` 在 frontier detection 后，只把当前 `ft_list` 交给 `FrontierDetector.get_SoM_img()`。后者在当前 RGB 上标出 A/B/C 等 frontier 标签，再调用 `detect_frontier_probabilities()`。

当前 Qwen 请求包含：

- 一张当前 RGB SoM 图；
- 目标类别；
- 当前图中 frontier 标签；
- 要求每个标签输出 target-related probability 和一句解释的 prompt。

当前 Qwen 不接收：

- BEV/occupancy map；
- agent 历史轨迹或 visit density；
- persistent frontier 的生命周期；
- 过去 RGB keyframes；
- 上一次 frontier 决策或显式 episodic memory。

因此，当前 prompt 虽要求考虑「larger unknown regions」和 longer-term navigation，但图像本身没有提供足够的地图或历史证据。模型只能从当前 RGB 进行推测。

相关代码：

- `nav/agent.py`：SoM 生成、Qwen 调用和 probability 回写；
- `frontier/detector.py:get_SoM_img()`：单张当前 RGB 标注；
- `vlm/utils.py:build_frontier_probability_prompt()`：当前打分语义；
- `frontier/manager.py:update_utility()`：probability、gain 和 distance 的确定性组合。

### 2.2 OpenFrontier 已经有历史，但历史不在 VLM 内

`FrontierManager` 会持久管理 frontiers，并利用 robot poses、地图可见空间和 frontier merge/filter 调整 exploration gain。也就是说，系统整体并非没有历史；缺少的是让 VLM 在解释 RGB 语义时看到相同空间上下文。

这个区别很重要：BEV signal 不能简单重复惩罚 visited area，否则会与 manager 已有的 `u_gain` 调整重复计数。

### 2.3 H2 实际实现

在历史 VLFM/H2 工作树的提交 `437dd7e` 中：

- `decision_panel.py` 用 VLFM 自己的 `ObstacleMap` 生成 BEV，不使用 Habitat oracle top-down map；
- BEV 包含 policy map、完整轨迹、repeat-visit density、robot pose/heading、全部合法 frontiers、候选编号和 active frontier；
- 非 BEV-only 模式会在右侧加入最多四张历史 RGB keyframe card；
- 请求仍是一张拼接后的 panel，而不是原生多图请求；
- `HybridFrontierPolicy` 以 geometry candidate 1 为默认，只在 active-frontier replan boundary 调用 VLM，并带 warmup、cooldown、调用次数限制、override 次数限制与部分二次确认；
- H1 的 `history_spatial_only` 模式只允许历史空间证据覆盖 geometry。

所以用户的核心理解——「VLM 可以利用标注 frontier 与历史轨迹的 BEV」——是正确的；需要纠正的只是：H2 和当前 OpenFrontier scorer 都不是原生多图。H2 使用的是单张复合 panel，当前 OpenFrontier 使用的是单张 RGB SoM。

### 2.4 现有实验能证明什么

已封存结果：

| 配置 | 数据 | SR | SPL | 能支持的结论 |
| --- | ---: | ---: | ---: | --- |
| Hybrid coherent（H1/H2 路线） | fixed random-100 | 55.0% | 0.2186 | 存在局部有效 override，但总体没有超过同集 T1 |
| T1 ApexFusion | fixed random-100 | 55.0% | 0.2189 | 同一小集 aggregate 基本持平 |
| H2 relaxed | full HM3Dv1, 2000 | 52.9% | 0.2251 | 高频历史 override 没有形成可靠整体收益，路径效率明显偏低 |

H2 full trace 还记录了：

- 1172 次 `KEEP_GEOMETRY`；
- 2554 次 `OVERRIDE HISTORY_SPATIAL`；
- 1935 次 `REJECT INVALID_EVIDENCE`；
- 292 次 `VALUE_TAKEOVER`。

这说明 BEV 能触发有意义的历史/回环判断，但不能证明让 VLM 高频接管 frontier 选择是可靠策略。现有结果更支持「BEV 是辅助证据」，不支持「BEV-VLM 是独立高层控制器」。

## 3. 与当前 OpenFrontier 结合时的核心设计

### 3.1 保持三个概念分离

对每个 candidate `i`：

```text
current RGB SoM --------> semantic_probability_i
policy-built BEV -------> history_spatial_score_i
FrontierManager --------> information_gain_i, distance_i, lifecycle_i

                              deterministic fusion
                                      |
                                  utility_i
```

不要让 VLM 输出一个含义混杂的「最终概率」。特别要避免把以下两件事合并：

- 这个方向在语义上是否更可能通向目标；
- 这个方向是否比已重复探索的分支更新颖。

OpenFrontier 当前 `probability` 会与 `u_gain` 相乘。若直接把 BEV 的 novelty 也写进 probability，manager 的 history penalty 和 VLM 的 history judgment 可能被重复计算，同时破坏 OpenFrontier 原本「目标价值」与「探索价值」的可解释分离。

### 3.2 第一版候选集合必须完全一致

RGB SoM 和 BEV 必须展示同一个 immutable candidate snapshot，并共享完全相同的 label。

第一版只允许当前 `ft_list` 中已经进入 RGB SoM 的候选参与评分。不要因为 BEV 能显示全局 persistent frontiers，就同时扩大可选集合。否则无法判断变化来自 BEV 上下文，还是来自 candidate eligibility 改变。

后续若需要给离开当前视野的 persistent frontier 补分，应作为单独实验。

### 3.3 第一版使用单张复合 panel

建议布局：

```text
+----------------------+----------------------+
| Current RGB SoM      | Policy-built BEV     |
| A/B/C markers        | same A/B/C markers   |
| visual semantics     | path + map + history |
+----------------------+----------------------+
```

理由：

- 当前 `QwenClient.generate()` 接口只接收一张图；
- 当前 vLLM 启动参数是 `--limit-mm-per-prompt {"image": 1, "video": 0}`；
- H2 已验证复合 panel 的工程可行性；
- 单张 panel 固定了图像顺序和 label correspondence；
- 不需要在第一轮改服务部署，且更容易做确定性 replay。

原生两图请求可以作为后续 ablation。它在概念上没有障碍，但会增加 vision token、延迟和服务变量，不适合作为最小实现。

### 3.4 BEV 只允许使用在线系统状态

BEV 必须由 Wavemap/FrontierManager/agent trajectory 生成，严禁调用 Habitat `TopDownMap`、semantic scene annotation 或 goal viewpoint。

最低内容：

- explored free / occupied / unknown；
- 当前机器人位置和朝向；
- episode 内历史轨迹；
- 当前候选 A/B/C；
- 当前 active/selected frontier（若有）；
- 清楚且固定的 legend。

可选但要单独消融：

- repeat-visit density；
- persistent frontier history/failure state；
- frontier age；
- 已失效 frontier 的淡化标记。

地图应采用固定 episode/world frame，而不是每次随机器人旋转。裁剪范围变化也应尽量稳定，并在 prompt 中明确地图只覆盖当前已建图区域。

### 3.5 建议输出协议

```json
{
  "A": {
    "semantic_probability": 0.62,
    "history_spatial_score": 0.35,
    "redundancy_risk": 0.70,
    "confidence": 0.66,
    "semantic_reason": "...",
    "spatial_reason": "..."
  }
}
```

约束：

- `semantic_probability` 只引用 RGB/scene evidence；
- `history_spatial_score` 只引用 BEV 中可见的 coverage、trajectory、branch 和 loop evidence；
- 没有充分证据时回到中性值，不允许把距离、现有 utility 或 geometry rank 伪装成视觉语义；
- VLM 不返回 `selected_id`、`STOP` 或 planner command；
- parser 对缺字段、越界值、错误 label 和非 JSON 必须 fail closed，回退当前 OpenFrontier score。

## 4. 主要风险与对应边界

### 风险 A：历史证据被双重计入

FrontierManager 已通过地图与 robot poses 调整 gain。若 BEV score 再泛化为「没走过就高分」，会重复奖励 novelty。

边界：shadow 阶段先记录 BEV score 与现有 `u_gain`、visit density 的相关性。若高度相关，BEV 不应进入 utility；只有 VLM 能可靠识别 manager 标量难以表达的 branch/loop topology 时才有增量价值。

### 风险 B：VLM 混淆 target likelihood 与 exploration utility

这是最严重的架构风险。一个标量概率无法区分「更像目标房间」与「更少探索」。

边界：强制双字段输出，确定性代码分别消费；禁止直接重用 H2 override。

### 风险 C：candidate label/坐标错配

当前 RGB label 属于当次 `ft_list`，persistent manager frontier 有自己的 identity。两者如果分别排序或 merge，A 在两张图上可能不是同一实体，且错误不会导致 crash。

边界：先冻结 candidate snapshot，再由同一个 renderer context 同时生成 RGB 和 BEV；保存 label、pixel position、world position、frontier ID 与 panel artifact，做一次人工 overlay 审查。

### 风险 D：BEV 图形语言不稳定

密集轨迹、动态 crop、小字和相近颜色会让 VLM 编造拓扑关系。

边界：低复杂度视觉规范、固定颜色/legend、较粗轨迹、限制候选数量；同时把必要的确定性 metadata 以紧凑文本附在 prompt 中，而不是依赖 VLM 从小字读取数值。

### 风险 E：多楼层投影混叠

单张 2.5D BEV 会把上下楼层投影到同一平面。

边界：当前阶段只渲染当前 floor slice，并在输出标记 floor unknown/current-slice limitation。真正 multi-floor 后必须使用 floor-indexed panel，不能延用全高投影。

### 风险 F：延迟与确定性回归

panel 会增加视觉 token。prompt/panel 的任何变更也会改变当前已冻结的 Qwen 打分轨迹。

边界：不在正在运行的 full HM3Dv1 中启用；新分支单独冻结 prompt hash、renderer version、Qwen model/revision、sampling 参数与图像尺寸。

## 5. 分阶段开发计划

### Phase 0：只做 renderer fixture

目标：证明 RGB/BEV label 和坐标一致，不接 VLM、不影响动作。

产物：

- `FrontierScoringSnapshot`：冻结 labels、current frontiers、world poses 与 map revision；
- 无 oracle 的 `FrontierContextPanelRenderer`；
- 5–10 个固定 episode step 的 panel 与 JSON sidecar；
- 人工确认 A/B/C 在 RGB 和 BEV 指向同一 frontier。

通过标准：无坐标翻转、无 label 漂移、轨迹和 robot heading 正确、panel 只含在线信息。

### Phase 1：shadow scoring

目标：调用 Qwen 并记录双分量输出，但完全不改变 `ft.probability`、utility 或 action。

需要记录：

- 原始 RGB-only response；
- BEV-panel response；
- per-label 双分量与原因；
- manager `gain/u_gain/distance/utility`；
- 最终实际选中的 frontier；
- Qwen latency、parse failure、hash 与 deterministic replay 结果。

分析重点：

- BEV 是否能稳定识别明显 loop/revisit；
- `history_spatial_score` 是否只是复述 `u_gain`；
- 加入 BEV 后 semantic ranking 是否被地图 novelty 污染；
- 相同输入是否输出相同结构与排序。

### Phase 2：离线反事实与小规模 guarded fusion

先对 frozen trace 离线扫描少量融合权重，不重新跑 simulator。只有 shadow 证据显示增量信息后，才在固定诊断 episodes 上启用 bounded fusion。

融合必须在代码中显式完成。示意而非预设公式：

```text
semantic_component = calibrated(semantic_probability)
history_component  = bounded(history_spatial_score, redundancy_risk)
base_component     = existing FrontierManager gain and distance
final_utility      = deterministic_fusion(base, semantic, history)
```

第一轮约束：

- history component 只能小幅重排 utility 接近的候选；
- 不允许 VLM 创建、删除或复活 frontier；
- 不允许影响 target lock、approach 或 STOP；
- 任意服务/解析失败时严格回退当前 frozen baseline；
- 不改变 FrontierNet、manager lifecycle、planner 和 executor。

### Phase 3：完全相同 exact random-100，再决定 full

比较：

- SR / SPL / SR@1m / SPL@1m；
- paired flips；
- target-never-visible；
- no-frontier、timeout 和路径长度；
- VLM 调用次数与延迟；
- override/重排发生的位置及可审计原因。

不能只看 SR。H2 的历史已经表明，小样本局部 rescue 可能伴随完整集路径效率损失。若 exact-100 仅靠少数 flips 变好，但路径长度、重复访问或不稳定性恶化，不应直接进入 full。

## 6. 最小代码边界建议

未来实现时建议新增独立模块，而不是把 BEV 绘图塞进 `vlm/utils.py` 或 `FrontierManager`：

```text
zson3/frontier_scoring/
  snapshot.py          # immutable candidate snapshot
  panel_renderer.py    # RGB SoM + online BEV composite
  schema.py            # strict response schema/parser
  fusion.py            # deterministic, unit-testable fusion
  trace.py             # shadow artifacts and hashes
```

调用关系：

```text
Agent builds candidate snapshot
        |-- current RGB SoM
        |-- map/trajectory BEV
        v
PanelRenderer -> Qwen scorer -> strict structured evidence
                                      |
                             shadow trace first
                                      |
                         later deterministic fusion
                                      |
                              FrontierManager utility
```

`FrontierManager` 仍是 frontier lifecycle 和最终 utility 的所有者；VLM adapter 只产生证据，不持有 navigation state。

## 7. 明确不做的事

第一阶段不做：

- 复刻 H2 的 geometry override 状态机；
- 让 Qwen 直接选择最终 frontier；
- 引入新的 SearchBelief 或 episodic VLM memory；
- 给 persistent but invisible frontiers 补造 RGB evidence；
- 用 Habitat oracle top-down map；
- 修改 target detection、ApexFusion、approach/STOP；
- 修改 FrontierNet、Wavemap、planner/executor 或 multi-floor；
- 在当前 full HM3Dv1 运行中插入 shadow hook。

## 8. Go / No-Go 标准

进入受控在线融合前，至少满足：

1. RGB/BEV candidate identity 在固定 fixtures 中完全一致；
2. Qwen structured output 可解析且相同输入可重复；
3. 对人工标注的明显 loop/new-branch cases，history judgment 有稳定区分；
4. semantic score 不因地图 novelty 系统性漂移；
5. BEV signal 与 manager 现有 `u_gain` 不只是高度重复；
6. 服务失败时行为逐步等价于 frozen baseline；
7. 额外延迟处于可接受范围。

如果第 3 或第 5 条不成立，应停止：BEV 仍可保留为诊断可视化，但不进入决策闭环。

## 9. 最终判断

BEV 增强不是一个收益「已经被 H2 证明」的组件，但它是一个有代码依据、失败模式已知、可以低风险验证的研究方向。

最有价值的变化不是让 Qwen 获得更大控制权，而是给当前 OpenFrontier 的局部视觉语义打分补上可审计的空间历史上下文。只要严格保持：

```text
semantic evidence != exploration history != final utility
```

并采用 `renderer fixture -> shadow scoring -> offline analysis -> bounded fusion` 的顺序，就不会重复 H2 高频 override 的主要问题，也不会破坏当前已冻结基线。
