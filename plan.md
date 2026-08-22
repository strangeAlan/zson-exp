# ZSON3 顶层目标规划

## 1. 总体目标

ZSON3 的目标不是从零重新设计一套导航系统，也不是把 OpenFrontier、ApexNav、ASCENT 等多个仓库机械拼接到一起。
(参考仓库在：/home/hsy/references，虚拟环境conda的zson3)
我们的目标是：

> **以 OpenFrontier 为现代单楼层 ZSON 的算法主干，将其迁移到现代、可维护、可完整评测的运行环境中；在保持其先进几何与语义探索能力的基础上，逐步补强目标识别、记忆和执行能力，并最终以 Multi-Floor Object Navigation 为优先研究方向寻找新的论文问题。**

整个项目应始终区分：

```text
已有成熟能力 → 尽量继承和保持
工程债务     → 有控制地迁移和清理
真实能力缺口 → 才进入研究和创新
```

---

## 2. 第一阶段：建立现代 OpenFrontier 基座

第一目标不是立即创新，而是获得一个可信的现代单楼层 baseline。

核心原则：

> **保留 OpenFrontier 的 exploration algorithm，迁移 runtime，而不是重新实现 exploration。**

重点继承：

```text
Visual Frontier Generation
        ↓
3D Frontier Grounding
        ↓
Information Gain
        ↓
Frontier Management
        ↓
VLM Semantic Grounding
        ↓
Geometry + Semantics Frontier Selection
        ↓
Navigation
```

这一阶段主要完成：

* Habitat-Sim / Lab 0.3.3 环境；
* HM3Dv1 evaluation protocol；
* OpenFrontier exploration 主链迁移；
* FrontierNet / Wavemap / PointNav / local VLM 运行闭包；
* 固定 episode trace；
* 单楼层完整 baseline。

第一阶段原则上不改变：

* visual frontier 算法；
* information gain 更新；
* VLM frontier scoring；
* frontier utility；
  -主要 manager 更新顺序。

先得到：

```text
OpenFrontier-derived ZSON3-Base
```

再谈结构性改进。

---

## 3. 第二阶段：补强明显短板，而不是重写整个系统

基座稳定后，通过完整 HM3Dv1 failure analysis 判断真正的性能瓶颈。

当前已知最值得优先观察的候选包括：

### Target / Object side

OpenFrontier 已有 target detection、association、VLM verification、approach 和 STOP，但长期目标记忆和多帧证据融合较弱。

如果实际 failure analysis 证明这是主要问题，则重点参考 ApexNav：

```text
Object association
Multi-view fusion
Positive evidence
Confidence management
Target approach / verification
Recovery
```

不是接入整个 ApexNav，而是针对具体缺陷吸收其成熟机制。

### Exploration side

OpenFrontier 的 visual frontier + VLM exploration 是当前默认主干。

只有实验发现明确问题后，才研究：

* semantic evidence 的长期管理；
* frontier identity / lifecycle；
* path-cost-aware utility；
* long-horizon planning；
* SearchBelief；
* uncertainty。

这些不是预先规定必须实现的组件。

---

## 4. 第三阶段：重点研究 Multi-Floor ObjectNav

当现代单楼层基座稳定后，Multi-Floor Object Navigation 作为当前优先研究方向。

不是简单复刻 ASCENT，而是利用更强的 OpenFrontier-style exploration substrate 重新研究：

```text
当前楼层还值不值得继续搜？
        ↓
是否存在有效 vertical transition？
        ↓
是否应该付出代价切换楼层？
        ↓
切换之后如何继续利用历史探索状态？
        ↓
如何在多个楼层之间分配长期搜索预算？
```

ASCENT 主要作为以下能力的参考：

* stair / transition perception；
* floor transition behavior；
* floor switching；
* multi-floor state management；
* failure handling。

但其具体实现不被预设为最终方案。

Multi-floor 阶段真正值得研究的问题包括：

* vertical transition detection；
* floor representation；
* when-to-switch-floor；
* cross-floor frontier / option selection；
* long-horizon exploration planning；
* multi-floor semantic memory；
* search completeness / uncertainty。

最终论文贡献应尽可能来自这些尚未被现有方法充分解决的问题，而不是来自重新实现已有单楼层组件。

---

## 5. 研究路线允许改变

Multi-floor 是当前最有希望的主线，但不是强制结局。

每完成一个稳定 baseline，都进行 failure analysis。

如果发现更明确、更有价值的问题，例如：

* frontier长期语义记忆；
* target verification；
* exploration uncertainty；
* long-horizon search；
* VLM decision instability；

可以转向该问题。

因此项目路线是：

```text
Modern Base
    ↓
Failure Analysis
    ↓
Identify Real Bottleneck
    ↓
Research
```

而不是：

```text
预先列出模块 A+B+C+D
    ↓
全部实现
    ↓
希望最后形成论文
```

---

## 6. 开发过程中始终遵守的边界

任何较大的改动都先问四个问题：

1. **这是迁移工作、能力补全，还是研究创新？**
2. **现有强方法是否已经有成熟实现可以参考？**
3. **这个修改是否会破坏 OpenFrontier 当前已验证的探索行为？**
4. **它是否服务于已经观察到的 failure，或者明确的研究 hypothesis？**

特别避免两种极端：

### 极端一：被 OpenFrontier 代码结构牵着走

不能因为 upstream 当前把多个职责写在一个 Manager 中，就默认 ZSON3 永远采用相同结构。

### 极端二：为了“架构干净”重写算法

不能因为某段代码结构不漂亮，就重新实现 visual frontier、gain、VLM scoring 等核心能力，最终得到一个结构漂亮但性能倒退的系统。

原则是：

> **先继承能力，再逐渐获得代码控制权。**

---

# 当前最近目标

当前只推进：

```text
Habitat 0.3.3 Environment
        ↓
OpenFrontier Runtime Migration
        ↓
HM3Dv1 Single-Floor Baseline
        ↓
Full Evaluation + Failure Analysis
```

在这一节点之前：

* 不急着做 multi-floor；
* 不急着接 ApexNav；
* 不急着做 SearchBelief；
* 不大规模重新设计 FrontierManager；
* 不同时替换 exploration、mapping 和 executor。

先得到一套真正可信、足够现代的单楼层 ZSON3-Base，再决定下一刀应该切在哪里。

## 7. 当前冻结点与下一评测门槛（2026-08-10）

已冻结的迁移基线：

```text
tag: openfrontier-base-sr55-random100-seed20260727
HM3Dv1 random-100: SR 55%, SPL 0.2532
```

基线后的允许改动暂限于：

* SAM3 无损传输和推理运行时加速；
* evaluator-only target visibility / evidence diagnostics；
* 固定 episode manifest 与可恢复评测运行器。

下一步先运行与 VLFM T1 完全相同的 100 个 `(scene, episode_id)`；
确认结果与失败结构后，才运行完整 2000-episode HM3Dv1 val。两者完成前不接入
ApexNav target fusion，不开始 multi-floor，也不调整 OpenFrontier 算法参数。
