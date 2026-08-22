# HM3Dv2：pre-Apex OpenFrontier 与 OF-ApexTarget full-1000 paired audit

状态：**最终只读配对审计（1000/1000）**<br>
审计日期：2026-08-22 UTC<br>
pre-Apex 代码基线：`a2d8d04551bad469261bd29d812594c096ded995` 加 HM3Dv2 evaluator/launcher 适配<br>
pre-Apex 结果：`results/openfrontier_base_sam3_full_hm3dv2_1000_seed20260727`<br>
ApexTarget 结果：`results/of_apex__full_hm3dv2_1000_seed20260727`

本文只读取冻结的 episode JSON，不修改 agent、target pipeline、planner、模型或 evaluator。HM3Dv2
以官方 **1 m Success/SPL** 为唯一主指标；0.1 m 只作附加诊断。

## 1. 完整性与配对身份

- 两侧各有 1000 个 episode JSON，index 均为连续的 `0..999`。
- pre-Apex 为 0 exceptions；ApexTarget 为 2 exceptions。
- 两个输出 manifest 的文件哈希不同，是因为 pre-Apex runner 把运行方式记为
  `explicit_manifest` 并写入 `source_manifest`；ApexTarget 原文件记为 `all`。
- 忽略上述运行元数据后，1000 个 `(index, scene basename, episode_id, target)` 的身份和顺序
  **逐项完全相同**，episode JSON 也与各自 manifest 1000/1000 对齐。因此这是严格 paired
  comparison，不是两个随机样本的比较。

冻结文件 SHA-256：

| File | SHA-256 |
| --- | --- |
| pre-Apex `manifest.json` | `c7ef8f4bcc42a54d29932c71ff6371e46bccb8e720ad9afaa9f89df2e6271374` |
| pre-Apex `summary.json` | `6ece633bd071e1748d04ad0079210299ac154fd454e103cf3a069b5164a940b7` |
| pre-Apex `progress.log` | `66fb5a74b108b4360d2452e6da05e48c238b6b45e8b4b8bf15fe814c617dd1a3` |
| ApexTarget source `manifest.json` | `606ebb74a19c970e92f319f3c2be2600000c6931d7ed038906d08250215c8b62` |
| ApexTarget `summary.json` | `88c5326e55b752ff14c1b48ec7e5b410be8a429fbafeb81364fe168b6501e0bb` |

## 2. 官方主结果与 paired flips

| Method | SR@1m | SPL@1m | Exceptions | Wall time |
| --- | ---: | ---: | ---: | ---: |
| **pre-Apex OF + SAM3 + Qwen** | **70.80%**（708/1000） | **0.3299** | 0 | 157,137 s |
| OF-ApexTarget | 65.20%（652/1000） | 0.2790 | 2 | 217,487 s |
| Paired difference | **+5.60 pp** | **+0.0509** | -2 | -60,350 s |

pre-Apex 的 0.1 m 附加参考也是 39.50% / 0.1821，高于 ApexTarget 的 37.60% / 0.1580；
因此当前结果不支持把 ApexTarget 解释成“牺牲 1 m、换取更近终点”的协议专用方案。

| Paired outcome | N |
| --- | ---: |
| 两者都成功 | 572 |
| **pre-Apex 独赢** | **136** |
| **ApexTarget 独赢** | **80** |
| 两者都失败 | 212 |

净 flips 为 `136 - 80 = +56`，恰好对应 +5.60 pp。216 个 discordant pairs 的双侧 exact
McNemar/binomial 检验为 `p=0.000169`；这不是合理的中途随机波动解释。

## 3. 按类别结果

| Category | N | pre SR@1m | Apex SR@1m | ΔSR | pre SPL | Apex SPL | pre-only | Apex-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bed | 165 | 80.61% | **81.82%** | -1.21 pp | **0.3711** | 0.3527 | 10 | 12 |
| chair | 195 | 82.05% | **85.64%** | -3.59 pp | **0.4154** | 0.3851 | 13 | 20 |
| plant | 152 | **59.87%** | 32.89% | **+26.97 pp** | **0.2708** | 0.0936 | 51 | 10 |
| sofa | 187 | 71.12% | **74.87%** | -3.74 pp | 0.3466 | **0.3763** | 13 | 20 |
| toilet | 166 | **66.27%** | 52.41% | **+13.86 pp** | **0.2730** | 0.2011 | 30 | 7 |
| tv_monitor | 135 | **60.00%** | 54.07% | **+5.93 pp** | **0.2695** | 0.2053 | 19 | 11 |

净 56 flips 的类别分解为：plant `+41`、toilet `+23`、TV `+8`，bed `-2`、chair
`-7`、sofa `-7`。因此总体提升不是六类平均的小增益，而是 pre-Apex 修复了 ApexTarget 在
plant/toilet/TV 上的大缺口，同时放弃了一部分 chair/sofa 的精度优势。

## 4. 共同成功 episode 的效率与时间线

阶段定义：

- `visible`：evaluator semantic target mask 在任一步达到 50 pixels；不供 policy 使用。
- `candidate`：可进入目标对象管线的 SAM/几何候选。pre-Apex 为 SAM3 composition 产生的
  `DetectedObject`；ApexTarget 为 target-label YOLO mask 通过深度/几何过滤。
- `takeover/reliable`：pre-Apex 为 Qwen verification accepted 后 object lock-in；ApexTarget 为
  fusion 首次给出 reliable target 并发生 `explore_to_target`。
- `approach`：接管后执行目标路径；`STOP` 由 trace 的 `termination_event` 定义。

下面的时间均为 navigation step。候选和接管只在双方都有该事件的 paired subset 上比较。

| Common-success metric | pre-Apex | ApexTarget | Paired difference |
| --- | ---: | ---: | ---: |
| N | 572 | 572 | — |
| SPL，mean / median | **0.4725 / 0.4669** | 0.4296 / 0.3911 | pre mean +0.0429 |
| Navigation steps，mean / median | **128.1 / 104** | 159.2 / 122 | Apex +31.1 mean |
| First visible step，mean / median | 68.1 / 39 | 68.0 / 38 | -0.1 mean；538/572 完全相同 |
| First usable candidate，mean / median | **89.6 / 65**（571） | 113.0 / 85（571） | Apex +23.4 mean |
| First takeover，mean / median | 123.5 / 99（571） | 126.5 / 98.5（570） | Apex +3.0 mean，median -12 |
| Candidate → takeover，mean / median | 33.8 / 21 | **13.4 / 1** | Apex fusion 在候选后更快 |
| Takeover → STOP，mean / median | **3.7 / 0**（570） | 20.2 / 14（534） | Apex approach 更长 |

ApexTarget 的第一张 target-label YOLO box 出现在 mean 98.7 / median 72 step；经过 mask/geometry
后，可用候选推迟到 mean 113.0 / median 85。

“Apex 接管普遍更晚”不是正确的总体解释：在 570 个双方都有接管事件的共同成功 pairs 中，
Apex 更早接管 419 个，pre-Apex 更早 150 个，1 个相同。真正的效率差异是：pre-Apex 先把
candidate 当作 object frontier 导航到验证视点，Qwen 接管时通常已经足够近；ApexTarget 较早
锁定 reliable cluster 后还要执行显式 approach，接管到 STOP 的中位数为 14 steps，而 pre-Apex
为 0。

官方 success 不要求 `reason=object_found`。共同成功中，pre-Apex 570/572 有显式 STOP；
ApexTarget 只有 534/572 有显式 STOP，另外 38 个以 stuck/max-steps 结束但最终位置仍在官方
1 m 成功区。这说明 ApexTarget 的终止闭环更脆弱，即使其中一部分没有损失 SR。

## 5. Discordant episodes 停在哪个阶段

### 5.1 pre-Apex 独赢：ApexTarget 的 first unresolved stage

Apex 的 `YOLO miss` 严格定义为：episode 曾有 `>=50 px` GT-visible frame，但所有这些 frame
都没有 target-label YOLO box；它不排除其他 frame 上存在 GT-unmatched detection。

| ApexTarget stopping stage | N | pre-only 占比 |
| --- | ---: | ---: |
| visible → **无同帧 target YOLO box** | **60** | **44.12%** |
| candidate geometry → **始终未 reliable/takeover** | **28** | **20.59%** |
| reliable/approach → **stuck 或 max-steps，无 STOP** | **17** | **12.50%** |
| target 从未达到 50 px visible | 12 | 8.82% |
| YOLO box → SAM/geometry rejection | 10 | 7.35% |
| approach → STOP，但官方失败 | 7 | 5.15% |
| runtime exception | 2 | 1.47% |
| **Total** | **136** | **100%** |

按类别看，60 个 visible-frame YOLO misses 中 41 个是 plant；17 个 downstream stuck/max-steps
中 9 个是 toilet。Apex 的两次 runtime exception 也都落在 pre-Apex 成功 flips 中。

28 个“有可用几何候选但从未 reliable”的 episode 中，最接近可靠的 target cluster 28/28
都没有达到 `positive_observation_count >= 2`；其中 1 个还同时被 confusable label 压过。
confidence 0.65 和 positive volume 8 并不是这 28 个的最终阻塞条件。它们的第一候选中位 step
为 218，结束前仍有中位 256 steps，说明主要问题是未获得/未关联第二次正观测，而不是单纯没有
剩余时间。

在已经通过 visible、YOLO、geometry、reliable 的 24 个 pre-only flips 中：

- 17 个没有 STOP：9 max-steps、8 robot-stuck；
- 7 个快速 STOP 但官方失败，全部记录为 false positive；
- 17 个无 STOP 的首次接管中位 step 为 246，比 pre-Apex 接管晚 84 steps；10/17 甚至晚于
  pre-Apex 已经成功 STOP 的时刻，但其接管后仍有中位 200 steps。因此这里同时存在失败尾部的
  接管延迟与 approach/recovery 失效，不能只归因于预算耗尽。

作为全集一致性检查，ApexTarget 的 348 个官方失败漏斗为：91 never-visible、158 visible-frame
YOLO miss、15 geometry rejection、36 never reliable、23 reliable 后 stuck/max-steps、23 STOP
false failure、2 exceptions，与独立 ApexTarget V2 审计逐项一致。

### 5.2 ApexTarget 独赢：pre-Apex 的 first unresolved stage

pre-Apex 的 SAM3 composition trace 没有保存 box-only 事件，因此这里不能把“无几何候选”继续
拆成 YOLO/SAM/geometry 三项。

| pre-Apex stopping stage | N | Apex-only 占比 |
| --- | ---: | ---: |
| approach → STOP，但官方失败 | **34** | **42.50%** |
| target 从未达到 50 px visible | 19 | 23.75% |
| candidate → Qwen rejected / 无 takeover | 16 | 20.00% |
| takeover/approach → stuck 或 max-steps，无 STOP | 7 | 8.75% |
| visible → 无 SAM3 geometry candidate | 4 | 5.00% |
| **Total** | **80** | **100%** |

34 个 pre-Apex 错误 STOP 中 chair 11、sofa 13；这正是 ApexTarget 的主要正面价值：它在
chair/sofa 上减少了 legacy Qwen verification 接受错误候选后的快速错误终止。但这一收益小于
plant/toilet/TV 的召回与闭环损失。

## 6. ApexTarget 损失的主因排序

对决定方法净差异的 136 个 pre-only flips，证据支持如下排序：

1. **YOLO target recall / adapter path：首要原因。** 60/136（44.1%），其中 plant 占 41。
   这是“同一 GT-visible step 上没有 target-label box”的 trace 结论，不应扩张成 YOLO 权重本身
   一定有问题；逐帧 Apex adapter 与 legacy SAM3 composition 的输入/调用路径也不同。
2. **可靠性门槛与候选形成：第二原因。** 28 个被 two-positive-observation gate 阻断，另有 10 个
   同帧 geometry rejection，合计 38/136（27.9%）。主要不是 confidence/volume threshold。
3. **stuck/max-steps：明确但较小的 downstream 原因。** 17/136（12.5%）已经 reliable，却没有
   完成 STOP。
4. **接管延迟：失败尾部的重要放大器，不是全局主因。** 共同成功时 Apex 通常更早接管；但上述
   17 个 downstream 无 STOP flips 中有 10 个直到 pre-Apex 已成功后才接管。
5. **approach/STOP 错误：7/136（5.1%）直接造成错误 STOP。** 另外共同成功集上 Apex 的显式
   approach 中位多 14 steps，是 SPL 和总步数退化的重要来源。

另有 12 个 never-visible 属于两条轨迹在较早决策后分叉形成的探索/覆盖差异，不能硬归因给某次
YOLO miss；2 个是已知 episode-over runtime exception。

## 7. 明确定位结论

**ApexTarget 不应作为 HM3Dv2 默认模块，也没有证据支持把它定位成当前的协议相关分支；它只应
保留为实验模块。**

理由是：同一 manifest 上，pre-Apex 在官方 1 m 指标上同时提高 SR（+5.60 pp）和 SPL
（+0.0509），在 0.1 m 附加指标上也仍然更好，同时运行时间少约 27.7%。ApexTarget 的价值集中在
chair/sofa 的错误 STOP 抑制，但其 plant/toilet/TV 召回、两观测可靠性门槛和 downstream
approach 成本造成更大的净损失。

因此主开发版本应回到 pre-Apex OpenFrontier + SAM3 + Qwen；ApexTarget 代码和冻结结果保留在
独立 experimental branch 供研究追溯。本轮不据此设计 hybrid、不调阈值，也不修改 target
pipeline。
