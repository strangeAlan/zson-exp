# OpenFrontier → ZSON3 迁移提示与边界

## 当前决定

ZSON3 采用 OpenFrontier-derived 路线：以 OpenFrontier 的现代探索算法为主干，迁移到 Habitat 0.3.3 和既有 HM3Dv1 评测协议。

这不是 clean-slate 重写，也不是把多个 reference 仓库同时拼接。第一阶段只回答：

> 能否在新 runtime 中保持 OpenFrontier 的 visual frontier、gain、VLM grounding、utility 和导航闭环行为？

## 已确认基础

- 上游基准：OpenFrontier commit `a3f8b83da6135a88247651534061df2ea05850f6`，已由本仓库 tag `openfrontier-upstream-a3f8b83` 固定。
- `zson3`：Python 3.10.20、Torch 2.5.1+cu121、Habitat-Sim 0.3.3、Habitat-Lab 0.3.3。
- Habitat-Lab 源码：`.local/src/habitat-lab`，tag `v0.3.3`。
- 现有 HM3Dv1 已完成 `Env -> reset -> RGB-D/GPS/Compass -> metrics -> close` 验收。
- HM3Dv1 数据：`data/datasets/objectnav/hm3d/v1`。
- HM3D scenes：`data/scene_datasets`。
- 数据只用软链接，不复制、不改原数据。

建议后续链接：

```text
data/datasets       -> <objectnav task dataset root>
data/scene_datasets -> <Habitat scene dataset root>
```

## 必须冻结的上游行为

首次单 episode 跑通前，不主动改变：

1. FrontierNet 输入、后处理、depth anchoring 和 HDBSCAN clustering。
2. `Frontier` 字段含义和 `FrontierManager` association/merge 语义。
3. history/map information-gain reduction。
4. set-of-marks 图像、OpenFrontier prompt 和 JSON parser。
5. semantic probability、gain、distance 的 utility 公式。
6. manager 更新顺序：

```text
detect -> add/filter/gain -> VLM score
       -> map update -> gain/filter -> merge/filter
       -> utility -> select -> plan/action
```

7. target approach、verification、STOP 的判定流程。

已知可疑行为也先冻结并记录，例如 depth-gradient 角度单位、frontier ID churn、active-goal merge 和直线距离 utility。没有固定 trace 前不修。

## 可以做的迁移修改

- Habitat 0.3.3 config、dataset、sensor、action、metrics 和 episode adapter。
- 将 simulator private state 访问收敛到 runtime adapter。
- 为现有服务编写 client adapter、health check、timeout 和结构化错误处理。
- 修复路径、配置文件名、依赖声明、资源定位和日志目录。
- 加入 trace，不改变决策输入与调用顺序。
- 为 Habitat-Baselines 0.3.3 适配 PointNav wrapper，但先保持同一 checkpoint、输入和 action 语义。

以上修改必须能够用固定 episode trace 单独验证。

## 当前禁止事项

- 不接 ApexNav、ASCENT 或 BeliefMapNav 算法。
- 不做 multi-floor、SearchBelief 或新 Frontier lifecycle。
- 不重新实现 visual frontier 或 information gain。
- 不同时替换 Wavemap、PointNav、VLM 策略和 target pipeline。
- 不把现有 VLFM、Qwen、detector 环境的包安装进 `zson3`。
- 不修改外部 OpenFrontier reference checkout 和既有 VLFM 仓库。
- 不先跑 full HM3D；顺序必须是 import smoke、组件 smoke、固定 1 episode、固定 10 episodes。

## 服务复用原则

先复用进程外服务；core 只依赖稳定 adapter，不依赖服务的 conda 环境。

ZSON3 源码建立后，必须在 `scripts/` 中提供项目自己的统一启动入口，不能要求开发者记住服务器上当前手工启动的进程。脚本只负责编排已有环境和模型，不复制模型、不向 `zson3` 安装服务端依赖。预期至少包含：

```text
scripts/launch_model_services.sh    # 统一检查/启动需要的服务
scripts/start_local_qwen.sh         # 调用现有 multi-agent-nav 环境
scripts/start_pointnav_legacy.sh    # 迁移期 PointNav 行为 oracle
scripts/check_model_services.sh     # health、端口和模型身份检查
scripts/stop_model_services.sh      # 只停止由本项目启动并记录 PID 的进程
```

启动脚本必须支持环境变量覆盖 Python、模型路径、GPU、端口和日志目录；必须先检查已有健康服务，不能重复加载模型；不得停止或接管由 VLFM/其它实验启动且没有 ZSON3 PID 记录的进程。

当前已发现：

| 能力 | 地址 | 状态与用途 |
|---|---|---|
| Qwen3-VL-8B | `127.0.0.1:18080/health`, `/generate` | health 已确认；优先作为 OpenFrontier frontier-scoring/verification 本地 VLM |
| GroundingDINO | `127.0.0.1:12181/gdino` | POST 服务存在；后续 target 侧候选，不替换 FrontierNet |
| BLIP2-ITM | `127.0.0.1:12182/blip2itm` | 可作为对照，不进入首个默认闭环 |
| MobileSAM | `127.0.0.1:12183/mobile_sam` | 只接受 bbox，不能等价替代 OpenFrontier text-prompt SAM3 |
| YOLOv7 | `127.0.0.1:12184/yolov7` | 可替换 detector 对照，不进入首个默认闭环 |

Qwen server 的真实脚本为：

```text
<external UniGoal checkout>/script/qwen_backend_server.py
```

历史 VLFM launcher 中旧的作者机默认路径已经过时，后续不要照搬。

OpenFrontier 原 client 与 Qwen `/generate` 协议不同；应写薄 adapter，保持 OpenFrontier prompt/parser，不修改 Qwen server。Qwen 替代 Gemini/InternVL 保持的是算法流程，不预设数值行为或论文性能等价。

SAM3 当前没有可复用的已确认环境或 checkpoint。优先单独验证上游 SAM3 server；若失败，再把 GroundingDINO + MobileSAM 定义为显式 target-pipeline variant，不能悄悄当成同一 baseline。

## PointNav 迁移策略

现有 `vlfm` 环境中的 PointNav 已知可运行，可作为迁移 oracle。OpenFrontier 的旧 wrapper 依赖 Habitat-Baselines 0.2.4；Habitat-Baselines 0.3.3 仍保留同一 PointNav policy 和大部分接口，但新版 `from_config()` 期待 multi-agent policy config，不能直接读取旧 checkpoint 内的 0.2.4 config。

因此分两步：

1. 迁移初期在 `vlfm` 环境中提供最小 PointNav action service，输入 `reset/depth/rho/theta`，输出离散 action。它只作为行为 oracle，不拥有 Habitat environment。
2. 在 `zson3` 中使用 Habitat-Baselines 0.3.3，按旧 checkpoint 的网络参数直接构造 policy、加载同一 state dict；用固定输入序列与 legacy service 逐 action 对齐。对齐后，本地 0.3.3 adapter 成为默认，legacy service 退出正常运行链。

不要把整个 Habitat 0.2.4 runtime 带入 ZSON3，也不要在未对齐前换 A* 或新的 PointNav checkpoint。

## 当前依赖状态

`zson3` 已安装并确认可 import：Habitat 0.3.3、Torch、NumPy 1.26.4、OpenCV 4.11、Scipy、NetworkX、YAML、Hydra/OmegaConf、scikit-learn、HDBSCAN、Open3D 0.19、segmentation-models-pytorch 0.3.3、Albumentations 2.0.6、pywavemap、Flask、requests 和 Habitat-Baselines 0.3.3。`pip check` 已通过；依赖安装后 HM3Dv1 reset 也已再次通过。

`google-genai==2.17.0` 已安装，OpenFrontier 的 `vlm.utils`、`nav.agent`、`nav.habitat_agent` 和 `nav.pointnav_agent` 已在 `zson3` 中完成 import smoke。迁移稳定后可再把 Gemini backend 改成真正的可选依赖。

PointNav 当前不是依赖缺失，而是 API/config 迁移问题：旧 checkpoint config 与 Habitat-Baselines 0.3.3 的 multi-agent `from_config()` 不兼容，原 loader 会退回不存在的 `nav/vlfm.yaml`。按上节的 oracle/parity 方案处理，不通过补装旧 Habitat 解决。

## 正式移植前最后门禁

开始复制或修改源码前完成：

1. `google-genai` 和四个 OpenFrontier agent import smoke（已完成）。
2. FrontierNet `rgbd_11cls.pth` 及 SHA256（已完成）。
3. `conda env export --no-builds` 和 `pip freeze --all`（已完成）。
4. 保留 OpenFrontier git 历史的迁移分支与 upstream tag（已完成）。
5. 冻结 HM3Dv1 protocol manifest：dataset/scene 路径、episode ID、success distance、step/turn、sensor shape/FOV/depth、max steps、STOP 和 metrics。
6. 固定一个 episode 和一组 RGB-D/pose fixture；后续每层迁移都基于同一输入比较 trace。
7. 记录当前服务的 endpoint、真实模型路径、启动环境和 GPU；正式 smoke 前安排 GPU 组合，不在两张卡接近满载时重复加载模型。

## 迁移门禁

按以下顺序推进，前一项失败时不进入后一项：

1. **Runtime gate**：Habitat 0.3.3 + HM3Dv1 reset（已通过）。
2. **Frontier gate**：固定 RGB-D/pose 得到可复现的 raw/anchored frontier trace。
3. **Mapping gate**：Wavemap integrate/query，记录耗时和 RSS。
4. **Semantic gate**：同一 SoM 图分别记录 upstream prompt、Qwen raw output 和 parsed probabilities。
5. **Executor gate**：PointNav checkpoint 在 0.3.3 下产生合法离散 action，并验证 recurrent reset。
6. **Episode gate**：固定 1 episode 完整闭环，无 crash/deadlock/stale state。
7. **Regression gate**：固定 10 episodes，记录 SR/SPL、终止原因、frontier/object/planner trace 和资源占用。

## 最小 trace contract

每次决策至少记录：

```text
episode_id / step / target
pose / action / collision-or-progress
raw frontier pixels
anchored XYZ / direction / base gain
frontier IDs before and after merge
VLM prompt hash / raw output / parsed probability
updated gain / utility / selected frontier
object candidate / verification / lock-in
planner goal / planner state / termination reason
```

## 给后续开发回合的提示

每次改动开始前先声明：

1. 这是 runtime migration、behavior-preserving refactor，还是 research change？
2. 修改影响上述哪一个 trace 字段？
3. 用哪个固定输入或 episode 证明行为没有意外变化？
4. 是否错误地把 upstream 可疑行为顺手“修好”了？

当前默认答案应是：先移植和测量，不创新。
