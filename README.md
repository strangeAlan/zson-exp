# ZSON3: Zero-Shot Object Navigation

ZSON3 is a research fork of
[OpenFrontier](https://github.com/cvg/OpenFrontier) for zero-shot ObjectNav in
Habitat. The current baseline keeps OpenFrontier exploration and adds:

- a Habitat 0.3.3 runtime for HM3Dv1 and HM3Dv2;
- local Qwen3-VL frontier scoring;
- YOLOv7/GroundingDINO + MobileSAM target perception;
- multi-frame ApexTarget geometry/fusion;
- resumable full-split evaluation with frozen manifests.

The imported OpenFrontier revision is
`a3f8b83da6135a88247651534061df2ea05850f6`. This is not the official
OpenFrontier repository.

## Current results

| Benchmark | Episodes | Primary SR | Primary SPL |
| --- | ---: | ---: | ---: |
| HM3Dv1 val | 2000 | 50.45% at 0.1 m | 0.2300 |
| HM3Dv2 val | 1000 | 65.20% at 1 m | 0.2790 |

Lightweight manifests and logs are under [results](results/README.md). Detailed
analysis is in [HM3Dv1 full audit](docs/HM3DV1_FULL_AUDIT.md) and
[HM3Dv2 full audit](docs/HM3DV2_FULL_AUDIT.md).

## Repository layout

```text
config/zson3/       experiment overlays
frontier/           frontier detection and lifecycle
mapping/            Wavemap integration
nav/                navigation and action execution
planner/            global and PointNav planning
zson3/runtime/      datasets, sensors and evaluator metrics
zson3/services/     local model service clients
zson3/target/       ApexTarget geometry and temporal fusion
scripts/            service and evaluation entry points
docs/               audits and implementation notes
```

Datasets, checkpoints, Conda environments, raw logs and episode traces are
local resources and are not stored in Git.

## 1. Clone

```bash
git clone --recursive git@github.com:strangeAlan/zson-exp.git
cd zson-exp
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Habitat environment

The validated navigation runtime uses Ubuntu 20.04, Python 3.10,
Habitat-Sim/Lab/Baselines 0.3.3, PyTorch 2.5.1+cu121 and NumPy 1.26.4.

```bash
mkdir -p .local/envs .local/src
conda create --prefix "$PWD/.local/envs/zson3" python=3.10 -y
conda activate "$PWD/.local/envs/zson3"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/facebookresearch/habitat-sim.git \
  .local/src/habitat-sim
git -C .local/src/habitat-sim checkout acbe6f4922e68145e401e55c30f9dfea460a3f24

git clone https://github.com/facebookresearch/habitat-lab.git \
  .local/src/habitat-lab
git -C .local/src/habitat-lab checkout 094d6be2f9d057e4781a68ae792132895fd4d3d0

python -m pip install -v .local/src/habitat-sim
python -m pip install -e .local/src/habitat-lab/habitat-lab
python -m pip install -e .local/src/habitat-lab/habitat-baselines
python -m pip install -r requirements_habitat.txt
```

`environment.zson3.yml` and `requirements.zson3.lock.txt` are provenance
snapshots. Habitat is intentionally installed from the pinned source revisions
above.

Download FrontierNet and PointNav weights:

```bash
bash scripts/download_weights.sh
```

Expected files:

```text
model_weights/rgbd_11cls.pth
model_weights/pointnav_weights.pth
```

## 3. HM3D data

Scene data requires acceptance of the
[HM3D terms](https://github.com/matterport/habitat-matterport-3dresearch).
ObjectNav episode archives are provided by Habitat-Lab:

- [HM3Dv1 ObjectNav episodes](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip)
- [HM3Dv2 ObjectNav episodes](https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip)
- [Habitat dataset instructions](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

After extraction, either use the default layout:

```text
data/datasets/objectnav/hm3d/v1/
data/external_datasets/objectnav/hm3d/v2/
data/scene_datasets/hm3d/          # HM3DSem v0.1 scenes/config
data/scene_datasets/hm3d_v0.2/     # HM3DSem v0.2 scenes/config
```

or point ZSON3 at existing downloads:

```bash
export ZSON3_HM3DV1_ROOT=/path/to/objectnav/hm3d/v1
export ZSON3_HM3DV2_ROOT=/path/to/objectnav/hm3d/v2
export ZSON3_SCENE_DATASETS_ROOT=/path/to/scene_datasets
```

The scene root must contain
`hm3d/hm3d_annotated_basis.scene_dataset_config.json` for V1 and
`hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json` for V2.

## 4. Local model services

Model servers use separate environments because their CUDA/Torch dependencies
differ from Habitat.

### Qwen3-VL

```bash
conda create --prefix "$PWD/.local/envs/qwen-vllm" python=3.12 -y
conda activate "$PWD/.local/envs/qwen-vllm"
python -m pip install vllm==0.15.1 huggingface_hub

mkdir -p .local/models
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir .local/models/qwen3-vl-8b
```

The launcher serves Qwen through an OpenAI-compatible endpoint on port 18080.
See [Qwen runtime notes](docs/QWEN_RUNTIME.md) for validated flags.

### ApexTarget detector services

ZSON3 reuses the lightweight model servers from
[VLFM](https://github.com/rai-opensource/vlfm):

```bash
git clone https://github.com/rai-opensource/vlfm.git .local/vlfm
git clone https://github.com/WongKinYiu/yolov7.git .local/vlfm/yolov7
git -C .local/vlfm/yolov7 checkout a207844b1ce82d204ab36d87d496728d3d2348e7
git clone https://github.com/IDEA-Research/GroundingDINO.git \
  .local/vlfm/GroundingDINO
git -C .local/vlfm/GroundingDINO checkout eeba084341aaa454ce13cb32fa7fd9282fc73a67

conda create --prefix "$PWD/.local/envs/vlfm" python=3.9 -y
conda activate "$PWD/.local/envs/vlfm"
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -e .local/vlfm/GroundingDINO salesforce-lavis==1.0.2
python -m pip install -e .local/vlfm
```

Place the detector checkpoints at:

```text
.local/vlfm/data/groundingdino_swint_ogc.pth
.local/vlfm/data/yolov7-e6e.pt
.local/vlfm/data/mobile_sam.pt
```

Download them from the official
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO),
[YOLOv7](https://github.com/WongKinYiu/yolov7), and
[MobileSAM](https://github.com/ChaoningZhang/MobileSAM) repositories.

Paths can instead be supplied through `ZSON3_PYTHON`,
`ZSON3_QWEN_VLLM_ENV`, `ZSON3_QWEN_MODEL_PATH`, `VLFM_ROOT` and
`VLFM_PYTHON`.

## 5. Run HM3Dv1 or HM3Dv2

The full wrappers start/check all required local services, freeze the episode
selection, write one JSON per episode, and resume only from a contiguous
completed prefix.

HM3Dv1 full validation (2000 episodes, primary success distance 0.1 m):

```bash
bash scripts/run_openfrontier_apextarget_full_hm3dv1.sh
```

HM3Dv2 full validation (1000 episodes, official 1 m SR/SPL is the primary
reported result):

```bash
bash scripts/run_openfrontier_apextarget_full_hm3dv2.sh
```

For HM3Dv2, read `sr_at_1m` and `spl_at_1m` as the primary metrics in
`summary.json`; the 0.1 m fields are retained only as a diagnostic reference.

Useful overrides:

```bash
export ZSON3_NAV_GPU=0
export ZSON3_QWEN_GPU=1
export ZSON3_EVAL_SEED=20260727
export ZSON3_RUN_ID=my_run_name
```

For a long remote run:

```bash
tmux new-session -d -s zson3-full \
  'cd /path/to/zson-exp && bash scripts/run_openfrontier_apextarget_full_hm3dv2.sh'
tmux attach -t zson3-full
```

Outputs are written to `results/<run-id>/`. `progress.log` contains the
per-episode heartbeat; `summary.json` and `summary.txt` contain aggregate
metrics.

An optional environment-only check is:

```bash
.local/envs/zson3/bin/python scripts/smoke_hm3dv1_runtime.py
```

## Reproducibility boundary

- Qwen, detector and segmentation inference runs in localhost services.
- Habitat semantic masks and top-down maps are evaluator diagnostics only and
  are not consumed by the navigation policy.
- The six evaluated target categories currently route through YOLOv7; the DINO
  adapter is available but is not part of the frozen full results.
- HM3Dv1 reports the frozen 0.1 m protocol. HM3Dv2 uses 1 m SR/SPL as its
  official primary result; 0.1 m is diagnostic only.
- Checkpoints, datasets, raw logs and episode traces must remain outside Git.

## Attribution

This repository retains substantial OpenFrontier code. Please cite the
original OpenFrontier work when using it:

```bibtex
@inproceedings{openfrontier2026,
  title     = {OpenFrontier: General Navigation with Visual-Language Grounded Frontiers},
  author    = {Padilla-Cerdio, Esteban and Sun, Boyang and Pollefeys, Marc and Blum, Hermann},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026}
}
```

The detector service runtime is derived from VLFM; please also follow its
license and citation requirements.
