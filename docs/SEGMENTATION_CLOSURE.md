# Target segmentation closure

## Confirmed state

- OpenFrontier pins `third_party/sam3` at
  `5dd401d1c5c1d5c3eedff06d41b77af824517619`; the submodule source is now
  initialized at that exact commit.
- No local SAM3 conda environment, installed package, checkpoint, or Hugging
  Face cache was found before initialization.
- The pinned SAM3 implementation uses text prompts and returns masks, boxes and
  scores. This is the contract consumed by OpenFrontier.
- The existing MobileSAM service at port 12183 consumes a detector-provided
  bounding box. It is not a faithful replacement for text-prompt SAM3.
- OpenFrontier hard-codes SAM3 to port 12184. The existing VLFM YOLOv7 service
  already owns port 12184. ZSON3 will use port **12186** for SAM3; the server and
  client adapter must be changed together after the environment exists.
- `build_sam3_image_model()` downloads `facebook/sam3/config.json` and
  `facebook/sam3/sam3.pt` when no explicit checkpoint is provided. The model
  repository is gated and requires accepted access plus Hugging Face login.

## Environment boundary

SAM3 remains an out-of-process service. Do not install its Torch or model
dependencies into `zson3`.

The pinned SAM3 README recommends Python 3.12 and currently specifies
Torch 2.10 with CUDA 12.8. This server reports NVIDIA driver 570.133.20 and
CUDA 12.8, so the advertised wheel is compatible at the driver level.

Minimal environment setup:

```bash
conda create -n zson3-sam3 python=3.12 -y
conda activate zson3-sam3

python -m pip install --upgrade "setuptools<82" wheel
python -m pip install torch==2.10.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e /home/hsy/zson-exp/third_party/sam3
python -m pip install flask

hf auth login
```

Do not install the optional notebook, training, FlashAttention or cc_torch
dependencies for the first service smoke.

After login, do not manually start the upstream server on port 12184. The next
migration batch will:

1. make the SAM3 server port and checkpoint path explicit;
2. default ZSON3 to port 12186;
3. add bounded timeouts and proxy-independent health/inference clients;
4. run one text-prompt segmentation on a fixed HM3D RGB frame;
5. record mask/box/score shapes and resource usage;
6. add it to project-owned start/check/stop scripts.

## GPU scheduling

At audit time GPU0 had only about 2.8 GiB free because of the VLFM detector
services. GPU1 had about 23.5 GiB free while Qwen occupied about 17 GiB. SAM3
should first be attempted on GPU1, but fit is not assumed until measured. Do not
duplicate Qwen or stop existing services merely to make SAM3 fit.

## Implemented runtime closure

- SAM3 server and client use the non-conflicting port 12186.
- The server exposes an identity-bearing `/health` endpoint.
- Client requests bypass machine HTTP proxies and use bounded timeouts.
- A checkpoint can be supplied explicitly through `ZSON3_SAM3_CHECKPOINT`;
  otherwise the pinned builder requests `facebook/sam3/sam3.pt`.
- Start/check/stop scripts preserve ownership through project PID records.
- The OpenFrontier mask/box parsing path passed a contract-level mock smoke.

While official `facebook/sam3` access is pending, the compatible `sam3.pt` was
downloaded from the user-provided `1038lab/sam3` mirror to
`/home/hsy/models/sam3/1038lab/sam3.pt`. Its SHA256 is
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
Several independently hosted Hugging Face mirrors advertise the same SHA256,
but byte identity with the inaccessible official repository has not been
directly verified. The mirror states that Meta's original license still
applies.

The pinned builder loaded the flat `detector.*`/`tracker.*` checkpoint without
conversion. A real service on GPU1 used approximately 5.85 GiB and passed two
requests on the fixed HM3Dv1 turn-6 RGB frame:

- `chair`: valid empty result, 2.37 seconds;
- `floor`: one mask `[1, 1, 480, 640]`, one box `[1, 4]`, score `0.91015625`,
  0.758 seconds.

The service is stopped after the gate. `scripts/start_sam3.sh` now discovers
this external checkpoint by default and never commits the model into Git.

## Gate status

`SEGMENTATION_COMPONENT_GATE_PASSED_WITH_MIRROR_CHECKPOINT`

Official access remains desirable for direct hash comparison, but no longer
blocks the executable integration gate. GroundingDINO + MobileSAM may later be
evaluated as a named pipeline variant, never as silent SAM3 equivalence.
