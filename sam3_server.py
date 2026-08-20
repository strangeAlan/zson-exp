import argparse
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path(__file__).resolve().parent / "third_party" / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from utils.server_wrapper import ServerMixin, host_agent
from utils.sam3_transport import decode_image, encode_masks

SAM3_BPE_PATH = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ZSON3_SAM3_PORT", "12186"))
    )
    parser.add_argument(
        "--checkpoint", default=os.environ.get("ZSON3_SAM3_CHECKPOINT")
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    class Sam3Server(ServerMixin):

        def __init__(self) -> None:
            super().__init__()

            # Load the model
            self.model = build_sam3_image_model(
                str(SAM3_BPE_PATH),
                checkpoint_path=args.checkpoint,
                load_from_HF=args.checkpoint is None,
            )
            self.processor = Sam3Processor(self.model)
            self.use_cuda_autocast = torch.cuda.is_available()

        def health_payload(self) -> dict:
            return {
                "transport": "packed-v1",
                "image_encoding": "raw-uint8-base64-v1",
                "mask_encoding": "packbits-base64-v1",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }

        def process_payload(self, payload: dict) -> dict:
            try:
                request_started = time.perf_counter()
                image = decode_image(payload)
                image = Image.fromarray(image)
                prompt = payload["prompt"]
                decode_finished = time.perf_counter()

                inference_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if self.use_cuda_autocast
                    else nullcontext()
                )
                with torch.inference_mode():
                    with inference_context:
                        inference_state = self.processor.set_image(image)
                        # Prompt the model with text
                        output = self.processor.set_text_prompt(
                            state=inference_state, prompt=prompt
                        )
                inference_finished = time.perf_counter()

                # Get the masks, bounding boxes, and scores
                masks, boxes, scores = (
                    output["masks"],
                    output["boxes"],
                    output["scores"],
                )

                # NumPy cannot serialize bfloat16 tensors produced under autocast.
                masks = masks.detach().to(dtype=torch.bool, device="cpu")
                boxes = boxes.detach().to(dtype=torch.float32, device="cpu")
                scores = scores.detach().to(dtype=torch.float32, device="cpu")

                response = {
                    "result": "success",
                    "boxes": boxes.numpy().tolist(),
                    "scores": scores.numpy().tolist(),
                    "timings": {
                        "decode_seconds": decode_finished - request_started,
                        "inference_seconds": inference_finished - decode_finished,
                    },
                }
                if payload.get("response_format") == "packed-v1":
                    response.update(encode_masks(masks.numpy()))
                else:
                    response["masks"] = masks.numpy().tolist()
                return response
            except Exception as e:
                return {"result": "error", "message": str(e)}

    server = Sam3Server()
    print("Sam3 loaded!")
    host_agent(server, name="sam3", port=args.port)
