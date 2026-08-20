#!/usr/bin/env python3
"""Local Qwen3-VL server owned by the ZSON3 runtime.

The HTTP contract matches the previously reused UniGoal server, while model
placement, attention backend, and timing telemetry are explicit.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def _strip_data_url(value: str) -> str:
    return value.split(",", 1)[1] if "," in value else value


def _strip_thinking(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


class QwenRunner:
    def __init__(
        self,
        *,
        model_path: str,
        device_map: str,
        dtype: str,
        attention_implementation: str,
        max_new_tokens: int,
        temperature: float,
        log_dir: Path,
    ) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype_map[dtype],
            device_map=device_map,
            attn_implementation=attention_implementation,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_path = model_path
        self.device_map = getattr(self.model, "hf_device_map", device_map)
        self.attention_implementation = getattr(
            self.model.config,
            "_attn_implementation",
            attention_implementation,
        )

    @staticmethod
    def _decode_images(images: list[str] | None) -> list[Image.Image]:
        decoded = []
        for item in images or []:
            raw = base64.b64decode(_strip_data_url(item))
            decoded.append(Image.open(BytesIO(raw)).convert("RGB"))
        return decoded

    def health(self) -> dict:
        return {
            "ok": True,
            "model_path": self.model_path,
            "device_map": self.device_map,
            "attention_implementation": self.attention_implementation,
        }

    def generate(
        self,
        *,
        prompt: str,
        images: list[str] | None,
        max_new_tokens: int | None,
    ) -> tuple[str, dict]:
        started = time.perf_counter()
        pil_images = self._decode_images(images)
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": image} for image in pil_images)
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        prepared = time.perf_counter()

        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature

        with self.lock, torch.inference_mode():
            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            generated = self.model.generate(**inputs, **generation_kwargs)
            torch.cuda.synchronize()
            generation_finished = time.perf_counter()

        input_tokens = int(inputs.input_ids.shape[-1])
        output_ids = generated[:, input_tokens:]
        output_tokens = int(output_ids.shape[-1])
        output = _strip_thinking(
            self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )
        finished = time.perf_counter()
        timings = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_images": len(pil_images),
            "prepare_seconds": prepared - started,
            "generate_seconds": generation_finished - generation_started,
            "decode_seconds": finished - generation_finished,
            "total_seconds": finished - started,
        }
        self._log(prompt=prompt, output=output, timings=timings)
        return output, timings

    def _log(self, *, prompt: str, output: str, timings: dict) -> None:
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "attention_implementation": self.attention_implementation,
            "prompt_preview": prompt[:500],
            "output": output,
            "timings": timings,
        }
        with (self.log_dir / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class QwenHTTPServer(ThreadingHTTPServer):
    runner: QwenRunner


class Handler(BaseHTTPRequestHandler):
    server: QwenHTTPServer

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, self.server.runner.health())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text, timings = self.server.runner.generate(
                prompt=payload.get("prompt", ""),
                images=payload.get("images", []),
                max_new_tokens=payload.get("max_new_tokens"),
            )
            self._send_json(200, {"text": text, "timings": timings})
        except Exception as error:
            self._send_json(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--attention-implementation",
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/qwen"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = QwenRunner(
        model_path=args.model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        log_dir=args.log_dir,
    )
    server = QwenHTTPServer((args.host, args.port), Handler)
    server.runner = runner
    print(f"Qwen backend ready: http://{args.host}:{args.port} {runner.health()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
