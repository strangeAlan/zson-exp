#!/usr/bin/env python3
"""Run a frozen HM3D ObjectNav episode set through OpenFrontier.

The historical filename is retained for command compatibility. The runner now
supports HM3Dv1/HM3Dv2, random samples, explicit manifests, and complete splits.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import habitat
import torch
from habitat.config.read_write import read_write
from habitat.datasets import make_dataset
from omegaconf import OmegaConf

from nav.pointnav_agent import PointnavAgent
from utils.frontier_utils import read_config_yaml
from zson3.runtime.datasets import build_objectnav_config
from zson3.runtime.metrics import success_spl_at_distance
from zson3.services.qwen import QwenClient, QwenServiceError
from zson3.services.sam3 import Sam3Client, Sam3ServiceError
from zson3.services.apex_target import (
    ApexTargetServiceClient,
    ApexTargetServiceError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("hm3dv1", "hm3dv2"),
        default="hm3dv1",
        help="HM3D ObjectNav dataset version (default preserves historical HM3Dv1 runs)",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        help="Exact scene/episode selection to replay instead of resampling",
    )
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Evaluate every episode in the selected HM3D validation split",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-time", type=int, default=3600)
    parser.add_argument(
        "--config", default="config/zson3/navigation_hm3dv1_qwen.yaml"
    )
    parser.add_argument(
        "--unet-weight",
        type=Path,
        default=Path("model_weights/rgbd_11cls.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/openfrontier_random100_seed20260727"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--log-level", type=int, default=20)
    return parser.parse_args()


def episode_identity(episode) -> dict:
    return {
        "scene_id": episode.scene_id,
        "episode_id": str(episode.episode_id),
        "target": episode.object_category,
    }


def scene_name(scene_id: str) -> str:
    return Path(scene_id).stem


def normalized_scene_name(scene_id: str) -> str:
    name = Path(scene_id).stem
    return name.removesuffix(".basis")


def normalized_target_name(target: str) -> str:
    aliases = {
        "couch": "sofa",
        "potted plant": "plant",
        "tv": "tv_monitor",
    }
    return aliases.get(target, target)


def configure_sequential_iteration(config) -> None:
    with read_write(config):
        iterator_options = config.habitat.environment.iterator_options
        iterator_options.cycle = False
        iterator_options.shuffle = False
        iterator_options.group_by_scene = False
        iterator_options.max_scene_repeat_episodes = -1
        iterator_options.max_scene_repeat_steps = -1
        iterator_options.num_episode_sample = -1


def select_random_episodes(config, *, count: int, seed: int):
    """Apply Habitat's EpisodeIterator sampling semantics without a simulator."""

    dataset_config = config.habitat.dataset
    dataset = make_dataset(dataset_config.type, config=dataset_config)
    options = OmegaConf.to_container(
        config.habitat.environment.iterator_options, resolve=True
    )
    options.update(
        {
            "cycle": False,
            "shuffle": True,
            "group_by_scene": True,
            "max_scene_repeat_episodes": 1,
            "max_scene_repeat_steps": -1,
            "num_episode_sample": count,
            "seed": seed,
        }
    )
    iterator = dataset.get_episode_iterator(**options)
    selected = [next(iterator) for _ in range(count)]
    identities = [episode_identity(episode) for episode in selected]
    if len({(item["scene_id"], item["episode_id"]) for item in identities}) != count:
        raise RuntimeError("Random episode selection contains duplicate identities")

    # The random selection is now frozen. Feed it to Env sequentially so resume
    # behavior cannot resample or reshuffle it.
    dataset.episodes = selected
    configure_sequential_iteration(config)
    return dataset, identities


def select_manifest_episodes(config, manifest_path: Path, dataset_label: str):
    """Resolve a frozen scene/episode manifest against the selected HM3D data."""

    payload = json.loads(manifest_path.read_text())
    requested = payload.get("selection")
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"Manifest has no non-empty selection: {manifest_path}")

    dataset_config = config.habitat.dataset
    dataset = make_dataset(dataset_config.type, config=dataset_config)
    episode_by_id = {
        (normalized_scene_name(episode.scene_id), str(episode.episode_id)): episode
        for episode in dataset.episodes
    }
    if len(episode_by_id) != len(dataset.episodes):
        raise RuntimeError(
            f"{dataset_label} contains duplicate (scene, episode_id) identities"
        )

    selected = []
    missing = []
    for item in requested:
        key = (
            normalized_scene_name(str(item["scene_id"])),
            str(item["episode_id"]),
        )
        episode = episode_by_id.get(key)
        if episode is None:
            missing.append(key)
        else:
            source_target = item.get("source_target")
            if source_target is not None and normalized_target_name(
                str(source_target)
            ) != normalized_target_name(str(episode.object_category)):
                raise RuntimeError(
                    f"Manifest target drift for {key}: source={source_target}, "
                    f"local={episode.object_category}"
                )
            selected.append(episode)
    if missing:
        raise RuntimeError(
            f"Manifest references {len(missing)} unavailable episodes: {missing[:5]}"
        )
    identities = [episode_identity(episode) for episode in selected]
    if len(set((x["scene_id"], x["episode_id"]) for x in identities)) != len(
        identities
    ):
        raise RuntimeError("Explicit episode manifest contains duplicate identities")

    dataset.episodes = selected
    configure_sequential_iteration(config)
    return dataset, identities, payload


def select_all_episodes(config):
    dataset_config = config.habitat.dataset
    dataset = make_dataset(dataset_config.type, config=dataset_config)
    selected = list(dataset.episodes)
    dataset.episodes = selected
    configure_sequential_iteration(config)
    return dataset, [episode_identity(episode) for episode in selected]


def read_completed(episodes_dir: Path) -> dict[int, dict]:
    completed = {}
    for path in sorted(episodes_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        completed[int(payload["index"])] = payload
    if completed and sorted(completed) != list(range(max(completed) + 1)):
        raise RuntimeError("Resume requires a contiguous completed episode prefix")
    return completed


def aggregate(results: list[dict]) -> dict:
    count = len(results)
    successes = sum(float(item["metrics"].get("success", 0.0)) for item in results)
    spl_sum = sum(float(item["metrics"].get("spl", 0.0)) for item in results)
    successes_at_1m = sum(
        float(item["metrics"].get("success_at_1m", 0.0)) for item in results
    )
    spl_at_1m_sum = sum(
        float(item["metrics"].get("spl_at_1m", 0.0)) for item in results
    )
    return {
        "episodes": count,
        "successes": int(successes),
        "sr": successes / count if count else 0.0,
        "spl": spl_sum / count if count else 0.0,
        "successes_at_1m": int(successes_at_1m),
        "sr_at_1m": successes_at_1m / count if count else 0.0,
        "spl_at_1m": spl_at_1m_sum / count if count else 0.0,
        "elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in results),
        "exceptions": sum(item["status"] != "ok" for item in results),
    }


def progress_line(result: dict, summary: dict, total: int) -> str:
    metrics = result["metrics"]
    return (
        f"[ZSON3 EVAL] {result['index'] + 1}/{total} "
        f"scene={result['scene']} episode={result['episode_id']} "
        f"target={result['target']} success={float(metrics.get('success', 0)):.0f} "
        f"spl={float(metrics.get('spl', 0)):.4f} reason={result['reason']} "
        f"steps={result['navigation_steps']} runtime={result['elapsed_seconds']:.1f}s "
        f"SR={summary['sr'] * 100:.2f}% ({summary['successes']}/{summary['episodes']}) "
        f"mean_SPL={summary['spl']:.4f} "
        f"SR@1m={summary['sr_at_1m'] * 100:.2f}% "
        f"SPL@1m={summary['spl_at_1m']:.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.all_episodes and args.episode_manifest is not None:
        raise ValueError("--all-episodes and --episode-manifest are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = args.output_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)
    failures_dir = args.output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)

    dataset_label = "HM3Dv1" if args.dataset == "hm3dv1" else "HM3Dv2"
    dataset_log_label = f"{dataset_label}-val"
    config = build_objectnav_config(
        args.dataset, seed=args.seed, top_down_map=False
    )
    source_manifest = None
    if args.all_episodes:
        dataset, manifest = select_all_episodes(config)
        selection_mode = "all"
    elif args.episode_manifest is not None:
        dataset, manifest, source_manifest = select_manifest_episodes(
            config, args.episode_manifest, dataset_label
        )
        selection_mode = "explicit_manifest"
    else:
        dataset, manifest = select_random_episodes(
            config, count=args.episodes, seed=args.seed
        )
        selection_mode = "habitat_random"
    total_episodes = len(manifest)
    manifest_payload = {
        "dataset": f"{dataset_label} val",
        "seed": args.seed,
        "episodes": total_episodes,
        "selection_mode": selection_mode,
        "source_manifest": (
            str(args.episode_manifest.resolve())
            if args.episode_manifest is not None
            else None
        ),
        "source_manifest_metadata": (
            source_manifest.get("metadata") if source_manifest else None
        ),
        "iterator": {
            "shuffle": True,
            "group_by_scene": True,
            "num_episode_sample": (
                args.episodes if selection_mode == "habitat_random" else -1
            ),
            "max_scene_repeat_episodes": 1,
            "max_scene_repeat_steps": -1,
        },
        "selection": manifest,
    }
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest_payload:
            raise RuntimeError(f"Existing manifest drifted: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")

    if args.manifest_only:
        print(
            f"[ZSON3 EVAL] manifest dataset={dataset_log_label} seed={args.seed} "
            f"episodes={total_episodes} mode={selection_mode} path={manifest_path}",
            flush=True,
        )
        return

    existing_results = read_completed(episodes_dir)
    if existing_results and not args.resume:
        raise RuntimeError("Output already contains episodes; pass --resume or use a new run id")
    completed = existing_results if args.resume else {}

    openfrontier_config = read_config_yaml(args.config)
    target_perception = openfrontier_config.get(
        "target_perception", "openfrontier_legacy"
    )
    qwen_health = QwenClient().health()
    service_payload = {"qwen": qwen_health}
    if target_perception == "t1_apex_fusion":
        service_payload["apex_target"] = ApexTargetServiceClient().health()
    else:
        service_payload["sam3"] = Sam3Client().health()
    (args.output_dir / "services.json").write_text(
        json.dumps(service_payload, indent=2) + "\n"
    )
    print(
        f"[ZSON3 EVAL] start dataset={dataset_log_label} seed={args.seed} "
        f"episodes={total_episodes} mode={selection_mode} "
        f"resume_from={len(completed)} "
        f"qwen_attention={qwen_health.get('attention_implementation', 'unknown')}",
        flush=True,
    )

    env = habitat.Env(config=config, dataset=dataset)
    results = [completed[index] for index in sorted(completed)]

    try:
        for index in range(total_episodes):
            env.reset()
            episode = env.current_episode
            identity = episode_identity(episode)
            if identity != manifest[index]:
                raise RuntimeError(
                    f"Episode order drift at {index}: {identity} != {manifest[index]}"
                )
            if index in completed:
                continue

            # Never begin an episode with a missing required model service. A
            # failed preflight does not create a completed episode, so resume
            # will retry the same manifest entry after the service is restored.
            QwenClient().health()
            if target_perception == "t1_apex_fusion":
                ApexTargetServiceClient().health()
            else:
                Sam3Client().health()

            scene = scene_name(episode.scene_id)
            episode_dir = args.output_dir / "episode_logs" / f"{index:03d}_{scene}_{episode.episode_id}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            args.write_path = None
            agent = None
            started = time.perf_counter()
            reason = "not_started"
            error = None
            fatal_service_error = None

            try:
                agent = PointnavAgent(
                    env,
                    args,
                    save_dir=str(episode_dir),
                    openfrontier_config=openfrontier_config,
                    habitat_config=config,
                    scene=scene,
                )
                agent.setup_system()
                agent.initialize()
                reason = "continue_navigation"
                while not env.episode_over:
                    if time.perf_counter() - started >= args.max_time:
                        reason = "max_time_reached"
                        break
                    navigate, reason = agent.navigation(save_images=args.save_images)
                    if not navigate:
                        env.step("stop")
                        break
                if env.episode_over and reason == "continue_navigation":
                    reason = "max_steps_reached"
            except KeyboardInterrupt:
                raise
            except BaseException as exception:
                error = {
                    "type": type(exception).__name__,
                    "message": str(exception),
                    "traceback": traceback.format_exc(),
                }
                if isinstance(
                    exception,
                    (QwenServiceError, Sam3ServiceError, ApexTargetServiceError),
                ):
                    reason = "required_service_unavailable"
                    fatal_service_error = exception
                else:
                    reason = "exception"
            finally:
                elapsed = time.perf_counter() - started
                metrics = dict(env.get_metrics())
                metrics.update(
                    success_spl_at_distance(env, metrics, success_distance=1.0)
                )
                if not metrics.get("success") and reason == "object_found":
                    reason = (
                        "object_found_at_1m_only"
                        if metrics.get("success_at_1m")
                        else "false_positive"
                    )
                result = {
                    "index": index,
                    "status": "error" if error else "ok",
                    "scene": scene,
                    "scene_id": episode.scene_id,
                    "episode_id": str(episode.episode_id),
                    "target": episode.object_category,
                    "reason": reason,
                    "navigation_steps": agent.navigation_steps if agent else 0,
                    "elapsed_seconds": elapsed,
                    "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "metrics": metrics,
                    "detected_objects": len(agent.detected_objects) if agent else 0,
                    "target_diagnostics": (
                        agent.get_target_diagnostics() if agent else None
                    ),
                    "timings": agent.timings if agent else None,
                    "error": error,
                }
                result_root = failures_dir if fatal_service_error else episodes_dir
                result_path = result_root / f"{index:03d}_{scene}_{episode.episode_id}.json"
                result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
                if fatal_service_error is None:
                    results.append(result)
                    summary = aggregate(results)
                    (args.output_dir / "summary.json").write_text(
                        json.dumps(summary, indent=2) + "\n"
                    )
                    print(progress_line(result, summary, total_episodes), flush=True)
                else:
                    print(
                        f"[ZSON3 ABORT] required service unavailable during "
                        f"episode {index + 1}/{total_episodes}: {fatal_service_error}",
                        flush=True,
                    )
                if agent is not None:
                    agent.close()
                del agent
                gc.collect()
                torch.cuda.empty_cache()
            if fatal_service_error is not None:
                raise fatal_service_error
    finally:
        env.close()

    summary = aggregate(results)
    summary_text = (
        f"episodes={summary['episodes']}\n"
        f"successes={summary['successes']}\n"
        f"sr={summary['sr']:.6f}\n"
        f"spl={summary['spl']:.6f}\n"
        f"successes_at_1m={summary['successes_at_1m']}\n"
        f"sr_at_1m={summary['sr_at_1m']:.6f}\n"
        f"spl_at_1m={summary['spl_at_1m']:.6f}\n"
        f"elapsed_seconds={summary['elapsed_seconds']:.3f}\n"
        f"exceptions={summary['exceptions']}\n"
    )
    (args.output_dir / "summary.txt").write_text(summary_text)
    print(
        f"[ZSON3 SUMMARY] episodes={summary['episodes']} SR={summary['sr'] * 100:.2f}% "
        f"SPL={summary['spl']:.4f} SR@1m={summary['sr_at_1m'] * 100:.2f}% "
        f"SPL@1m={summary['spl_at_1m']:.4f} exceptions={summary['exceptions']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
