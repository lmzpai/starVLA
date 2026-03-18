import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from examples.Camera.eval_files.hstar_env import (
    CameraConfig,
    HStarOnlineEnv,
    build_hstar_val_parquet,
    expand_target_ranges_like_hstar,
    load_hstar_episodes,
    load_hstar_episodes_from_parquet,
)
from starVLA.model.framework.base_framework import baseframework


DEFAULT_BASE_CONFIG = "/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/examples/Camera/eval_files/eval_during_training/hstar_eval_during_training.yaml"

CKPT_LIST = [
    "/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/checkpoints/QwenCamera_v2/QwenCamera_chunk_size_4_history_stride_4_sample_stride_4_history_mode_max_gen_v2_norm_90_to_1_ca1m+hstar-l_4B/checkpoints/steps_5000_pytorch_model.pt",
    "/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/checkpoints/QwenCamera_v2/QwenCamera_chunk_size_4_history_stride_4_sample_stride_4_history_mode_max_gen_v2_norm_90_to_1_ca1m+hstar-l_4B/checkpoints/steps_10000_pytorch_model.pt",
    "/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/checkpoints/QwenCamera_v2/QwenCamera_chunk_size_4_history_stride_4_sample_stride_4_history_mode_max_gen_v2_norm_90_to_1_ca1m+hstar-l_4B/checkpoints/steps_15000_pytorch_model.pt",
    "/share/project/zhouenshen/hpfs/code/ActivePerception/starVLA/checkpoints/QwenCamera_v2/QwenCamera_chunk_size_4_history_stride_4_sample_stride_4_history_mode_max_gen_v2_norm_90_to_1_ca1m+hstar-l_4B/checkpoints/steps_20000_pytorch_model.pt",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially evaluate ckpts on hstar_bench_mini (single-process Python, no shell subprocess)."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory that contains multiple ckpts; all entries will be sorted and evaluated in order.",
    )
    parser.add_argument(
        "--base_eval_config",
        type=str,
        default=DEFAULT_BASE_CONFIG,
        help=f"Base eval config YAML, default: {DEFAULT_BASE_CONFIG}",
    )
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Whether to run policy server with bf16.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/hstar_eval_ckpt_list",
        help="Root directory to store per-ckpt eval results.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU ids to use for parallel evaluation, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of parallel evaluation processes per GPU. "
        "Total workers will be workers_per_gpu * num_gpus (capped by number of ckpts).",
    )
    return parser.parse_args()


def _load_ckpts_from_dir(ckpt_dir: str) -> List[str]:
    """从目录中读取 ckpt 路径，按名称排序。"""
    p = Path(ckpt_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"ckpt_dir is not a directory: {ckpt_dir}")
    # 允许目录下是 ckpt 目录或文件，统一按名称排序
    entries = sorted(p.iterdir(), key=lambda x: x.name)
    ckpts: List[str] = [str(e) for e in entries]
    return ckpts



def _prepare_eval_config(
    base_config_path: Path,
    output_dir: Path,
) -> Path:
    with base_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = dict(cfg or {})
    cfg["output_dir"] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = output_dir / "hstar_eval_config.yaml"
    with tmp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return tmp_cfg_path


# ==== 以下是基于 eval_hstar_vla.py 的本地评测实现（去掉 websocket，直接调用模型） ====

ACTION_MAX_MAG_RAD = 1
PANO_DELTA_MAX_DEG = 90


def action_space_to_pano_delta_deg(actions: np.ndarray) -> np.ndarray:
    clipped = np.clip(actions, -ACTION_MAX_MAG_RAD, ACTION_MAX_MAG_RAD)
    return clipped / ACTION_MAX_MAG_RAD * PANO_DELTA_MAX_DEG


def should_stop_by_zero_tail(
    action_chunk: np.ndarray,
    infer_action_num: int,
    tail: int,
    eps_deg: float,
) -> bool:
    if infer_action_num <= 0 or tail <= 0 or action_chunk.shape[0] <= 0:
        return False

    exec_num = min(infer_action_num, action_chunk.shape[0])
    if exec_num <= 0:
        return False

    exec_actions = action_chunk[:exec_num, :]

    tail_len = min(tail, exec_num)
    tail_actions = exec_actions[-tail_len:, :]

    tail_actions_deg = action_space_to_pano_delta_deg(tail_actions)
    return bool(np.all(np.abs(tail_actions_deg) <= float(eps_deg)))


def _save_keyframes(frames: list[np.ndarray], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for i, frame_rgb in enumerate(frames):
        p = out_dir / f"frame_{i:04d}.jpg"
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(p), frame_bgr)
        saved_paths.append(str(p))
    return saved_paths


def _get_bench_tolerance(cfg: dict, bench_name: str) -> dict:
    tol_raw = cfg.get("tolerance", {})
    hos = tol_raw.get("hos", {})
    hps = tol_raw.get("hps", {})
    if bench_name == "hos":
        return hos
    if bench_name == "hps":
        return hps
    return {"yaw_tolerance": 30.0, "pitch_tolerance": 20.0}


def _load_eval_raw_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return dict(raw or {})


def run_hstar_eval_with_model(
    model: Any,
    cfg_path: str,
) -> None:
    raw_cfg = _load_eval_raw_config(cfg_path)

    dataset_cfg = raw_cfg.get("dataset", {})
    server_cfg = raw_cfg.get("server", {})
    policy_cfg = raw_cfg.get("policy", {})
    rollout_cfg = raw_cfg.get("rollout", {})
    camera_cfg_dict = raw_cfg.get("camera", {})
    visualization_cfg = raw_cfg.get("visualization", {})

    out_root = Path(raw_cfg.get("output_dir", "experiments/hstar_eval"))
    out_root.mkdir(parents=True, exist_ok=True)

    camera_cfg = CameraConfig(**camera_cfg_dict)

    np.random.seed(dataset_cfg.get("seed", 42))

    if dataset_cfg.get("auto_generate_val_parquet", False):
        target_parquet = dataset_cfg.get("val_parquet") or dataset_cfg.get("generated_val_parquet_path")
        if not target_parquet:
            target_parquet = str(out_root / "generated_val.parquet")
        build_hstar_val_parquet(
            dataset_root=dataset_cfg["root"],
            benches=dataset_cfg.get("benches", ["hos", "hps"]),
            output_parquet_path=target_parquet,
            default_hps_initial_yaws=dataset_cfg.get("hps_default_initial_yaws", [0.0, 90.0, 180.0, 270.0]),
            init_pitch=dataset_cfg.get("init_pitch", 0.0),
            split="test",
            env_name="hstar",
            bench_shuffle_seed=dataset_cfg.get("seed", 42),
            test_size_by_bench=dataset_cfg.get("parquet_test_size_by_bench"),
        )
        dataset_cfg["val_parquet"] = target_parquet
        logging.info("Generated val parquet at: %s", target_parquet)

    if dataset_cfg.get("val_parquet"):
        episodes = load_hstar_episodes_from_parquet(
            val_parquet_path=dataset_cfg["val_parquet"],
            default_hps_initial_yaws=dataset_cfg.get("hps_default_initial_yaws", [0.0, 90.0, 180.0, 270.0]),
            init_pitch=dataset_cfg.get("init_pitch", 0.0),
        )
    else:
        episodes = load_hstar_episodes(
            dataset_root=dataset_cfg["root"],
            benches=dataset_cfg.get("benches", ["hos", "hps"]),
            default_hps_initial_yaws=dataset_cfg.get("hps_default_initial_yaws", [0.0, 90.0, 180.0, 270.0]),
            init_pitch=dataset_cfg.get("init_pitch", 0.0),
        )
    max_episodes = dataset_cfg.get("max_episodes")
    if max_episodes is not None:
        episodes = episodes[: max_episodes]
    if not episodes:
        raise RuntimeError("No episodes found. Please check dataset_root / benches config.")

    results: list[dict[str, Any]] = []
    bench_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    episode_level_values = sorted({int(ep.level) for ep in episodes})
    stat_levels = dataset_cfg.get("stat_levels")
    if stat_levels is None:
        stat_levels = episode_level_values
    else:
        stat_levels = sorted({int(x) for x in stat_levels})
    if not stat_levels:
        raise ValueError("No valid levels found. Please check dataset.stat_levels / dataset source.")
    logging.info(
        "Episode level distribution (pre-eval): %s",
        {lvl: sum(1 for ep in episodes if int(ep.level) == lvl) for lvl in episode_level_values},
    )
    level_to_idx = {lvl: idx for idx, lvl in enumerate(stat_levels)}
    level_count = [0 for _ in stat_levels]
    level_success = [0 for _ in stat_levels]
    per_bench_level_count: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    per_bench_level_success: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    skipped_level_count: dict[int, int] = defaultdict(int)
    step_success = [0] * 11

    max_inference_rounds = rollout_cfg.get("max_inference_rounds", 20)
    history_max_frames = rollout_cfg.get("history_max_frames", 10)
    infer_action_num = policy_cfg.get("infer_action_num", 4)
    stop_zero_tail = policy_cfg.get("stop_zero_tail", 8)
    stop_zero_eps_deg = policy_cfg.get("stop_zero_eps_deg", 2.25)

    vis_enable = visualization_cfg.get("enable", False)
    vis_fps = visualization_cfg.get("video_fps", 20)

    for ep in tqdm(episodes, desc="HSTAR online eval", unit="episode"):
        env = HStarOnlineEnv(episode=ep, camera_cfg=camera_cfg)
        obs = env.reset()

        episode_dir = out_root / "task" / ep.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        history_queue: deque[np.ndarray] = deque(maxlen=history_max_frames)
        all_frames: list[np.ndarray] = [obs]
        keyframe_frames: list[np.ndarray] = [obs]

        video_writer = None
        if vis_enable:
            video_path = episode_dir / "rollout.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                float(vis_fps),
                (camera_cfg.front_width, camera_cfg.front_height),
            )
            if video_writer.isOpened():
                video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        inference_round = 0
        stop_reason = "max_inference_rounds"
        action_trace_by_round: list[dict[str, Any]] = []
        executed_action_steps = 0

        while inference_round < max_inference_rounds:
            model_inputs = {"image": list(history_queue) + [obs], "lang": ep.instruction}
            with torch.no_grad():
                output = model.predict_action(type="predict_action", examples=[model_inputs])
            data = output

            actions_rad = np.asarray(data.get("normalized_actions"), dtype=np.float32)
            if actions_rad.ndim == 3:
                actions_rad = actions_rad[0]
            if actions_rad.ndim != 2 or actions_rad.shape[1] != 2:
                raise ValueError(f"Unexpected normalized_actions (rad) shape: {actions_rad.shape}")

            action_chunk = np.clip(actions_rad, -ACTION_MAX_MAG_RAD, ACTION_MAX_MAG_RAD)

            exec_num = min(infer_action_num, action_chunk.shape[0])
            exec_actions_rad = action_chunk[:exec_num, :]
            exec_actions_deg = action_space_to_pano_delta_deg(exec_actions_rad)

            if should_stop_by_zero_tail(
                action_chunk=action_chunk,
                infer_action_num=infer_action_num,
                tail=stop_zero_tail,
                eps_deg=stop_zero_eps_deg,
            ):
                round_record: dict[str, Any] = {
                    "round": inference_round,
                    "exec_num": 0,
                    "action_norm": exec_actions_rad.tolist(),
                    "action_delta_rad": exec_actions_rad.tolist(),
                    "action_delta_deg": exec_actions_deg.tolist(),
                    "state_after_action": [],
                    "stop_by_zero_tail": True,
                }
                action_trace_by_round.append(round_record)
                stop_reason = "Stop by zero tail"
                break

            round_states_after_action: list[dict[str, Any]] = []
            for idx in range(exec_num):
                act_rad = action_chunk[idx]
                act_deg = action_space_to_pano_delta_deg(act_rad)
                obs = env.step(delta_yaw_deg=float(act_deg[0]), delta_pitch_deg=float(act_deg[1]))
                executed_action_steps += 1

                state = env.get_state()
                round_states_after_action.append(state)

                all_frames.append(obs)
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            if exec_num > 0:
                history_queue.append(obs)

            if exec_num > 0:
                round_record = {
                    "round": inference_round,
                    "exec_num": exec_num,
                    "action_norm": exec_actions_rad.tolist(),
                    "action_delta_rad": exec_actions_rad.tolist(),
                    "action_delta_deg": exec_actions_deg.tolist(),
                    "state_after_action": round_states_after_action,
                }
                action_trace_by_round.append(round_record)

            keyframe_frames.append(obs)

            inference_round += 1

        if video_writer is not None:
            video_writer.release()

        tol_cfg = _get_bench_tolerance(raw_cfg, ep.bench_name)
        eval_target_yaw, eval_target_pitch = expand_target_ranges_like_hstar(
            yaw_range=ep.target_yaw,
            pitch_range=ep.target_pitch,
            yaw_tolerance=tol_cfg.get("yaw_tolerance", 30.0),
            pitch_tolerance=tol_cfg.get("pitch_tolerance", 20.0),
        )
        success = env.is_success(
            yaw_tolerance=tol_cfg.get("yaw_tolerance", 30.0),
            pitch_tolerance=tol_cfg.get("pitch_tolerance", 20.0),
        )
        bench = ep.bench_name
        bench_stats[bench]["total"] += 1
        bench_stats[bench]["success"] += int(success)
        if ep.level in level_to_idx:
            idx = level_to_idx[ep.level]
            level_count[idx] += 1
            per_bench_level_count[bench][idx] += 1
            if success:
                level_success[idx] += 1
                per_bench_level_success[bench][idx] += 1
                if inference_round <= 10:
                    step_success[inference_round] += 1
        else:
            skipped_level_count[int(ep.level)] += 1

        keyframe_paths = _save_keyframes(keyframe_frames, episode_dir / "keyframes")
        record = {
            "episode_id": ep.episode_id,
            "bench_name": ep.bench_name,
            "scene_id": ep.scene_id,
            "task_id": ep.task_id,
            "instruction": ep.instruction,
            "level": ep.level,
            "init_yaw": ep.init_yaw,
            "init_pitch": ep.init_pitch,
            "target_yaw": list(ep.target_yaw),
            "target_pitch": list(ep.target_pitch),
            "eval_target_yaw": list(eval_target_yaw),
            "eval_target_pitch": list(eval_target_pitch),
            "yaw_tolerance": float(tol_cfg.get("yaw_tolerance", 30.0)),
            "pitch_tolerance": float(tol_cfg.get("pitch_tolerance", 20.0)),
            "stop_reason": stop_reason,
            "inference_rounds_used": inference_round,
            "executed_action_steps": executed_action_steps,
            "final_state": env.get_state(),
            "success": bool(success),
            "keyframes": keyframe_paths,
            "action_trace_by_round": action_trace_by_round,
        }
        with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        results.append(record)

    total = len(results)
    success_num = sum(int(r["success"]) for r in results)
    summary = {
        "total_episodes": total,
        "total_success": success_num,
        "success_rate": float(success_num / max(total, 1)),
        "per_bench": {
            bench: {
                "total": stat["total"],
                "success": stat["success"],
                "success_rate": float(stat["success"] / max(stat["total"], 1)),
            }
            for bench, stat in bench_stats.items()
        },
        "hstar_style": {
            "stat_levels": stat_levels,
            "level_count": level_count,
            "level_success": level_success,
            "step_success": step_success,
        },
        "per_bench_levels": {
            bench: {
                "stat_levels": stat_levels,
                "level_count": per_bench_level_count[bench],
                "level_success": per_bench_level_success[bench],
            }
            for bench in bench_stats.keys()
        },
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_root / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def evaluate_single_ckpt(
    ckpt_path: str,
    args: argparse.Namespace,
    device: str | None = None,
) -> dict:
    ckpt_path_str = str(ckpt_path)
    ckpt_name = Path(ckpt_path_str).name
    logging.info("==== Evaluating ckpt: %s ====", ckpt_path_str)

    output_root = Path(args.output_root)
    ckpt_out_dir = output_root / ckpt_name

    base_cfg_path = Path(args.base_eval_config)
    if not base_cfg_path.is_file():
        raise FileNotFoundError(f"Base eval config not found: {base_cfg_path}")

    tmp_cfg_path = _prepare_eval_config(base_cfg_path, ckpt_out_dir)

    # 本地直接加载模型
    logging.info("Loading model from ckpt: %s", ckpt_path_str)
    model = baseframework.from_pretrained(ckpt_path_str)
    if args.use_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device or "cuda").eval()

    try:
        logging.info("Running local hstar eval for ckpt %s ...", ckpt_path_str)
        run_hstar_eval_with_model(model, str(tmp_cfg_path))
        status = "success"
    except Exception as e:  # noqa: BLE001
        logging.exception("Eval failed for ckpt %s: %s", ckpt_path_str, e)
        status = "eval_failed"
    finally:
        # 释放 GPU 资源
        del model
        torch.cuda.empty_cache()

    summary_path = ckpt_out_dir / "summary.json"
    summary_data = None
    if summary_path.is_file():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to load summary.json for %s: %s", ckpt_path_str, e)

    return {
        "ckpt": ckpt_path_str,
        "status": status,
        "summary": summary_data,
        "output_dir": str(ckpt_out_dir),
    }


def _worker_eval_single_ckpt(
    ckpt: str,
    args_dict: dict,
    gpu_id: str,
) -> dict:
    """
    子进程入口：绑定到指定 GPU 并评测单个 ckpt。
    """
    try:
        gpu_int = int(str(gpu_id))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Invalid gpu_id: {gpu_id}") from e

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if gpu_int < 0 or gpu_int >= device_count:
            raise ValueError(
                f"Invalid device id {gpu_int}. Visible cuda device_count={device_count}. "
                f"If you set CUDA_VISIBLE_DEVICES externally, please pass gpu indices within that visibility."
            )
        torch.cuda.set_device(gpu_int)
        logging.info(
            "[worker pid=%s] bind to cuda:%d (%s)",
            os.getpid(),
            gpu_int,
            torch.cuda.get_device_name(gpu_int),
        )
    worker_args = argparse.Namespace(**args_dict)
    return evaluate_single_ckpt(ckpt, worker_args, device=f"cuda:{gpu_int}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )

    all_ckpts: List[str] = []
    if args.ckpt_dir:
        dir_ckpts = _load_ckpts_from_dir(args.ckpt_dir)
        logging.info("Loaded %d ckpts from directory %s", len(dir_ckpts), args.ckpt_dir)
        all_ckpts.extend(dir_ckpts)

    all_ckpts.extend(CKPT_LIST)

    if not all_ckpts:
        raise ValueError("No ckpt found. Please specify --ckpt_dir or CKPT_LIST.")

    # 解析 GPU 列表
    gpu_ids = [g.strip() for g in str(args.gpus).split(",") if g.strip() != ""]
    if not gpu_ids:
        raise ValueError("Parsed empty GPU list from --gpus.")

    # 生成“GPU 槽位”，用于把多个 worker 分配到同一张卡（workers_per_gpu>1）
    # 例：gpus=0,1 且 workers_per_gpu=2 -> slots: [0,0,1,1]
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be positive.")
    gpu_slots: list[str] = [gid for gid in gpu_ids for _ in range(int(args.workers_per_gpu))]

    max_workers = min(len(all_ckpts), len(gpu_slots))

    logging.info(
        "Start parallel evaluation for %d ckpts with %d workers on GPUs: %s",
        len(all_ckpts),
        max_workers,
        ",".join(gpu_ids),
    )

    # 打印最终分配（ckpt -> gpu）
    assignment: list[tuple[int, str, str]] = []
    for idx, ckpt in enumerate(all_ckpts):
        gpu_id = gpu_slots[idx % len(gpu_slots)]
        assignment.append((idx, gpu_id, ckpt))
    print("\n===== CKPT -> GPU 分配结果 =====")
    for idx, gpu_id, ckpt in assignment:
        print(f"[{idx:03d}] gpu={gpu_id}  ckpt={ckpt}")
    print("===== 分配结束 =====\n")

    # Namespace 在多进程之间复制一份字典即可，避免在子进程里意外修改
    args_dict = vars(args).copy()

    all_results: list[dict] = []

    # 使用 spawn 上下文创建进程池，避免 PyTorch/CUDA 在 fork 模式下的潜在问题
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:
        futures: list[concurrent.futures.Future] = []
        for idx, ckpt in enumerate(all_ckpts):
            gpu_id = gpu_slots[idx % len(gpu_slots)]
            fut = executor.submit(_worker_eval_single_ckpt, ckpt, args_dict, gpu_id)
            futures.append(fut)

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Evaluating ckpts"):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                logging.exception("Unexpected error in worker process: %s", e)
                continue
            all_results.append(res)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_file = output_root / "ckpt_list_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logging.info("All ckpt evaluations finished. Summary saved to %s", summary_file)


if __name__ == "__main__":
    main()

