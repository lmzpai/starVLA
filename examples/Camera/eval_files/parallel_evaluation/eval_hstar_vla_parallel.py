import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from examples.Camera.eval_files.eval_hstar_vla import (
    EvalConfig,
    _get_bench_tolerance,
    _save_keyframes,
    action_space_to_pano_delta_deg,
    load_config as load_eval_config,
    should_stop_by_zero_tail,
)
from examples.Camera.eval_files.hstar_env import (
    HStarOnlineEnv,
    build_hstar_val_parquet,
    expand_target_ranges_like_hstar,
    load_hstar_episodes,
    load_hstar_episodes_from_parquet,
)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

ACTION_MAX_MAG_RAD = 1


@dataclass
class GpuModelPlanItem:
    gpu_id: int
    num_models: int


@dataclass
class ModelServersConfig:
    mode: str = "launch"
    host: str = "127.0.0.1"
    endpoints: list[str] = field(default_factory=list)
    python_bin: str = "python3"
    server_script: str = "deployment/model_server/server_policy.py"
    ckpt_path: str = ""
    use_bf16: bool = True
    idle_timeout: int = -1
    auto_shutdown: bool = True
    startup_timeout_sec: int = 300
    startup_poll_interval_sec: float = 2.0
    gpu_model_plan: list[GpuModelPlanItem] = field(default_factory=list)


@dataclass
class ParallelEvalConfig:
    eval_config_path: str
    mp_start_method: str = "spawn"
    output_dir: str | None = None
    model_servers: ModelServersConfig = field(default_factory=ModelServersConfig)


def load_parallel_config(config_path: str) -> ParallelEvalConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "eval_config_path" not in raw:
        raise ValueError("Parallel config must contain `eval_config_path`.")

    ms_raw = raw.get("model_servers", {}) or {}
    plan = [GpuModelPlanItem(**item) for item in ms_raw.get("gpu_model_plan", [])]
    model_servers = ModelServersConfig(
        mode=ms_raw.get("mode", "launch"),
        host=ms_raw.get("host", "127.0.0.1"),
        endpoints=list(ms_raw.get("endpoints", []) or []),
        python_bin=ms_raw.get("python_bin", "python3"),
        server_script=ms_raw.get("server_script", "deployment/model_server/server_policy.py"),
        ckpt_path=ms_raw.get("ckpt_path", ""),
        use_bf16=bool(ms_raw.get("use_bf16", True)),
        idle_timeout=int(ms_raw.get("idle_timeout", -1)),
        auto_shutdown=bool(ms_raw.get("auto_shutdown", True)),
        startup_timeout_sec=int(ms_raw.get("startup_timeout_sec", 300)),
        startup_poll_interval_sec=float(ms_raw.get("startup_poll_interval_sec", 2.0)),
        gpu_model_plan=plan,
    )
    return ParallelEvalConfig(
        eval_config_path=str(raw["eval_config_path"]),
        mp_start_method=str(raw.get("mp_start_method", "spawn")),
        output_dir=raw.get("output_dir"),
        model_servers=model_servers,
    )


def _parse_endpoint(endpoint: str, fallback_host: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_s = endpoint.rsplit(":", 1)
        return host, int(port_s)
    return fallback_host, int(endpoint)


def _wait_endpoint_ready(host: str, port: int, timeout_sec: float, poll_interval_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(poll_interval_sec)
    raise TimeoutError(f"Endpoint not ready in time: {host}:{port}")


def _split_episodes_evenly(episodes: list[Any], num_splits: int) -> list[list[Any]]:
    if num_splits <= 0:
        return [episodes]
    splits: list[list[Any]] = [[] for _ in range(num_splits)]
    for idx, ep in enumerate(episodes):
        splits[idx % num_splits].append(ep)
    return [x for x in splits if x]


def _launch_model_servers(
    ms_cfg: ModelServersConfig,
    default_port: int,
    default_host: str,
) -> tuple[list[tuple[str, int]], list[subprocess.Popen[Any]]]:
    mode = ms_cfg.mode.strip().lower()
    if mode not in {"launch", "external"}:
        raise ValueError(f"Unsupported model_servers.mode: {ms_cfg.mode}")

    if mode == "external":
        if ms_cfg.endpoints:
            endpoints = [_parse_endpoint(ep, ms_cfg.host) for ep in ms_cfg.endpoints]
        else:
            # 默认继承原始 eval yaml 的 host/port
            endpoints = [(default_host, int(default_port))]
        return endpoints, []

    gpu_ids: list[int] = []
    for item in ms_cfg.gpu_model_plan:
        gpu_ids.extend([item.gpu_id] * item.num_models)
    if not gpu_ids:
        gpu_ids = [0]

    endpoints = [(ms_cfg.host, default_port + i) for i in range(len(gpu_ids))]
    launched: list[subprocess.Popen[Any]] = []
    for i, gpu_id in enumerate(gpu_ids):
        host, port = endpoints[i]
        cmd = [
            ms_cfg.python_bin,
            ms_cfg.server_script,
            "--ckpt_path",
            ms_cfg.ckpt_path,
            "--port",
            str(port),
            "--idle_timeout",
            str(ms_cfg.idle_timeout),
        ]
        if ms_cfg.use_bf16:
            cmd.append("--use_bf16")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(cmd, env=env)
        launched.append(proc)
        logging.info("Launched model server pid=%s on %s:%d (gpu=%d)", proc.pid, host, port, gpu_id)

    for host, port in endpoints:
        _wait_endpoint_ready(
            host=host,
            port=port,
            timeout_sec=float(ms_cfg.startup_timeout_sec),
            poll_interval_sec=float(ms_cfg.startup_poll_interval_sec),
        )
        logging.info("Model server ready: %s:%d", host, port)

    return endpoints, launched


def _terminate_model_servers(processes: list[subprocess.Popen[Any]]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    for p in processes:
        logging.info("Server process pid=%s exited with code=%s", p.pid, p.poll())


def _run_worker_eval(
    worker_id: int,
    episodes: list[Any],
    cfg: EvalConfig,
    endpoint: tuple[str, int],
    out_root_str: str,
    progress_q: Any,
) -> list[dict[str, Any]]:
    np.random.seed((cfg.dataset.seed if cfg.dataset else 42) + worker_id)
    host, port = endpoint
    out_root = Path(out_root_str)
    client = WebsocketClientPolicy(host=host, port=port)
    results: list[dict[str, Any]] = []

    for ep in episodes:
        env = HStarOnlineEnv(episode=ep, camera_cfg=cfg.camera)
        obs = env.reset()

        episode_dir = out_root / "task" / ep.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        history_queue: deque[np.ndarray] = deque(maxlen=cfg.rollout.history_max_frames)
        keyframe_frames: list[np.ndarray] = [obs]

        video_writer = None
        if cfg.visualization.enable:
            video_path = episode_dir / "rollout.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                float(cfg.visualization.video_fps),
                (cfg.camera.front_width, cfg.camera.front_height),
            )
            if video_writer.isOpened():
                video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        inference_round = 0
        stop_reason = "max_inference_rounds"
        action_trace_by_round: list[dict[str, Any]] = []
        executed_action_steps = 0

        while inference_round < cfg.rollout.max_inference_rounds:
            model_inputs = {"image": list(history_queue) + [obs], "lang": ep.instruction}
            response = client.predict_action({"type": "predict_action", "examples": [model_inputs]})
            data = response.get("data", {})
            actions_rad = np.asarray(data.get("normalized_actions"), dtype=np.float32)

            if actions_rad.ndim == 3:
                actions_rad = actions_rad[0]
            if actions_rad.ndim != 2 or actions_rad.shape[1] != 2:
                raise ValueError(f"Unexpected normalized_actions (rad) shape: {actions_rad.shape}")

            action_chunk = np.clip(actions_rad, -ACTION_MAX_MAG_RAD, ACTION_MAX_MAG_RAD)
            exec_num = min(cfg.policy.infer_action_num, action_chunk.shape[0])
            exec_actions_rad = action_chunk[:exec_num, :]
            exec_actions_deg = action_space_to_pano_delta_deg(exec_actions_rad)

            if should_stop_by_zero_tail(
                action_chunk=action_chunk,
                infer_action_num=cfg.policy.infer_action_num,
                tail=cfg.policy.stop_zero_tail,
                eps_deg=cfg.policy.stop_zero_eps_deg,
            ):
                action_trace_by_round.append(
                    {
                        "round": inference_round,
                        "exec_num": 0,
                        "action_norm": exec_actions_rad.tolist(),
                        "action_delta_rad": exec_actions_rad.tolist(),
                        "action_delta_deg": exec_actions_deg.tolist(),
                        "state_after_action": [],
                        "stop_by_zero_tail": True,
                    }
                )
                stop_reason = "Stop by zero tail"
                break

            round_states_after_action: list[dict[str, Any]] = []
            for idx in range(exec_num):
                act_rad = action_chunk[idx]
                act_deg = action_space_to_pano_delta_deg(act_rad)
                obs = env.step(delta_yaw_deg=float(act_deg[0]), delta_pitch_deg=float(act_deg[1]))
                executed_action_steps += 1
                round_states_after_action.append(env.get_state())
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            if exec_num > 0:
                history_queue.append(obs)
                action_trace_by_round.append(
                    {
                        "round": inference_round,
                        "exec_num": exec_num,
                        "action_norm": exec_actions_rad.tolist(),
                        "action_delta_rad": exec_actions_rad.tolist(),
                        "action_delta_deg": exec_actions_deg.tolist(),
                        "state_after_action": round_states_after_action,
                    }
                )

            keyframe_frames.append(obs)
            inference_round += 1

        if video_writer is not None:
            video_writer.release()

        tol_cfg = _get_bench_tolerance(cfg, ep.bench_name)
        eval_target_yaw, eval_target_pitch = expand_target_ranges_like_hstar(
            yaw_range=ep.target_yaw,
            pitch_range=ep.target_pitch,
            yaw_tolerance=tol_cfg.yaw_tolerance,
            pitch_tolerance=tol_cfg.pitch_tolerance,
        )
        success = env.is_success(yaw_tolerance=tol_cfg.yaw_tolerance, pitch_tolerance=tol_cfg.pitch_tolerance)
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
            "yaw_tolerance": float(tol_cfg.yaw_tolerance),
            "pitch_tolerance": float(tol_cfg.pitch_tolerance),
            "stop_reason": stop_reason,
            "inference_rounds_used": inference_round,
            "executed_action_steps": executed_action_steps,
            "final_state": env.get_state(),
            "success": bool(success),
            "keyframes": keyframe_paths,
            "action_trace_by_round": action_trace_by_round,
            "worker_id": worker_id,
            "server_endpoint": f"{host}:{port}",
        }
        with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        results.append(record)
        progress_q.put(1)

    return results


def _build_summary(cfg: EvalConfig, episodes: list[Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    bench_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    episode_level_values = sorted({int(ep.level) for ep in episodes})
    if cfg.dataset and cfg.dataset.stat_levels is not None:
        stat_levels = sorted({int(x) for x in cfg.dataset.stat_levels})
    else:
        stat_levels = episode_level_values
    if not stat_levels:
        raise ValueError("No valid levels found. Please check dataset.stat_levels / dataset source.")

    level_to_idx = {lvl: idx for idx, lvl in enumerate(stat_levels)}
    level_count = [0 for _ in stat_levels]
    level_success = [0 for _ in stat_levels]
    per_bench_level_count: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    per_bench_level_success: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    skipped_level_count: dict[int, int] = defaultdict(int)
    step_success = [0] * 11

    for r in results:
        bench = str(r["bench_name"])
        level = int(r["level"])
        success = bool(r["success"])
        rounds = int(r["inference_rounds_used"])
        bench_stats[bench]["total"] += 1
        bench_stats[bench]["success"] += int(success)

        if level in level_to_idx:
            idx = level_to_idx[level]
            level_count[idx] += 1
            per_bench_level_count[bench][idx] += 1
            if success:
                level_success[idx] += 1
                per_bench_level_success[bench][idx] += 1
                if rounds <= 10:
                    step_success[rounds] += 1
        else:
            skipped_level_count[level] += 1

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
    if skipped_level_count:
        summary["skipped_level_count"] = dict(skipped_level_count)
    return summary


def run_parallel_eval(parallel_cfg: ParallelEvalConfig) -> None:
    cfg = load_eval_config(parallel_cfg.eval_config_path)
    if cfg.dataset is None:
        raise ValueError("dataset config is required")

    # output_dir 默认继承原始 eval yaml，可被 parallel yaml 覆盖
    cfg.output_dir = parallel_cfg.output_dir if parallel_cfg.output_dir else cfg.output_dir

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.dataset.seed)

    if cfg.dataset.auto_generate_val_parquet:
        target_parquet = cfg.dataset.val_parquet or cfg.dataset.generated_val_parquet_path
        if not target_parquet:
            target_parquet = str(out_root / "generated_val.parquet")
        build_hstar_val_parquet(
            dataset_root=cfg.dataset.root,
            benches=cfg.dataset.benches,
            output_parquet_path=target_parquet,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
            split="test",
            env_name="hstar",
            bench_shuffle_seed=cfg.dataset.seed,
            test_size_by_bench=cfg.dataset.parquet_test_size_by_bench,
        )
        cfg.dataset.val_parquet = target_parquet
        logging.info("Generated val parquet at: %s", target_parquet)

    if cfg.dataset.val_parquet:
        episodes = load_hstar_episodes_from_parquet(
            val_parquet_path=cfg.dataset.val_parquet,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
        )
    else:
        episodes = load_hstar_episodes(
            dataset_root=cfg.dataset.root,
            benches=cfg.dataset.benches,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
        )
    if cfg.dataset.max_episodes is not None:
        episodes = episodes[: cfg.dataset.max_episodes]
    if not episodes:
        raise RuntimeError("No episodes found. Please check dataset_root / benches config.")

    # port 默认继承原始 eval yaml 的 server.port
    endpoints, launched_processes = _launch_model_servers(
        ms_cfg=parallel_cfg.model_servers,
        default_port=cfg.server.port,
        default_host=cfg.server.host,
    )
    if not endpoints:
        raise RuntimeError("No server endpoints available.")

    episode_splits = _split_episodes_evenly(episodes, len(endpoints))
    worker_endpoints = endpoints[: len(episode_splits)]
    logging.info("Prepared %d workers for %d episodes.", len(worker_endpoints), len(episodes))

    ctx = mp.get_context(parallel_cfg.mp_start_method)
    manager = ctx.Manager()
    progress_q = manager.Queue()
    all_results: list[dict[str, Any]] = []

    try:
        with ProcessPoolExecutor(max_workers=len(worker_endpoints), mp_context=ctx) as executor:
            futures = []
            for worker_id, (eps, endpoint) in enumerate(zip(episode_splits, worker_endpoints)):
                futures.append(
                    executor.submit(
                        _run_worker_eval,
                        worker_id,
                        eps,
                        cfg,
                        endpoint,
                        str(out_root),
                        progress_q,
                    )
                )

            pending = set(futures)
            with tqdm(total=len(episodes), desc="HSTAR parallel eval", unit="episode") as pbar:
                while pending:
                    done_now = {f for f in pending if f.done()}
                    for f in done_now:
                        all_results.extend(f.result())
                    pending -= done_now

                    updated = False
                    while True:
                        try:
                            pbar.update(int(progress_q.get_nowait()))
                            updated = True
                        except queue.Empty:
                            break

                    if not updated and pending:
                        try:
                            pbar.update(int(progress_q.get(timeout=0.2)))
                        except queue.Empty:
                            pass

                # 排空队列，避免遗漏最后若干个已完成 episode
                while True:
                    try:
                        pbar.update(int(progress_q.get_nowait()))
                    except queue.Empty:
                        break
    finally:
        try:
            manager.shutdown()
        except Exception:
            pass
        if parallel_cfg.model_servers.mode.strip().lower() == "launch" and parallel_cfg.model_servers.auto_shutdown:
            _terminate_model_servers(launched_processes)

    summary = _build_summary(cfg, episodes, all_results)
    all_results.sort(key=lambda x: str(x["episode_id"]))

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_root / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with (out_root / "hstar_style_summary.txt").open("w", encoding="utf-8") as f:
        stat_levels = summary["hstar_style"]["stat_levels"]
        level_count = summary["hstar_style"]["level_count"]
        level_success = summary["hstar_style"]["level_success"]
        for i, lvl in enumerate(stat_levels):
            f.write(f"{lvl} {level_success[i]} {level_count[i]}\n")
        f.write("step_success " + " ".join(map(str, summary["hstar_style"]["step_success"])) + "\n")

    logging.info(
        "HSTAR parallel eval done: %d / %d (%.4f)",
        summary["total_success"],
        summary["total_episodes"],
        summary["success_rate"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel online HSTAR evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to parallel YAML config")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args()
    parallel_cfg = load_parallel_config(args.config)
    run_parallel_eval(parallel_cfg)
import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from examples.Camera.eval_files.eval_hstar_vla import (
    EvalConfig,
    _get_bench_tolerance,
    _save_keyframes,
    action_space_to_pano_delta_deg,
    load_config as load_eval_config,
    should_stop_by_zero_tail,
)
from examples.Camera.eval_files.hstar_env import (
    HStarOnlineEnv,
    build_hstar_val_parquet,
    expand_target_ranges_like_hstar,
    load_hstar_episodes,
    load_hstar_episodes_from_parquet,
)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


@dataclass
class GpuModelPlanItem:
    gpu_id: int
    num_models: int


@dataclass
class ModelServersConfig:
    mode: str = "launch"
    host: str = "127.0.0.1"
    endpoints: list[str] = field(default_factory=list)
    python_bin: str = "python3"
    server_script: str = "deployment/model_server/server_policy.py"
    ckpt_path: str = ""
    use_bf16: bool = True
    idle_timeout: int = -1
    auto_shutdown: bool = True
    startup_timeout_sec: int = 300
    startup_poll_interval_sec: float = 2.0
    gpu_model_plan: list[GpuModelPlanItem] = field(default_factory=list)


@dataclass
class ParallelEvalConfig:
    eval_config_path: str
    mp_start_method: str = "spawn"
    output_dir: str | None = None
    model_servers: ModelServersConfig = field(default_factory=ModelServersConfig)


def load_parallel_config(config_path: str) -> ParallelEvalConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "eval_config_path" not in raw:
        raise ValueError("Parallel config must contain `eval_config_path`.")

    ms_raw = raw.get("model_servers", {}) or {}
    plan = [GpuModelPlanItem(**item) for item in ms_raw.get("gpu_model_plan", [])]
    model_servers = ModelServersConfig(
        mode=ms_raw.get("mode", "launch"),
        host=ms_raw.get("host", "127.0.0.1"),
        endpoints=list(ms_raw.get("endpoints", []) or []),
        python_bin=ms_raw.get("python_bin", "python3"),
        server_script=ms_raw.get("server_script", "deployment/model_server/server_policy.py"),
        ckpt_path=ms_raw.get("ckpt_path", ""),
        use_bf16=bool(ms_raw.get("use_bf16", True)),
        idle_timeout=int(ms_raw.get("idle_timeout", -1)),
        auto_shutdown=bool(ms_raw.get("auto_shutdown", True)),
        startup_timeout_sec=int(ms_raw.get("startup_timeout_sec", 300)),
        startup_poll_interval_sec=float(ms_raw.get("startup_poll_interval_sec", 2.0)),
        gpu_model_plan=plan,
    )
    return ParallelEvalConfig(
        eval_config_path=str(raw["eval_config_path"]),
        mp_start_method=str(raw.get("mp_start_method", "spawn")),
        output_dir=raw.get("output_dir"),
        model_servers=model_servers,
    )


def _parse_endpoint(endpoint: str, fallback_host: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_s = endpoint.rsplit(":", 1)
        return host, int(port_s)
    return fallback_host, int(endpoint)


def _wait_endpoint_ready(host: str, port: int, timeout_sec: float, poll_interval_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(poll_interval_sec)
    raise TimeoutError(f"Endpoint not ready in time: {host}:{port}")


def _split_episodes_evenly(episodes: list[Any], num_splits: int) -> list[list[Any]]:
    if num_splits <= 0:
        return [episodes]
    splits: list[list[Any]] = [[] for _ in range(num_splits)]
    for idx, ep in enumerate(episodes):
        splits[idx % num_splits].append(ep)
    return [x for x in splits if x]


def _launch_model_servers(
    ms_cfg: ModelServersConfig,
    default_port: int,
    default_host: str,
) -> tuple[list[tuple[str, int]], list[subprocess.Popen[Any]]]:
    mode = ms_cfg.mode.strip().lower()
    if mode not in {"launch", "external"}:
        raise ValueError(f"Unsupported model_servers.mode: {ms_cfg.mode}")

    if mode == "external":
        if ms_cfg.endpoints:
            endpoints = [_parse_endpoint(ep, ms_cfg.host) for ep in ms_cfg.endpoints]
        else:
            endpoints = [(default_host, int(default_port))]
        return endpoints, []

    gpu_ids: list[int] = []
    for item in ms_cfg.gpu_model_plan:
        gpu_ids.extend([item.gpu_id] * item.num_models)
    if not gpu_ids:
        gpu_ids = [0]

    endpoints = [(ms_cfg.host, default_port + i) for i in range(len(gpu_ids))]
    launched: list[subprocess.Popen[Any]] = []
    for i, gpu_id in enumerate(gpu_ids):
        host, port = endpoints[i]
        cmd = [
            ms_cfg.python_bin,
            ms_cfg.server_script,
            "--ckpt_path",
            ms_cfg.ckpt_path,
            "--port",
            str(port),
            "--idle_timeout",
            str(ms_cfg.idle_timeout),
        ]
        if ms_cfg.use_bf16:
            cmd.append("--use_bf16")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(cmd, env=env)
        launched.append(proc)
        logging.info("Launched model server pid=%s on %s:%d (gpu=%d)", proc.pid, host, port, gpu_id)

    for host, port in endpoints:
        _wait_endpoint_ready(
            host=host,
            port=port,
            timeout_sec=float(ms_cfg.startup_timeout_sec),
            poll_interval_sec=float(ms_cfg.startup_poll_interval_sec),
        )
        logging.info("Model server ready: %s:%d", host, port)

    return endpoints, launched


def _terminate_model_servers(processes: list[subprocess.Popen[Any]]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    for p in processes:
        logging.info("Server process pid=%s exited with code=%s", p.pid, p.poll())


def _run_worker_eval(
    worker_id: int,
    episodes: list[Any],
    cfg: EvalConfig,
    endpoint: tuple[str, int],
    out_root_str: str,
    progress_q: Any,
) -> list[dict[str, Any]]:
    np.random.seed((cfg.dataset.seed if cfg.dataset else 42) + worker_id)
    host, port = endpoint
    out_root = Path(out_root_str)
    client = WebsocketClientPolicy(host=host, port=port)
    results: list[dict[str, Any]] = []

    for ep in episodes:
        env = HStarOnlineEnv(episode=ep, camera_cfg=cfg.camera)
        obs = env.reset()

        episode_dir = out_root / "task" / ep.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        history_queue: deque[np.ndarray] = deque(maxlen=cfg.rollout.history_max_frames)
        all_frames: list[np.ndarray] = [obs]
        keyframe_frames: list[np.ndarray] = [obs]

        video_writer = None
        if cfg.visualization.enable:
            video_path = episode_dir / "rollout.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                float(cfg.visualization.video_fps),
                (cfg.camera.front_width, cfg.camera.front_height),
            )
            if video_writer.isOpened():
                video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        inference_round = 0
        stop_reason = "max_inference_rounds"
        action_trace_by_round: list[dict[str, Any]] = []
        executed_action_steps = 0

        while inference_round < cfg.rollout.max_inference_rounds:
            model_inputs = {"image": list(history_queue) + [obs], "lang": ep.instruction}
            response = client.predict_action({"type": "predict_action", "examples": [model_inputs]})
            data = response.get("data", {})
            actions_rad = np.asarray(data.get("normalized_actions"), dtype=np.float32)

            if actions_rad.ndim == 3:
                actions_rad = actions_rad[0]
            if actions_rad.ndim != 2 or actions_rad.shape[1] != 2:
                raise ValueError(f"Unexpected normalized_actions (rad) shape: {actions_rad.shape}")

            action_chunk = np.clip(actions_rad, -4.0 * np.pi, 4.0 * np.pi)
            exec_num = min(cfg.policy.infer_action_num, action_chunk.shape[0])
            exec_actions_rad = action_chunk[:exec_num, :]
            exec_actions_deg = action_space_to_pano_delta_deg(exec_actions_rad)

            if should_stop_by_zero_tail(
                action_chunk=action_chunk,
                infer_action_num=cfg.policy.infer_action_num,
                tail=cfg.policy.stop_zero_tail,
                eps_deg=cfg.policy.stop_zero_eps_deg,
            ):
                action_trace_by_round.append(
                    {
                        "round": inference_round,
                        "exec_num": 0,
                        "action_norm": exec_actions_rad.tolist(),
                        "action_delta_rad": exec_actions_rad.tolist(),
                        "action_delta_deg": exec_actions_deg.tolist(),
                        "state_after_action": [],
                        "stop_by_zero_tail": True,
                    }
                )
                stop_reason = "Stop by zero tail"
                break

            round_states_after_action: list[dict[str, Any]] = []
            for idx in range(exec_num):
                act_rad = action_chunk[idx]
                act_deg = action_space_to_pano_delta_deg(act_rad)
                obs = env.step(delta_yaw_deg=float(act_deg[0]), delta_pitch_deg=float(act_deg[1]))
                executed_action_steps += 1
                round_states_after_action.append(env.get_state())
                all_frames.append(obs)
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            if exec_num > 0:
                history_queue.append(obs)
                action_trace_by_round.append(
                    {
                        "round": inference_round,
                        "exec_num": exec_num,
                        "action_norm": exec_actions_rad.tolist(),
                        "action_delta_rad": exec_actions_rad.tolist(),
                        "action_delta_deg": exec_actions_deg.tolist(),
                        "state_after_action": round_states_after_action,
                    }
                )

            keyframe_frames.append(obs)
            inference_round += 1

        if video_writer is not None:
            video_writer.release()

        tol_cfg = _get_bench_tolerance(cfg, ep.bench_name)
        eval_target_yaw, eval_target_pitch = expand_target_ranges_like_hstar(
            yaw_range=ep.target_yaw,
            pitch_range=ep.target_pitch,
            yaw_tolerance=tol_cfg.yaw_tolerance,
            pitch_tolerance=tol_cfg.pitch_tolerance,
        )
        success = env.is_success(yaw_tolerance=tol_cfg.yaw_tolerance, pitch_tolerance=tol_cfg.pitch_tolerance)
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
            "yaw_tolerance": float(tol_cfg.yaw_tolerance),
            "pitch_tolerance": float(tol_cfg.pitch_tolerance),
            "stop_reason": stop_reason,
            "inference_rounds_used": inference_round,
            "executed_action_steps": executed_action_steps,
            "final_state": env.get_state(),
            "success": bool(success),
            "keyframes": keyframe_paths,
            "action_trace_by_round": action_trace_by_round,
            "worker_id": worker_id,
            "server_endpoint": f"{host}:{port}",
        }
        with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        results.append(record)
        progress_q.put(1)

    return results


def _build_summary(cfg: EvalConfig, episodes: list[Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    bench_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    episode_level_values = sorted({int(ep.level) for ep in episodes})
    if cfg.dataset and cfg.dataset.stat_levels is not None:
        stat_levels = sorted({int(x) for x in cfg.dataset.stat_levels})
    else:
        stat_levels = episode_level_values
    if not stat_levels:
        raise ValueError("No valid levels found. Please check dataset.stat_levels / dataset source.")

    level_to_idx = {lvl: idx for idx, lvl in enumerate(stat_levels)}
    level_count = [0 for _ in stat_levels]
    level_success = [0 for _ in stat_levels]
    per_bench_level_count: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    per_bench_level_success: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    skipped_level_count: dict[int, int] = defaultdict(int)
    step_success = [0] * 11

    for r in results:
        bench = str(r["bench_name"])
        level = int(r["level"])
        success = bool(r["success"])
        rounds = int(r["inference_rounds_used"])
        bench_stats[bench]["total"] += 1
        bench_stats[bench]["success"] += int(success)

        if level in level_to_idx:
            idx = level_to_idx[level]
            level_count[idx] += 1
            per_bench_level_count[bench][idx] += 1
            if success:
                level_success[idx] += 1
                per_bench_level_success[bench][idx] += 1
                if rounds <= 10:
                    step_success[rounds] += 1
        else:
            skipped_level_count[level] += 1

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
    if skipped_level_count:
        summary["skipped_level_count"] = dict(skipped_level_count)
    return summary


def run_parallel_eval(parallel_cfg: ParallelEvalConfig) -> None:
    cfg = load_eval_config(parallel_cfg.eval_config_path)
    if cfg.dataset is None:
        raise ValueError("dataset config is required")

    output_dir = parallel_cfg.output_dir if parallel_cfg.output_dir else cfg.output_dir
    cfg.output_dir = output_dir

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.dataset.seed)

    if cfg.dataset.auto_generate_val_parquet:
        target_parquet = cfg.dataset.val_parquet or cfg.dataset.generated_val_parquet_path
        if not target_parquet:
            target_parquet = str(out_root / "generated_val.parquet")
        build_hstar_val_parquet(
            dataset_root=cfg.dataset.root,
            benches=cfg.dataset.benches,
            output_parquet_path=target_parquet,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
            split="test",
            env_name="hstar",
            bench_shuffle_seed=cfg.dataset.seed,
            test_size_by_bench=cfg.dataset.parquet_test_size_by_bench,
        )
        cfg.dataset.val_parquet = target_parquet
        logging.info("Generated val parquet at: %s", target_parquet)

    if cfg.dataset.val_parquet:
        episodes = load_hstar_episodes_from_parquet(
            val_parquet_path=cfg.dataset.val_parquet,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
        )
    else:
        episodes = load_hstar_episodes(
            dataset_root=cfg.dataset.root,
            benches=cfg.dataset.benches,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
        )
    if cfg.dataset.max_episodes is not None:
        episodes = episodes[: cfg.dataset.max_episodes]
    if not episodes:
        raise RuntimeError("No episodes found. Please check dataset_root / benches config.")

    default_host = cfg.server.host
    default_port = cfg.server.port
    endpoints, launched_processes = _launch_model_servers(
        ms_cfg=parallel_cfg.model_servers,
        default_port=default_port,
        default_host=default_host,
    )
    if not endpoints:
        raise RuntimeError("No server endpoints available.")

    logging.info("Using %d model server endpoints.", len(endpoints))
    episode_splits = _split_episodes_evenly(episodes, len(endpoints))
    worker_endpoints = endpoints[: len(episode_splits)]
    logging.info("Prepared %d workers for %d episodes.", len(worker_endpoints), len(episodes))

    ctx = mp.get_context(parallel_cfg.mp_start_method)
    progress_q = ctx.Queue()
    all_results: list[dict[str, Any]] = []

    try:
        with ProcessPoolExecutor(max_workers=len(worker_endpoints), mp_context=ctx) as executor:
            futures = []
            for worker_id, (eps, endpoint) in enumerate(zip(episode_splits, worker_endpoints)):
                futures.append(
                    executor.submit(
                        _run_worker_eval,
                        worker_id,
                        eps,
                        cfg,
                        endpoint,
                        str(out_root),
                        progress_q,
                    )
                )

            pending = set(futures)
            with tqdm(total=len(episodes), desc="HSTAR parallel eval", unit="episode") as pbar:
                while pending:
                    done_now = {f for f in pending if f.done()}
                    for f in done_now:
                        all_results.extend(f.result())
                    pending -= done_now

                    updated = False
                    while True:
                        try:
                            pbar.update(int(progress_q.get_nowait()))
                            updated = True
                        except queue.Empty:
                            break
                    if not updated and pending:
                        try:
                            pbar.update(int(progress_q.get(timeout=0.2)))
                        except queue.Empty:
                            pass

                while True:
                    try:
                        pbar.update(int(progress_q.get_nowait()))
                    except queue.Empty:
                        break
    finally:
        if parallel_cfg.model_servers.mode.strip().lower() == "launch" and parallel_cfg.model_servers.auto_shutdown:
            _terminate_model_servers(launched_processes)

    summary = _build_summary(cfg, episodes, all_results)
    all_results.sort(key=lambda x: str(x["episode_id"]))

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_root / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with (out_root / "hstar_style_summary.txt").open("w", encoding="utf-8") as f:
        stat_levels = summary["hstar_style"]["stat_levels"]
        level_count = summary["hstar_style"]["level_count"]
        level_success = summary["hstar_style"]["level_success"]
        for i, lvl in enumerate(stat_levels):
            f.write(f"{lvl} {level_success[i]} {level_count[i]}\n")
        f.write("step_success " + " ".join(map(str, summary["hstar_style"]["step_success"])) + "\n")

    logging.info(
        "HSTAR parallel eval done: %d / %d (%.4f)",
        summary["total_success"],
        summary["total_episodes"],
        summary["success_rate"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel online HSTAR evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to parallel YAML config")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args()
    parallel_cfg = load_parallel_config(args.config)
    run_parallel_eval(parallel_cfg)
import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from examples.Camera.eval_files.eval_hstar_vla import (
    EvalConfig,
    _get_bench_tolerance,
    _save_keyframes,
    action_space_to_pano_delta_deg,
    load_config as load_eval_config,
    should_stop_by_zero_tail,
)
from examples.Camera.eval_files.hstar_env import (
    HStarOnlineEnv,
    build_hstar_val_parquet,
    expand_target_ranges_like_hstar,
    load_hstar_episodes,
    load_hstar_episodes_from_parquet,
)
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


@dataclass
class GpuModelPlanItem:
    gpu_id: int
    num_models: int


@dataclass
class ModelServersConfig:
    mode: str = "launch"
    host: str = "127.0.0.1"
    endpoints: list[str] = field(default_factory=list)
    python_bin: str = "python3"
    server_script: str = "deployment/model_server/server_policy.py"
    ckpt_path: str = ""
    use_bf16: bool = True
    idle_timeout: int = -1
    auto_shutdown: bool = True
    startup_timeout_sec: int = 300
    startup_poll_interval_sec: float = 2.0
    gpu_model_plan: list[GpuModelPlanItem] = field(default_factory=list)


@dataclass
class ParallelEvalConfig:
    eval_config_path: str
    mp_start_method: str = "spawn"
    output_dir: str | None = None
    model_servers: ModelServersConfig = field(default_factory=ModelServersConfig)


def load_parallel_config(config_path: str) -> ParallelEvalConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "eval_config_path" not in raw:
        raise ValueError("Parallel config must contain `eval_config_path`.")

    ms_raw = raw.get("model_servers", {}) or {}
    plan = [GpuModelPlanItem(**item) for item in ms_raw.get("gpu_model_plan", [])]
    model_servers = ModelServersConfig(
        mode=ms_raw.get("mode", "launch"),
        host=ms_raw.get("host", "127.0.0.1"),
        endpoints=list(ms_raw.get("endpoints", []) or []),
        python_bin=ms_raw.get("python_bin", "python3"),
        server_script=ms_raw.get("server_script", "deployment/model_server/server_policy.py"),
        ckpt_path=ms_raw.get("ckpt_path", ""),
        use_bf16=bool(ms_raw.get("use_bf16", True)),
        idle_timeout=int(ms_raw.get("idle_timeout", -1)),
        auto_shutdown=bool(ms_raw.get("auto_shutdown", True)),
        startup_timeout_sec=int(ms_raw.get("startup_timeout_sec", 300)),
        startup_poll_interval_sec=float(ms_raw.get("startup_poll_interval_sec", 2.0)),
        gpu_model_plan=plan,
    )
    return ParallelEvalConfig(
        eval_config_path=str(raw["eval_config_path"]),
        mp_start_method=str(raw.get("mp_start_method", "spawn")),
        output_dir=raw.get("output_dir"),
        model_servers=model_servers,
    )


def _parse_endpoint(endpoint: str, fallback_host: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_s = endpoint.rsplit(":", 1)
        return host, int(port_s)
    return fallback_host, int(endpoint)


def _wait_endpoint_ready(host: str, port: int, timeout_sec: float, poll_interval_sec: float) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(poll_interval_sec)
    raise TimeoutError(f"Endpoint not ready in time: {host}:{port}")


def _split_episodes_evenly(episodes: list[Any], num_splits: int) -> list[list[Any]]:
    if num_splits <= 0:
        return [episodes]
    splits: list[list[Any]] = [[] for _ in range(num_splits)]
    for idx, ep in enumerate(episodes):
        splits[idx % num_splits].append(ep)
    return [x for x in splits if x]


def _launch_model_servers(
    ms_cfg: ModelServersConfig,
    default_port: int,
    default_host: str,
) -> tuple[list[tuple[str, int]], list[subprocess.Popen[Any]]]:
    mode = ms_cfg.mode.strip().lower()
    if mode not in {"launch", "external"}:
        raise ValueError(f"Unsupported model_servers.mode: {ms_cfg.mode}")

    if mode == "external":
        if ms_cfg.endpoints:
            endpoints = [_parse_endpoint(ep, ms_cfg.host) for ep in ms_cfg.endpoints]
        else:
            # 默认继承原始 eval yaml 的 host/port
            endpoints = [(default_host, int(default_port))]
        return endpoints, []

    # mode == launch
    gpu_ids: list[int] = []
    for item in ms_cfg.gpu_model_plan:
        gpu_ids.extend([item.gpu_id] * item.num_models)
    if not gpu_ids:
        gpu_ids = [0]

    endpoints = [(ms_cfg.host, default_port + i) for i in range(len(gpu_ids))]
    launched: list[subprocess.Popen[Any]] = []
    for i, gpu_id in enumerate(gpu_ids):
        host, port = endpoints[i]
        cmd = [
            ms_cfg.python_bin,
            ms_cfg.server_script,
            "--ckpt_path",
            ms_cfg.ckpt_path,
            "--port",
            str(port),
            "--idle_timeout",
            str(ms_cfg.idle_timeout),
        ]
        if ms_cfg.use_bf16:
            cmd.append("--use_bf16")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(cmd, env=env)
        launched.append(proc)
        logging.info("Launched model server pid=%s on %s:%d (gpu=%d)", proc.pid, host, port, gpu_id)

    for host, port in endpoints:
        _wait_endpoint_ready(
            host=host,
            port=port,
            timeout_sec=float(ms_cfg.startup_timeout_sec),
            poll_interval_sec=float(ms_cfg.startup_poll_interval_sec),
        )
        logging.info("Model server ready: %s:%d", host, port)

    return endpoints, launched


def _terminate_model_servers(processes: list[subprocess.Popen[Any]]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    for p in processes:
        exit_code = p.poll()
        logging.info("Server process pid=%s exited with code=%s", p.pid, exit_code)


def _run_worker_eval(
    worker_id: int,
    episodes: list[Any],
    cfg: EvalConfig,
    endpoint: tuple[str, int],
    out_root_str: str,
    progress_q: Any,
) -> list[dict[str, Any]]:
    np.random.seed((cfg.dataset.seed if cfg.dataset else 42) + worker_id)
    host, port = endpoint
    out_root = Path(out_root_str)
    client = WebsocketClientPolicy(host=host, port=port)
    results: list[dict[str, Any]] = []

    for ep in episodes:
        env = HStarOnlineEnv(episode=ep, camera_cfg=cfg.camera)
        obs = env.reset()

        episode_dir = out_root / "task" / ep.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        history_queue: deque[np.ndarray] = deque(maxlen=cfg.rollout.history_max_frames)
        all_frames: list[np.ndarray] = [obs]
        keyframe_frames: list[np.ndarray] = [obs]

        video_writer = None
        if cfg.visualization.enable:
            video_path = episode_dir / "rollout.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                float(cfg.visualization.video_fps),
                (cfg.camera.front_width, cfg.camera.front_height),
            )
            if video_writer.isOpened():
                video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

        inference_round = 0
        stop_reason = "max_inference_rounds"
        action_trace_by_round: list[dict[str, Any]] = []
        executed_action_steps = 0

        while inference_round < cfg.rollout.max_inference_rounds:
            model_inputs = {"image": list(history_queue) + [obs], "lang": ep.instruction}
            response = client.predict_action({"type": "predict_action", "examples": [model_inputs]})
            data = response.get("data", {})
            actions_rad = np.asarray(data.get("normalized_actions"), dtype=np.float32)

            if actions_rad.ndim == 3:
                actions_rad = actions_rad[0]

            if actions_rad.ndim != 2 or actions_rad.shape[1] != 2:
                raise ValueError(f"Unexpected normalized_actions (rad) shape: {actions_rad.shape}")

            action_chunk = np.clip(actions_rad, -4.0 * np.pi, 4.0 * np.pi)
            exec_num = min(cfg.policy.infer_action_num, action_chunk.shape[0])
            exec_actions_rad = action_chunk[:exec_num, :]
            exec_actions_deg = action_space_to_pano_delta_deg(exec_actions_rad)

            if should_stop_by_zero_tail(
                action_chunk=action_chunk,
                infer_action_num=cfg.policy.infer_action_num,
                tail=cfg.policy.stop_zero_tail,
                eps_deg=cfg.policy.stop_zero_eps_deg,
            ):
                action_trace_by_round.append(
                    {
                        "round": inference_round,
                        "exec_num": 0,
                        "action_norm": exec_actions_rad.tolist(),
                        "action_delta_rad": exec_actions_rad.tolist(),
                        "action_delta_deg": exec_actions_deg.tolist(),
                        "state_after_action": [],
                        "stop_by_zero_tail": True,
                    }
                )
                stop_reason = "Stop by zero tail"
                break

            round_states_after_action: list[dict[str, Any]] = []
            for idx in range(exec_num):
                act_rad = action_chunk[idx]
                act_deg = action_space_to_pano_delta_deg(act_rad)
                obs = env.step(delta_yaw_deg=float(act_deg[0]), delta_pitch_deg=float(act_deg[1]))
                executed_action_steps += 1
                round_states_after_action.append(env.get_state())

                all_frames.append(obs)
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))

            if exec_num > 0:
                history_queue.append(obs)
                action_trace_by_round.append(
                    {
                        "round": inference_round,
                        "exec_num": exec_num,
                        "action_norm": exec_actions_rad.tolist(),
                        "action_delta_rad": exec_actions_rad.tolist(),
                        "action_delta_deg": exec_actions_deg.tolist(),
                        "state_after_action": round_states_after_action,
                    }
                )

            keyframe_frames.append(obs)
            inference_round += 1

        if video_writer is not None:
            video_writer.release()

        tol_cfg = _get_bench_tolerance(cfg, ep.bench_name)
        eval_target_yaw, eval_target_pitch = expand_target_ranges_like_hstar(
            yaw_range=ep.target_yaw,
            pitch_range=ep.target_pitch,
            yaw_tolerance=tol_cfg.yaw_tolerance,
            pitch_tolerance=tol_cfg.pitch_tolerance,
        )
        success = env.is_success(yaw_tolerance=tol_cfg.yaw_tolerance, pitch_tolerance=tol_cfg.pitch_tolerance)
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
            "yaw_tolerance": float(tol_cfg.yaw_tolerance),
            "pitch_tolerance": float(tol_cfg.pitch_tolerance),
            "stop_reason": stop_reason,
            "inference_rounds_used": inference_round,
            "executed_action_steps": executed_action_steps,
            "final_state": env.get_state(),
            "success": bool(success),
            "keyframes": keyframe_paths,
            "action_trace_by_round": action_trace_by_round,
            "worker_id": worker_id,
            "server_endpoint": f"{host}:{port}",
        }
        with (episode_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        results.append(record)
        progress_q.put(1)

    return results


def _build_summary(cfg: EvalConfig, episodes: list[Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    bench_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    episode_level_values = sorted({int(ep.level) for ep in episodes})
    if cfg.dataset and cfg.dataset.stat_levels is not None:
        stat_levels = sorted({int(x) for x in cfg.dataset.stat_levels})
    else:
        stat_levels = episode_level_values
    if not stat_levels:
        raise ValueError("No valid levels found. Please check dataset.stat_levels / dataset source.")

    level_to_idx = {lvl: idx for idx, lvl in enumerate(stat_levels)}
    level_count = [0 for _ in stat_levels]
    level_success = [0 for _ in stat_levels]
    per_bench_level_count: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    per_bench_level_success: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    skipped_level_count: dict[int, int] = defaultdict(int)
    step_success = [0] * 11

    for r in results:
        bench = str(r["bench_name"])
        level = int(r["level"])
        success = bool(r["success"])
        rounds = int(r["inference_rounds_used"])
        bench_stats[bench]["total"] += 1
        bench_stats[bench]["success"] += int(success)

        if level in level_to_idx:
            idx = level_to_idx[level]
            level_count[idx] += 1
            per_bench_level_count[bench][idx] += 1
            if success:
                level_success[idx] += 1
                per_bench_level_success[bench][idx] += 1
                if rounds <= 10:
                    step_success[rounds] += 1
        else:
            skipped_level_count[level] += 1

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
    if skipped_level_count:
        summary["skipped_level_count"] = dict(skipped_level_count)
    return summary


def run_parallel_eval(parallel_cfg: ParallelEvalConfig) -> None:
    cfg = load_eval_config(parallel_cfg.eval_config_path)
    if cfg.dataset is None:
        raise ValueError("dataset config is required")

    # output_dir 默认继承原始 eval yaml，可被 parallel yaml 覆盖
    output_dir = parallel_cfg.output_dir if parallel_cfg.output_dir else cfg.output_dir
    cfg.output_dir = output_dir

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.dataset.seed)

    if cfg.dataset.auto_generate_val_parquet:
        target_parquet = cfg.dataset.val_parquet or cfg.dataset.generated_val_parquet_path
        if not target_parquet:
            target_parquet = str(out_root / "generated_val.parquet")
        build_hstar_val_parquet(
            dataset_root=cfg.dataset.root,
            benches=cfg.dataset.benches,
            output_parquet_path=target_parquet,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
            split="test",
            env_name="hstar",
            bench_shuffle_seed=cfg.dataset.seed,
            test_size_by_bench=cfg.dataset.parquet_test_size_by_bench,
        )
        cfg.dataset.val_parquet = target_parquet
        logging.info("Generated val parquet at: %s", target_parquet)

    if cfg.dataset.val_parquet:
        episodes = load_hstar_episodes_from_parquet(
            val_parquet_path=cfg.dataset.val_parquet,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
        )
    else:
        episodes = load_hstar_episodes(
            dataset_root=cfg.dataset.root,
            benches=cfg.dataset.benches,
            default_hps_initial_yaws=cfg.dataset.hps_default_initial_yaws,
            init_pitch=cfg.dataset.init_pitch,
        )
    if cfg.dataset.max_episodes is not None:
        episodes = episodes[: cfg.dataset.max_episodes]
    if not episodes:
        raise RuntimeError("No episodes found. Please check dataset_root / benches config.")

    default_host = cfg.server.host
    default_port = cfg.server.port  # 默认继承原始 eval yaml 的 server.port
    endpoints, launched_processes = _launch_model_servers(
        ms_cfg=parallel_cfg.model_servers,
        default_port=default_port,
        default_host=default_host,
    )
    if not endpoints:
        raise RuntimeError("No server endpoints available.")

    logging.info("Using %d model server endpoints.", len(endpoints))
    episode_splits = _split_episodes_evenly(episodes, len(endpoints))
    worker_endpoints = endpoints[: len(episode_splits)]
    logging.info("Prepared %d workers for %d episodes.", len(worker_endpoints), len(episodes))

    ctx = mp.get_context(parallel_cfg.mp_start_method)
    progress_q = ctx.Queue()
    all_results: list[dict[str, Any]] = []

    try:
        with ProcessPoolExecutor(max_workers=len(worker_endpoints), mp_context=ctx) as executor:
            futures = []
            for worker_id, (eps, endpoint) in enumerate(zip(episode_splits, worker_endpoints)):
                futures.append(
                    executor.submit(
                        _run_worker_eval,
                        worker_id,
                        eps,
                        cfg,
                        endpoint,
                        str(out_root),
                        progress_q,
                    )
                )

            pending = set(futures)
            with tqdm(total=len(episodes), desc="HSTAR parallel eval", unit="episode") as pbar:
                while pending:
                    done_now = {f for f in pending if f.done()}
                    for f in done_now:
                        all_results.extend(f.result())
                    pending -= done_now

                    updated = False
                    while True:
                        try:
                            pbar.update(int(progress_q.get_nowait()))
                            updated = True
                        except queue.Empty:
                            break
                    if not updated and pending:
                        try:
                            pbar.update(int(progress_q.get(timeout=0.2)))
                        except queue.Empty:
                            pass

                while True:
                    try:
                        pbar.update(int(progress_q.get_nowait()))
                    except queue.Empty:
                        break
    finally:
        if parallel_cfg.model_servers.mode.strip().lower() == "launch" and parallel_cfg.model_servers.auto_shutdown:
            _terminate_model_servers(launched_processes)

    summary = _build_summary(cfg, episodes, all_results)
    all_results.sort(key=lambda x: str(x["episode_id"]))

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_root / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with (out_root / "hstar_style_summary.txt").open("w", encoding="utf-8") as f:
        stat_levels = summary["hstar_style"]["stat_levels"]
        level_count = summary["hstar_style"]["level_count"]
        level_success = summary["hstar_style"]["level_success"]
        for i, lvl in enumerate(stat_levels):
            f.write(f"{lvl} {level_success[i]} {level_count[i]}\n")
        f.write("step_success " + " ".join(map(str, summary["hstar_style"]["step_success"])) + "\n")

    logging.info(
        "HSTAR parallel eval done: %d / %d (%.4f)",
        summary["total_success"],
        summary["total_episodes"],
        summary["success_rate"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel online HSTAR evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to parallel YAML config")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args()
    parallel_cfg = load_parallel_config(args.config)
    run_parallel_eval(parallel_cfg)
