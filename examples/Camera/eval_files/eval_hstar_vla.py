import argparse
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

ACTION_MAX_MAG_RAD = float(4.0 * np.pi)
PANO_DELTA_MAX_DEG = 180.0


def action_space_to_pano_delta_deg(actions: np.ndarray) -> np.ndarray:
    """
    将模型动作空间值（默认 [-4π, 4π]）反变换为 pano 旋转角度增量（度）。
    """
    clipped = np.clip(actions, -ACTION_MAX_MAG_RAD, ACTION_MAX_MAG_RAD)
    return clipped / ACTION_MAX_MAG_RAD * PANO_DELTA_MAX_DEG


@dataclass
class DatasetConfig:
    root: str
    benches: list[str] = field(default_factory=lambda: ["hos", "hps"])
    val_parquet: str | None = None
    auto_generate_val_parquet: bool = False
    generated_val_parquet_path: str | None = None
    parquet_test_size_by_bench: dict[str, int] | None = None
    max_episodes: int | None = None
    seed: int = 42
    init_pitch: float = 0.0
    hps_default_initial_yaws: list[float] = field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    # 若不显式配置，则按当前 episodes 中实际出现的 level 自动推断。
    stat_levels: list[int] | None = None


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 10093


@dataclass
class PolicyConfig:
    action_chunk_size: int = 4
    infer_action_num: int = 4
    stop_zero_tail: int = 8
    stop_zero_eps: float = 0.05


@dataclass
class RolloutConfig:
    max_inference_rounds: int = 20
    history_max_frames: int = 10


@dataclass
class BenchToleranceConfig:
    yaw_tolerance: float = 30.0
    pitch_tolerance: float = 20.0


@dataclass
class ToleranceConfig:
    hos: BenchToleranceConfig = field(default_factory=lambda: BenchToleranceConfig(yaw_tolerance=30.0, pitch_tolerance=20.0))
    hps: BenchToleranceConfig = field(default_factory=lambda: BenchToleranceConfig(yaw_tolerance=10.0, pitch_tolerance=20.0))


@dataclass
class VisualizationConfig:
    enable: bool = False
    video_fps: int = 20


@dataclass
class EvalConfig:
    output_dir: str = "experiments/hstar_eval"
    dataset: DatasetConfig | None = None
    server: ServerConfig = field(default_factory=ServerConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    tolerance: ToleranceConfig = field(default_factory=ToleranceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: str) -> EvalConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if "dataset" not in raw or not isinstance(raw["dataset"], dict):
        raise ValueError("Config must contain top-level `dataset` section.")

    dataset = DatasetConfig(**raw.get("dataset", {}))
    server = ServerConfig(**raw.get("server", {}))
    policy = PolicyConfig(**raw.get("policy", {}))
    rollout = RolloutConfig(**raw.get("rollout", {}))

    tol_raw = raw.get("tolerance", {})
    tolerance = ToleranceConfig(
        hos=BenchToleranceConfig(**tol_raw.get("hos", {})),
        hps=BenchToleranceConfig(**tol_raw.get("hps", {})),
    )

    camera = CameraConfig(**raw.get("camera", {}))
    visualization = VisualizationConfig(**raw.get("visualization", {}))
    return EvalConfig(
        output_dir=raw.get("output_dir", "experiments/hstar_eval"),
        dataset=dataset,
        server=server,
        policy=policy,
        rollout=rollout,
        tolerance=tolerance,
        camera=camera,
        visualization=visualization,
    )


def should_stop_by_zero_tail(
    action_chunk: np.ndarray,
    infer_action_num: int,
    tail: int,
    eps: float,
) -> bool:
    """
    判断是否根据“尾部接近 0”提前停止。
    语义：
    - 只考虑“将要执行的前 infer_action_num 个动作”；
    - 在这些将要执行的动作中，取其“尾部 tail 个”（不足则取全部）；
    - 要求尾部这些动作中，每一步的 yaw / pitch 分量都在 [-eps, eps] 之内。
    """
    if infer_action_num <= 0 or tail <= 0 or action_chunk.shape[0] <= 0:
        return False

    exec_num = min(infer_action_num, action_chunk.shape[0])
    if exec_num <= 0:
        return False

    # 只看将要执行的动作
    exec_actions = action_chunk[:exec_num, :]  # (exec_num, 2)

    # 再从中取尾部 tail 个（不足则取全部）
    tail_len = min(tail, exec_num)
    tail_actions = exec_actions[-tail_len:, :]  # (tail_len, 2)

    # 要求尾部所有动作在两个维度上都“几乎为 0”
    return bool(np.all(np.abs(tail_actions) <= float(eps)))


def _save_keyframes(frames: list[np.ndarray], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for i, frame_rgb in enumerate(frames):
        p = out_dir / f"frame_{i:04d}.jpg"
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(p), frame_bgr)
        saved_paths.append(str(p))
    return saved_paths


def _get_bench_tolerance(cfg: EvalConfig, bench_name: str) -> BenchToleranceConfig:
    if bench_name == "hos":
        return cfg.tolerance.hos
    if bench_name == "hps":
        return cfg.tolerance.hps
    return BenchToleranceConfig()


def run_eval(cfg: EvalConfig) -> None:
    if cfg.dataset is None:
        raise ValueError("dataset config is required")

    np.random.seed(cfg.dataset.seed)
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    camera_cfg = cfg.camera
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

    client = WebsocketClientPolicy(host=cfg.server.host, port=cfg.server.port)
    
    results: list[dict[str, Any]] = []
    bench_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
    episode_level_values = sorted({int(ep.level) for ep in episodes})
    if cfg.dataset.stat_levels is None:
        stat_levels = episode_level_values
    else:
        stat_levels = sorted({int(x) for x in cfg.dataset.stat_levels})
    if not stat_levels:
        raise ValueError("No valid levels found. Please check dataset.stat_levels / dataset source.")
    logging.info("Episode level distribution (pre-eval): %s", {lvl: sum(1 for ep in episodes if int(ep.level) == lvl) for lvl in episode_level_values})
    level_to_idx = {lvl: idx for idx, lvl in enumerate(stat_levels)}
    level_count = [0 for _ in stat_levels]
    level_success = [0 for _ in stat_levels]
    # 按 bench（hos/hps）分别统计各个 level 的成功率
    per_bench_level_count: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    per_bench_level_success: dict[str, list[int]] = defaultdict(lambda: [0 for _ in stat_levels])
    skipped_level_count: dict[int, int] = defaultdict(int)
    step_success = [0] * 11

    for ep in tqdm(episodes, desc="HSTAR online eval", unit="episode"):
        env = HStarOnlineEnv(episode=ep, camera_cfg=camera_cfg)
        obs = env.reset()

        episode_dir = out_root / "task" / ep.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        history_queue: deque[np.ndarray] = deque(maxlen=cfg.rollout.history_max_frames)
        all_frames: list[np.ndarray] = [obs]  # 每步一帧，用于视频
        keyframe_frames: list[np.ndarray] = [obs]  # 仅每个 action chunk 执行完后的帧，用于 keyframes 文件夹

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
        # 结构：list[round_record]；每个 round_record 是一个 dict，里面的 action_* 字段都是 (exec_num, 2) 的二维列表
        action_trace_by_round: list[dict[str, Any]] = []
        executed_action_steps = 0

        while inference_round < cfg.rollout.max_inference_rounds:
            # 本轮将产生的 action 序列（一个 action chunk 内的若干步）
            model_inputs = {"image": list(history_queue) + [obs], "lang": ep.instruction}
            response = client.predict_action({"type": "predict_action", "examples": [model_inputs]})
            data = response.get("data", {})
            # NOTE: 这里的 `normalized_actions` 约定为动作空间值，范围 [-4π, 4π]。
            #       执行前需反变换成 pano 的角度增量（度）。
            actions_rad = np.asarray(data.get("normalized_actions"), dtype=np.float32)

            if actions_rad.ndim == 3:
                actions_rad = actions_rad[0]
            
            if actions_rad.ndim != 2 or actions_rad.shape[1] != 2:
                raise ValueError(f"Unexpected normalized_actions (rad) shape: {actions_rad.shape}")

            action_chunk = np.clip(actions_rad, -ACTION_MAX_MAG_RAD, ACTION_MAX_MAG_RAD)

            # 先确定本轮实际会尝试执行的步数
            exec_num = min(cfg.policy.infer_action_num, action_chunk.shape[0])
            exec_actions_rad = action_chunk[:exec_num, :]  # (exec_num, 2)
            exec_actions_deg = action_space_to_pano_delta_deg(exec_actions_rad)

            # 基于“将要执行的动作”的尾部，判断是否可以提前停止
            if should_stop_by_zero_tail(
                action_chunk=action_chunk,
                infer_action_num=cfg.policy.infer_action_num,
                tail=cfg.policy.stop_zero_tail,
                eps=cfg.policy.stop_zero_eps,
            ):
                # 对于 stop_by_zero_tail 的情况，也先把模型本轮预测的动作记录到 metadata 里，
                # 但不实际执行（exec_num 记为 0，state_after_action 为空）。
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

            # 如果 action_chunk 步数小于 infer_action_num，就只执行前面可用的若干步
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

            # 执行完整个 action chunk 后，再把最终 obs 加入历史队列
            if exec_num > 0:
                history_queue.append(obs)

            # 如果本轮实际执行了 action，则记录到“按轮聚合”的结构中
            if exec_num > 0:
                round_record: dict[str, Any] = {
                    "round": inference_round,
                    "exec_num": exec_num,
                    # 下面三个字段都是 (exec_num, 2) 的二维列表，方便后处理
                    "action_norm": exec_actions_rad.tolist(),
                    "action_delta_rad": exec_actions_rad.tolist(),
                    "action_delta_deg": exec_actions_deg.tolist(),
                    # 实际执行后每一步的环境状态，长度为 exec_num
                    "state_after_action": round_states_after_action,
                }
                action_trace_by_round.append(round_record)

            # keyframe：仅保存本 round 执行完整个 action chunk 后的观测（即当前 obs）
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
            "yaw_tolerance": float(tol_cfg.yaw_tolerance),
            "pitch_tolerance": float(tol_cfg.pitch_tolerance),
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
        # 对于每个 bench（如 hos / hps），分别给出各 level 的统计结果
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

    with (out_root / "hstar_style_summary.txt").open("w", encoding="utf-8") as f:
        for i, lvl in enumerate(stat_levels):
            f.write(f"{lvl} {level_success[i]} {level_count[i]}\n")
        f.write("step_success " + " ".join(map(str, step_success)) + "\n")

    logging.info("HSTAR online eval done: %d / %d (%.4f)", success_num, total, summary["success_rate"])
    for i, lvl in enumerate(stat_levels):
        if level_count[i] > 0:
            logging.info(
                "Level %d (overall): %d/%d, success_rate=%.4f",
                lvl,
                level_success[i],
                level_count[i],
                level_success[i] / max(level_count[i], 1),
            )
    # 额外打印 hos / hps 在各个 level 上的成功率
    for bench in bench_stats.keys():
        b_counts = per_bench_level_count[bench]
        b_success = per_bench_level_success[bench]
        for i, lvl in enumerate(stat_levels):
            if b_counts[i] > 0:
                logging.info(
                    "Bench %s, Level %d: %d/%d, success_rate=%.4f",
                    bench,
                    lvl,
                    b_success[i],
                    b_counts[i],
                    b_success[i] / max(b_counts[i], 1),
                )
    logging.info("step_success: %s", " ".join(map(str, step_success)))
    if skipped_level_count:
        logging.warning("Levels not in stat_levels were skipped in hstar_style stats: %s", dict(skipped_level_count))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online HSTAR evaluation for VLA/fake policy server")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_parser().parse_args()
    config = load_config(args.config)
    run_eval(config)
