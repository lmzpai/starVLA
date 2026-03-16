import json
import math
import ast
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def normalize_yaw_deg(yaw: float) -> float:
    return float((yaw + 360.0) % 360.0)


def clamp_pitch_deg(pitch: float) -> float:
    return float(max(-90.0, min(90.0, pitch)))


def shortest_yaw_distance_deg(a: float, b: float) -> float:
    a = normalize_yaw_deg(a)
    b = normalize_yaw_deg(b)
    d = abs(a - b)
    return min(d, 360.0 - d)


def in_yaw_range(yaw: float, yaw_range: tuple[float, float], tolerance: float = 0.0) -> bool:
    left = normalize_yaw_deg(yaw_range[0] - tolerance)
    right = normalize_yaw_deg(yaw_range[1] + tolerance)
    yaw = normalize_yaw_deg(yaw)
    if left <= right:
        return left <= yaw <= right
    return yaw >= left or yaw <= right


def in_pitch_range(pitch: float, pitch_range: tuple[float, float], tolerance: float = 0.0) -> bool:
    low = max(-90.0, pitch_range[0] - tolerance)
    high = min(90.0, pitch_range[1] + tolerance)
    return low <= pitch <= high


def expand_target_ranges_like_hstar(
    yaw_range: tuple[float, float],
    pitch_range: tuple[float, float],
    yaw_tolerance: float,
    pitch_tolerance: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    yaw_l, yaw_r = float(yaw_range[0]), float(yaw_range[1])
    pitch_l, pitch_r = float(pitch_range[0]), float(pitch_range[1])

    if yaw_l > yaw_r:
        yaw_r = yaw_r + 360.0

    if (yaw_r - yaw_l) < 2.0 * yaw_tolerance:
        yaw_c = (yaw_l + yaw_r) / 2.0
        new_yaw = (normalize_yaw_deg(yaw_c - yaw_tolerance), normalize_yaw_deg(yaw_c + yaw_tolerance))
    else:
        new_yaw = (normalize_yaw_deg(yaw_l), normalize_yaw_deg(yaw_r))

    if (pitch_r - pitch_l) < 2.0 * pitch_tolerance:
        pitch_c = (pitch_l + pitch_r) / 2.0
        new_pitch = (
            max(-90.0, pitch_c - pitch_tolerance),
            min(90.0, pitch_c + pitch_tolerance),
        )
    else:
        new_pitch = (pitch_l, pitch_r)

    return new_yaw, new_pitch


@dataclass
class CameraConfig:
    pano_max_long_side: int = 2048
    front_width: int = 640
    front_height: int = 480
    fov_deg: float = 90.0
    fx: float | None = None
    draw_center_cross: bool = True


@dataclass
class EpisodeSpec:
    bench_name: str
    scene_id: str
    task_id: int
    level: int
    instruction: str
    pano_path: str
    init_yaw: float
    init_pitch: float
    target_yaw: tuple[float, float]
    target_pitch: tuple[float, float]

    @property
    def episode_id(self) -> str:
        return (
            f"{self.bench_name}_scene{self.scene_id}_task{self.task_id}"
            f"_initYaw{int(round(self.init_yaw))}_initPitch{int(round(self.init_pitch))}"
        )


def _find_pano_image(scene_dir: Path) -> Path:
    candidates = [p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not candidates:
        raise FileNotFoundError(f"No pano image found in {scene_dir}")
    return sorted(candidates)[0]


def _build_episodes_from_bench_dir(
    bench: str,
    bench_dir: Path,
    default_hps_initial_yaws: list[float],
    init_pitch: float,
) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    for scene_dir in sorted([p for p in bench_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        anno_path = scene_dir / "annotation.json"
        if not anno_path.exists():
            continue
        pano_path = _find_pano_image(scene_dir)
        with anno_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
        if not isinstance(annotation, list):
            continue

        for task_id, item in enumerate(annotation):
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("task", ""))
            yaw = item.get("yaw", [0, 0])
            pitch = item.get("pitch", [-90, 90])
            yaw_range = (float(yaw[0]), float(yaw[1])) if isinstance(yaw, list) and len(yaw) >= 2 else (0.0, 0.0)
            pitch_range = (
                (float(pitch[0]), float(pitch[1])) if isinstance(pitch, list) and len(pitch) >= 2 else (-90.0, 90.0)
            )

            if "initial yaw" in item:
                init_yaws = item.get("initial yaw", [])
                levels = item.get("level", [0] * len(init_yaws))
                # 对齐 hstar 原始逻辑：使用 zip，长度不一致时以最短为准。
                for start_yaw, level in zip(init_yaws, levels):
                    episodes.append(
                        EpisodeSpec(
                            bench_name=bench,
                            scene_id=scene_dir.name,
                            task_id=task_id,
                            level=int(level),
                            instruction=instruction,
                            pano_path=str(pano_path),
                            init_yaw=float(start_yaw),
                            init_pitch=float(init_pitch),
                            target_yaw=yaw_range,
                            target_pitch=pitch_range,
                        )
                    )
            else:
                level = int(item.get("level", 0))
                for start_yaw in default_hps_initial_yaws:
                    episodes.append(
                        EpisodeSpec(
                            bench_name=bench,
                            scene_id=scene_dir.name,
                            task_id=task_id,
                            level=level,
                            instruction=instruction,
                            pano_path=str(pano_path),
                            init_yaw=float(start_yaw),
                            init_pitch=float(init_pitch),
                            target_yaw=yaw_range,
                            target_pitch=pitch_range,
                        )
                    )
    return episodes


def load_hstar_episodes(
    dataset_root: str,
    benches: list[str],
    default_hps_initial_yaws: list[float],
    init_pitch: float = 0.0,
) -> list[EpisodeSpec]:
    root = Path(dataset_root)
    episodes: list[EpisodeSpec] = []

    for bench in benches:
        bench_dir = root / f"{bench}_bench"
        if not bench_dir.exists():
            continue
        episodes.extend(
            _build_episodes_from_bench_dir(
                bench=bench,
                bench_dir=bench_dir,
                default_hps_initial_yaws=default_hps_initial_yaws,
                init_pitch=init_pitch,
            )
        )

    return episodes


def _safe_parse_extra_info(raw_extra: Any) -> dict[str, Any]:
    if isinstance(raw_extra, dict):
        return raw_extra
    if isinstance(raw_extra, str):
        if not raw_extra.strip():
            return {}
        try:
            parsed = ast.literal_eval(raw_extra)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            return {}
    return {}


def _infer_bench_name_from_data_path(data_path: str) -> str:
    bench_dir_name = Path(data_path).name.strip().lower()
    if bench_dir_name.endswith("_bench"):
        return bench_dir_name[: -len("_bench")]
    return bench_dir_name


def load_hstar_episodes_from_parquet(
    val_parquet_path: str,
    default_hps_initial_yaws: list[float],
    init_pitch: float = 0.0,
) -> list[EpisodeSpec]:
    """
    按 hstar 的 parquet + seed 语义恢复评测样本顺序：
    - 每行通过 extra_info 提供 data_path/env_config/seed；
    - 对每个 data_path 先构建完整 bench episodes，再按 env_config.seed 做固定 shuffle；
    - 最后使用 row.seed % len(episodes) 取单条 episode，与 HstarEnv.reset(seed=...) 行为一致。
    """
    import pandas as pd

    df = pd.read_parquet(val_parquet_path)
    episodes: list[EpisodeSpec] = []
    cache: dict[tuple[str, int], list[EpisodeSpec]] = {}

    for _, row in df.iterrows():
        extra = _safe_parse_extra_info(row.get("extra_info", {}))
        env_cfg = extra.get("env_config", {}) if isinstance(extra.get("env_config", {}), dict) else {}
        data_path = str(env_cfg.get("data_path", "")).strip()
        if not data_path:
            continue
        bench_dir = Path(data_path)
        if not bench_dir.exists():
            continue

        bench_name = _infer_bench_name_from_data_path(data_path)
        shuffle_seed = int(env_cfg.get("seed", 42))
        cache_key = (str(bench_dir.resolve()), shuffle_seed)
        if cache_key not in cache:
            bench_eps = _build_episodes_from_bench_dir(
                bench=bench_name,
                bench_dir=bench_dir,
                default_hps_initial_yaws=default_hps_initial_yaws,
                init_pitch=init_pitch,
            )
            # 对齐 hstar 原始环境：random.seed + random.shuffle
            rng = random.Random(shuffle_seed)
            rng.shuffle(bench_eps)
            cache[cache_key] = bench_eps

        bench_eps = cache[cache_key]
        if not bench_eps:
            continue
        row_seed = int(extra.get("seed", 42))
        episodes.append(bench_eps[row_seed % len(bench_eps)])

    return episodes


def build_hstar_val_parquet(
    dataset_root: str,
    benches: list[str],
    output_parquet_path: str,
    default_hps_initial_yaws: list[float],
    init_pitch: float = 0.0,
    split: str = "test",
    env_name: str = "hstar",
    bench_shuffle_seed: int = 42,
    test_size_by_bench: dict[str, int] | None = None,
) -> str:
    """
    在 starVLA 内生成与 hstar create_dataset 语义兼容的 val parquet：
    - 每条记录使用 extra_info 组织 env_name/env_config/seed；
    - env_config 里写入 data_path 与 seed（用于后续恢复 episode 顺序）。
    """
    import pandas as pd

    root = Path(dataset_root)
    records: list[dict[str, Any]] = []
    for bench in benches:
        bench_dir = root / f"{bench}_bench"
        if not bench_dir.exists():
            continue

        bench_episodes = _build_episodes_from_bench_dir(
            bench=bench,
            bench_dir=bench_dir,
            default_hps_initial_yaws=default_hps_initial_yaws,
            init_pitch=init_pitch,
        )
        default_size = len(bench_episodes)
        bench_size = default_size if test_size_by_bench is None else int(test_size_by_bench.get(bench, default_size))
        if bench_size <= 0:
            continue

        env_config = {
            "data_path": str(bench_dir),
            "seed": int(bench_shuffle_seed),
        }
        for seed in range(bench_size):
            records.append(
                {
                    "data_source": env_name,
                    "prompt": [{"role": "user", "content": ""}],
                    "extra_info": {
                        "split": split,
                        "env_name": env_name,
                        "env_config": env_config,
                        "seed": int(seed),
                    },
                }
            )

    out_path = Path(output_parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(str(out_path), index=False)
    return str(out_path)


class PanoProjector:
    def __init__(self, camera_cfg: CameraConfig):
        self.cfg = camera_cfg
        self._fov_h_deg, self._fov_v_deg = self._compute_fov()

    def _compute_fov(self) -> tuple[float, float]:
        if self.cfg.fx is not None and self.cfg.fx > 0:
            fov_h_rad = 2.0 * math.atan(self.cfg.front_width / (2.0 * float(self.cfg.fx)))
        else:
            fov_h_rad = math.radians(float(self.cfg.fov_deg))

        fov_v_rad = 2.0 * math.atan(
            math.tan(fov_h_rad / 2.0) * (float(self.cfg.front_height) / float(self.cfg.front_width))
        )
        return math.degrees(fov_h_rad), math.degrees(fov_v_rad)

    @property
    def fov_h_deg(self) -> float:
        return self._fov_h_deg

    @property
    def fov_v_deg(self) -> float:
        return self._fov_v_deg

    def render(self, pano_bgr: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
        h_out, w_out = self.cfg.front_height, self.cfg.front_width

        fov_h = math.radians(self._fov_h_deg)
        fov_v = math.radians(self._fov_v_deg)
        focal_x = w_out / (2.0 * math.tan(fov_h / 2.0))
        focal_y = h_out / (2.0 * math.tan(fov_v / 2.0))

        ii, jj = np.meshgrid(np.arange(w_out), np.arange(h_out))
        x = (ii - w_out / 2.0) / focal_x
        y = (jj - h_out / 2.0) / focal_y
        z = np.ones_like(x)
        norm = np.sqrt(x * x + y * y + z * z)
        x, y, z = x / norm, y / norm, z / norm

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        rx = np.array(
            [[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]]
        )
        r = ry @ rx

        xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3).T
        x2, y2, z2 = r @ xyz
        lon = np.arctan2(x2, z2)
        lat = np.arcsin(np.clip(y2, -1.0, 1.0))

        pano_h, pano_w = pano_bgr.shape[:2]
        map_x = ((lon / np.pi + 1.0) * 0.5 * pano_w).reshape(h_out, w_out).astype(np.float32)
        map_y = ((lat / np.pi + 0.5) * pano_h).reshape(h_out, w_out).astype(np.float32)
        out = cv2.remap(pano_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        if self.cfg.draw_center_cross:
            cy, cx = h_out // 2, w_out // 2
            cv2.line(out, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
            cv2.line(out, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)
        return out


class HStarOnlineEnv:
    def __init__(self, episode: EpisodeSpec, camera_cfg: CameraConfig):
        self.episode = episode
        self.camera_cfg = camera_cfg
        self.projector = PanoProjector(camera_cfg)

        pano = cv2.imread(episode.pano_path, cv2.IMREAD_COLOR)
        if pano is None:
            raise RuntimeError(f"Failed to read pano image: {episode.pano_path}")
        self.pano_bgr = self._resize_long_side(pano, camera_cfg.pano_max_long_side)

        self.yaw = float(episode.init_yaw)
        self.pitch = float(episode.init_pitch)

    def _resize_long_side(self, image: np.ndarray, max_long_side: int) -> np.ndarray:
        h, w = image.shape[:2]
        long_side = max(h, w)
        if long_side <= max_long_side:
            return image
        scale = float(max_long_side) / float(long_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def reset(self) -> np.ndarray:
        self.yaw = float(self.episode.init_yaw)
        self.pitch = float(self.episode.init_pitch)
        return self._render_rgb()

    def _render_rgb(self) -> np.ndarray:
        view_bgr = self.projector.render(self.pano_bgr, yaw_deg=self.yaw, pitch_deg=self.pitch)
        return cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)

    def step(self, delta_yaw_deg: float, delta_pitch_deg: float) -> np.ndarray:
        self.yaw = normalize_yaw_deg(self.yaw + float(delta_yaw_deg))
        self.pitch = clamp_pitch_deg(self.pitch + float(delta_pitch_deg))
        return self._render_rgb()

    def is_success(self, yaw_tolerance: float = 0.0, pitch_tolerance: float = 0.0) -> bool:
        eval_yaw_range, eval_pitch_range = expand_target_ranges_like_hstar(
            yaw_range=self.episode.target_yaw,
            pitch_range=self.episode.target_pitch,
            yaw_tolerance=yaw_tolerance,
            pitch_tolerance=pitch_tolerance,
        )
        yaw_ok = in_yaw_range(self.yaw, eval_yaw_range, tolerance=0.0)
        pitch_ok = in_pitch_range(self.pitch, eval_pitch_range, tolerance=0.0)
        return bool(yaw_ok and pitch_ok)

    def get_state(self) -> dict[str, Any]:
        return {
            "yaw": float(self.yaw),
            "pitch": float(self.pitch),
            "target_yaw": list(self.episode.target_yaw),
            "target_pitch": list(self.episode.target_pitch),
            "level": int(self.episode.level),
        }
