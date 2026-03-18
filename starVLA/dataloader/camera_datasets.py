from __future__ import annotations

import bisect
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch):
    return batch


def _cfg_get(data_cfg, key, default=None):
    getter = getattr(data_cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(data_cfg, key, default)


def _to_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"false", "0", "no", ""}
    return bool(value)


def _source_cfg_get(source_cfg, global_cfg, key, default=None):
    source_value = _cfg_get(source_cfg, key, None)
    if source_value is not None:
        return source_value
    if global_cfg is not None:
        return _cfg_get(global_cfg, key, default)
    return default


def _infer_num_steps_from_metadata(item: dict) -> Optional[int]:
    candidate_keys = ("frame_count", "num_steps", "action_length", "num_actions", "length")
    for key in candidate_keys:
        value = item.get(key, None)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _normalize_action_matrix(
    parquet_path: Path,
    action_dim: Optional[int] = None,
    action_columns: Optional[List[str]] = None,
) -> np.ndarray:
    df = pd.read_parquet(parquet_path)

    if action_columns:
        cols = [col for col in action_columns if col in df.columns]
        if not cols:
            raise ValueError(f"Action columns {action_columns} not found in {parquet_path}. Available: {list(df.columns)}")
        if len(cols) == 1 and not np.issubdtype(df[cols[0]].dtype, np.number):
            matrix = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df[cols[0]]], axis=0)
        else:
            matrix = np.stack([df[col].to_numpy(dtype=np.float32) for col in cols], axis=-1)
    elif "actions" in df.columns:
        matrix = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df["actions"]], axis=0)
    elif "action" in df.columns:
        matrix = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in df["action"]], axis=0)
    else:
        numeric_cols = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
        if not numeric_cols:
            raise ValueError(f"No supported action columns found in {parquet_path}. Columns: {list(df.columns)}")
        matrix = df[numeric_cols].to_numpy(dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"Expected action matrix to be 2D, got shape {matrix.shape} from {parquet_path}")

    if action_dim is not None and matrix.shape[1] != action_dim:
        if matrix.shape[1] > action_dim:
            matrix = matrix[:, :action_dim]
        else:
            matrix = np.pad(matrix, ((0, 0), (0, action_dim - matrix.shape[1])), mode="constant")

    return matrix.astype(np.float32)


def _get_frames_by_indices(video_path: str, indices: List[int], video_backend: str = "opencv") -> np.ndarray:
    backend = str(video_backend).lower()

    if backend == "decord":
        try:
            import decord
        except ImportError as exc:
            raise ImportError("video_backend='decord' requested, but decord is not installed.") from exc

        vr = decord.VideoReader(video_path)
        return vr.get_batch(indices).asnumpy()

    if backend in {"opencv", "cv2", "pyav"}:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames = []
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return np.asarray(frames)

    raise ValueError(f"Unsupported video backend: {video_backend}")


class _HStarPanoSourceDataset(Dataset):
    def __init__(self, source_name: str, source_cfg, global_cfg=None):
        self.source_name = source_name
        self.global_cfg = global_cfg
        self.root = Path(_source_cfg_get(source_cfg, global_cfg, "data_root_dir"))
        self.metadata_path = self.root / _source_cfg_get(source_cfg, global_cfg, "metadata_filename", "metadata.json")
        self.max_history_images = max(1, int(_source_cfg_get(source_cfg, global_cfg, "max_history_images", 6)))
        self.min_history_images = max(1, int(_source_cfg_get(source_cfg, global_cfg, "min_history_images", 1)))
        self.action_chunk_len = int(
            _source_cfg_get(
                source_cfg,
                global_cfg,
                "action_chunk_len",
                _source_cfg_get(source_cfg, global_cfg, "chunk_len", 4),
            )
        )
        self.action_dim = _source_cfg_get(source_cfg, global_cfg, "action_dim", None)
        self.state_dim = _source_cfg_get(source_cfg, global_cfg, "state_dim", None)
        self.include_state = _to_bool(_source_cfg_get(source_cfg, global_cfg, "include_state", False))
        self.video_backend = _source_cfg_get(source_cfg, global_cfg, "video_backend", "opencv")
        self.action_columns = _source_cfg_get(source_cfg, global_cfg, "action_columns", None)
        self.history_mode = str(_source_cfg_get(source_cfg, global_cfg, "history_mode", "random")).lower()
        self.history_stride = max(1, int(_source_cfg_get(source_cfg, global_cfg, "history_stride", 1)))
        self.sample_stride = max(1, int(_source_cfg_get(source_cfg, global_cfg, "sample_stride", 1)))
        self.preserve_image_size = _to_bool(_source_cfg_get(source_cfg, global_cfg, "preserve_image_size", True), default=True)
        self.skip_file_existence_check = _to_bool(
            _source_cfg_get(source_cfg, global_cfg, "skip_file_existence_check", False),
            default=False,
        )

        image_size = _source_cfg_get(source_cfg, global_cfg, "image_size", None)
        self.image_size = tuple(image_size) if image_size is not None else None

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        self.items = raw_items if isinstance(raw_items, list) else [raw_items]

        self.valid_items = []
        self.item_sample_counts = []
        self.cumulative_sample_counts = []
        self._action_cache = {}
        running = 0

        for item in self.items:
            video_path = self.root / item["video_path"]
            action_path = self.root / item["action_path"]
            if not self.skip_file_existence_check and (not video_path.exists() or not action_path.exists()):
                continue

            effective_len = _infer_num_steps_from_metadata(item)
            if effective_len is None or effective_len <= 0:
                # Fallback for incomplete metadata. This is slower, but keeps compatibility.
                action_matrix = _normalize_action_matrix(
                    action_path,
                    action_dim=self.action_dim,
                    action_columns=self.action_columns,
                )
                effective_len = int(action_matrix.shape[0])
            if effective_len <= 0:
                continue

            packed = dict(item)
            packed["_video_path"] = video_path
            packed["_action_path"] = action_path
            packed["_num_steps"] = effective_len
            self.valid_items.append(packed)

            sample_count = (effective_len + self.sample_stride - 1) // self.sample_stride
            self.item_sample_counts.append(sample_count)
            running += sample_count
            self.cumulative_sample_counts.append(running)

    def __len__(self) -> int:
        return self.cumulative_sample_counts[-1] if self.cumulative_sample_counts else 0

    def _load_action_matrix(self, item: dict) -> np.ndarray:
        action_path = item["_action_path"]
        cache_key = str(action_path)
        if cache_key not in self._action_cache:
            self._action_cache[cache_key] = _normalize_action_matrix(
                action_path,
                action_dim=self.action_dim,
                action_columns=self.action_columns,
            )
        return self._action_cache[cache_key]

    def _history_length_for_step(self, step_idx: int) -> int:
        max_available = min(self.max_history_images, step_idx // self.history_stride + 1)
        min_available = min(self.min_history_images, max_available)
        if self.history_mode == "random":
            return random.randint(min_available, max_available)
        elif self.history_mode == "min":
            return min_available
        elif self.history_mode == "max":
            return max_available
        else:
            raise ValueError(f"Invalid history mode: {self.history_mode}")

    def _get_history_images(self, item: dict, step_idx: int) -> List[Image.Image]:
        history_len = self._history_length_for_step(step_idx)
        history_start = step_idx - (history_len - 1) * self.history_stride
        indices = list(range(history_start, step_idx + 1, self.history_stride))

        frames = _get_frames_by_indices(
            str(item["_video_path"]),
            indices,
            video_backend=self.video_backend,
        )
        if len(frames) != len(indices):
            raise RuntimeError(f"Expected {len(indices)} frames, got {len(frames)} for {item['_video_path']}")

        images = []
        for frame in frames:
            image = Image.fromarray(frame)
            if not self.preserve_image_size and self.image_size is not None:
                image = image.resize(self.image_size)
            images.append(image)
        return images

    def _get_action_chunk(self, action_matrix: np.ndarray, step_idx: int) -> np.ndarray:
        chunk = action_matrix[step_idx : step_idx + self.action_chunk_len]
        if chunk.shape[0] == self.action_chunk_len:
            return chunk.astype(np.float16)

        if chunk.shape[0] == 0:
            pad_frame = action_matrix[-1]
            chunk = np.repeat(pad_frame[None, :], self.action_chunk_len, axis=0)
        else:
            # NOTE(zhouenshen): 当前能这么做是因为我手动补齐了最后相机不动数据
            pad_frame = chunk[-1]
            pad_len = self.action_chunk_len - chunk.shape[0]
            chunk = np.concatenate([chunk, np.repeat(pad_frame[None, :], pad_len, axis=0)], axis=0)
        return chunk.astype(np.float16)

    def _build_state(self, item: dict) -> np.ndarray:
        """
        Currently, we don't have state in the dataset, so we return an empty state.
        """
        state_values = []
        if "initial_yaw" in item:
            state_values.append(float(item["initial_yaw"]))
        if "initial_pitch" in item:
            state_values.append(float(item["initial_pitch"]))
        while len(state_values) < (self.state_dim or len(state_values)):
            state_values.append(0.0)
        return np.asarray(state_values[: self.state_dim], dtype=np.float16).reshape(1, -1)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        item_idx = bisect.bisect_right(self.cumulative_sample_counts, index)
        prev_cum = 0 if item_idx == 0 else self.cumulative_sample_counts[item_idx - 1]
        sample_idx = index - prev_cum
        step_idx = sample_idx * self.sample_stride

        item = self.valid_items[item_idx]
        action_matrix = self._load_action_matrix(item)
        if action_matrix.shape[0] <= 0:
            raise RuntimeError(f"Action matrix is empty for {item['_action_path']}")
        # Metadata length may be stale; clamp to valid range.
        step_idx = min(step_idx, action_matrix.shape[0] - 1)

        task_names = item.get("task_names", [])

        # NOTE(zhouenshen): 现在指令只有一条，后续可以考虑扩充多条
        language = task_names[0] if task_names else ""

        sample = {
            "image": self._get_history_images(item, step_idx),
            "lang": language,
            "language": language,
            "action": self._get_action_chunk(action_matrix, step_idx),
            "camera_source": self.source_name,
        }

        if self.include_state and self.state_dim:
            sample["state"] = self._build_state(item)
        return sample

    def summary(self) -> dict:
        return {
            "source_name": self.source_name,
            "dataset_type": "hstar_pano_dataset",
            "metadata_path": str(self.metadata_path),
            "num_episodes": len(self.valid_items),
            "num_samples": len(self),
            "max_history_images": self.max_history_images,
            "min_history_images": self.min_history_images,
            "history_mode": self.history_mode,
            "history_stride": self.history_stride,
            "action_chunk_len": self.action_chunk_len,
            "action_dim": self.action_dim,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "preserve_image_size": self.preserve_image_size,
        }


CAMERA_SOURCE_BUILDERS = {
    "hstar_pano_dataset": _HStarPanoSourceDataset,
}


class CameraDataset(Dataset):
    """
    Camera-centric dataset wrapper that supports multiple source datasets.

    Config style:
      camera_data:
        dataset_py: camera_datasets
        dataset_use: source_a,source_b
        dataset_sources:
          source_a:
            dataset_type: hstar_pano_dataset
            ...
    """

    def __init__(self, data_cfg):
        self.data_cfg = data_cfg
        dataset_use = _cfg_get(data_cfg, "dataset_use", "")
        if isinstance(dataset_use, str):
            source_names = [name.strip() for name in dataset_use.split(",") if name.strip()]
        else:
            source_names = list(dataset_use or [])

        if not source_names:
            raise ValueError("camera_data.dataset_use must specify at least one camera dataset source.")

        dataset_sources = _cfg_get(data_cfg, "dataset_sources", None) or {}
        if not dataset_sources and _cfg_get(data_cfg, "data_root_dir", None):
            dataset_sources = {
                source_names[0]: {
                    "dataset_type": source_names[0],
                    "data_root_dir": _cfg_get(data_cfg, "data_root_dir"),
                    "metadata_filename": _cfg_get(data_cfg, "metadata_filename", "metadata.json"),
                    "max_history_images": _cfg_get(data_cfg, "max_history_images", 9),
                    "min_history_images": _cfg_get(data_cfg, "min_history_images", 1),
                    "history_mode": _cfg_get(data_cfg, "history_mode", "random"),
                    "history_stride": _cfg_get(data_cfg, "history_stride", 1),
                    "sample_stride": _cfg_get(data_cfg, "sample_stride", 1),
                    "action_chunk_len": _cfg_get(data_cfg, "action_chunk_len", 4),
                    "action_dim": _cfg_get(data_cfg, "action_dim", None),
                    "state_dim": _cfg_get(data_cfg, "state_dim", None),
                    "include_state": _cfg_get(data_cfg, "include_state", False),
                    "video_backend": _cfg_get(data_cfg, "video_backend", "opencv"),
                    "action_columns": _cfg_get(data_cfg, "action_columns", None),
                    "image_size": _cfg_get(data_cfg, "image_size", None),
                    "preserve_image_size": _cfg_get(data_cfg, "preserve_image_size", True),
                }
            }

        self.source_names = source_names
        self.datasets = []
        self.cumulative_sizes = []
        running = 0

        for source_name in source_names:
            if source_name not in dataset_sources:
                raise KeyError(f"camera_data.dataset_sources missing config for source `{source_name}`")
            source_cfg = dataset_sources[source_name]
            dataset_type = _cfg_get(source_cfg, "dataset_type", source_name)
            if dataset_type not in CAMERA_SOURCE_BUILDERS:
                raise KeyError(f"Unsupported camera dataset type `{dataset_type}` for source `{source_name}`")

            dataset = CAMERA_SOURCE_BUILDERS[dataset_type](source_name=source_name, source_cfg=source_cfg, global_cfg=data_cfg)
            self.datasets.append(dataset)
            running += len(dataset)
            self.cumulative_sizes.append(running)

    def get_dataset_summary_for_log(self) -> List[Tuple[str, int]]:
        """返回 (数据集名称, 样本数) 的列表，便于外部打印加载信息。"""
        return [(name, len(ds)) for name, ds in zip(self.source_names, self.datasets)]

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        prev_cum = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        sample_idx = index - prev_cum
        return self.datasets[dataset_idx][sample_idx]

    def save_dataset_statistics(self, path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset_type": "camera_datasets",
            "num_sources": len(self.datasets),
            "num_samples": len(self),
            "sources": [dataset.summary() for dataset in self.datasets],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def get_camera_dataset(data_cfg, **kwargs) -> CameraDataset:
    return CameraDataset(data_cfg=data_cfg)
