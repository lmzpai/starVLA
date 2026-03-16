# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare Active Camera training data."""
    logger.info(f"Creating Active Camera Dataset with Mixture `{cfg.datasets.camera_data.dataset_use}`")
    camera_train_dataloader = build_dataloader(
        cfg=cfg,
        dataset_py=cfg.datasets.camera_data.dataset_py,
        dataset_cfg_key="camera_data",
    )

    dataset = camera_train_dataloader.dataset
    if hasattr(dataset, "get_dataset_summary_for_log"):
        summary = dataset.get_dataset_summary_for_log()
        accelerator.print("[Data] 当前使用的数据集及样本数量：")
        for name, count in summary:
            accelerator.print(f"  - {name}: {count} 条样本")
        total = len(dataset)
        accelerator.print(f"[Data] 总样本数量: {total}")
    else:
        num_samples = len(dataset)
        accelerator.print(f"[Data] 总样本数量: {num_samples}")

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return camera_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._last_saved_epoch = -1  # 用于按 epoch 保存时去重

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.camera_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self, save_as_epoch=None):
        """Save current training state.
        save_as_epoch: if set, checkpoint is named epoch_{save_as_epoch}; otherwise steps_{completed_steps}.
        """
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            if save_as_epoch is not None:
                checkpoint_name = f"epoch_{save_as_epoch}"
            else:
                checkpoint_name = f"steps_{self.completed_steps}"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

            state_dict = self.accelerator.get_state_dict(self.model)
            checkpoint_file_path = None
            if save_format == "safetensors":
                from safetensors.torch import save_file

                checkpoint_file_path = checkpoint_path + "_model.safetensors"
                save_file(state_dict, checkpoint_file_path)
            elif save_format == "pt":
                checkpoint_file_path = checkpoint_path + "_pytorch_model.pt"
                torch.save(state_dict, checkpoint_file_path)
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            if save_as_epoch is not None:
                summary_data["epoch"] = save_as_epoch
            auto_eval_summary = self._run_auto_hstar_eval_if_enabled(
                checkpoint_file_path=checkpoint_file_path,
                checkpoint_name=checkpoint_name,
            )
            if auto_eval_summary is not None:
                summary_data["auto_hstar_eval"] = auto_eval_summary
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _resolve_path(self, maybe_path: str) -> str:
        p = Path(maybe_path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return str(p)

    def _wait_tcp_ready(self, host: str, port: int, timeout_sec: float, poll_interval_sec: float = 1.0) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                try:
                    sock.connect((host, int(port)))
                    return
                except OSError:
                    time.sleep(poll_interval_sec)
        raise TimeoutError(f"Policy server not ready in time: {host}:{port}")

    def _terminate_subprocess(self, proc: subprocess.Popen, name: str) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self.accelerator.print(f"🧹 {name} stopped with code={proc.returncode}")

    def _run_auto_hstar_eval_if_enabled(self, checkpoint_file_path: str, checkpoint_name: str) -> dict[str, Any] | None:
        auto_eval_cfg = getattr(self.config.trainer, "auto_hstar_eval", None)
        if auto_eval_cfg is None or not bool(getattr(auto_eval_cfg, "enabled", False)):
            return None

        started_at = time.perf_counter()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_eval_root = getattr(auto_eval_cfg, "output_root", None)
        if auto_eval_root:
            auto_eval_root = self._resolve_path(str(auto_eval_root))
        else:
            auto_eval_root = os.path.join(self.config.output_dir, "auto_eval")
        os.makedirs(auto_eval_root, exist_ok=True)

        run_output_dir = os.path.join(auto_eval_root, f"{checkpoint_name}_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        eval_python_bin = str(getattr(auto_eval_cfg, "python_bin", "python3"))
        eval_script = self._resolve_path(
            str(
                getattr(
                    auto_eval_cfg,
                    "eval_script",
                    "examples/Camera/eval_files/eval_during_training/eval_hstar_vla.py",
                )
            )
        )
        eval_config_template = self._resolve_path(
            str(
                getattr(
                    auto_eval_cfg,
                    "eval_config",
                    "examples/Camera/eval_files/eval_during_training/hstar_eval_during_training.yaml",
                )
            )
        )
        server_script = self._resolve_path(
            str(
                getattr(
                    auto_eval_cfg,
                    "server_script",
                    "deployment/model_server/server_policy.py",
                )
            )
        )
        server_python_bin = str(getattr(auto_eval_cfg, "server_python_bin", eval_python_bin))
        server_gpu_id = getattr(auto_eval_cfg, "server_gpu_id", None)

        generated_eval_cfg_dir = os.path.join(self.config.output_dir, "auto_eval_configs")
        os.makedirs(generated_eval_cfg_dir, exist_ok=True)
        generated_eval_cfg_path = os.path.join(generated_eval_cfg_dir, f"{checkpoint_name}_{timestamp}.yaml")
        eval_log_path = os.path.join(run_output_dir, "auto_eval.log")
        server_log_path = os.path.join(run_output_dir, "policy_server.log")

        return_payload: dict[str, Any] = {
            "enabled": True,
            "checkpoint": checkpoint_file_path,
            "checkpoint_name": checkpoint_name,
            "output_dir": run_output_dir,
            "config_path": generated_eval_cfg_path,
            "eval_log_path": eval_log_path,
            "server_log_path": server_log_path,
        }

        server_proc: subprocess.Popen | None = None
        try:
            eval_cfg = OmegaConf.load(eval_config_template)
            eval_cfg.output_dir = run_output_dir
            if "server" not in eval_cfg:
                raise ValueError(f"`server` section missing in {eval_config_template}")
            server_host = str(eval_cfg.server.host)
            server_port = int(eval_cfg.server.port)
            OmegaConf.save(eval_cfg, generated_eval_cfg_path)

            server_cmd = [
                server_python_bin,
                server_script,
                "--ckpt_path",
                checkpoint_file_path,
                "--port",
                str(server_port),
                "--idle_timeout",
                str(int(getattr(auto_eval_cfg, "server_idle_timeout", -1))),
            ]
            if bool(getattr(auto_eval_cfg, "server_use_bf16", True)):
                server_cmd.append("--use_bf16")

            env = os.environ.copy()
            env["PYTHONPATH"] = f"{Path.cwd()}:{env.get('PYTHONPATH', '')}"
            if server_gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(server_gpu_id)
            with open(server_log_path, "w", encoding="utf-8") as server_log_file:
                server_proc = subprocess.Popen(
                    server_cmd,
                    stdout=server_log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

                start_timeout = float(getattr(auto_eval_cfg, "server_start_timeout_sec", 300))
                start_poll = float(getattr(auto_eval_cfg, "server_start_poll_interval_sec", 1.0))
                self._wait_tcp_ready(server_host, server_port, timeout_sec=start_timeout, poll_interval_sec=start_poll)

            cmd = [eval_python_bin, eval_script, "--config", generated_eval_cfg_path]
            timeout_sec = getattr(auto_eval_cfg, "timeout_sec", None)
            timeout = float(timeout_sec) if timeout_sec is not None else None

            self.accelerator.print(f"🚀 Auto-eval start: {checkpoint_name}")
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
            with open(eval_log_path, "w", encoding="utf-8") as f:
                f.write(proc.stdout or "")

            elapsed_sec = time.perf_counter() - started_at
            return_payload["elapsed_sec"] = round(elapsed_sec, 2)
            return_payload["return_code"] = int(proc.returncode)

            summary_json_path = os.path.join(run_output_dir, "summary.json")
            if os.path.exists(summary_json_path):
                with open(summary_json_path, "r", encoding="utf-8") as f:
                    eval_summary = json.load(f)
                return_payload["summary_path"] = summary_json_path
                return_payload["success_rate"] = eval_summary.get("success_rate")
                return_payload["total_episodes"] = eval_summary.get("total_episodes")
                return_payload["total_success"] = eval_summary.get("total_success")
                if bool(getattr(auto_eval_cfg, "log_to_wandb", True)) and self.accelerator.is_main_process:
                    wandb.log(
                        {
                            "auto_hstar_eval/success_rate": float(eval_summary.get("success_rate", 0.0)),
                            "auto_hstar_eval/total_episodes": int(eval_summary.get("total_episodes", 0)),
                            "auto_hstar_eval/total_success": int(eval_summary.get("total_success", 0)),
                        },
                        step=self.completed_steps,
                    )
            else:
                return_payload["summary_path"] = None

            if proc.returncode == 0:
                self.accelerator.print(
                    "✅ Auto-eval done: "
                    f"checkpoint={checkpoint_name}, success_rate={return_payload.get('success_rate')}, "
                    f"log={eval_log_path}"
                )
            else:
                self.accelerator.print(
                    f"❌ Auto-eval failed (code={proc.returncode}) for {checkpoint_name}. Check log: {eval_log_path}"
                )
            return return_payload
        except Exception as e:
            elapsed_sec = time.perf_counter() - started_at
            return_payload["elapsed_sec"] = round(elapsed_sec, 2)
            return_payload["error"] = repr(e)
            self.accelerator.print(f"❌ Auto-eval exception for {checkpoint_name}: {e}")
            return return_payload
        finally:
            if server_proc is not None:
                self._terminate_subprocess(server_proc, "auto-eval policy server")

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            save_interval_epochs = getattr(self.config.trainer, "save_interval_epochs", None)
            if save_interval_epochs is not None and save_interval_epochs > 0:
                steps_per_epoch = max(
                    1,
                    len(self.vla_train_dataloader) // self.accelerator.gradient_accumulation_steps,
                )
                current_epoch = (self.completed_steps + 1) // steps_per_epoch
                if (
                    (self.completed_steps + 1) % steps_per_epoch == 0
                    and current_epoch > 0
                    and current_epoch % save_interval_epochs == 0
                    and current_epoch != self._last_saved_epoch
                ):
                    self._last_saved_epoch = current_epoch
                    self._save_checkpoint(save_as_epoch=current_epoch)

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.camera_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  Save every N steps = {self.config.trainer.save_interval}")
            save_interval_epochs = getattr(self.config.trainer, "save_interval_epochs", None)
            logger.info(f"  Save every N epochs = {save_interval_epochs if save_interval_epochs else 'disabled'}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            self.accelerator.backward(total_loss)

            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            self.lr_scheduler.step()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
