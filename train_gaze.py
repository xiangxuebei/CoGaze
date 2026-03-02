from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cogaze.config import CoGazeConfig
from cogaze.data.gaze_dataset import NPZSequenceDataset
from cogaze.losses.cogaze_loss import CoGazeLoss
from cogaze.models.cogaze import CoGaze
from cogaze.utils.checkpoint import load_checkpoint, save_checkpoint
from cogaze.utils.gaze_metrics import summarize_gaze_metrics
from cogaze.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train CoGaze")
    p.add_argument("--train-manifest", type=str, required=True)
    p.add_argument("--val-manifest", type=str, default=None)
    p.add_argument("--cfg-json", type=str, default=None)
    p.add_argument("--outdir", type=str, default="runs/cogaze_gaze")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=10000, help="Total optimizer steps. <=0 means use epochs only.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="cosine")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_cfg(path: str | None) -> CoGazeConfig:
    cfg = CoGazeConfig()
    if path:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in obj.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


@torch.no_grad()
def evaluate(model: CoGaze, loader: DataLoader, loss_fn: CoGazeLoss, device: str) -> Dict[str, float]:
    model.eval()
    total = reg = cons = 0.0
    n = 0
    pred_list = []
    gt_list = []
    valid_list = []
    px_list = []
    task_list = []

    for batch in loader:
        tensor_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = model(tensor_batch)
        loss = loss_fn(out["pred_pog_cm"], tensor_batch["gt_pog_cm"], tensor_batch["valid_mask"], tensor_batch["segment_id"])
        total += float(loss.total.item())
        reg += float(loss.reg.item())
        cons += float(loss.cons.item())
        n += 1

        pred_list.append(out["pred_pog_cm"].detach().cpu())
        gt_list.append(batch["gt_pog_cm"])
        valid_list.append(batch["valid_mask"])
        px_list.append(batch.get("px_per_cm"))
        task_list.extend([str(x) for x in batch.get("task", [])])

    if n == 0:
        return {"total": 0.0, "reg": 0.0, "cons": 0.0}

    metrics: Dict[str, float] = {"total": total / n, "reg": reg / n, "cons": cons / n}
    pred = torch.cat(pred_list, dim=0)
    gt = torch.cat(gt_list, dim=0)
    valid = torch.cat(valid_list, dim=0)
    px = torch.cat(px_list, dim=0) if px_list and px_list[0] is not None else None
    metrics.update(summarize_gaze_metrics(pred_pog_cm=pred, gt_pog_cm=gt, valid_mask=valid, task_names=task_list, px_per_cm=px))
    return metrics


def build_optimizer(args: argparse.Namespace, model: CoGaze) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(args: argparse.Namespace, opt: torch.optim.Optimizer, total_steps: int):
    if args.scheduler == "cosine" and total_steps > 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=args.min_lr)
    return None


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    cfg = load_cfg(args.cfg_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_ds = NPZSequenceDataset(args.train_manifest, max_seq_len=cfg.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = None
    if args.val_manifest:
        val_ds = NPZSequenceDataset(args.val_manifest, max_seq_len=cfg.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CoGaze(cfg).to(args.device)
    loss_fn = CoGazeLoss(cfg.lambda_cons, cfg.smooth_l1_beta)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    opt = build_optimizer(args, model)
    scheduler = build_scheduler(args, opt, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")  # lower is better, preferably by fix_err_cm

    if args.resume:
        ckpt = load_checkpoint(args.resume, "cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt.get("global_step", global_step))
        best_val = float(ckpt.get("best_val", best_val))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:03d}", ncols=120)
        for batch in pbar:
            if total_steps > 0 and global_step >= total_steps:
                break
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                out = model(batch)
                loss_out = loss_fn(out["pred_pog_cm"], batch["gt_pog_cm"], batch["valid_mask"], batch["segment_id"])

            scaler.scale(loss_out.total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            global_step += 1
            if scheduler is not None:
                scheduler.step()

            lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(
                total=f"{loss_out.total.item():.4f}",
                reg=f"{loss_out.reg.item():.4f}",
                cons=f"{loss_out.cons.item():.4f}",
                lr=f"{lr:.2e}",
                step=f"{global_step}/{total_steps if total_steps > 0 else '-'}",
            )

        save_checkpoint(
            outdir / "last.pt",
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
            },
        )

        if val_loader is not None:
            metrics = evaluate(model, val_loader, loss_fn, args.device)
            msg = (
                f"[val] total={metrics['total']:.4f} reg={metrics['reg']:.4f} cons={metrics['cons']:.4f} "
                f"fix_err={metrics.get('fix_err_cm', np.nan):.4f} "
                f"pix100={metrics.get('pix100_acc', np.nan):.4f} "
                f"pix200={metrics.get('pix200_acc', np.nan):.4f}"
            )
            print(msg)
            score = metrics.get("fix_err_cm", float("nan"))
            score = score if np.isfinite(score) else metrics["total"]
            if score < best_val:
                best_val = score
                save_checkpoint(
                    outdir / "best.pt",
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_val": best_val,
                        "val_metrics": metrics,
                    },
                )

        if total_steps > 0 and global_step >= total_steps:
            break


if __name__ == "__main__":
    main()
