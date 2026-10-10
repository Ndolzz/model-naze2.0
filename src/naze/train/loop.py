"""Training loop penuh (M008-T007, REQ-007/REQ-101/REQ-102/REQ-103).

TrainingRun menyusun komponen M-008 (batches/batches_pos per-epoch seed,
loss/loss_pos, evaluate, checkpoint v2, RunLog) menjadi loop epoch-batch
dengan SGD. Optimizer tetap SGD (no overengineering; OD-106 terbuka).
Reproducible: seed batch epoch ke-e = config.seed + e; resume penuh via
start_epoch + load_checkpoint_v2 (REQ-102). Checkpoint disimpan setiap
epoch (M-010) agar run yang terputus (mis. timeout CI) tetap dapat
dilanjutkan tanpa kehilangan bobot.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from naze.train.checkpoint import save_checkpoint_v2
from naze.train.config import TrainConfig
from naze.train.evaluate import evaluate
from naze.train.runlog import RunLog, checkpoint_size, measure_peak_rss


class TrainingRun:
    """Satu run training sesuai TrainConfig; model & data disuntikkan."""

    def __init__(self, config: TrainConfig, model, data, *, holdout=None) -> None:
        self.config = config
        self.model = model
        self.data = data
        self.holdout = holdout

    def run(self, *, start_epoch: int = 0, log: RunLog | None = None) -> dict:
        cfg = self.config
        if start_epoch < 0 or start_epoch > cfg.epochs:
            raise ValueError(f"start_epoch harus dalam [0, {cfg.epochs}], dapat {start_epoch}")
        per_pos = cfg.model == "transformer"
        if start_epoch == cfg.epochs:
            warnings.warn("start_epoch == epochs: tidak ada epoch tersisa (no-op)")
            if log is not None:
                log.write({"type": "no_op", "start_epoch": start_epoch, "epochs": cfg.epochs})
            return self._summary(cfg, start_epoch, epochs_run=0, steps=0, no_op=True,
                                 final_train_loss=None, eval_loss=None, perplexity=None,
                                 checkpoint_dir=None, checkpoint_size_bytes=None)
        steps = 0
        epoch_loss = None
        eval_loss = None
        perplexity = None
        for e in range(start_epoch, cfg.epochs):
            batches = (self.data.batches_pos(seed=cfg.seed + e) if per_pos
                       else self.data.batches(seed=cfg.seed + e))
            losses = []
            for x, y in batches:
                logits = self.model.forward(x)
                loss = (self.model.loss_pos(logits, y) if per_pos
                        else self.model.loss(logits, y))
                grads = (self.model.backward_pos() if per_pos
                         else self.model.backward())
                for name, p in self.model.params().items():
                    p -= cfg.lr * grads[name]
                steps += 1
                losses.append(float(loss))
            epoch_loss = sum(losses) / len(losses)
            if self.holdout is not None and (e + 1) % cfg.eval_every == 0:
                res = evaluate(self.model, self.holdout, per_pos=per_pos, seed=cfg.seed)
                eval_loss = res.mean_loss
                perplexity = res.perplexity
                if log is not None:
                    log.write({"type": "eval", "epoch": e, "step": steps,
                               "train_loss": epoch_loss, "eval_loss": eval_loss,
                               "perplexity": perplexity, "n_tokens": res.n_tokens})
            if cfg.checkpoint_path is not None:
                save_checkpoint_v2(cfg.checkpoint_path, self.model,
                                   step=steps, epoch=e + 1, config=cfg)
        checkpoint_dir = None
        checkpoint_bytes = None
        if cfg.checkpoint_path is not None:
            save_checkpoint_v2(cfg.checkpoint_path, self.model,
                               step=steps, epoch=cfg.epochs, config=cfg)
            checkpoint_dir = str(cfg.checkpoint_path)
            checkpoint_bytes = checkpoint_size(Path(cfg.checkpoint_path) / "params.npz")
        return self._summary(cfg, start_epoch, cfg.epochs - start_epoch, steps, False,
                             epoch_loss, eval_loss, perplexity,
                             checkpoint_dir, checkpoint_bytes)

    @staticmethod
    def _summary(cfg, start_epoch, epochs_run, steps, no_op, final_train_loss,
                 eval_loss, perplexity, checkpoint_dir, checkpoint_size_bytes) -> dict:
        return {"model": cfg.model, "start_epoch": start_epoch, "epochs_run": epochs_run,
                "steps": steps, "no_op": no_op, "final_train_loss": final_train_loss,
                "eval_loss": eval_loss, "perplexity": perplexity,
                "peak_rss_kb": measure_peak_rss(),
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_size_bytes": checkpoint_size_bytes}
