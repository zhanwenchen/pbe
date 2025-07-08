#!/usr/bin/env python
# coding: utf-8
"""
main.py  ──  Lightning-free rewrite that follows the original coding style.
"""

# -------------------------------------------------------------------------
# stdlib
# -------------------------------------------------------------------------
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from functools import partial
import datetime
import glob
import os
import sys
from pprint import pprint

# -------------------------------------------------------------------------
# third-party
# -------------------------------------------------------------------------
import torch
from torch import set_float32_matmul_precision
from torch.utils.data import DataLoader, Dataset, get_worker_info

from numpy.random import (
    seed     as np_random_seed,
    get_state as np_random_get_state,
    choice   as np_random_choice,
)

from omegaconf import OmegaConf

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin
from transformers import AdamW, get_cosine_schedule_with_warmup, set_seed

# -------------------------------------------------------------------------
# project
# -------------------------------------------------------------------------
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config


# -------------------------------------------------------------------------
# constants & misc
# -------------------------------------------------------------------------
NUM_WORKERS         = 11        # 0 ⇒ single-process dataloading
PERSISTENT_WORKERS  = NUM_WORKERS > 0

set_float32_matmul_precision("high")


# -------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------
def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"yes", "true", "t", "y", "1"}:
        return True
    if v in {"no", "false", "f", "n", "0"}:
        return False
    raise ArgumentTypeError(f"Boolean value expected. {v=}")


def worker_init_fn(_: int) -> None:
    """
    Provides worker-specific seeding and iterable-dataset sharding.
    Mirrors original behaviour that relied on numpy RNG-state.
    """
    w_info   = get_worker_info()
    worker_id = w_info.id
    dataset  = w_info.dataset

    rng_state = np_random_get_state()[1]
    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split = dataset.num_records // w_info.num_workers
        dataset.sample_ids = dataset.valid_ids[worker_id * split:(worker_id + 1) * split]
        current_idx = np_random_choice(len(rng_state), 1)
        seed = rng_state[current_idx] + worker_id
    else:
        seed = rng_state[0] + worker_id

    np_random_seed(seed)


class WrappedDataset(Dataset):
    """Wrap any iterable/sequence to satisfy PyTorch Dataset API."""
    def __init__(self, data): self.data = data
    def __len__(self):        return len(self.data)
    def __getitem__(self, i): return self.data[i]


# -------------------------------------------------------------------------
# DataModule analogue  (drop-in for the former Lightning one)
# -------------------------------------------------------------------------
class DataModuleFromConfig:
    """
    Same public surface as the original LightningDataModule variant, minus
    the inheritance. Keeping the original method names so configs need
    **zero** changes.
    """

    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        self.batch_size            = batch_size
        self.wrap                  = wrap
        self.num_workers           = NUM_WORKERS if num_workers is None else num_workers
        self.use_worker_init_fn    = use_worker_init_fn
        self.shuffle_test_loader   = shuffle_test_loader
        self.shuffle_val_dataloader= shuffle_val_dataloader

        self.dataset_configs = {
            k: v
            for k, v in zip(
                ("train", "validation", "test", "predict"),
                (train, validation, test, predict),
            ) if v is not None
        }
        self.datasets = {}

        # preserve original API surface
        if "train" in self.dataset_configs:
            self.train_dataloader   = self._train_dataloader
        if "validation" in self.dataset_configs:
            self.val_dataloader     = self._val_dataloader
        if "test" in self.dataset_configs:
            self.test_dataloader    = self._test_dataloader
        if "predict" in self.dataset_configs:
            self.predict_dataloader = self._predict_dataloader

    # ----- lightning-compat stubs -------------------------------------
    def prepare_data(self):
        for cfg in self.dataset_configs.values():
            instantiate_from_config(cfg)

    def setup(self, stage=None):
        self.datasets = {
            k: instantiate_from_config(cfg) for k, cfg in self.dataset_configs.items()
        }
        if self.wrap:
            self.datasets = {k: WrappedDataset(v) for k, v in self.datasets.items()}

    # ----- internal loader builders -----------------------------------
    def _make_loader(self, name: str, shuffle: bool):
        ds           = self.datasets[name]
        iterable     = isinstance(ds, Txt2ImgIterableBaseDataset)
        init_fn      = worker_init_fn if (iterable or self.use_worker_init_fn) else None
        should_shuffle = shuffle and not iterable
        return DataLoader(
            ds,
            batch_size         = self.batch_size,
            shuffle            = should_shuffle,
            num_workers        = self.num_workers,
            worker_init_fn     = init_fn,
            persistent_workers = PERSISTENT_WORKERS,
            pin_memory         = True,
        )

    # public shortcuts
    def _train_dataloader  (self): return self._make_loader("train",        shuffle=True)
    def _val_dataloader    (self): return self._make_loader("validation",   shuffle=self.shuffle_val_dataloader)
    def _test_dataloader   (self): return self._make_loader("test",         shuffle=self.shuffle_test_loader)
    def _predict_dataloader(self): return self._make_loader("predict",      shuffle=False)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def get_parser(**kwargs):
    parser = ArgumentParser(**kwargs)
    parser.add_argument("-n", "--name",    type=str, nargs="?", const=True, default="")
    parser.add_argument("-r", "--resume",  type=str, nargs="?", const=True, default="")
    parser.add_argument("-b", "--base",    nargs="*", default=["configs/stable-diffusion/v1-inference-inpaint.yaml"],
                        metavar="base_config.yaml")
    parser.add_argument("-t", "--train",   type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--no-test",       type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("-p", "--project")
    parser.add_argument("-d", "--debug",   type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("-s", "--seed",    type=int, default=23)
    parser.add_argument("-f", "--postfix", type=str, default="")
    parser.add_argument("-l", "--logdir",  type=str, default="logs")
    parser.add_argument("--pretrained_model",      type=str, default="")
    parser.add_argument("--scale_lr",              type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--train_from_scratch",    type=str2bool, nargs="?", const=True, default=False)
    return parser


# -------------------------------------------------------------------------
# log/ckpt dir handling  (mirrors original logic)
# -------------------------------------------------------------------------
def prepare_log_dirs(opt: Namespace, run_stamp: str):
    if opt.name and opt.resume:
        raise ValueError("Specify only one of --name or --resume, not both.")

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")

        if os.path.isfile(opt.resume):
            logdir = "/".join(opt.resume.split("/")[:-2])
            ckpt   = opt.resume
        else:
            logdir = opt.resume.rstrip("/")
            ckpt   = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        opt.base = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml"))) + opt.base
        run_name = os.path.basename(logdir)
    else:
        if opt.name:
            suffix = "_" + opt.name
        elif opt.base:
            suffix = "_" + os.path.splitext(os.path.basename(opt.base[0]))[0]
        else:
            suffix = ""
        run_name = f"{run_stamp}{suffix}{opt.postfix}"
        logdir   = os.path.join(opt.logdir, run_name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir  = os.path.join(logdir, "configs")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir,  exist_ok=True)
    return logdir, ckptdir, cfgdir


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    run_stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    sys.path.append(os.getcwd())

    # ----- parse ----------------------------------------------------------------
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # ----- config merge ---------------------------------------------------------
    cfg_files = [OmegaConf.load(p) for p in opt.base]
    cli_cfg   = OmegaConf.from_dotlist(unknown)
    cfg       = OmegaConf.merge(*cfg_files, cli_cfg)

    logdir, ckptdir, _ = prepare_log_dirs(opt, run_stamp)

    # ----- seeding --------------------------------------------------------------
    set_seed(opt.seed)

    # ----- model ----------------------------------------------------------------
    model = instantiate_from_config(cfg.model)

    if not opt.resume:
        sd = torch.load(opt.pretrained_model, map_location="cpu", weights_only=False)["state_dict"]
        if opt.train_from_scratch:
            sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            print("Train from scratch -> loading non-textual weights only.")
        else:
            print("Loading Stable Diffusion v1-4 weights.")
        model.load_state_dict(sd, strict=False)

    # ----- data -----------------------------------------------------------------
    data: DataModuleFromConfig = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()

    train_loader = data.train_dataloader()
    val_loader   = None if opt.no_test else data.val_dataloader()
    test_loader  = None if opt.no_test else data.test_dataloader()

    print("#### Data ####")
    for k, v in data.datasets.items():
        print(f"{k:12s}  {v.__class__.__name__:40s}  {len(v):>7d}")

    # ----- learning rate scaling ------------------------------------------------
    bs, base_lr = cfg.data.params.batch_size, cfg.model.base_learning_rate
    grad_acc    = getattr(cfg.trainer, "accumulate_grad_batches", 1)
    num_gpus    = torch.cuda.device_count() or 1
    if opt.scale_lr:
        model.learning_rate = grad_acc * num_gpus * bs * base_lr
    else:
        model.learning_rate = base_lr
    print(f"learning_rate = {model.learning_rate:.2e}")

    # ----- optimiser & scheduler -----------------------------------------------
    optimizer = AdamW(model.parameters(), lr=model.learning_rate, weight_decay=1e-2)

    epochs          = getattr(cfg.trainer, "max_epochs", 1)
    warmup_steps    = getattr(cfg.trainer, "warmup_steps", 0)
    total_steps     = len(train_loader) * epochs // grad_acc
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = warmup_steps,
        num_training_steps = total_steps,
    )

    # ----- accelerator ----------------------------------------------------------
    ddp_kwargs  = DistributedDataParallelKwargs(find_unused_parameters=False)
    ds_plugin   = DeepSpeedPlugin(zero_stage=2)
    accelerator = Accelerator(
        gradient_accumulation_steps = grad_acc,
        mixed_precision             = "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        kwargs_handlers             = [ddp_kwargs],
        deepspeed_plugin            = ds_plugin,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ----- training loop --------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_loader, 1):
            with accelerator.accumulate(model):
                outs = model(**batch)
                loss = outs["loss"] if isinstance(outs, dict) else outs.loss
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and step % 50 == 0:
                accelerator.print(f"epoch {epoch:02d} step {step:05d} loss {loss.item():.4f}")

        # ----- validation -------------------------------------------------------
        if val_loader is not None and accelerator.is_main_process:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outs = model(**batch)
                    val_loss += (outs["loss"] if isinstance(outs, dict) else outs.loss).item()
            val_loss /= len(val_loader)
            accelerator.print(f"epoch {epoch:02d} val_loss {val_loss:.4f}")
            model.train()

        # ----- checkpoint -------------------------------------------------------
        if accelerator.is_main_process:
            ckpt_path = os.path.join(ckptdir, f"epoch-{epoch:03d}.pt")
            accelerator.save_state(ckpt_path)

    # ----- test -----------------------------------------------------------------
    if test_loader is not None:
        model.eval()
        with torch.no_grad():
            for _batch in test_loader:
                _ = model(**_batch)
        accelerator.print("testing complete")


# -------------------------------------------------------------------------
# entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
