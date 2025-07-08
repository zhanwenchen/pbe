Below is the updated **`main.py`** – identical in logic to the previous version, but all string literals now use **single quotes** (and no visual alignment around `=`).

```python
#!/usr/bin/env python
# coding: utf-8
'''
main.py — Lightning-free rewrite that uses the vanilla Hugging Face Trainer.
All string literals use single quotes and no padded alignment is applied.
'''

# ── standard library ────────────────────────────────────────────────────
from argparse import ArgumentParser, ArgumentTypeError, Namespace
import datetime
import glob
import os
import sys

# ── third-party ─────────────────────────────────────────────────────────
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
from numpy.random import seed as np_random_seed, get_state as np_random_get_state, choice as np_random_choice
from omegaconf import OmegaConf
from transformers import (
    Trainer,
    TrainingArguments,
    AdamW,
    get_cosine_schedule_with_warmup,
    set_seed,
)

torch.set_float32_matmul_precision('high')

# ── project-local ───────────────────────────────────────────────────────
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

NUM_WORKERS = 11
PERSISTENT_WORKERS = NUM_WORKERS > 0


# ───────────────────────── helpers ──────────────────────────────────────
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {'yes', 'true', 't', 'y', '1'}:
        return True
    if v in {'no', 'false', 'f', 'n', '0'}:
        return False
    raise ArgumentTypeError(f'Boolean value expected; got {v!r}')


def worker_init_fn(_):
    info = get_worker_info()
    wid = info.id
    dataset = info.dataset
    rng_state = np_random_get_state()[1]
    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split = dataset.num_records // info.num_workers
        dataset.sample_ids = dataset.valid_ids[wid * split:(wid + 1) * split]
        seed = rng_state[np_random_choice(len(rng_state), 1)] + wid
    else:
        seed = rng_state[0] + wid
    np_random_seed(seed)


class WrappedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────── DataModule replacement ────────────────────────────────
class DataModuleFromConfig:
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
        self.batch_size = batch_size
        self.wrap = wrap
        self.num_workers = NUM_WORKERS if num_workers is None else num_workers
        self.use_worker_init_fn = use_worker_init_fn
        self.shuffle_test_loader = shuffle_test_loader
        self.shuffle_val_dataloader = shuffle_val_dataloader

        self.dataset_cfgs = {
            k: v
            for k, v in zip(
                ('train', 'validation', 'test', 'predict'),
                (train, validation, test, predict),
            )
            if v is not None
        }
        self.datasets = {}
        if 'train' in self.dataset_cfgs:
            self.train_dataloader = self._train_dl
        if 'validation' in self.dataset_cfgs:
            self.val_dataloader = self._val_dl
        if 'test' in self.dataset_cfgs:
            self.test_dataloader = self._test_dl
        if 'predict' in self.dataset_cfgs:
            self.predict_dataloader = self._pred_dl

    def prepare_data(self):
        for cfg in self.dataset_cfgs.values():
            instantiate_from_config(cfg)

    def setup(self, stage=None):
        self.datasets = {k: instantiate_from_config(v) for k, v in self.dataset_cfgs.items()}
        if self.wrap:
            self.datasets = {k: WrappedDataset(v) for k, v in self.datasets.items()}

    # loader factory
    def _make_loader(self, name, shuffle):
        ds = self.datasets[name]
        iterable = isinstance(ds, Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if (iterable or self.use_worker_init_fn) else None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle and not iterable,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            persistent_workers=PERSISTENT_WORKERS,
            pin_memory=True,
        )

    def _train_dl(self):
        return self._make_loader('train', True)

    def _val_dl(self):
        return self._make_loader('validation', self.shuffle_val_dataloader)

    def _test_dl(self):
        return self._make_loader('test', self.shuffle_test_loader)

    def _pred_dl(self):
        return self._make_loader('predict', False)


# ─────────────────────── CLI helpers ───────────────────────────────────
def get_parser(**kwargs):
    p = ArgumentParser(**kwargs)
    p.add_argument('-n', '--name', type=str, nargs='?', const=True, default='')
    p.add_argument('-r', '--resume', type=str, nargs='?', const=True, default='')
    p.add_argument(
        '-b',
        '--base',
        nargs='*',
        default=['configs/stable-diffusion/v1-inference-inpaint.yaml'],
        metavar='base_config.yaml',
    )
    p.add_argument('-t', '--train', type=str2bool, nargs='?', const=True, default=True)
    p.add_argument('--no-test', type=str2bool, nargs='?', const=True, default=False)
    p.add_argument('-p', '--project')
    p.add_argument('-d', '--debug', type=str2bool, nargs='?', const=True, default=False)
    p.add_argument('-s', '--seed', type=int, default=23)
    p.add_argument('-f', '--postfix', type=str, default='')
    p.add_argument('-l', '--logdir', type=str, default='logs')
    p.add_argument('--pretrained_model', type=str, default='')
    p.add_argument('--scale_lr', type=str2bool, nargs='?', const=True, default=True)
    p.add_argument('--train_from_scratch', type=str2bool, nargs='?', const=True, default=False)
    return p


def prepare_log_dirs(opt, stamp):
    if opt.name and opt.resume:
        raise ValueError('Specify either --name or --resume, not both.')

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f'Cannot find {opt.resume}')
        if os.path.isfile(opt.resume):
            logdir = '/'.join(opt.resume.split('/')[:-2])
            ckpt = opt.resume
        else:
            logdir = opt.resume.rstrip('/')
            ckpt = os.path.join(logdir, 'checkpoints', 'last.ckpt')
        opt.resume_from_checkpoint = ckpt
        opt.base = sorted(glob.glob(os.path.join(logdir, 'configs/*.yaml'))) + opt.base
        run_name = os.path.basename(logdir)
    else:
        suffix = (
            '_' + opt.name
            if opt.name
            else '_' + os.path.splitext(os.path.basename(opt.base[0]))[0]
            if opt.base
            else ''
        )
        run_name = f'{stamp}{suffix}{opt.postfix}'
        logdir = os.path.join(opt.logdir, run_name)

    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    return logdir, ckptdir, cfgdir


# ─────────────────────────── main ──────────────────────────────────────
def main():
    run_stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    cfgs = [OmegaConf.load(p) for p in opt.base]
    cli_cfg = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(*cfgs, cli_cfg)

    logdir, ckptdir, _ = prepare_log_dirs(opt, run_stamp)
    set_seed(opt.seed)

    model = instantiate_from_config(cfg.model)
    if not opt.resume:
        state_dict = torch.load(opt.pretrained_model, map_location='cpu', weights_only=False)['state_dict']
        if opt.train_from_scratch:
            state_dict = {k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(state_dict, strict=False)

    data = instantiate_from_config(cfg.data)
    data.prepare_data()
    data.setup()
    train_dataset = data.datasets['train']
    eval_dataset = None if opt.no_test else data.datasets.get('validation')

    print('#### Data ####')
    for k, v in data.datasets.items():
        print(f'{k:12s} {v.__class__.__name__:40s} {len(v):7d}')

    batch_size = cfg.data.params.batch_size
    base_lr = cfg.model.base_learning_rate
    grad_acc = getattr(cfg.trainer, 'accumulate_grad_batches', 1)
    num_gpus = torch.cuda.device_count() or 1
    lr = grad_acc * num_gpus * batch_size * base_lr if opt.scale_lr else base_lr
    print(f'learning_rate = {lr:.2e}')

    epochs = getattr(cfg.trainer, 'max_epochs', 1)
    warmup = getattr(cfg.trainer, 'warmup_steps', 0)
    total_steps = (len(train_dataset) // batch_size) * epochs // grad_acc
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    ds_cfg = {
        'zero_optimization': {
            'stage': 2,
            'allgather_partitions': True,
            'reduce_scatter': True,
            'overlap_comm': True,
        },
        'bf16': {'enabled': torch.cuda.is_bf16_supported()},
    }

    training_args = TrainingArguments(
        output_dir=ckptdir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        dataloader_num_workers=NUM_WORKERS,
        learning_rate=lr,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        logging_steps=50,
        save_strategy='epoch',
        evaluation_strategy='epoch' if eval_dataset is not None else 'no',
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        deepspeed=ds_cfg,
        seed=opt.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
    )

    trainer.train(resume_from_checkpoint=getattr(opt, 'resume_from_checkpoint', None))
    if eval_dataset is not None:
        trainer.evaluate()


if __name__ == '__main__':
    main()
```
