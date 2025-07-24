from argparse import ArgumentParser, ArgumentTypeError, Namespace
from functools import partial
import os
import sys
import datetime
import glob
from pprint import pprint
# from warnings import warn
from numpy.random import seed as np_random_seed, get_state as np_random_get_state, choice as np_random_choice
# import pytorch_lightning as pl
from omegaconf import OmegaConf
from lightning import LightningDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks import ModelCheckpoint

# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
# from lightning.pytorch.strategies import DeepSpeedStrategy
from torch import set_float32_matmul_precision, load as torch_load
from torch.utils.data import DataLoader, Dataset, get_worker_info
# from tqdm import tqdm
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.models.diffusion.latent_diffusion import LatentDiffusion
from ldm.util import instantiate_from_config

# import timg
# from timg import Ansi24Hblock


# (pbe) ubuntu@ip-172-31-42-214:~/pbe$ file dataset/multitote/v1/after/valid/SHV1-paKivaT02-2109,ConsolidateTote,hcX0e4qxpt9,2025-05-26T18:08:13Z.Ind3,4.jpg
# dataset/multitote/v1/after/valid/SHV1-paKivaT02-2109,ConsolidateTote,hcX0e4qxpt9,2025-05-26T18:08:13Z.Ind3,4.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 1024x670, components 3
# (pbe) ubuntu@ip-172-31-42-214:~/pbe$ cat dataset/multitote/v1/bbox/valid/SHV1-paKivaT02-2109,ConsolidateTote,hcX0e4qxpt9,2025-05-26T18:08:13Z.Ind3,4.txt
# 685 134 1047 369(pbe) ubuntu@ip-172-31-42-214:~/pbe$

# NUM_WORKERS = 0 # 11
NUM_WORKERS = 11 # 11
PERSISTENT_WORKERS = NUM_WORKERS > 0


set_float32_matmul_precision('high')


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if (v:= v.lower()) in ("yes", "true", "t", "y", "1"):
            return True
        if v in {"no", "false", "f", "n", "0"}:
            return False
        raise ArgumentTypeError(f"Boolean value expected. {v=}")

    parser = ArgumentParser(**parser_kwargs)
    parser.add_argument('-n', '--name', type=str, const=True, default="", nargs="?", help="postfix for logdir")
    parser.add_argument('-r', '--resume', type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir")
    parser.add_argument('-b', '--base', nargs="*", metavar="base_config.yaml", help="paths to base configs. Loaded from left-to-right. " "Parameters can be overwritten or added with command-line options of the form `--key value`.", default=["configs/stable-diffusion/v1-inference-inpaint.yaml"],)
    parser.add_argument('-t', '--train', type=str2bool, const=True, default=True, nargs="?", help="train")
    parser.add_argument('--no-test', type=str2bool, const=True, default=False, nargs="?", help="disable test")
    parser.add_argument('-p', '--project', help="name of new or path to existing project")
    parser.add_argument('-d', '--debug', type=str2bool, nargs="?", const=True, default=False, help="enable post-mortem debugging")
    parser.add_argument('-s', '--seed', type=int, default=23, help="seed for seed_everything")
    parser.add_argument('-f', '--postfix', type=str, default="", help="post-postfix for default name")
    parser.add_argument('-l', '--logdir', type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument('--pretrained_model', type=str, default="", help="path to pretrained model")
    parser.add_argument('--scale_lr', type=str2bool, nargs="?", const=True, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument('--train_from_scratch', type=str2bool, nargs="?", const=True, default=False, help="Train from scratch")
    return parser


def nondefault_trainer_args(opt):
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np_random_choice(len(np_random_get_state()[1]), 1)
        return np_random_seed(np_random_get_state()[1][current_id] + worker_id)
    return np_random_seed(np_random_get_state()[1][0] + worker_id)


class DataModuleFromConfig(LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        num_workers = NUM_WORKERS
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap
        self.datasets = {}

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, persistent_workers=PERSISTENT_WORKERS)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=PERSISTENT_WORKERS)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=PERSISTENT_WORKERS)

    def _predict_dataloader(self, shuffle=False):
        predict = self.datasets['predict']
        if isinstance(predict, Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(predict, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=PERSISTENT_WORKERS,)


def main():
    now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    args = parser.parse_args()

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f'Cannot find {opt.resume}')
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # try:
        # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp
    trainer_config["accelerator"] = "ddp"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config["accelerator"]
        cpu = True
    else:
        gpuinfo = trainer_config["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False
    trainer_opt = Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(config.model)
    # model = torch_compile(model)
    if not opt.resume:
        if opt.train_from_scratch:
            ckpt_file=torch_load(opt.pretrained_model, map_location='cpu', weights_only=False)['state_dict']
            ckpt_file={key:value for key,value in ckpt_file.items() if not key[:6]=='model.'}
            model.load_state_dict(ckpt_file,strict=False)
            print("Train from scratch!")
        else:
            model.load_state_dict(torch_load(opt.pretrained_model, map_location='cpu', weights_only=False)['state_dict'], strict=False)
            print("Load Stable Diffusion v1-4!")

    # trainer and callbacks
    trainer_kwargs = {}
    # trainer_opt
    args_trainer_dict = vars(trainer_opt) | trainer_kwargs
    args_trainer_dict.pop('gpus')
    # print(f'{args_trainer_dict["accelerator"]=}, {gpus=}')
    args_trainer_dict['accelerator'] = 'cuda'

    # Find two GPUs on the system that are not already occupied
    # strategy = DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(minutes=1))
    # strategy=DeepSpeedStrategy(
    #     stage=3,
    #     offload_optimizer=True,
    #     offload_parameters=True,
    #     # remote_device="nvme",
    #     # offload_params_device="nvme",
    #     # offload_optimizer_device="nvme",
    #     # nvme_path="/local_nvme",
    # ),
    # trainer = Trainer(**args_trainer_dict, devices=device_count(), strategy='deepspeed_stage_3')
    # trainer = Trainer(**args_trainer_dict, devices=8, strategy='deepspeed_stage_1', precision='bf16-mixed')
    # trainer = Trainer(**args_trainer_dict, devices=8, strategy='deepspeed_stage_1', precision='fp16')
    # trainer = Trainer(**args_trainer_dict, devices=8, strategy='deepspeed_stage_1')
    devices = find_usable_cuda_devices()
    pprint(args_trainer_dict)

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor='val/loss_simple_ema', verbose=True, save_top_k=5, auto_insert_metric_name=True)
    trainer = Trainer(**args_trainer_dict, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback])
    # Namespace(name='', resume='', base=['configs/v1.yaml'], train=True, no_test=False, project=None, debug=False, seed=23, postfix='', logdir='models/Paint-by-Example', pretrained_model='pretrained_models/sd-v1-4-modified-9channel.ckpt', scale_lr=False, train_from_scratch=False)
    # trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    # trainer = Trainer.add_argparse_args(trainer_opt, **trainer_kwargs)

    # trainer.plugins = [MyCluster()]
    trainer.logdir = logdir  ###

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()
    print(data.batch_size)
    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # bads = set()
    # for k in data.datasets:
    #     dataset_ = data.datasets[k]
    #     print(f"{k}, {dataset_.__class__.__name__}, {len(dataset_)}")

    #     # bads[k] = {}
    #     for idx in tqdm(range(len(dataset_))):
    #         # try:
    #         bbox_path_bad = dataset_[idx]
    #         if isinstance(bbox_path_bad, str):
    #             bads.add(bbox_path_bad)
    #         # except:
    #         #     warn(f"Dataset {k} at index {idx} failed. {bbox_path_bad=}")
    #         #     continue
    # print(bads)
    # breakpoint()


    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        # ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        ngpu = len(devices)
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    # if 'num_nodes' in lightning_config.trainer:
    #     num_nodes = lightning_config.trainer.num_nodes
    # else:
    num_nodes = 1
    print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * num_nodes * ngpu * bs * base_lr
        print(f'Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches=} * {num_nodes=} * {ngpu=} * {bs} * {base_lr:.2e} (base_lr)')
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")


    # # allow checkpointing via USR1
    # def melk(*args, **kwargs):
    #     # run all checkpoint hooks
    #     if trainer.global_rank == 0:
    #         print("Summoning checkpoint.")
    #         ckpt_path = os.path.join(ckptdir, "last.ckpt")
    #         trainer.save_checkpoint(ckpt_path)


    # def divein(*args, **kwargs):
    #     if trainer.global_rank == 0:
    #         import pudb;
    #         pudb.set_trace()


    # import signal

    # signal.signal(signal.SIGUSR1, melk)
    # signal.signal(signal.SIGUSR2, divein)


    # run
    trainer.fit(model, data)
    # if opt.train:
    #     try:
    #     except Exception:
    #         melk()
    #         raise
    # if not opt.no_test and not trainer.interrupted:
    trainer.predict(model, data.test_dataloader)


if __name__ == '__main__':
    main()
