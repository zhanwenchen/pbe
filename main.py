from argparse import ArgumentParser, ArgumentTypeError, Namespace
from datetime import datetime
from functools import partial
import os
from os import getenv
from pathlib import Path
from pprint import pprint
import sys
from lightning import LightningDataModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from numpy.random import seed as np_random_seed, get_state as np_random_get_state, choice as np_random_choice
from omegaconf import OmegaConf
from torch import set_float32_matmul_precision, load as torch_load
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm
# from ldm.data.base import Txt2ImgIterableBaseDataset
# from ldm.models.diffusion.latent_diffusion import LatentDiffusion
from ldm.util import instantiate_from_config
# fromFitLoop
# from pl_image_logger_callback import WandbImageLogger

DEBUG = getenv('DEBUG', '0') == '1'
print(f'main: {DEBUG=}')
# NUM_WORKERS = 0 # 11
NUM_WORKERS = 11 # 11
# NUM_WORKERS = 0 if DEBUG else NUM_WORKERS
# NUM_WORKERS = 47 # 11
# NUM_WORKERS = 1 # 11 # 0.29it/s
PERSISTENT_WORKERS = NUM_WORKERS > 0


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if (v:= v.lower()) in ('yes', 'true', 't', 'y', '1'):
            return True
        if v in {'no', 'false', 'f', 'n', '0'}:
            return False
        raise ArgumentTypeError(f'Boolean value expected. {v=}')

    parser = ArgumentParser(**parser_kwargs)
    parser.add_argument('-n', '--name', type=str, const=True, default='', nargs='?', help='postfix for logdir')
    parser.add_argument('-r', '--resume', type=str, const=True, default='', nargs='?', help='resume from logdir or checkpoint in logdir')
    parser.add_argument('-b', '--base', nargs='*', metavar='base_config.yaml', help='paths to base configs. Loaded from left-to-right. ' 'Parameters can be overwritten or added with command-line options of the form `--key value`.', default=['configs/stable-diffusion/v1-inference-inpaint.yaml'],)
    parser.add_argument('-t', '--train', type=str2bool, const=True, default=True, nargs='?', help='train')
    parser.add_argument('--no-test', type=str2bool, const=True, default=False, nargs='?', help='disable test')
    parser.add_argument('-p', '--project', help='name of new or path to existing project')
    parser.add_argument('-d', '--debug', type=str2bool, nargs='?', const=True, default=False, help='enable post-mortem debugging')
    parser.add_argument('-s', '--seed', type=int, default=23, help='seed for seed_everything')
    parser.add_argument('--val_check_interval', type=int, default=1, help='How many steps to run validation, in number of steps or 1 for every epoch. If None, defaults to 1 epoch.')
    parser.add_argument('-f', '--postfix', type=str, default='', help='post-postfix for default name')
    parser.add_argument('-l', '--logdir', type=str, default='logs', help='directory for logging dat shit')
    parser.add_argument('--pretrained_model', type=str, default='', help='path to pretrained model')
    parser.add_argument('--scale_lr', type=str2bool, nargs='?', const=True, default=True, help='scale base-lr by ngpu * batch_size * n_accumulate')
    parser.add_argument('--train_from_scratch', type=str2bool, nargs='?', const=True, default=False, help='Train from scratch')
    return parser


def nondefault_trainer_args(opt):
    parser = ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# class WrappedDataset(Dataset):
#     """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

#     def __init__(self, dataset):
#         self.data = dataset

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]


# def worker_init_fn(_):
#     worker_info = get_worker_info()

#     dataset = worker_info.dataset
#     worker_id = worker_info.id

#     if isinstance(dataset, Txt2ImgIterableBaseDataset):
#         split_size = dataset.num_records // worker_info.num_workers
#         # reset num_records to the true number to retain reliable length information
#         dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
#         current_id = np_random_choice(len(np_random_get_state()[1]), 1)
#         return np_random_seed(np_random_get_state()[1][current_id] + worker_id)
#     return np_random_seed(np_random_get_state()[1][0] + worker_id)


class DataModuleFromConfig(LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=NUM_WORKERS, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        # num_workers = NUM_WORKERS
        # self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        print(f'DataModuleFromConfig: {self.num_workers=}, {self.batch_size=}, {self.use_worker_init_fn=}')
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs['predict'] = predict
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
        # if self.wrap:
            # for k in self.datasets:
            #     self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        # is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # if is_iterable_dataset or self.use_worker_init_fn:
        #     init_fn = worker_init_fn
        # else:
        #     init_fn = None
        init_fn = None
        return DataLoader(self.datasets['train'], batch_size=self.batch_size,
                        #   num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          num_workers=self.num_workers, shuffle=True,
                          worker_init_fn=init_fn, persistent_workers=PERSISTENT_WORKERS)

    def _val_dataloader(self, shuffle=False):
        # if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
        #     init_fn = worker_init_fn
        # else:
        #     init_fn = None
        init_fn = None
        return DataLoader(self.datasets['validation'],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, persistent_workers=PERSISTENT_WORKERS)

    def _test_dataloader(self, shuffle=False):
        # is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        # if is_iterable_dataset or self.use_worker_init_fn:
        #     init_fn = worker_init_fn
        # else:
        #     init_fn = None
        init_fn = None

        # do not shuffle dataloader for iterable dataset
        # shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets['test'], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, persistent_workers=PERSISTENT_WORKERS)

    def _predict_dataloader(self, shuffle=False):
        predict = self.datasets['predict']
        # if isinstance(predict, Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
        #     init_fn = worker_init_fn
        # else:
        #     init_fn = None
        init_fn = None
        return DataLoader(predict, batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, persistent_workers=PERSISTENT_WORKERS,)


def main():
    set_float32_matmul_precision('medium')
    # set_float32_matmul_precision('high')

    now = datetime.now().strftime('%Y%m%dT%H%M%S')
    sys.path.append(os.getcwd())

    parser = get_parser()
    args = parser.parse_args()

    opt, unknown = parser.parse_known_args()
    assert not opt.resume
    opt_name = opt.name
    if opt_name:
        name = f'_{opt_name}'
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = f'_{cfg_name}'
    else:
        name = ''
    nowname = now + name + opt.postfix
    # logdir = os.path.join(opt.logdir, nowname)

    logdir = Path(opt.logdir) / nowname
    ckptdir = logdir / 'checkpoints'

    # ckptdir = os.path.join(logdir, 'checkpoints')
    # cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop('lightning', OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get('trainer', OmegaConf.create())
    # default to ddp
    trainer_config['accelerator'] = 'ddp'
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    if not "gpus" in trainer_config:
        del trainer_config['accelerator']
        print('Running on CPU')
        cpu = True
    else:
        gpuinfo = trainer_config['gpus']
        print(f'Running on GPUs {gpuinfo}')
        cpu = False
    trainer_opt = Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    config.model.params.log_every_t = 10 if DEBUG else config.model.params.log_every_t

    model = instantiate_from_config(config.model)
    # model = torch_compile(model)
    if not opt.resume:
        if opt.train_from_scratch:
            ckpt_file=torch_load(opt.pretrained_model, map_location='cpu', weights_only=False)['state_dict']
            ckpt_file={key:value for key,value in ckpt_file.items() if not key[:6]=='model.'}
            model.load_state_dict(ckpt_file,strict=False)
            print('Train from scratch!')
        else:
            model.load_state_dict(torch_load(opt.pretrained_model, map_location='cpu', weights_only=False)['state_dict'], strict=False)
            print('Load Stable Diffusion v1-4!')

    # trainer and callbacks
    # trainer_kwargs = {}
    # trainer_opt
    # args_trainer_dict = vars(trainer_opt) | trainer_kwargs
    args_trainer_dict = vars(trainer_opt)
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
    if not cpu:
        # ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        devices = find_usable_cuda_devices()
        print(f'Found {len(devices)} usable CUDA devices: {devices}')
        ngpu = len(devices)
    else:
        ngpu = 1
    print(f'{args_trainer_dict=}') # {'accelerator': 'cuda', 'max_epochs': 20, 'num_nodes': 1}

    # Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor='val/loss_simple_ema', verbose=True, save_top_k=5, auto_insert_metric_name=True)
    # checkpoint_callback = None
    # wandb_logger = WandbLogger(log_model="all")
    # batch_frequency, max_images = 500, 4
    project = args.project
    # wandb_logger = WandbImageLogger(project, batch_frequency, max_images)
    wandb_logger = WandbLogger(project=project)

    limit_val_batches = 10 if DEBUG else 1.0

    print(f'{args.val_check_interval=}')
    # trainer = Trainer(**args_trainer_dict, logger=wandb_logger, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval)
    # trainer = Trainer(**args_trainer_dict, logger=wandb_logger, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches, enable_model_summary=True)
    # trainer = Trainer(**args_trainer_dict, logger=wandb_logger, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches)
    # trainer = Trainer(**args_trainer_dict, logger=wandb_logger, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches)
    log_every_n_steps = 50
    # model.logger.experiment.define_metric('*', step_metric='trainer/global_step')

    # trainer = Trainer(**args_trainer_dict, logger=wandb_logger, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches, log_every_n_steps=log_every_n_steps)
    max_epochs = args_trainer_dict['max_epochs']
    num_nodes = args_trainer_dict['num_nodes']
    # accelerator = args_trainer_dict['accelerator']
    accelerator = 'cuda' # gpu

    # strategy = 'deepspeed_stage_3' # deepspeed_stage_3_offload
    strategy = 'deepspeed_stage_3_offload' # deepspeed_stage_3_offload
    # precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
    # 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
    # Can be used on CPU, GPU, TPUs, or HPUs.
    # Default: ``'32-true'``.
    precision = '32' # "bf16-mixed" # "16-mixed" # "16-true" # "32"
    # trainer = Trainer(max_epochs=max_epochs, num_nodes=num_nodes, accelerator=accelerator, logger=wandb_logger, devices=devices, strategy='deepspeed_stage_3', default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches, log_every_n_steps=log_every_n_steps)
    # devices = [4,5,6,7]
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, devices))
    devices = find_usable_cuda_devices()
    print(f'Found {len(devices)} usable CUDA devices: {devices}')

    # os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get('LOCAL_RANK')

    # trainer = Trainer(max_epochs=max_epochs, num_nodes=num_nodes, accelerator=accelerator, devices=devices, logger=wandb_logger, strategy=strategy, default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches, log_every_n_steps=log_every_n_steps)
    trainer = Trainer(max_epochs=max_epochs, num_nodes=num_nodes, accelerator=accelerator, logger=wandb_logger, strategy=strategy, default_root_dir=ckptdir, callbacks=[checkpoint_callback], val_check_interval=args.val_check_interval, limit_val_batches=limit_val_batches, log_every_n_steps=log_every_n_steps)
    # trainer = Trainer(**args_trainer_dict, devices=devices, strategy='deepspeed_stage_2', default_root_dir=ckptdir, callbacks=[checkpoint_callback])
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
    print(f'#### Data #####\n{data.batch_size=}')
    for k in data.datasets:
        print(f'{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}')

    # for datapoint in tqdm(data.datasets['train']):
    #     pass

    # for datapoint in tqdm(data.datasets['validation']):
    #     pass

    # for datapoint in tqdm(data.datasets['test']):
    #     pass
    # return
    # configure learning rate
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate

    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    print(f'accumulate_grad_batches = {accumulate_grad_batches}')
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    num_nodes = 1
    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * num_nodes * ngpu * bs * base_lr
        print(f'Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches=} * {num_nodes=} * {ngpu=} * {bs} * {base_lr:.2e} (base_lr)')
    else:
        model.learning_rate = base_lr
        print('++++ NOT USING LR SCALING ++++')
        print(f'Setting learning rate to {model.learning_rate:.2e}')

    # run
    trainer.fit(model, data)
    wandb_logger.experiment.finish()
    trainer.test(model, data.test_dataloader)


if __name__ == '__main__':
    main()
