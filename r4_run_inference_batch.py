# python "./models/20250627T035445_v1/checkpoints/lightning_logs/version_0/checkpoints/epoch=39-step=40000.ckpt/zero_to_fp32.py" "./models/20250627T035445_v1/checkpoints/lightning_logs/version_0/checkpoints/epoch=39-step=40000.ckpt" ./models/20250627T035445_v1/checkpoints/lightning_logs/version_0/checkpoints/epoch=39-step=40000.ckpt --max_shard_size 100GB
# bash run_inference.sh

from argparse import ArgumentParser, Namespace
from logging import warning
import os
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from PIL.Image import open as Image_open
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import (
    autocast, inference_mode, ones as torch_ones, float32 as torch_float32, Tensor, device as torch_device,
    float16 as torch_float16,
    stack as torch_stack,
)
from torch.nn import Module
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.v2.functional import resize, to_dtype
from torchvision.utils import save_image
from tqdm import tqdm
from contextlib import nullcontext
from ldm.models.diffusion.latent_diffusion import LatentDiffusion
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class UnNormalize(Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


MEAN = (0.5, 0.5, 0.5)
MEAN_CLIP = (0.48145466, 0.4578275, 0.40821073)
STD = (0.5, 0.5, 0.5)
STD_CLIP = (0.26862954, 0.26130258, 0.27577711)


un_norm = UnNormalize(MEAN, STD)
un_norm_clip = UnNormalize(MEAN_CLIP, STD_CLIP)


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list.append(ToTensor())

    if normalize:
        transform_list.append(Normalize(MEAN, STD))
    return Compose(transform_list) if len(transform_list) > 1 else transform_list[0]


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list.append(ToTensor())

    if normalize:
        transform_list.append(Normalize(MEAN_CLIP, STD_CLIP))
    return Compose(transform_list) if len(transform_list) > 1 else transform_list[0]


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd is False else pl_sd
    model = instantiate_from_config(config.model).eval()
    m, u = model.load_state_dict(sd, strict=False)
    if m and verbose:
        print(f"missing keys:\n{m}")
    if u and verbose:
        print(f"unexpected keys:\n{u}")

    model.eval()
    return model.to('cuda', non_blocking=True)


def un_norm_old(x):
    return (x+1.0)/2.0


def get_image_cirep_fnsku(image_tensor: Tensor, dirpath_cirep_fnsku: Path, get_tensor_clip_fn, device) -> Tensor:
    '''
    Ideally, we want to get the most similar to the item (crop tray image with the mmid mask)
    '''
    # from ldm.models.clip_rep import CIREP
    # /home/ubuntu/AFTAI_Intern25_Tote_Diffusion_backup_mine/data/v2/cirep_afex
    # for fpath_cirep in sorted(dirpath_cirep_fnsku.iterdir()):
        # return fpath_cirep
    try:
        fpath_first_cirep = sorted(dirpath_cirep_fnsku.iterdir())[0]
    except:
        fpath_first_cirep = dirpath_cirep_fnsku.with_suffix('.jpg')

    # image_tensor_cirep = decode_image(fpath_first_cirep, mode=ImageReadMode.RGB).unsqueeze(0).to(image_tensor.device, non_blocking=True)
            # Load reference image
    with Image_open(fpath_first_cirep) as ref_p:
        ref_p = ref_p.convert("RGB").resize((224, 224)) # why resize? because otherwise ValueError: Input image size (1328*1170) doesn't match model (224*224).
        # ref_p = ref_p.convert("RGB")
    # ref_p = Image.open(opt.reference_path).convert("RGB").resize((224, 224))
    ref_tensor = get_tensor_clip_fn(ref_p)
    ref_tensor = ref_tensor.unsqueeze(0).to(device, non_blocking=True)
    return ref_tensor


# def infer_one(image_tensor: Tensor, mask_tensor: Tensor, ref_tensor: Tensor, model: Module, sampler, scale: float, precision: str, sample_path, result_path, grid_path, filename):
# def infer_one(image_tensor: Tensor, mask_tensor: Tensor, ref_tensor: Tensor, model: Module, sampler, scale: float, precision: str, sample_path, result_path, grid_path, filename, num_channels_latent, H, W, f):
def infer_one(model: Module, sampler, image_inpaint: Tensor, image_tensor: Tensor, mask_tensor: Tensor, ref_tensor: Tensor, scale: float, precision: str, sample_path, result_path, grid_path, filename, num_channels_latent, H, W, f):
    '''
    Run inference for one image with the given model and sampler.
    '''
    # Prepare inpaint image
    # inpaint_image = image_tensor * mask_tensor
    inpaint_image = image_inpaint

    # Prepare model kwargs
    test_model_kwargs = {
        'inpaint_image': inpaint_image,
        'inpaint_mask': mask_tensor,
    }

    # Run inference
    precision_scope = autocast if precision == 'autocast' else nullcontext

    un_norm = UnNormalize(MEAN, STD)
    un_norm_clip = UnNormalize(MEAN_CLIP, STD_CLIP)
    # with torch.no_grad():
    with precision_scope('cuda'), model.ema_scope():
        # Get conditioning
        uc = None
        if scale != 1.0:
            uc = model.learnable_vector

        c = model.get_learned_conditioning(ref_tensor.to(torch.float16, non_blocking=True))
        c = model.proj_out(c)

        # Encode inpaint image
        z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
        z_inpaint = model.get_first_stage_encoding(z_inpaint)
        test_model_kwargs['inpaint_image'] = z_inpaint
        test_model_kwargs['inpaint_mask'] = resize(test_model_kwargs['inpaint_mask'], (z_inpaint.shape[-2:]))

        # Sample
        shape = (num_channels_latent, H // f, W // f)
        samples_ddim, _ = sampler.sample(
            S=opt.ddim_steps,
            conditioning=c,
            batch_size=1,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=None,
            test_model_kwargs=test_model_kwargs
        )

        # Decode samples
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).to('cpu', non_blocking=True)
        # x_samples_ddim = x_samples_ddim.to('cpu', non_blocking=True).permute(0, 2, 3, 1).numpy()
        # return x_samples_ddim
        # Convert to torch tensor
        # x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

        # Save results
        for i, x_sample in enumerate(x_samples_ddim):
            # Create grid of all images
            # all_img = []
            image_tensor_i_un_norm = un_norm(image_tensor[i]).to('cpu', non_blocking=True)
            # image_tensor_i_un_norm_old = un_norm_old(image_tensor[i]).to('cpu', non_blocking=True)
            # all_img.append(image_tensor_i_un_norm)
            inpaint_image_i_un_norm = un_norm(inpaint_image[i]).to('cpu', non_blocking=True)
            # inpaint_image_i_un_norm_old = un_norm_old(inpaint_image[i]).to('cpu', non_blocking=True)
            # all_img.append(inpaint_image_i_un_norm)
            ref_img = ref_tensor
            ref_img = resize(ref_img, (H, W))
            ref_image_i_unnorm = un_norm_clip(ref_img[i]).to('cpu', non_blocking=True)
            # ref_image_i_unnorm_old = un_norm_clip_old(ref_img[i]).to('cpu', non_blocking=True)
            # breakpoint()
            # all_img.append(ref_image_i_unnorm)
            # all_img.append(x_sample)

            # Save grid
            grid = torch.stack((image_tensor_i_un_norm, inpaint_image_i_un_norm, ref_image_i_unnorm, x_sample), 0)
            grid = make_grid(grid)
            grid = 255. * rearrange(grid, 'c h w -> h w c').to('cpu', non_blocking=True).numpy()
            img = Image.fromarray(grid.astype(np.uint8))
            fpath_out_grid = os.path.join(grid_path, f'grid-{os.path.splitext(filename)[0]}_{opt.seed}.png')
            img.save(fpath_out_grid)

            # Save result
            x_sample = 255. * rearrange(x_sample.to('cpu', non_blocking=True).numpy(), 'c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(result_path / f'{os.path.splitext(filename)[0]}_{opt.seed}.png')

            # Save mask
            # mask_save = 255. * rearrange(un_norm(mask_tensor[i]).to('cpu', non_blocking=True), 'c h w -> h w c').numpy()
            mask_save = 255. * rearrange(un_norm_old(mask_tensor[i]).to('cpu', non_blocking=True), 'c h w -> h w c').numpy()
            mask_save = np.repeat(mask_save, 3, axis=2)  # Convert to RGB
            mask_save = Image.fromarray(mask_save.astype(np.uint8))
            mask_save.save(sample_path / f'{os.path.splitext(filename)[0]}_{opt.seed}_mask.png')

            # Save original image
            GT_img = 255. * rearrange(image_tensor_i_un_norm, 'c h w -> h w c').numpy()
            GT_img = Image.fromarray(GT_img.astype(np.uint8))
            GT_img.save(sample_path / f'{os.path.splitext(filename)[0]}_{opt.seed}_GT.png')

            # Save inpaint image
            inpaint_img = 255. * rearrange(inpaint_image_i_un_norm, 'c h w -> h w c').numpy()
            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
            inpaint_img.save(sample_path / f'{os.path.splitext(filename)[0]}_{opt.seed}_inpaint.png')

            # Save reference image
            ref_img = 255. * rearrange(ref_image_i_unnorm, 'c h w -> h w c').numpy()
            ref_img = Image.fromarray(ref_img.astype(np.uint8))
            ref_img.save(sample_path / f'{os.path.splitext(filename)[0]}_{opt.seed}_ref.png')

            # Save GT image
            gt_img = 255. * rearrange(ref_image_i_unnorm, 'c h w -> h w c').numpy()
            gt_img = Image.fromarray(gt_img.astype(np.uint8))
            gt_img.save(sample_path / f'{os.path.splitext(filename)[0]}_{opt.seed}_ref.png')

            # print(f'Saved grid to {fpath_out_grid=}')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--fpath_config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--fpath_checkpoint', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--dirpath_empty', type=Path, required=True, help='Path to empty tote images')
    parser.add_argument('--dirpath_mask', type=Path, required=True, help='Path to mask images')
    parser.add_argument('--dirpath_cireps', type=Path, required=True, help='Path to all reference images per fnsku')
    parser.add_argument('--outdir', type=Path, default='results', help='Output directory')
    parser.add_argument('--ddim_steps', type=int, default=50, help='Number of DDIM sampling steps')
    parser.add_argument('--use_plms', action='store_true', help='Use PLMS sampling')
    parser.add_argument('--scale', type=float, default=5.0, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--H', type=int, default=512, help='Image height')
    parser.add_argument('--W', type=int, default=512, help='Image width')
    parser.add_argument('--num_channels_latent', type=int, default=4, help='Latent channels')
    parser.add_argument('--f', type=int, default=8, help='Downsampling factor')
    parser.add_argument('--precision', type=str, choices=('full', 'autocast'), default='autocast', help='Precision')
    return parser.parse_args()


def load_mask_from_image_or_txt(fpath_in_mask: Path, H, W, device) -> Tensor:
    # Load mask from bbox file
    if fpath_in_mask.endswith('.txt'):
        with open(fpath_in_mask, 'r') as f:
            bbox_lines = f.readlines()
        if not bbox_lines:
            raise ValueError(f"Empty bbox file: {fpath_in_mask=}")
        # Use the first bbox in the file
        bbox_coords = [int(float(coord)) for coord in bbox_lines[0].strip().split()]
        if len(bbox_coords) != 4:
            raise ValueError(f"Invalid bbox format in {fpath_in_mask=}. Expected 4 coordinates, got {len(bbox_coords)}")
            # Create mask from bbox
        x_min, y_min, x_max, y_max = bbox_coords
        # Check if bbox is in xywh format (width, height) instead of xyxy (x_max, y_max)
        if x_max < x_min or y_max < y_min:
            # Convert from xywh to xyxy
            raise ValueError(f"Invalid bbox coordinates: {bbox_coords}. Expected format is x_min, y_min, width, height.")
        # Create mask (0 for area to inpaint, 1 for area to keep)
        mask_tensor = torch_ones((1, 1, H, W), dtype=torch_float32, device=device)
        mask_tensor[:, :, y_min:y_max, x_min:x_max] = 0
        # mask = mask[None, None]  # Add batch and channel dimensions
        # mask_tensor = torch.from_numpy(mask).to(device)
    else:
        print(f'Reading mask from a mask image file: {fpath_in_mask=}')
        # Load mask from image file
        # with Image.open(fpath_in_mask) as mask_tensor_pil:
        #     mask_tensor_pil = np.asarray(mask_tensor_pil.convert('L'))
        # mask_tensor = decode_image(fpath_in_mask, mode=ImageReadMode.GRAY).unsqueeze(0).to(device, non_blocking=True)
        mask_tensor = decode_image(fpath_in_mask).unsqueeze(0).to(device=device, dtype=torch_float32, non_blocking=True) == 0 # mask_tensor.dtype: bool. mask_tensor.size(): (1, 1, 512, 512)

        # mask_tensor = to_dtype(mask_tensor, torch_float32, scale=True)
        # from torchvision.transforms.v2.functional import convert_image_dtype
        # breakpoint()
    return mask_tensor.to(torch_float32, non_blocking=True)

def get_model_sampler(fpath_config: Path, fpath_checkpoint: Path, use_plms: bool, device: torch_device | None = None):
    config = OmegaConf.load(fpath_config)
    model = load_model_from_config(config, fpath_checkpoint)
    if device is not None:
        model = model.to(device, non_blocking=True)

    # Choose sampler
    sampler = PLMSSampler(model) if use_plms else DDIMSampler(model)

    return model, sampler


def infer_all(dirpath_input_image_empty: Path, dirpath_input_image_mask: Path, dirpath_cirep: Path, model: Module, sampler, scale: float, precision: str, sample_path, result_path, grid_path):
    '''
    Run inference for all images in the dataset.
    '''

    device = torch_device('cuda')
    get_tensor_fn = get_tensor()
    get_tensor_clip_fn = get_tensor_clip()


    for fpath_empty in tqdm(list(dirpath_input_image_empty.iterdir())):
        fnsku = fpath_empty.stem
        # fpath_ref = dirpath_input_image_empty / fpath_empty.name
        # image_tensor_empty = decode_image(fpath_empty, mode=ImageReadMode.RGB).unsqueeze(0).to(device, non_blocking=True)
        # Load input image
        with Image_open(fpath_empty) as img_p:
            img_p = img_p.convert("RGB")
            W, H = img_p.size
            image_tensor_empty = get_tensor_fn(img_p)
        image_tensor_empty = image_tensor_empty.unsqueeze(0).to(device, non_blocking=True)


        # mask_tensor = load_mask_from_image_or_txt(opt.mask_path, H, W, device)
        dirpath_cirep_fnsku = dirpath_cirep / fnsku
        image_tensor_cirep = get_image_cirep_fnsku(image_tensor_empty, dirpath_cirep_fnsku, get_tensor_clip_fn, device)
        fpath_in_mask = dirpath_input_image_mask / f'{fnsku}.png'
        mask_tensor = decode_image(fpath_in_mask).unsqueeze(0).to(device=device, dtype=torch_float32, non_blocking=True) == 0 # mask_tensor.dtype: bool. mask_tensor.size(): (1, 1, 512, 512)
        # base, mask, ref = image_tensor_empty, image_tensor_base, image_tensor_cirep
        infer_one(image_tensor_empty, mask_tensor, image_tensor_cirep, model, sampler, scale, precision, sample_path, result_path, grid_path, fpath_empty.name)


@inference_mode()
def run_directory(fpath_config: str, fpath_checkpoint: str, use_plms: bool, dirpath_empty: Path, dirpath_mask: Path, dirpath_cireps: Path, outdir: Path, ddim_steps: int, scale: float, seed: int, H: int, W: int, C: int, f: int, precision: str, sample_path, result_path: str, grid_path: str):
    model, sampler = get_model_sampler(Path(fpath_config), Path(fpath_checkpoint), use_plms, device=torch_device('cuda'))
    infer_all(Path(dirpath_empty), Path(dirpath_mask), Path(dirpath_cireps), model, sampler, scale, precision, Path(sample_path), Path(result_path), Path(grid_path))


# @inference_mode()
def infer_batch(model: LatentDiffusion, sampler: DDIMSampler, batch_collated: dict[str, Tensor], scale: float, precision: str, ddim_steps: int, f: int, num_channels_latent: int, device: torch_device) -> Tensor:
    '''
    Run inference for a batch of images with the given model and sampler.
    '''
    if device is None:
        device = torch_device('cuda')
    # inpaint_image = batch_collated['inpaint_image'].to(device, non_blocking=True)
    inpaint_image = batch_collated['images_inpaint'].to(device, non_blocking=True)
    # inpaint_mask = batch_collated['inpaint_mask'].to(device, non_blocking=True)
    inpaint_mask = batch_collated['images_mask'].to(device, non_blocking=True)
    # images_pt_ref = batch_collated['ref_image'].to(torch_float16, non_blocking=True)
    images_pt_ref = batch_collated['images_ref'].to(device, torch_float16, non_blocking=True)

    assert inpaint_image.ndim == 4, f'inpaint_image must be 4D, got {inpaint_image.size()=}'
    # breakpoint()
    # Run inference
    precision_scope = autocast if precision == 'autocast' else nullcontext
    batch_size, _, height, width = inpaint_image.size()

    with inference_mode(), precision_scope('cuda'), model.ema_scope():
        # Get conditioning
        uc = None if scale == 1.0 else model.learnable_vector

        c = model.get_learned_conditioning(images_pt_ref)
        c = model.proj_out(c)

        # Encode inpaint image
        lol = inpaint_image
        print(f'input to model.encode_first_stage: {lol.size()=}, {lol.dtype=}, {lol.device=}, {lol.min()=}, {lol.max()=}, {lol.mean()=}, {lol.std()=}')
        z_inpaint = model.encode_first_stage(inpaint_image)
        mode = z_inpaint.mode()
        print(f'z_inpaint" after model.encode_first_stage(inpaint_image): {mode.size()=}, {mode.dtype=}, {mode.device=}, {mode.min()=}, {mode.max()=}, {mode.mean()=}, {mode.std()=}')
        z_inpaint = model.get_first_stage_encoding(z_inpaint)
        print(f'z_inpaint" after  model.get_first_stage_encoding(z_inpaint): {z_inpaint.size()=}, {z_inpaint.dtype=}, {z_inpaint.device=}, {z_inpaint.min()=}, {z_inpaint.max()=}, {z_inpaint.mean()=}, {z_inpaint.std()=}')

        # Sample
        # shape = (batch_size, num_channels_latent, height // f, width // f)
        shape = (num_channels_latent, height // f, width // f)
        num_samples = 1
        print(f'{sampler=}')
        print(f'Call sampler.sample with {ddim_steps=}, {c.size()=}, {shape=}, {num_samples=}, {scale=}, {uc.size()=}')
        test_model_kwargs = {'images_inpaint': z_inpaint, 'images_mask': resize(inpaint_mask, (z_inpaint.shape[-2:]))}
        for k, v in test_model_kwargs.items():
            print(f'infer_batch: {k=} {v.size()=}, {v.dtype=}, {v.device=}, {v.min()=}, {v.max()=}, {v.mean()=}, {v.std()=}')
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=num_samples,
            shape=shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=0.0,
            x_T=None,
            test_model_kwargs=test_model_kwargs, # TODO: maybe the size across batch isn't even. How does the sampler handle it?
        )

        # Decode samples
        x_samples_ddim = model.decode_first_stage(samples_ddim)
        # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).to('cpu', non_blocking=True)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        print(f'after x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0), {x_samples_ddim.size()=}, {x_samples_ddim.dtype=}, {x_samples_ddim.min()=}, {x_samples_ddim.max()=}, {x_samples_ddim.mean()=}, {x_samples_ddim.std()=}')

        # x_samples_ddim = x_samples_ddim.to('cpu', non_blocking=True).permute(0, 2, 3, 1).numpy()

        # Convert to torch tensor
        # x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

        # Save results
    return x_samples_ddim


def visualize_batch(batch_collated: dict[str, Tensor], images_after_pred: Tensor, result_path: Path, filenames: list[Path], do_save: bool) -> list[Tensor]:
    images_before = batch_collated['images_before']
    # images_inpaint = batch_collated['inpaint_image']
    images_inpaint = batch_collated['images_inpaint']
    # inpaint_mask = batch_collated['inpaint_mask']
    images_mask = batch_collated['images_mask'].expand(-1, 3, -1, -1).to('cpu', non_blocking=True)  # Expand mask to 3 channels for visualization
    # images_ref = batch_collated['ref_image']
    images_ref = batch_collated['images_ref']
    images_after_gt = batch_collated['images_after_gt']

    assert images_before.ndim == 4, f'images_before must be 4D, got {images_before.size()=}'
    # H, W = images_before.size()[-2:]
    batch_size, _, H, W = images_before.size()

    images_after_pred = images_after_pred.to('cpu', non_blocking=True)
    images_before_un_normed = un_norm(images_before).to('cpu', non_blocking=True)
    images_inpaint_un_normed = un_norm(images_inpaint).to('cpu', non_blocking=True)
    images_after_gt_un_normed = un_norm(images_after_gt)
    images_after_gt_un_normed_cpu = images_after_gt_un_normed.to('cpu', non_blocking=True)
    images_ref_un_normed_resized = resize(un_norm_clip(images_ref), (H, W))
    images_ref_un_normed_resized_cpu = images_ref_un_normed_resized.to('cpu', non_blocking=True)

    if do_save is True:
        assert result_path.is_dir(), f'result_path must be a directory, got {result_path=}'
        assert filenames is not None, f'filenames must not be None, got {filenames=}'
    if filenames is None:
        filenames = [None for _ in range(batch_size)]

    # images_stacked = torch_stack((images_before_un_normed, images_mask, images_inpaint_un_normed, images_ref_un_normed_resized, images_after_gt, images_after_pred))
    # print(f'visualize_batch: {images_stacked.size()=}, {images_stacked.dtype=}, {images_stacked.device=}')
    # images_grid = make_grid(images_stacked, nrow=batch_size, scale_each=True)
    image_grids_each = []
    for filenames_i, images_before_un_normed_i, image_mask_i, images_inpaint_un_normed_i, images_ref_un_normed_resized_i, images_after_gt_un_normed_i, images_after_pred_i in zip(filenames, images_before_un_normed, images_mask, images_inpaint_un_normed, images_ref_un_normed_resized_cpu, images_after_gt_un_normed_cpu, images_after_pred):
        # Create grid of all images
        # Save grid
        # print(f'{images_before_un_normed_i.size()=}, {image_mask_i.size()=}, {images_inpaint_un_normed_i.size()=}, {images_ref_un_normed_resized_i.size()=}, {images_after_gt_un_normed_i.size()=}, {images_after_pred_i.size()=}')
        # grid = torch_stack((images_before_un_normed_i.to('cpu', non_blocking=True), image_mask_i.to('cpu', non_blocking=True), images_inpaint_un_normed_i.to('cpu', non_blocking=True), images_ref_un_normed_resized_i.to('cpu', non_blocking=True), images_after_gt_un_normed_i.to('cpu', non_blocking=True), images_after_pred_i.to('cpu', non_blocking=True)), 0)
        grid = torch_stack((images_before_un_normed_i, image_mask_i, images_inpaint_un_normed_i, images_ref_un_normed_resized_i, images_after_gt_un_normed_i, images_after_pred_i), 0)
        image_grid_i = make_grid(grid)
        # image_grid_i = images_grid[i]
        image_grids_each.append(image_grid_i)

        if do_save is True:
            # dirname = os.path.splitext(filenames_i)[0]
            # breakpoint()
            dirname = Path(filenames_i).stem
            dirpath_example = result_path / dirname
            # dirpath_example.mkdir(parents=True, exist_ok=True)
            try:
                dirpath_example.mkdir(parents=True, exist_ok=False)
            except:
                warning(f'Warning: Directory {dirpath_example} already exists. Overwriting files in it.')
                dirpath_example.mkdir(parents=True, exist_ok=True)

            save_image(image_grid_i, dirpath_example / 'grid.jpg')
            save_image(images_before_un_normed_i, dirpath_example / 'image_before.jpg')
            save_image(image_mask_i, dirpath_example / 'image_mask.jpg')
            save_image(images_inpaint_un_normed_i, dirpath_example / 'image_inpaint.jpg')
            save_image(images_ref_un_normed_resized_i, dirpath_example / 'image_ref.jpg')
            save_image(images_after_gt_un_normed_i, dirpath_example / 'image_after_gt.jpg')
            save_image(images_after_pred_i, dirpath_example / 'image_after_pred.jpg')
            # print(f'visualize_batch: Saved grid to {(dirpath_example / 'grid.jpg')=}')
    return image_grids_each, images_ref_un_normed_resized, images_after_gt_un_normed


def run_batch(model: LatentDiffusion, sampler: DDIMSampler, batch_collated: dict[str, Tensor], scale: float, precision: str, ddim_steps: int, f: int, result_path: Path, filenames: list[Path], do_save: bool, num_channels_latent: int, device: torch_device) -> list[Tensor]:
    images_after_pred = infer_batch(model, sampler, batch_collated, scale, precision, ddim_steps, f, num_channels_latent, device)
    images_grid = visualize_batch(batch_collated, images_after_pred, result_path, filenames, do_save)
    return images_grid


# def setup_model_sampler_and_run_batch(fpath_config: Path, fpath_checkpoint: Path, use_plms: bool, outdir: Path) -> list[Tensor]:

#     # Load model and sampler

#     model, sampler = get_model_sampler(fpath_config, fpath_checkpoint, use_plms, device=torch_device('cuda'))

#     # Create output directories
#     images_after_pred = run_batch(
#         model=model,
#         sampler=sampler,
#         batch_collated=batch_collated,
#         scale=opt.scale,
#         precision=opt.precision,
#         ddim_steps=opt.ddim_steps,
#         f=opt.f,
#         result_path=result_path,
#         filenames=filenames,
#         do_save=True
#     )
#     images_grid = visualize_batch(batch_collated, images_after_pred, result_path, filenames, do_save=True)
#     return images_grid


def main():
    args = parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Create output directories
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    sample_path = outdir / 'source'
    sample_path.mkdir(parents=True, exist_ok=True)
    result_path = outdir / 'results'
    result_path.mkdir(parents=True, exist_ok=True)
    grid_path = outdir / 'grid'
    grid_path.mkdir(parents=True, exist_ok=True)
    run(
        fpath_config=args.fpath_config,
        fpath_checkpoint=args.fpath_checkpoint,
        use_plms=args.use_plms,
        dirpath_empty=args.dirpath_empty,
        dirpath_mask=args.dirpath_mask,
        dirpath_cireps=args.dirpath_cireps,
        outdir=outdir,
        ddim_steps=args.ddim_steps,
        scale=args.scale,
        seed=args.seed,
        H=args.H,
        W=args.W,
        C=args.C,
        f=args.f,
        precision=args.precision
    )

if __name__ == "__main__":
    main()
