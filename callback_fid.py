from pathlib import Path
from lightning.pytorch.callbacks import Callback
from torch import Tensor as torch_Tensor, inference_mode as torch_inference_mode, device as torch_device
from torchmetrics.image.fid import FrechetInceptionDistance
from torch import (
    arange as torch_arange,
    cat as torch_cat,
    float32 as torch_float32,
    inference_mode as torch_inference_mode,
    Tensor as torch_Tensor,
    uint8 as torch_uint8,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2.functional import to_dtype
from torchvision.io import decode_image
from pandas import DataFrame
from PIL.Image import Image
from torchvision.ops import masks_to_boxes, roi_align   # NEW (top of file)
from r4_run_inference_batch import visualize_batch


def _bbox_from_mask(mask: torch_Tensor) -> torch_Tensor:
    """
    Binary mask (1xHxW or 3xHxW) ➜ (x0,y0,x1,y1). Empty mask → full frame.
    """
    if mask.ndim == 4:                     # (B,1,H,W) -> (B,H,W)
        assert mask.size(1) == 1, f'Expected mask to be (B,1,H,W) if ndim==4, got {mask.size()=}'
        mask = mask.squeeze(1)
    try:
        assert mask.ndim == 3, f'Expected masks to be (B,H,W), got {mask.size()=}'
    except:
        breakpoint()
    return masks_to_boxes(mask)


def _roi_crop_and_resize(imgs: torch_Tensor, bboxes_or_masks: torch_Tensor) -> torch_Tensor:
    """
    imgs  : (B,3,H,W) float in [0,1]  or uint8/float
    masks : (B,1,H,W) or (B,H,W)  binary / {0,1}
    Returns (B,3,299,299) uint8
    """
    # if masks.ndim == 3:                     # (B,H,W) -> (B,1,H,W)
    #     masks = masks.unsqueeze(1)
    # if masks.ndim == 4:                     # (B,1,H,W) -> (B,H,W)
    #     assert masks.size(1) == 1, f'Expected masks to be (B,1,H,W) if ndim==4, got {masks.size()=}'
    #     masks = masks.squeeze(1)
    # assert masks.ndim == 3, f'Expected masks to be (B,H,W), got {masks.size()=}'

    # # breakpoint()
    # boxes = masks_to_boxes(masks)           # (B,4) xyxy, float32
    if bboxes_or_masks.ndim == 4:            # (B,1,H,W) -> (B,H,W)
        # masks
        assert bboxes_or_masks.size(1) == 1, f'Expected masks to be (B,1,H,W) if ndim==4, got {bboxes_or_masks.size()=}'
        masks = bboxes_or_masks.squeeze(1)
        boxes = _bbox_from_mask(masks)           # (B,4) xyxy, float32
    else:
        assert bboxes_or_masks.ndim == 2 and bboxes_or_masks.size(1) == 4, f'Expected bboxes to be (B,4), got {bboxes_or_masks.size()=}'
        boxes = bboxes_or_masks  # (B,4) xyxy, float32

    B = imgs.size(0)
    try:
        rois = torch_cat((torch_arange(B, device=imgs.device, dtype=torch_float32).unsqueeze(1), boxes), dim=1)
    except  Exception as e:
        breakpoint()

    crops = roi_align(
        imgs.float(),                       # expects float32
        rois,
        output_size=(299, 299),             # same as default Inception input (3, 299, 299)
        spatial_scale=1.0,
        aligned=True,
    )                                       # (B,3,299,299) float32 [0,1]

    # return (crops * 255.0).clamp(0, 255).to(torch_uint8)   # TorchMetrics wants uint8
    return to_dtype(crops, torch_uint8, scale=True)  # (B,3,299,299) uint8 [0,255]


class FIDCallback(Callback):
    """
    FID over train/val/test (per-epoch + full-run).
    Automatically normalizes images if they are in [0,255] range.

    Args:
        batch_real_key: key in `batch` for ground-truth images [B,3,H,W].
        outputs_fake_key: key in `outputs` for generated images [B,3,H,W].
        feature: Inception feature dim (2048 typical).
        prefix: metric name prefix.
    """
    def __init__(self, split_fnames: dict[str, DataFrame] = None, result_path: Path = None, batch_real_key: str = "images_after_gt", outputs_fake_key: str = "imgs_after_pred_unnorm", feature: int = 2048, prefix: str = "fid"):
        super().__init__()
        self.batch_real_key = batch_real_key
        self.outputs_fake_key = outputs_fake_key
        self.prefix = prefix
        self.feature = feature
        # normalize=True ensures inputs in [0,255] get rescaled to [0,1]
        self.split_fnames = split_fnames
        self.result_path = result_path
        self.filenames_test = split_fnames['test']['pair_name']
        with torch_inference_mode():
            self.fid_test_global = FrechetInceptionDistance(feature=feature, normalize=True, sync_on_compute=False).eval().requires_grad_(False)
            self.fid_test_local = FrechetInceptionDistance(feature=feature, normalize=True, sync_on_compute=False).eval().requires_grad_(False)
            self.fid_test_ref = FrechetInceptionDistance(feature=feature, normalize=True, sync_on_compute=False).eval().requires_grad_(False)

    # --- device / resets ---
    def on_fit_start(self, trainer, pl_module) -> None:
        self._move_all_to(pl_module.device)
        for s in ("train", "val", "test"):
            self.fid_epoch[s].reset()
            self.fid_full[s].reset()

    # --- train ---
    def on_train_epoch_start(self, trainer, pl_module):
        self.fid_epoch["train"].reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        real = batch[self.batch_real_key].to(pl_module.device, non_blocking=True)
        fake = outputs[self.outputs_fake_key].to(pl_module.device, non_blocking=True)
        self._update_stage("train", real, fake)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log(f"{self.prefix}/train", self.fid_train.compute(), prog_bar=True, sync_dist=True)

    # --- val ---
    def on_validation_epoch_start(self, trainer, pl_module):
        device = pl_module.device
        self.fid_test_global.to(device, non_blocking=True).reset()
        fid_test = self.fid_test.to(pl_module.device, non_blocking=True)
        # self.fid_epoch["val"].reset()
        fid_test.reset()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        real = batch[self.batch_real_key].to(pl_module.device, non_blocking=True)
        fake = outputs[self.outputs_fake_key].to(pl_module.device, non_blocking=True)
        self._update_stage("val", real, fake)

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.log('val/fid', self.fid_epoch["val"].compute(), prog_bar=True, sync_dist=True)
    # def on_validation_end(self, trainer, pl_module):
    #     pl_module.log(f"{self.prefix}/val_full_run", self.fid_full["val"].compute(), sync_dist=True)

    # --- test ---

    def on_test_epoch_start(self, _trainer, pl_module) -> None:
        device = pl_module.device
        self.fid_test_global.to(device, non_blocking=True).reset()
        self.fid_test_local.to(device, non_blocking=True).reset()
        self.fid_test_ref.to(device, non_blocking=True).reset()
    #     self.fid_epoch["test"].reset()
    # def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
    #     print('on_test_batch_start')
    #     fid_test = self.fid_test.to(pl_module.device, non_blocking=True)
    #     fid_test.reset()
        # self.fid_batch['test'].reset()

    def on_test_batch_end(self, trainer, pl_module, outputs: dict[str, torch_Tensor], batch, _batch_idx, _dataloader_idx=0):
        # print('on_test_batch_end')
        # breakpoint()
        device = pl_module.device

        images_mask = batch['images_mask'].to(device, non_blocking=True)
        images_ref = batch['images_ref'].to(device, non_blocking=True)
        index = batch['index'].to('cpu', non_blocking=True).numpy()

        # 1. Global FIDs
        images_after_gt_global = batch[self.batch_real_key].to(device, non_blocking=True)
        images_after_pred_global = outputs[self.outputs_fake_key].to(device, non_blocking=True)
        self.fid_test_global.update(images_after_gt_global, real=True)
        self.fid_test_global.update(images_after_pred_global, real=False)

        # 2. Local FIDs
        images_after_gt_local = _roi_crop_and_resize(images_after_gt_global, images_mask)
        images_after_pred_local = _roi_crop_and_resize(images_after_pred_global, images_mask)   # uint8
        assert len(images_after_gt_global) == len(images_after_pred_global) == len(images_mask) == len(images_ref), f'Batch size mismatch: {len(images_after_gt_global)=}, {len(images_after_pred_local)=}, {len(images_mask)=}, {len(images_ref)=}'
        self.fid_test_local.update(images_after_gt_local, real=True)
        self.fid_test_local.update(images_after_pred_local, real=False)

        # 3. Reference vs Local FIDs
        self.fid_test_ref.update(images_ref, real=True)
        self.fid_test_ref.update(images_after_pred_local, real=False)
        _images_grid = visualize_batch(batch, images_after_pred_global, self.result_path, self.filenames_test[index], do_save=True)


    def on_test_epoch_end(self, _trainer, pl_module):
        # print('on_test_epoch_end')
        # pl_module.log('test/fid', self.fid_test.compute(), prog_bar=True, sync_dist=True)
        pl_module.log_dict({
            'test/fid_global': float(self.fid_test_global.compute()),
            'test/fid_local': float(self.fid_test_local.compute()),
            'test/fid_ref': float(self.fid_test_ref.compute()),
        }, prog_bar=True, sync_dist=True)
        self.fid_test_global.to('cpu', non_blocking=True).reset()
        self.fid_test_local.to('cpu', non_blocking=True).reset()
        self.fid_test_ref.to('cpu', non_blocking=True).reset()

    # def on_test_end(self, trainer, pl_module):
    #     print('on_test_end')
    #     pl_module.log('test/fid', self.fid_test.compute(), sync_dist=True)

# class FIDCallback(Callback):
#     # def __init__(self, feature: int = 2048, prefix: str = "fid"):
#     def __init__(self, feature: int = 2048, prefix: str = "fid"):
#         super().__init__()
#         self.prefix = prefix
#         self.fid = FrechetInceptionDistance(feature=feature, normalize=True, sync_on_compute=False).requires_grad_(False)

#     def on_fit_start(self, trainer, pl_module):
#         self.fid = self.fid.to(pl_module.device)
#         self.fid.reset()

#     def on_train_epoch_start(self, trainer, pl_module):
#         self.fid.reset()

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         # <-- adapt these two lines to your code structure
#         device = pl_module.device
#         real = batch['image_after_gt'].to(device, non_blocking=True)
#         fake = outputs['image_after_pred'].to(device, non_blocking=True)

#         self.fid.update(real, real=True)
#         self.fid.update(fake, real=False)

#     def on_train_epoch_end(self, trainer, pl_module):
#         value = self.fid.compute()
#         pl_module.log('train/fid', value, prog_bar=True, sync_dist=True)


    # @torch_inference_mode()
    # def validation_step(self, batch, batch_idx):
    #     _, loss_dict_no_ema = self.shared_step(batch)
    #     with self.ema_scope():
    #         _, loss_dict_ema = self.shared_step(batch)
    #         loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
    #     # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    #     self.log_dict(loss_dict_no_ema | loss_dict_ema, prog_bar=True, logger=USE_LOGGER_PL, on_step=True, on_epoch=True)
    #     # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    #     # self.log_dict(loss_dict_ema, prog_bar=False, logger=USE_LOGGER_PL, on_step=False, on_epoch=True)
    #     # if self.trainer.global_rank == 0:
    #         # print(f'DDPM.validation_step: {self.trainer.global_rank == 0=}', 'log_images')
    #     global_step = self.global_step if hasattr(self, 'global_step') else self.trainer.global_step
    #     log_every_n_steps = self.log_every_n_steps if hasattr(self, 'log_every_n_steps') else self.trainer.log_every_n_steps
    #     # if global_step % log_every_n_steps == 0 and rank_zero_only.rank == 0:
    #     if global_step % log_every_n_steps == 0:
    #         self.log_images(batch, N=4, n_row=2, sample=True, split='val', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
