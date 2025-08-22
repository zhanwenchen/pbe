'''
Clean LatentDiffusion implementation for inpainting without PyTorch Lightning dependencies.

This module provides a simplified, standalone implementation of latent diffusion
for inpainting tasks, following the core algorithmic flow without framework overhead.
'''
# from itertools import chain
from random import random
from einops import rearrange
from numpy import log as np_log
from torch import (
    as_tensor as torch_as_tensor, # type: ignore
    Tensor as torch_Tensor, # type: ignore
    cat as torch_cat, # type: ignore
    randint as torch_randint, # type: ignore
    randn as torch_randn, # type: ignore
    randn_like as torch_randn_like, # type: ignore
    no_grad as torch_no_grad, # type: ignore
    linspace as torch_linspace, # type: ignore
    float32 as torch_float32, # type: ignore
    stack as torch_stack, # type: ignore
    full as torch_full, # type: ignore
    int64 as torch_int64, # type: ignore
    round as torch_round, # type: ignore
    # vstack as torch_vstack, # type: ignore
    randint as torch_randint, # type: ignore
    exp as torch_exp, # type: ignore
    inference_mode as torch_inference_mode, # type: ignore
    arange as torch_arange, # type: ignore
    clip as torch_clip, # type: ignore
    argmax as torch_argmax, # type: ignore
    min as torch_min, # type: ignore
)
from torch.nn import Fold, Linear, Parameter, Unfold
from torch.nn.functional import conv2d, dropout
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.v2 import Normalize
from torchvision.transforms.v2.functional import resize # type: ignore
from tqdm import tqdm # type: ignore
from ldm.models.diffusion.ddpm import DDPM
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.modules.diffusionmodules.util import extract_into_tensor, noise_like
from ldm.util import exists, default, mean_flat, instantiate_from_config
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddim import DDIMSampler

# from ldm.util import exists, default, count_params, instantiate_from_config

DDIM_STEPS_LOGGING = 50
USE_LOGGER_PL = True


COLUMNS = ['split', 'global_step', 's3path_image_before', 'image_before_masked', 'image_cirep', 'image_after_gt', 'image_after_pred']
__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

class UnNormalize(Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m/s for m, s in zip(mean, std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

MEAN = (0.5, 0.5, 0.5)
MEAN_CLIP = (0.48145466, 0.4578275, 0.40821073)
STD = (0.5, 0.5, 0.5)
STD_CLIP = (0.26862954, 0.26130258, 0.27577711)
ALPHA = 0.4
DDIM_STEPS_LOGGING = 50

# norm = Normalize(MEAN, STD)
norm_clip = Normalize(MEAN_CLIP, STD_CLIP)

un_norm = UnNormalize(MEAN, STD)
un_norm_clip = UnNormalize(MEAN_CLIP, STD_CLIP)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        self.num_timesteps_cond = num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        print(f"LatentDiffusion.__init__: {self.scale_by_std=}, {scale_factor=}")
        assert num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.learnable_vector = Parameter(torch_randn((1,1,768)), requires_grad=True)
        self.proj_out = Linear(1024, 768)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch_as_tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.strict_loading = False

        # with torch_no_grad():
        #     # self.fid = FrechetInceptionDistance(feature=64, normalize=True).eval().to(self.device, non_blocking=True)
        #     # self.fid = FrechetInceptionDistance(feature=64, normalize=True).eval().to(self.device, non_blocking=True)
        #     self.fid = FrechetInceptionDistance(sync_on_compute=False).requires_grad_(False)
        # self.fid = fid
        # self.wandb_image_logger = wandb_image_logger = WandbImageLogger()
        # self.log_image_pair = wandb_image_logger.log_image_pair
        # self.logger_experiment_log = self.logger.experiment.log
        # self.logger.experiment.define_metric('')
                # if trainer.global_step == 0:
            # wandb.define_metric("val_accuracy", summary="max")
    def on_train_start(self, *args, **kwargs) -> None:
        self.logger.experiment.define_metric('*', step_metric='trainer/global_step')
        self.logger.experiment.log_code('.')

    def make_cond_schedule(self):
        num_timesteps = self.num_timesteps
        self.cond_ids = torch_full(size=(num_timesteps,), fill_value=num_timesteps - 1, dtype=torch_int64)
        num_timesteps_cond = self.num_timesteps_cond
        ids = torch_round(torch_linspace(0, num_timesteps - 1, num_timesteps_cond)).to(torch_int64, non_blocking=True)
        self.cond_ids[:num_timesteps_cond] = ids

    # @rank_zero_only
    # @torch_no_grad()
    # def on_train_batch_start(self, batch, batch_idx):
    #     # only for very first batch
    #     if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
    #         assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
    #         # set rescale weight to 1./std of encodings
    #         print("### USING STD-RESCALING ###")
    #         x = super().get_input(batch, self.first_stage_key)
    #         x = x.to(self.device, non_blocking=True)
    #         encoder_posterior = self.encode_first_stage(x)
    #         z = self.get_first_stage_encoding(encoder_posterior).detach()
    #         del self.scale_factor
    #         self.register_buffer('scale_factor', 1. / z.flatten().std())
    #         print(f"setting self.scale_factor to {self.scale_factor}")
    #         print("### USING STD-RESCALING ###")

    # def training_step(self, batch, batch_idx):
    #     loss = super().training_step(batch, batch_idx)
    #     if batch_idx % self.log_every_t_image_train == 0:
    #         # if self.trainer.global_rank == 0:
    #         # print(f'{getenv("LOCAL_RANK")=} LatentDiffusion.training_step: {self.trainer.global_rank=} calling log_images')

    #         with torch_inference_mode():
    #             self.log_images(batch, N=4, n_row=2, sample=True, split='train', ddim_steps=DDIM_STEPS_LOGGING)
    #             dict_images = self.log_images(batch, N=4, n_row=2, sample=True)
    #             dict_images = self.log_images(batch, N=4, n_row=2, sample=True)
    #             self.log_dict(dict_images)
    #         print(f'{getenv("LOCAL_RANK")=} LatentDiffusion.training_step: {self.trainer.global_rank=} DONE log_images')

    #     return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
    #     super().on_train_batch_end(self, outputs, batch, batch_idx)
    #     global_step = self.global_step if hasattr(self, 'global_step') else self.trainer.global_step
    #     log_every_n_steps = self.log_every_n_steps if hasattr(self, 'log_every_n_steps') else self.trainer.log_every_n_steps
    #     if global_step % log_every_n_steps == 0:
    #     # if self.global_step % self.log_every_t == 0 and rank_zero_only.rank == 0:
    #         # with torch_inference_mode():
    #         self.log_images(batch, N=4, n_row=2, sample=True, split='train', ddim_steps=DDIM_STEPS_LOGGING)
    #     if batch_idx % self.log_every_t_image_train == 0:
    #         # if self.trainer.global_rank == 0:
    #         # print(f'{getenv("LOCAL_RANK")=} LatentDiffusion.training_step: {self.trainer.global_rank=} calling log_images')

    #         with torch_inference_mode():
    #             self.log_images(batch, N=4, n_row=2, sample=True, split='train', ddim_steps=DDIM_STEPS_LOGGING)

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = shorten_cond_schedule = self.num_timesteps_cond > 1
        if shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        # breakpoint()
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config not in ('__is_first_stage__', '__is_unconditional__')
            self.cond_stage_model = instantiate_from_config(config)

    # def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
    #     denoise_row = [self.decode_first_stage(zd.to(self.device, non_blocking=True), force_not_quantize=force_no_decoder_quantization) for zd in tqdm(samples, desc=desc)]
    #     # n_imgs_per_row = len(denoise_row)
    #     # denoise_row = torch_vstack(denoise_row)  # n_log_step, n_row, C, H, W
    #     denoise_row = torch_stack(denoise_row, axis=1)  # n_log_step, n_row, C, H, W
    #     b, n, c, h, w = denoise_row.size()
    #     denoise_row = denoise_row.view(b * n, c, h, w)  # (b * n) c h w
    #     # denoise_row = rearrange(denoise_row, 'n b c h w -> b n c h w')
    #     # denoise_row = rearrange(denoise_row, 'b n c h w -> (b n) c h w')
    #     # denoise_row = make_grid(denoise_row, nrow=n_imgs_per_row)
    #     denoise_row = make_grid(denoise_row, nrow=n)
    #     return denoise_row

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch_Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        cond_stage_model = self.cond_stage_model
        if self.cond_stage_forward is None:
            if hasattr(cond_stage_model, 'encode') and callable(cond_stage_model.encode):
                c = cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = cond_stage_model(c)
        else:
            assert hasattr(cond_stage_model, self.cond_stage_forward)
            c = getattr(cond_stage_model, self.cond_stage_forward)(c)
        return c


    def meshgrid(self, h, w):
        y = torch_arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch_arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch_cat((y, x), dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch_as_tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch_min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch_min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch_min(torch_cat((dist_left_up, dist_right_down), dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device, dtype):
        weighting = self.delta_border(h, w)
        split_input_params = self.split_input_params
        weighting = torch_clip(weighting, split_input_params["clip_min_weight"],
                               split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device, dtype, non_blocking=True)

        if split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch_clip(L_weighting,
                                     split_input_params["clip_min_tie_weight"],
                                     split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device, dtype, non_blocking=True)
            weighting *= L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = Unfold(**fold_params)

            fold = Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device, x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device, x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device, x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting
    # val_loop.run() => ddpm.validation_step (b/c ld doesn't have validation step) => ld.shared_step (WHAT?? Not ddpm.shared_step?) => ld.forward
    # x ld.validation_step => ld.shared_step => ld.forward => validation_step
    # @torch_inference_mode()
    @torch_no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None,get_mask=False,get_reference=False):

        x,inpaint,mask,reference = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
            inpaint = inpaint[:bs]
            mask = mask[:bs]
            reference = reference[:bs]
        device = self.device
        x = x.to(device, non_blocking=True)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        encoder_posterior_inpaint = self.encode_first_stage(inpaint) # torch.Size([4, 1, 512, 512])
        z_inpaint = self.get_first_stage_encoding(encoder_posterior_inpaint).detach()
        z_shape_neg1 = z.shape[-1]
        # mask_resize = Resize([z_shape_neg1, z_shape_neg1])(mask)
        mask_resize = resize(mask, (z_shape_neg1, z_shape_neg1))
        z_new = torch_cat((z, z_inpaint,mask_resize), dim=1)
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in {'txt','caption', 'coordinates_bbox'}:
                    xc = batch[cond_key]
                elif cond_key == 'image':
                    xc = reference
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(device, non_blocking=True)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, (dict, list)):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(device, non_blocking=True))
                    c = self.proj_out(c).to(dtype=torch_float32, non_blocking=True)
                    # c = c.float()
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z_new, c]
        if return_first_stage_outputs:
            if self.first_stage_key=='inpaint':
                xrec = self.decode_first_stage(z[:,:4,:,:])
            else:
                xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        if get_mask:
            out.append(mask)
        if get_reference:
            out.append(reference)
        return out

    @torch_no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch_argmax(z.exp(), dim=1).to(torch_int64, non_blocking=True)
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            # z = rearrange(z, 'b h w c -> b c h w').contiguous()
            z = z.moveaxis(3, 1).contiguous()

        # z = 1. / self.scale_factor * z
        z *= 1. / self.scale_factor

        if hasattr(self, "split_input_params"):
            split_input_params = self.split_input_params
            if split_input_params["patch_distributed_vq"]:
                ks = split_input_params["ks"]  # eg. (128, 128)
                stride = split_input_params["stride"]  # eg. (64, 64)
                uf = split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                # if isinstance(self.first_stage_model, VQModelInterface):
                #     output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                #                                                  force_not_quantize=predict_cids or force_not_quantize)
                #                    for i in range(z.shape[-1])]
                # else:
                first_stage_model = self.first_stage_model
                assert isinstance(first_stage_model, AutoencoderKL)
                first_stage_model_decode = first_stage_model.decode
                output_list = [first_stage_model_decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch_stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o *= weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded /= normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                assert isinstance(self.first_stage_model, AutoencoderKL)
                # if isinstance(self.first_stage_model, AutoencoderKL):
                #     return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                return self.first_stage_model.decode(z)

        else:
            assert isinstance(self.first_stage_model, AutoencoderKL)
            # if isinstance(self.first_stage_model, VQModelInterface):
            #     return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            if self.first_stage_key=='inpaint':
                return self.first_stage_model.decode(z[:,:4,:,:])
            return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch_argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            # z = rearrange(z, 'b h w c -> b c h w').contiguous()
            # z = rearrange(z, 'b h w c -> b c h w').contiguous()
            z = z.moveaxis(3, 1).contiguous()
        z *= 1. / self.scale_factor

        if hasattr(self, 'split_input_params'):
            split_input_params = self.split_input_params
            if split_input_params["patch_distributed_vq"]:
                ks = split_input_params["ks"]  # eg. (128, 128)
                stride = split_input_params["stride"]  # eg. (64, 64)
                uf = split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print('reducing Kernel')

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                # if isinstance(self.first_stage_model, VQModelInterface):
                #     output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                #                                                  force_not_quantize=predict_cids or force_not_quantize)
                #                    for i in range(z.shape[-1])]
                # else:

                first_stage_model = self.first_stage_model
                assert isinstance(first_stage_model, AutoencoderKL)
                output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

                o = torch_stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o *= weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded /= normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                assert isinstance(self.first_stage_model, AutoencoderKL)
                    # return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                return self.first_stage_model.decode(z)

        else:
            assert isinstance(self.first_stage_model, AutoencoderKL)
                # return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            return self.first_stage_model.decode(z)

    @torch_no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            split_input_params = self.split_input_params
            if split_input_params["patch_distributed_vq"]:
                ks = split_input_params["ks"]  # eg. (128, 128)
                stride = split_input_params["stride"]  # eg. (64, 64)
                df = split_input_params["vqf"]
                split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch_stack(output_list, axis=-1)
                o *= weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded /= normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key) # why in ddpm.shared step instead of here?
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch_randint(0, self.num_timesteps, (x.size(0),), device=self.device, dtype=torch_int64)
        # self.u_cond_prop=random(0, 1)
        self.u_cond_prop=random()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
                c = self.proj_out(c)

            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device, non_blocking=True)
                c = self.q_sample(x_start=c, t=tc, noise=torch_randn_like(c, dtype=torch_float32))

        if self.u_cond_prop<self.u_cond_percent:
            return self.p_losses(x, self.learnable_vector.repeat(x.size(0),1,1), t, *args, **kwargs)
        else:
            return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict): # true
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, 'split_input_params'):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            split_input_params = self.split_input_params
            ks = split_input_params["ks"]  # eg. (128, 128)
            stride = split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]
            # len(cond['c_crossattn']): 1
            # c_crossattn = cond['c_crossattn'][0].size() # torch.Size([4, 1, 768])
            # c_crossattn_0 = c_crossattn[0].size()
            if self.cond_stage_key in {"image", "LR_image", "segmentation", 'bbox_img'} and self.model.conditioning_key:  # todo check for completeness # image, crossattn
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left positions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch_as_tensor(self.bbox_tokenizer._crop_encoder(bbox), dtype=torch_int64, device=self.device, non_blocking=True).unsqueeze(0)  # (1, 2)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device, non_blocking=True)  # cut last two tokens (x_tl, y_tl)

                adapted_cond = torch_stack([torch_cat((cut_cond, p), dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch_stack(output_list, axis=-1)
            o *= weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch_as_tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np_log(2.0)

    def p_losses(self, x_start, cond, t, noise=None): # noise: torch.Size([4, 4, 64, 64])
        if self.first_stage_key == 'inpaint':
            noise = default(noise, lambda: torch_randn_like(x_start[:,:4,:,:]))
            x_noisy = self.q_sample(x_start=x_start[:,:4,:,:], t=t, noise=noise)
            x_noisy = torch_cat((x_noisy,x_start[:,4:,:,:]),dim=1)
        else:
            noise = default(noise, lambda: torch_randn_like(x_start))
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)


        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)) # model_output is none?
        loss_dict = {
            f'{prefix}/loss_simple': loss_simple.mean().item(),
        }
        # loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # breakpoint()
        logvar_t = self.logvar[t]
        loss = loss_simple / torch_exp(logvar_t) + logvar_t
        # loss = loss_simple / torch_exp(self.logvar) + self.logvar
        loss_mean = loss.mean()
        if self.learn_logvar:
            loss_dict[f'{prefix}/loss_gamma'] = loss_mean.item()
            # loss_dict.update({f'{prefix}/loss_gamma': loss.mean(), 'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss_mean

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss_vlb = self.lvlb_weights[t] * self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)).mean()
        loss_dict[f'{prefix}/loss_vlb'] = loss_vlb.item()
        # loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        # loss_dict.update({f'{prefix}/loss': loss})
        loss_dict[f'{prefix}/loss'] = loss.item()

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            # x_recon, _, [_, _, _] = self.first_stage_model.quantize(x_recon)
            x_recon, *_ = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch_no_grad()
    # @torch_inference_mode()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        # b, *_, device = *x.shape, x.device
        b = x.size(0)
        device = x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            # model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch_no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch_randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if isinstance(temperature, float):
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch_full((b,), i, device=self.device, dtype=torch_int64)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device, non_blocking=True)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch_randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if intermediates and i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)
        return img, intermediates

    @torch_no_grad()
    # @torch_inference_mode()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch_randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img] if return_intermediates else None
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        shorten_cond_schedule = self.shorten_cond_schedule
        for i in iterator:
            ts = torch_full((b,), i, device=device, dtype=torch_int64)
            if shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device, non_blocking=True)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch_randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if return_intermediates and i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch_no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch_no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps, return_intermediates=True,**kwargs):

        shape = (self.channels, self.image_size, self.image_size)
        if ddim:
            # self.ddim_sampler = ddim_sampler = DDIMSampler(self) if not hasattr(self, 'ddim_sampler') or self.ddim_sampler is None else self.ddim_sampler
            ddim_sampler = DDIMSampler(self)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,disable_tqdm=True,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, shape=shape,
                                                 return_intermediates=return_intermediates,disable_tqdm=True,**kwargs)

        return samples, intermediates

    @torch_no_grad()
    def log_images(self, batch, N=-1, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., use_ddim=True, test_model_kwargs=None, return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=True, split: str = None, **kwargs) -> dict[str, torch_Tensor]:

        # <<< FIX: HELPER FUNCTION TO NORMALIZE IMAGES FOR LOGGING >>>
        # Models output in [-1, 1] range, but visualization needs [0, 1]
        # def normalize_for_log(img_tensor):
        #     return torch_clamp((img_tensor + 1.0) / 2.0, min=0.0, max=1.0)

        # use_ddim = ddim_steps is not None

        z, c, x, xrec_raw, xc, mask, reference = self.get_input(
            batch, self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            get_mask=True,
            get_reference=True,
            bs=N if N > 0 else None,
        )

        len_x = x.size(0)
        N = min(len_x, N) if N > 0 else len_x
        n_row = min(len_x, n_row)
        # try:
        assert len(z) == len(c) == len(x), f'Batch size mismatch: {len(z)=}, {len(c)=}, {len(x)=}'
        # except AssertionError as e:
        #     print(f'Batch size mismatch: {len(z)=}, {len(c)=}, {len(x)=}')
        #     [print(f'{k=}: {v.size()=}') for k, v in batch.items()]
        #     breakpoint()

        # if self.model.conditioning_key is not None:
        #     if hasattr(self.cond_stage_model, 'decode'):
        #         xc = self.cond_stage_model.decode(c)

        # logger_experiment_log = self.logger.experiment.log
        if sample:
            with self.ema_scope('Plotting'):
                if self.first_stage_key == 'inpaint':
                    samples_z, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                   ddim_steps=ddim_steps, eta=ddim_eta, rest=z[:, 4:, :, :], return_intermediates=False, unconditional_guidance_scale=5.0, unconditional_conditioning=unconditional_conditioning, test_model_kwargs=test_model_kwargs)
                else:
                    samples_z, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                   ddim_steps=ddim_steps, eta=ddim_eta, return_intermediates=False)

            samples_raw = self.decode_first_stage(samples_z)

            # if rank_zero_only.rank == 0:
            # print(f"{getenv("LOCAL_RANK")=} [{split}/raw_sample_stats] min: {samples_raw.min():.3f}, max: {samples_raw.max():.3f}, mean: {samples_raw.mean():.3f}")

            # Normalize samples for logging
            # imgs_after_pred_unnorm = normalize_for_log(samples_raw)
            imgs_after_pred_unnorm = un_norm(samples_raw).clamp(min=0, max=1)
            # log['samples'] = imgs_after_pred_unnorm

            # <<< COMBINED SOLUTION: CREATE THE INPAINTING RESULTS GRID >>>
            # 1. Prepare mask for visualization (ensure 3 channels for RGB)
            # vis_mask = mask if mask.shape[1] == 3 else mask.repeat(1, 3, 1, 1)

            # 2. Create masked input using the ORIGINAL [-1, 1] ground truth image
            # masked_input = x * (1. - mask)

            # 3. Assemble all 4 images for each sample into a list for the grid

            # COLUMNS = ['split', 's3path_image_before', 'image_before_masked', 'image_after_gt', 'image_after_pred']
            # table_image = wandb_Table(columns=COLUMNS)

            # ref_imgs = batch['images_ref']
            # mask_bool = mask.squeeze(1) < 0.5
            # # image_before_unnorm = un_norm(masked_input).clamp(min=0, max=1)  # Ensure masked input is in [0, 1] range for visualization
            # image_before_unnorm = un_norm(batch['images_inpaint']).clamp(min=0, max=1)  # Ensure GT is in [0, 1] range for visualization
            # image_after_gt_unnorm = un_norm(x).clamp(min=0, max=1)  # Ensure GT is in [0, 1] range for visualization
            # image_cirep_i = un_norm_clip(ref_imgs[i])
            # image_cirep_unnorm = resize(un_norm_clip(ref_imgs), image_after_gt_unnorm.size()[-2:])
            # image_cirep_unnorm = resize(un_norm_clip(ref_imgs), (512, 512))
            # image_cirep_unnorm = un_norm_clip(ref_imgs)
            # grid_items = []
            # global_step = self.global_step if hasattr(self, 'global_step') else self.trainer.global_step
            # log_every_n_steps = self.log_every_n_steps if hasattr(self, 'log_every_n_steps') else self.trainer.log_every_n_steps
            # if global_step % log_every_n_steps == 0 and rank_zero_only.rank == 0:
            # if global_step % log_every_n_steps == 0:
            # for i in range(N):
            #     # Order: [Masked Input, Mask, Ground Truth, Prediction]
            #     # Normalize each component for visualization in the grid
            #     s3path_image_before = 'TODO'
            #     # image_before_masked = normalize_for_log(masked_input[i])
            #     # breakpoint()
            #     mask_bool_i = mask_bool[i]
            #     image_before_unnorm_i_masked = draw_segmentation_masks(image_before_unnorm[i], mask_bool_i, alpha=ALPHA, colors='red')

            #     image_cirep_i = image_cirep_unnorm[i]
            #     image_after_gt_i = image_after_gt_unnorm[i]
            #     image_after_pred_i = imgs_after_pred_unnorm[i]
                    # grid_items.extend([image_before_unnorm_i_masked, image_cirep_i, image_after_gt_i, image_after_pred_i])
                    # grid_items.append(normalize_for_log(masked_input[i]))
                    # grid_items.append(normalize_for_log(masked_input[i]))
                    # grid_items.append(normalize_for_log(vis_mask[i]))
                    # grid_items.append(normalize_for_log(x[i]))
                    # grid_items.append(imgs_after_pred_unnorm[i]) # Already normalized
                    # wandb_Image(image_data, masks={"": {"mask_data": mask_data}})

                    # table_image.add_data(split, global_step, s3path_image_before, wandb_Image(image_before_unnorm_i_masked), wandb_Image(image_cirep_i), wandb_Image(image_after_gt_i), wandb_Image(image_after_pred_i))
                # logger_experiment_log({f'{split}/inpainting_results_table': table_image}, step=global_step)
        return {'imgs_after_pred_unnorm': imgs_after_pred_unnorm}

    @torch_inference_mode()
    def validation_step(self, batch, batch_idx) -> torch_Tensor:
        _, loss_dict_no_ema, _ = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema, _ = self.shared_step(batch)
            # loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            loss_dict_ema = {f'{key}_ema': v for key, v in loss_dict_ema.items()}
        # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # dict_merged = loss_dict_no_ema | loss_dict_ema
        # dict_merged = {k.replace('val/', 'test/'): v for k, v in dict_merged.items()}
        self.log_dict(loss_dict_no_ema | loss_dict_ema, prog_bar=True, logger=USE_LOGGER_PL, on_step=True, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=USE_LOGGER_PL, on_step=False, on_epoch=True)
        # if self.trainer.global_rank == 0:
            # print(f'DDPM.validation_step: {self.trainer.global_rank == 0=}', 'log_images')
        # global_step = self.global_step if hasattr(self, 'global_step') else self.trainer.global_step
        # log_every_n_steps = self.log_every_n_steps if hasattr(self, 'log_every_n_steps') else self.trainer.log_every_n_steps
        # if global_step % log_every_n_steps == 0 and rank_zero_only.rank == 0:
        # if global_step % log_every_n_steps == 0:
        # images = self.log_images(batch, N=4, n_row=2, sample=True, split='val', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        images = self.log_images(batch, n_row=2, sample=True, split='val', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        return images

    @torch_inference_mode()
    def test_step(self, batch, _batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            # loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            loss_dict_ema = {f'{key}_ema': v for key, v in loss_dict_ema.items()}
        # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        dict_merged = loss_dict_no_ema | loss_dict_ema
        dict_merged = {k.replace('val/', 'test/'): v for k, v in dict_merged.items()}
        self.log_dict(dict_merged, prog_bar=True, logger=USE_LOGGER_PL, on_step=False, on_epoch=True)
            # def log_images(self, batch, N=-1, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
            #        quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
            #        plot_diffusion_rows=True, split: str = None, **kwargs) -> dict[str, torch_Tensor]:

        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=USE_LOGGER_PL, on_step=False, on_epoch=True)
        # if self.trainer.global_rank == 0:
            # print(f'DDPM.validation_step: {self.trainer.global_rank == 0=}', 'log_images')
        # global_step = self.global_step if hasattr(self, 'global_step') else self.trainer.global_step
        # log_every_n_steps = self.log_every_n_steps if hasattr(self, 'log_every_n_steps') else self.trainer.log_every_n_steps
        # if global_step % log_every_n_steps == 0 and rank_zero_only.rank == 0:
        # if global_step % log_every_n_steps == 0:
        # images = self.log_images(batch, N=4, n_row=2, sample=True, split='test', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        # images = self.log_images(batch, n_row=2, sample=True, split='test', ddim_steps=DDIM_STEPS_LOGGING, eta=0.0, unconditional_guidance_scale=5.0, unconditional_conditioning, test_model_kwargs) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        test_model_kwargs = {
            '': None,  # TODO,
            '': None,  # TODO,
        }

        batch[''] # TODO
        unconditional_conditioning = None # TODO
        images = self.log_images(batch, n_row=2, sample=True, split='test', ddim_steps=DDIM_STEPS_LOGGING, eta=0.0, unconditional_guidance_scale=5.0, unconditional_conditioning=unconditional_conditioning, test_model_kwargs=test_model_kwargs) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        # images = self.log_images(batch, n_row=2, sample=True, split='test', use_ddim=False) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        return images

    def configure_optimizers(self):
        lr = self.learning_rate
        # params = list(self.model.parameters())
        params = list(param for param in self.model.parameters() if param.requires_grad is True)

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params.extend(list(self.cond_stage_model.final_ln.parameters())+list(self.cond_stage_model.mapper.parameters())+list(self.proj_out.parameters()))
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        params.append(self.learnable_vector)

        opt = AdamW(params, lr=lr, fused=True)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch_no_grad()
    def to_rgb(self, x):
        x = x.to(torch_float32, non_blocking=True)
        if not hasattr(self, 'colorize'):
            self.colorize = torch_randn(3, x.shape[1], 1, 1, device=x.device, dtype=torch_float32)
        x = conv2d(x, weight=self.colorize)
        x_min = x.min()
        x = 2. * (x - x_min) / (x.max() - x_min) - 1.
        return x


class LatentInpaintDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys


    @torch_no_grad()
    def get_input(
        self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False
    ):
        # note: restricted to non-trainable encoders currently
        assert (
            not self.cond_stage_trainable
        ), "trainable cond stages not yet supported for inpainting"
        z, c, x, xrec, xc = super().get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=bs,
        )

        assert exists(self.concat_keys)
        c_cat = []
        for ck in self.concat_keys:
            cc = (
                rearrange(batch[ck], "b h w c -> b c h w")
                # .to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
                .to(memory_format=contiguous_format, non_blocking=True)
            )
            if bs is not None:
                cc = cc[:bs].to(self.device, non_blocking=True)
            bchw = z.shape
            if ck != self.masked_image_key:
                cc = interpolate(cc, size=bchw[-2:])
            else:
                cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch_cat(c_cat, dim=1)
        all_conds = {'c_concat': [c_cat], 'c_crossattn': [c]}
        if return_first_stage_outputs:
            return z, all_conds, x, xrec, xc
        return z, all_conds
