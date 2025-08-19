"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
from contextlib import contextmanager
from functools import partial
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm
import torch
from torch import (
    inference_mode as torch_inference_mode,
    int64 as torch_int64,
    cat as torch_cat,
    contiguous_format,
    float32 as torch_float32,
    no_grad as torch_no_grad,
    full as torch_full,
    as_tensor as torch_as_tensor,
    Tensor as torch_Tensor,
    randn_like as torch_randn_like,
    randint as torch_randint,
    randn as torch_randn,
)
from torch.nn import Parameter
from torch.nn.functional import conv2d, dropout, interpolate, mse_loss
from torch.optim import AdamW
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.v2 import Normalize
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from ldm.util import exists, default, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
# from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
# from ldm.wandb_image_logger import WandbImageLogger
# from hunter import wrap, StackPrinter
# DEPTH = 4

torch_dtype = None # torch_bfloat16
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


def subtract_generators(gen1, gen2):
    # Convert the second generator to a set for efficient lookups
    elements_to_remove = set(gen2)

    # Filter elements from the first generator
    for item in gen1:
        if item not in elements_to_remove:
            yield item

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class DDPM(LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 u_cond_percent=0,
                 ):
        super().__init__()
        self.torch_dtype = torch_dtype
        assert parameterization in {"eps", "x0"}, 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size
        self.channels = channels
        self.u_cond_percent=u_cond_percent
        self.use_positional_encodings = use_positional_encodings
        self.model = model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = use_scheduler = scheduler_config is not None
        if use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        logvar = torch_full(fill_value=logvar_init, size=(self.num_timesteps,))
        # if self.learn_logvar:
        self.logvar = Parameter(logvar, requires_grad=learn_logvar)
        self.log_every_t_image_train = log_every_t
        print(f'DDPM.__init__: {self.log_every_t_image_train=}')
        # self.log_every_t_train = log_every_t // 2 if log_every_t > 10 else log_every_t
        # else:
        #     self.logvar = Parameter(logvar, requires_grad=self.learn_logvar)
            # self.register_buffer('logvar', logvar, persistent=True)
        # with torch_inference_mode(): #         While copying the parameter named "fid.inception.Mixed_7c.branch3x3_1.bn.num_batches_tracked", whose dimensions in the model are torch.Size([]) and whose dimensions in the checkpoint are torch.Size([]), an exception occurred : ('Inplace update to inference tensor outside InferenceMode is not allowed.You can make a clone to get a normal tensor before doing inplace update.See https://github.com/pytorch/rfcs/pull/17 for more details.',).

    # @property
    # def fid(self):
    #     with torch_inference_mode():
    #         fid = FrechetInceptionDistance(feature=64, normalize=True).eval().to(self.device, non_blocking=True)
    #     return fid

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert len(alphas_cumprod) == num_timesteps, 'alphas have to be defined for each timestep'

        # to_torch = partial(torch_as_tensor, dtype=self.torch_dtype)
        to_torch = partial(torch_as_tensor, dtype=torch_float32)
        register_buffer = self.register_buffer
        register_buffer('betas', to_torch(betas))
        register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch_Tensor(alphas_cumprod)) / (2. * 1 - torch_Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if (use_ema := self.use_ema):
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f'{context}: Restored training weights')

    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        sd = torch.load(path, map_location="cpu")
        if 'state_dict' in sd.keys():
            sd = sd['state_dict']
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f'Deleting key {k} from state_dict.')
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f'Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys')
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch_no_grad()
    # @torch_inference_mode()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        # nonzero_mask = (1 - (t == 0).to(torch_dtype, non_blocking=True)).reshape(b, *((1,) * (len(x.shape) - 1)))
        nonzero_mask = (1 - (t == 0).to(torch_float32, non_blocking=True)).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch_no_grad()
    # @torch_inference_mode()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch_randn(shape, device=device)
        intermediates = [img] if return_intermediates is True else None
        log_every_t = self.log_every_t
        num_timesteps = self.num_timesteps
        clip_denoised = self.clip_denoised
        for i in tqdm(reversed(range(0, num_timesteps)), desc='Sampling t', total=num_timesteps):
            img = self.p_sample(img, torch_full((b,), i, device=device, dtype=torch_int64), clip_denoised=clip_denoised)
            if return_intermediates is True and i % log_every_t == 0 or i == num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch_no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        # noise = default(noise, lambda: torch_randn_like(x_start, dtype=torch_dtype))
        noise = default(noise, lambda: torch_randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        if (loss_type := self.loss_type) == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            loss = mse_loss(target, pred, reduction='mean' if mean else 'none')
        else:
            raise NotImplementedError(f"unknown {loss_type=}")

        return loss

    def p_losses(self, x_start, t, noise=None) -> tuple[torch_Tensor, dict, torch_Tensor]:
        noise = default(noise, lambda: torch_randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if (parameterization := self.parameterization) == "eps":
            target = noise
        elif parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'
        loss_mean = loss.mean()
        loss_dict.update({f'{log_prefix}/loss_simple': loss_mean.item()})
        loss_simple = loss_mean * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb.item()})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss.item()})

        return loss, loss_dict, model_out

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch_randint(0, self.num_timesteps, (b,), device=self.device, dtype=torch_int64)
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        # torch_dtype = self.torch_dtype
        torch_dtype = torch_float32
        if k == "inpaint":
            x = batch['images_after_gt'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
            # mask = batch['inpaint_mask'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
            mask = batch['images_mask'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
            # inpaint = batch['inpaint_image'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
            inpaint = batch['images_inpaint'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
            # reference = batch['ref_imgs'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
            reference = batch['images_ref'].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        else:
            x = batch[k].to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        if x.ndim == 3:
            x = x.unsqueeze(-1).to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x.to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        # mask = mask.to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        # inpaint = inpaint.to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        # reference = reference.to(memory_format=contiguous_format, dtype=torch_dtype, non_blocking=True)
        return x, inpaint, mask, reference

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        # loss, loss_dict = self(x)
        return self(x)
        # return loss, loss_dict

    def training_step(self, batch, batch_idx) -> tuple[torch_Tensor, torch_Tensor]:
        loss, loss_dict, model_out = self.shared_step(batch)

        # self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict, prog_bar=True, logger=USE_LOGGER_PL, on_step=True, on_epoch=True)

        # self.log('global_step', self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # self.log('global_step', self.global_step, prog_bar=True, logger=USE_LOGGER_PL, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            # self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log('lr_abs', lr, prog_bar=True, logger=USE_LOGGER_PL, on_step=True, on_epoch=False)

        return loss, model_out

    # @rank_zero_only
    # @torch_no_grad()
    @torch_inference_mode()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema, model_out = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema, model_out = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        # self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_no_ema | loss_dict_ema, prog_bar=True, logger=USE_LOGGER_PL, on_step=True, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        # self.log_dict(loss_dict_ema, prog_bar=False, logger=USE_LOGGER_PL, on_step=False, on_epoch=True)
        # if self.trainer.global_rank == 0:
            # print(f'DDPM.validation_step: {self.trainer.global_rank == 0=}', 'log_images')
        global_step = self.global_step if hasattr(self, 'global_step') else self.trainer.global_step
        log_every_n_steps = self.log_every_n_steps if hasattr(self, 'log_every_n_steps') else self.trainer.log_every_n_steps
        # if global_step % log_every_n_steps == 0 and rank_zero_only.rank == 0:
        if global_step % log_every_n_steps == 0:
            self.log_images(batch, N=4, n_row=2, sample=True, split='val', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        # dict_images = self.log_images(batch, N=4, n_row=2, sample=True) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
        # self.log_dict(dict_images)

    # @rank_zero_only
    # @torch_no_grad()
    # def test_step(self, batch, batch_idx):
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
    #         self.log_images(batch, N=4, n_row=2, sample=True, split='test', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
    #     # dict_images = self.log_images(batch, N=4, n_row=2, sample=True) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
    #     # self.log_dict(dict_images)
    #     # self.log_images(batch, N=4, n_row=2, sample=True, split='test', ddim_steps=DDIM_STEPS_LOGGING) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
    #     # dict_images = self.log_images(batch, N=4, n_row=2, sample=True) #, return_keys=['inputs', 'samples', 'diffusion_row'],)
    #     # self.log_dict(dict_images)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    # def _get_rows_from_list(self, samples):
    #     n_imgs_per_row = len(samples)
    #     denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
    #     denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
    #     # denoise_grid_new = rearrange(samples, 'n b c h w -> (b n) c h w')
    #     denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
    #     return denoise_grid

    # @torch_no_grad()
    # def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, split: str = None, **kwargs) -> dict:
    #     x = self.get_input(batch, self.first_stage_key)
    #     len_x = x.size(0)
    #     N = min(len_x, N)
    #     x = x[:N].to(self.device, non_blocking=True)
    #     n_row = min(len_x, n_row)
    #     x_start = x[:n_row]
    #     log = {'inputs': x}

    #     # get diffusion row
    #     diffusion_row = []

    #     for t in range(self.num_timesteps):
    #         if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
    #             t = repeat(torch_as_tensor([t]), '1 -> b', b=n_row)
    #             t = t.to(self.device, dtype=torch_int64, non_blocking=True)
    #             noise = torch_randn_like(x_start)
    #             x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #             diffusion_row.append(x_noisy)

    #     log['diffusion_row'] = self._get_rows_from_list(diffusion_row)

    #     if sample:
    #         # get denoise row
    #         with self.ema_scope("Plotting"):
    #             samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

    #         log["samples"] = samples
    #         log["denoise_row"] = self._get_rows_from_list(denoise_row)

    #     try:
    #         logger_experiment_log = self.logger.experiment.log
    #         for key, value in log.items():
    #             # self.log_image(key=f'{split}/{key}', images=[value])
    #             logger_experiment_log({f'{split}/{key}': [wandb_Image(image) for image in value]})
    #             # (key=f'{split}/{key}', images=[value])
    #     except:
    #         breakpoint()

    #     if return_keys:
    #         if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
    #             return log
    #         else:
    #             return {key: log[key] for key in return_keys}
    #     return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        return AdamW(params, lr=lr, fused=True)


# class LatentDiffusion(DDPM):
#     """main class"""
#     def __init__(self,
#                  first_stage_config,
#                  cond_stage_config,
#                  num_timesteps_cond=None,
#                  cond_stage_key="image",
#                  cond_stage_trainable=False,
#                  concat_mode=True,
#                  cond_stage_forward=None,
#                  conditioning_key=None,
#                  scale_factor=1.0,
#                  scale_by_std=False,
#                  *args, **kwargs):
#         self.num_timesteps_cond = default(num_timesteps_cond, 1)
#         self.scale_by_std = scale_by_std
#         assert self.num_timesteps_cond <= kwargs['timesteps']
#         # for backwards compatibility after implementation of DiffusionWrapper
#         if conditioning_key is None:
#             conditioning_key = 'concat' if concat_mode else 'crossattn'
#         if cond_stage_config == '__is_unconditional__':
#             conditioning_key = None
#         ckpt_path = kwargs.pop("ckpt_path", None)
#         ignore_keys = kwargs.pop("ignore_keys", [])
#         super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
#         self.learnable_vector = Parameter(torch.randn((1,1,768)), requires_grad=True)
#         self.proj_out = Linear(1024, 768)
#         self.concat_mode = concat_mode
#         self.cond_stage_trainable = cond_stage_trainable
#         self.cond_stage_key = cond_stage_key
#         try:
#             self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
#         except:
#             self.num_downs = 0
#         if not scale_by_std:
#             self.scale_factor = scale_factor
#         else:
#             self.register_buffer('scale_factor', torch_as_tensor(scale_factor))
#         self.instantiate_first_stage(first_stage_config)
#         self.instantiate_cond_stage(cond_stage_config)
#         self.cond_stage_forward = cond_stage_forward
#         self.clip_denoised = False
#         self.bbox_tokenizer = None

#         self.restarted_from_ckpt = False
#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys)
#             self.restarted_from_ckpt = True
#         # self.wandb_image_logger = wandb_image_logger = WandbImageLogger()
#         # self.log_image_pair = wandb_image_logger.log_image_pair
#         # self.logger_experiment_log = self.logger.experiment.log

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def make_cond_schedule(self, ):
#         self.cond_ids = torch_full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch_int64)
#         ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
#         self.cond_ids[:self.num_timesteps_cond] = ids

#     @rank_zero_only
#     @torch_no_grad()
#     def on_train_batch_start(self, batch, batch_idx):
#         # only for very first batch
#         if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
#             assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
#             # set rescale weight to 1./std of encodings
#             print("### USING STD-RESCALING ###")
#             x = super().get_input(batch, self.first_stage_key)
#             x = x.to(self.device, non_blocking=True)
#             encoder_posterior = self.encode_first_stage(x)
#             z = self.get_first_stage_encoding(encoder_posterior).detach()
#             del self.scale_factor
#             self.register_buffer('scale_factor', 1. / z.flatten().std())
#             print(f"setting self.scale_factor to {self.scale_factor}")
#             print("### USING STD-RESCALING ###")

#     def training_step(self, batch, batch_idx):
#         loss = super().training_step(batch, batch_idx)
#         if batch_idx % self.log_every_t_image_train == 0:
#             if self.trainer.global_rank == 0:
#                 print(f'LatentDiffusion.training_step: {self.trainer.global_rank == 0=}', 'log_images')

#                 with torch_inference_mode():
#                     self.log_images(batch, N=4, n_row=2, sample=True, split='train', ddim_steps=DDIM_STEPS_LOGGING)
#                 # dict_images = self.log_images(batch, N=4, n_row=2, sample=True)
#                 # self.log_dict(dict_images)

#         return loss

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def register_schedule(self,
#                           given_betas=None, beta_schedule="linear", timesteps=1000,
#                           linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
#         super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

#         self.shorten_cond_schedule = shorten_cond_schedule = self.num_timesteps_cond > 1
#         if shorten_cond_schedule:
#             self.make_cond_schedule()

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def instantiate_first_stage(self, config):
#         model = instantiate_from_config(config)
#         # breakpoint()
#         self.first_stage_model = model.eval()
#         self.first_stage_model.train = disabled_train
#         for param in self.first_stage_model.parameters():
#             param.requires_grad = False

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def instantiate_cond_stage(self, config):
#         if not self.cond_stage_trainable:
#             if config == "__is_first_stage__":
#                 print("Using first stage also as cond stage.")
#                 self.cond_stage_model = self.first_stage_model
#             elif config == "__is_unconditional__":
#                 print(f"Training {self.__class__.__name__} as an unconditional model.")
#                 self.cond_stage_model = None
#                 # self.be_unconditional = True
#             else:
#                 model = instantiate_from_config(config)
#                 self.cond_stage_model = model.eval()
#                 self.cond_stage_model.train = disabled_train
#                 for param in self.cond_stage_model.parameters():
#                     param.requires_grad = False
#         else:
#             assert config not in ('__is_first_stage__', '__is_unconditional__')
#             self.cond_stage_model = instantiate_from_config(config)

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
#         denoise_row = [self.decode_first_stage(zd.to(self.device, non_blocking=True), force_not_quantize=force_no_decoder_quantization) for zd in tqdm(samples, desc=desc)]
#         # n_imgs_per_row = len(denoise_row)
#         # denoise_row = torch_vstack(denoise_row)  # n_log_step, n_row, C, H, W
#         denoise_row = torch_stack(denoise_row, axis=1)  # n_log_step, n_row, C, H, W
#         b, n, c, h, w = denoise_row.size()
#         denoise_row = denoise_row.view(b * n, c, h, w)  # (b * n) c h w
#         # denoise_row = rearrange(denoise_row, 'n b c h w -> b n c h w')
#         # denoise_row = rearrange(denoise_row, 'b n c h w -> (b n) c h w')
#         # denoise_row = make_grid(denoise_row, nrow=n_imgs_per_row)
#         denoise_row = make_grid(denoise_row, nrow=n)
#         return denoise_row

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def get_first_stage_encoding(self, encoder_posterior):
#         if isinstance(encoder_posterior, DiagonalGaussianDistribution):
#             z = encoder_posterior.sample()
#         elif isinstance(encoder_posterior, torch_Tensor):
#             z = encoder_posterior
#         else:
#             raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
#         return self.scale_factor * z

#     def get_learned_conditioning(self, c):
#         if self.cond_stage_forward is None:
#             if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
#                 c = self.cond_stage_model.encode(c)
#                 if isinstance(c, DiagonalGaussianDistribution):
#                     c = c.mode()
#             else:
#                 c = self.cond_stage_model(c)
#         else:
#             assert hasattr(self.cond_stage_model, self.cond_stage_forward)
#             c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
#         return c


#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def meshgrid(self, h, w):
#         y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
#         x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

#         arr = torch_cat((y, x), dim=-1)
#         return arr

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def delta_border(self, h, w):
#         """
#         :param h: height
#         :param w: width
#         :return: normalized distance to image border,
#          wtith min distance = 0 at border and max dist = 0.5 at image center
#         """
#         lower_right_corner = torch_as_tensor([h - 1, w - 1]).view(1, 1, 2)
#         arr = self.meshgrid(h, w) / lower_right_corner
#         dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
#         dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
#         edge_dist = torch.min(torch_cat((dist_left_up, dist_right_down), dim=-1), dim=-1)[0]
#         return edge_dist

#     def get_weighting(self, h, w, Ly, Lx, device):
#         weighting = self.delta_border(h, w)
#         split_input_params = self.split_input_params
#         weighting = torch.clip(weighting, split_input_params["clip_min_weight"],
#                                split_input_params["clip_max_weight"], )
#         weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

#         if split_input_params["tie_braker"]:
#             L_weighting = self.delta_border(Ly, Lx)
#             L_weighting = torch.clip(L_weighting,
#                                      split_input_params["clip_min_tie_weight"],
#                                      split_input_params["clip_max_tie_weight"])

#             L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device, non_blocking=True)
#             weighting *= L_weighting
#         return weighting

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
#         """
#         :param x: img of size (bs, c, h, w)
#         :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
#         """
#         bs, nc, h, w = x.shape

#         # number of crops in image
#         Ly = (h - kernel_size[0]) // stride[0] + 1
#         Lx = (w - kernel_size[1]) // stride[1] + 1

#         if uf == 1 and df == 1:
#             fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
#             unfold = Unfold(**fold_params)

#             fold = Fold(output_size=x.shape[2:], **fold_params)

#             weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype, non_blocking=True)
#             normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
#             weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

#         elif uf > 1 and df == 1:
#             fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
#             unfold = Unfold(**fold_params)

#             fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
#                                 dilation=1, padding=0,
#                                 stride=(stride[0] * uf, stride[1] * uf))
#             fold = Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

#             weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype, non_blocking=True)
#             normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
#             weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

#         elif df > 1 and uf == 1:
#             fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
#             unfold = Unfold(**fold_params)

#             fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
#                                 dilation=1, padding=0,
#                                 stride=(stride[0] // df, stride[1] // df))
#             fold = Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

#             weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype, non_blocking=True)
#             normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
#             weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

#         else:
#             raise NotImplementedError

#         return fold, unfold, normalization, weighting

#     @torch_no_grad()
#     def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
#                   cond_key=None, return_original_cond=False, bs=None,get_mask=False,get_reference=False):

#         x,inpaint,mask,reference = super().get_input(batch, k)
#         if bs is not None:
#             x = x[:bs]
#             inpaint = inpaint[:bs]
#             mask = mask[:bs]
#             reference = reference[:bs]
#         x = x.to(self.device, non_blocking=True)
#         # breakpoint()
#         encoder_posterior = self.encode_first_stage(x)
#         z = self.get_first_stage_encoding(encoder_posterior).detach()
#         encoder_posterior_inpaint = self.encode_first_stage(inpaint)
#         z_inpaint = self.get_first_stage_encoding(encoder_posterior_inpaint).detach()
#         z_shape_neg1 = z.shape[-1]
#         # mask_resize = Resize([z_shape_neg1, z_shape_neg1])(mask)
#         mask_resize = resize(mask, (z_shape_neg1, z_shape_neg1))
#         z_new = torch_cat((z, z_inpaint,mask_resize), dim=1)
#         # breakpoint()
#         if self.model.conditioning_key is not None:
#             if cond_key is None:
#                 cond_key = self.cond_stage_key
#             if cond_key != self.first_stage_key:
#                 if cond_key in {'txt','caption', 'coordinates_bbox'}:
#                     xc = batch[cond_key]
#                 elif cond_key == 'image':
#                     xc = reference
#                 elif cond_key == 'class_label':
#                     xc = batch
#                 else:
#                     xc = super().get_input(batch, cond_key).to(self.device, non_blocking=True)
#                     # breakpoint()
#             else:
#                 xc = x
#             if not self.cond_stage_trainable or force_c_encode:
#                 if isinstance(xc, (dict, list)):
#                     # import pudb; pudb.set_trace()
#                     c = self.get_learned_conditioning(xc)
#                 else:
#                     c = self.get_learned_conditioning(xc.to(self.device, non_blocking=True))
#                     c = self.proj_out(c).to(dtype=torch_float32, non_blocking=True)
#                     # c = c.float()
#             else:
#                 c = xc
#             if bs is not None:
#                 c = c[:bs]

#             if self.use_positional_encodings:
#                 pos_x, pos_y = self.compute_latent_shifts(batch)
#                 ckey = __conditioning_keys__[self.model.conditioning_key]
#                 c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

#         else:
#             c = None
#             xc = None
#             if self.use_positional_encodings:
#                 pos_x, pos_y = self.compute_latent_shifts(batch)
#                 c = {'pos_x': pos_x, 'pos_y': pos_y}
#         out = [z_new, c]
#         if return_first_stage_outputs:
#             if self.first_stage_key=='inpaint':
#                 xrec = self.decode_first_stage(z[:,:4,:,:])
#             else:
#                 xrec = self.decode_first_stage(z)
#             out.extend([x, xrec])
#         if return_original_cond:
#             out.append(xc)
#         if get_mask:
#             out.append(mask)
#         if get_reference:
#             out.append(reference)
#         return out

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
#         if predict_cids:
#             if z.dim() == 4:
#                 z = torch.argmax(z.exp(), dim=1).to(torch_int64, non_blocking=True)
#             z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
#             # z = rearrange(z, 'b h w c -> b c h w').contiguous()
#             z = z.moveaxis(3, 1).contiguous()

#         # z = 1. / self.scale_factor * z
#         z *= 1. / self.scale_factor

#         if hasattr(self, "split_input_params"):
#             split_input_params = self.split_input_params
#             if split_input_params["patch_distributed_vq"]:
#                 ks = split_input_params["ks"]  # eg. (128, 128)
#                 stride = split_input_params["stride"]  # eg. (64, 64)
#                 uf = split_input_params["vqf"]
#                 bs, nc, h, w = z.shape
#                 if ks[0] > h or ks[1] > w:
#                     ks = (min(ks[0], h), min(ks[1], w))
#                     print("reducing Kernel")

#                 if stride[0] > h or stride[1] > w:
#                     stride = (min(stride[0], h), min(stride[1], w))
#                     print("reducing stride")

#                 fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

#                 z = unfold(z)  # (bn, nc * prod(**ks), L)
#                 # 1. Reshape to img shape
#                 z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

#                 # 2. apply model loop over last dim
#                 # if isinstance(self.first_stage_model, VQModelInterface):
#                 #     output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
#                 #                                                  force_not_quantize=predict_cids or force_not_quantize)
#                 #                    for i in range(z.shape[-1])]
#                 # else:
#                 first_stage_model = self.first_stage_model
#                 assert isinstance(first_stage_model, AutoencoderKL)
#                 first_stage_model_decode = first_stage_model.decode
#                 output_list = [first_stage_model_decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

#                 o = torch_stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
#                 o *= weighting
#                 # Reverse 1. reshape to img shape
#                 o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
#                 # stitch crops together
#                 decoded = fold(o)
#                 decoded /= normalization  # norm is shape (1, 1, h, w)
#                 return decoded
#             else:
#                 assert isinstance(self.first_stage_model, AutoencoderKL)
#                 # if isinstance(self.first_stage_model, AutoencoderKL):
#                 #     return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
#                 return self.first_stage_model.decode(z)

#         else:
#             assert isinstance(self.first_stage_model, AutoencoderKL)
#             # if isinstance(self.first_stage_model, VQModelInterface):
#             #     return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
#             if self.first_stage_key=='inpaint':
#                 return self.first_stage_model.decode(z[:,:4,:,:])
#             return self.first_stage_model.decode(z)

#     # same as above but without decorator
#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
#         if predict_cids:
#             if z.dim() == 4:
#                 z = torch.argmax(z.exp(), dim=1).long()
#             z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
#             z = rearrange(z, 'b h w c -> b c h w').contiguous()

#         z *= 1. / self.scale_factor

#         if hasattr(self, 'split_input_params'):
#             split_input_params = self.split_input_params
#             if split_input_params["patch_distributed_vq"]:
#                 ks = split_input_params["ks"]  # eg. (128, 128)
#                 stride = split_input_params["stride"]  # eg. (64, 64)
#                 uf = split_input_params["vqf"]
#                 bs, nc, h, w = z.shape
#                 if ks[0] > h or ks[1] > w:
#                     ks = (min(ks[0], h), min(ks[1], w))
#                     print('reducing Kernel')

#                 if stride[0] > h or stride[1] > w:
#                     stride = (min(stride[0], h), min(stride[1], w))
#                     print("reducing stride")

#                 fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

#                 z = unfold(z)  # (bn, nc * prod(**ks), L)
#                 # 1. Reshape to img shape
#                 z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

#                 # 2. apply model loop over last dim
#                 # if isinstance(self.first_stage_model, VQModelInterface):
#                 #     output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
#                 #                                                  force_not_quantize=predict_cids or force_not_quantize)
#                 #                    for i in range(z.shape[-1])]
#                 # else:

#                 first_stage_model = self.first_stage_model
#                 assert isinstance(first_stage_model, AutoencoderKL)
#                 output_list = [self.first_stage_model.decode(z[:, :, :, :, i]) for i in range(z.shape[-1])]

#                 o = torch_stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
#                 o *= weighting
#                 # Reverse 1. reshape to img shape
#                 o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
#                 # stitch crops together
#                 decoded = fold(o)
#                 decoded /= normalization  # norm is shape (1, 1, h, w)
#                 return decoded
#             else:
#                 assert isinstance(self.first_stage_model, AutoencoderKL)
#                     # return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
#                 return self.first_stage_model.decode(z)

#         else:
#             assert isinstance(self.first_stage_model, AutoencoderKL)
#                 # return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
#             return self.first_stage_model.decode(z)

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def encode_first_stage(self, x):
#         if hasattr(self, "split_input_params"):
#             split_input_params = self.split_input_params
#             if split_input_params["patch_distributed_vq"]:
#                 ks = split_input_params["ks"]  # eg. (128, 128)
#                 stride = split_input_params["stride"]  # eg. (64, 64)
#                 df = split_input_params["vqf"]
#                 split_input_params['original_image_size'] = x.shape[-2:]
#                 bs, nc, h, w = x.shape
#                 if ks[0] > h or ks[1] > w:
#                     ks = (min(ks[0], h), min(ks[1], w))
#                     print("reducing Kernel")

#                 if stride[0] > h or stride[1] > w:
#                     stride = (min(stride[0], h), min(stride[1], w))
#                     print("reducing stride")

#                 fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
#                 z = unfold(x)  # (bn, nc * prod(**ks), L)
#                 # Reshape to img shape
#                 z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

#                 output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
#                                for i in range(z.shape[-1])]

#                 o = torch_stack(output_list, axis=-1)
#                 o *= weighting

#                 # Reverse reshape to img shape
#                 o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
#                 # stitch crops together
#                 decoded = fold(o)
#                 decoded /= normalization
#                 return decoded

#             else:
#                 return self.first_stage_model.encode(x)
#         else:
#             return self.first_stage_model.encode(x)

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def shared_step(self, batch, **kwargs):
#         x, c = self.get_input(batch, self.first_stage_key)
#         loss = self(x, c)
#         return loss

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def forward(self, x, c, *args, **kwargs):
#         # x: torch.Size([4, 9, 64, 64])
#         # c: torch.Size([4, 3, 224, 224])
#         # breakpoint()

#         t = torch_randint(0, self.num_timesteps, (x.size(0),), device=self.device, dtype=torch_int64)
#         self.u_cond_prop=random.uniform(0, 1)
#         if self.model.conditioning_key is not None:
#             assert c is not None
#             if self.cond_stage_trainable:
#                 c = self.get_learned_conditioning(c)
#                 c = self.proj_out(c)

#             if self.shorten_cond_schedule:  # TODO: drop this option
#                 tc = self.cond_ids[t].to(self.device, non_blocking=True)
#                 c = self.q_sample(x_start=c, t=tc, noise=torch_randn_like(c, dtype=torch_float32))

#         if self.u_cond_prop<self.u_cond_percent:
#             return self.p_losses(x, self.learnable_vector.repeat(x.size(0),1,1), t, *args, **kwargs)
#         else:
#             return self.p_losses(x, c, t, *args, **kwargs)

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
#         def rescale_bbox(bbox):
#             x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
#             y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
#             w = min(bbox[2] / crop_coordinates[2], 1 - x0)
#             h = min(bbox[3] / crop_coordinates[3], 1 - y0)
#             return x0, y0, w, h

#         return [rescale_bbox(b) for b in bboxes]

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def apply_model(self, x_noisy, t, cond, return_ids=False):

#         if isinstance(cond, dict): # true in single-gpu # {c_crossattn}
#             # hybrid case, cond is exptected to be a dict
#             pass
#         else:
#             if not isinstance(cond, list):
#                 cond = [cond]
#             key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
#             cond = {key: cond}

#         if hasattr(self, 'split_input_params'): # False in both single and multi-gpu
#             assert len(cond) == 1  # todo can only deal with one conditioning atm
#             assert not return_ids
#             ks = self.split_input_params["ks"]  # eg. (128, 128)
#             stride = self.split_input_params["stride"]  # eg. (64, 64)

#             h, w = x_noisy.shape[-2:]

#             fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

#             z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
#             # Reshape to img shape
#             z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
#             z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

#             if self.cond_stage_key in {"image", "LR_image", "segmentation", 'bbox_img'} and self.model.conditioning_key:  # todo check for completeness
#                 c_key = next(iter(cond.keys()))  # get key
#                 c = next(iter(cond.values()))  # get value
#                 assert (len(c) == 1)  # todo extend to list with more than one elem
#                 c = c[0]  # get element

#                 c = unfold(c)
#                 c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

#                 cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

#             elif self.cond_stage_key == 'coordinates_bbox': # False in both single and multi-gpu
#                 assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

#                 # assuming padding of unfold is always 0 and its dilation is always 1
#                 n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
#                 full_img_h, full_img_w = self.split_input_params['original_image_size']
#                 # as we are operating on latents, we need the factor from the original image size to the
#                 # spatial latent size to properly rescale the crops for regenerating the bbox annotations
#                 num_downs = self.first_stage_model.encoder.num_resolutions - 1
#                 rescale_latent = 2 ** (num_downs)

#                 # get top left positions of patches as conforming for the bbbox tokenizer, therefore we
#                 # need to rescale the tl patch coordinates to be in between (0,1)
#                 tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
#                                          rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
#                                         for patch_nr in range(z.shape[-1])]

#                 # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
#                 patch_limits = [(x_tl, y_tl,
#                                  rescale_latent * ks[0] / full_img_w,
#                                  rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
#                 # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

#                 # tokenize crop coordinates for the bounding boxes of the respective patches
#                 patch_limits_tknzd = [torch_as_tensor(self.bbox_tokenizer._crop_encoder(bbox), dtype=torch_int64, device=self.device, non_blocking=True).unsqueeze(0)  # (1, 2)
#                                       for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
#                 print(patch_limits_tknzd[0].shape)
#                 # cut tknzd crop position from conditioning
#                 assert isinstance(cond, dict), 'cond must be dict to be fed into model'
#                 cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device, non_blocking=True)  # cut last two tokens (x_tl, y_tl)

#                 adapted_cond = torch_stack([torch_cat((cut_cond, p), dim=1) for p in patch_limits_tknzd])
#                 adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
#                 adapted_cond = self.get_learned_conditioning(adapted_cond)
#                 adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])

#                 cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

#             else:
#                 cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

#             # apply model by loop over crops
#             output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
#             assert not isinstance(output_list[0],
#                                   tuple)  # todo cant deal with multiple model outputs check this never happens

#             o = torch_stack(output_list, axis=-1)
#             o *= weighting
#             # Reverse reshape to img shape
#             o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
#             # stitch crops together
#             x_recon = fold(o) / normalization

#         else:
#             x_recon = self.model(x_noisy, t, **cond)

#         # breakpoint()
#         assert x_recon is not None, "Model output is None, check model definition and input"
#         # x_recon.size(): torch.Size([4, 4, 64, 64])
#         if isinstance(x_recon, tuple) and not return_ids:
#             return x_recon[0]
#         return x_recon

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
#         return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
#                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def _prior_bpd(self, x_start):
#         """
#         Get the prior KL term for the variational lower-bound, measured in
#         bits-per-dim.
#         This term can't be optimized, as it only depends on the encoder.
#         :param x_start: the [N x C x ...] tensor of inputs.
#         :return: a batch of [N] KL values (in bits), one per batch element.
#         """
#         batch_size = x_start.shape[0]
#         t = torch_as_tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
#         qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
#         kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
#         return mean_flat(kl_prior) / np.log(2.0)

#     def p_losses(self, x_start, cond, t, noise=None, ):
#         if self.first_stage_key == 'inpaint':
#             # x_start=x_start[:,:4,:,:]
#             noise = default(noise, lambda: torch_randn_like(x_start[:,:4,:,:]))
#             x_noisy = self.q_sample(x_start=x_start[:,:4,:,:], t=t, noise=noise)
#             x_noisy = torch_cat((x_noisy,x_start[:,4:,:,:]),dim=1)
#         else:
#             noise = default(noise, lambda: torch_randn_like(x_start))
#             x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#         model_output = self.apply_model(x_noisy, t, cond)

#         loss_dict = {}
#         prefix = 'train' if self.training else 'val'

#         if self.parameterization == "x0":
#             target = x_start
#         elif self.parameterization == "eps":
#             target = noise
#         else:
#             raise NotImplementedError()

#         loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
#         loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

#         # breakpoint()
#         logvar_t = self.logvar[t]
#         loss = loss_simple / torch.exp(logvar_t) + logvar_t
#         # loss = loss_simple / torch.exp(self.logvar) + self.logvar
#         if self.learn_logvar:
#             loss_dict.update({f'{prefix}/loss_gamma': loss.mean(), 'logvar': self.logvar.data.mean()})

#         loss = self.l_simple_weight * loss.mean()

#         loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
#         loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
#         # loss_vlb = self.lvlb_weights[t] * self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)).mean()
#         loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
#         loss += (self.original_elbo_weight * loss_vlb)
#         loss_dict.update({f'{prefix}/loss': loss})

#         return loss, loss_dict

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
#                         return_x0=False, score_corrector=None, corrector_kwargs=None):
#         t_in = t
#         model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

#         if score_corrector is not None:
#             assert self.parameterization == "eps"
#             model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

#         if return_codebook_ids:
#             model_out, logits = model_out

#         if self.parameterization == "eps":
#             x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
#         elif self.parameterization == "x0":
#             x_recon = model_out
#         else:
#             raise NotImplementedError()

#         if clip_denoised:
#             x_recon.clamp_(-1., 1.)
#         if quantize_denoised:
#             # x_recon, _, [_, _, _] = self.first_stage_model.quantize(x_recon)
#             x_recon, *_ = self.first_stage_model.quantize(x_recon)
#         model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
#         if return_codebook_ids:
#             return model_mean, posterior_variance, posterior_log_variance, logits
#         if return_x0:
#             return model_mean, posterior_variance, posterior_log_variance, x_recon
#         else:
#             return model_mean, posterior_variance, posterior_log_variance

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
#                  return_codebook_ids=False, quantize_denoised=False, return_x0=False,
#                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
#         b, *_, device = *x.shape, x.device
#         outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
#                                        return_codebook_ids=return_codebook_ids,
#                                        quantize_denoised=quantize_denoised,
#                                        return_x0=return_x0,
#                                        score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
#         if return_codebook_ids:
#             raise DeprecationWarning("Support dropped.")
#             model_mean, _, model_log_variance, logits = outputs
#         elif return_x0:
#             model_mean, _, model_log_variance, x0 = outputs
#         else:
#             model_mean, _, model_log_variance = outputs

#         noise = noise_like(x.shape, device, repeat_noise) * temperature
#         if noise_dropout > 0.:
#             noise = dropout(noise, p=noise_dropout)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (x.ndim - 1)))

#         if return_codebook_ids:
#             return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
#         if return_x0:
#             return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
#         else:
#             return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
#                               img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
#                               score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
#                               log_every_t=None):
#         if not log_every_t:
#             log_every_t = self.log_every_t
#         timesteps = self.num_timesteps
#         if batch_size is not None:
#             b = batch_size if batch_size is not None else shape[0]
#             shape = [batch_size] + list(shape)
#         else:
#             b = batch_size = shape[0]
#         if x_T is None:
#             img = torch.randn(shape, device=self.device)
#         else:
#             img = x_T
#         intermediates = []
#         if cond is not None:
#             if isinstance(cond, dict):
#                 cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
#                 list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
#             else:
#                 cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

#         if start_T is not None:
#             timesteps = min(timesteps, start_T)
#         iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
#                         total=timesteps) if verbose else reversed(
#             range(0, timesteps))
#         if isinstance(temperature, float):
#             temperature = [temperature] * timesteps

#         for i in iterator:
#             ts = torch_full((b,), i, device=self.device, dtype=torch_int64)
#             if self.shorten_cond_schedule:
#                 assert self.model.conditioning_key != 'hybrid'
#                 tc = self.cond_ids[ts].to(cond.device, non_blocking=True)
#                 cond = self.q_sample(x_start=cond, t=tc, noise=torch_randn_like(cond))

#             img, x0_partial = self.p_sample(img, cond, ts,
#                                             clip_denoised=self.clip_denoised,
#                                             quantize_denoised=quantize_denoised, return_x0=True,
#                                             temperature=temperature[i], noise_dropout=noise_dropout,
#                                             score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
#             if mask is not None:
#                 assert x0 is not None
#                 img_orig = self.q_sample(x0, ts)
#                 img = img_orig * mask + (1. - mask) * img

#             if i % log_every_t == 0 or i == timesteps - 1:
#                 intermediates.append(x0_partial)
#             if callback:
#                 callback(i)
#             if img_callback:
#                 img_callback(img, i)
#         return img, intermediates

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def p_sample_loop(self, cond, shape, return_intermediates=False,
#                       x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
#                       mask=None, x0=None, img_callback=None, start_T=None,
#                       log_every_t=None):

#         if not log_every_t:
#             log_every_t = self.log_every_t
#         device = self.betas.device
#         b = shape[0]
#         if x_T is None:
#             img = torch.randn(shape, device=device)
#         else:
#             img = x_T

#         intermediates = [img]
#         if timesteps is None:
#             timesteps = self.num_timesteps

#         if start_T is not None:
#             timesteps = min(timesteps, start_T)
#         iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
#             range(0, timesteps))

#         if mask is not None:
#             assert x0 is not None
#             assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

#         shorten_cond_schedule = self.shorten_cond_schedule
#         for i in iterator:
#             ts = torch_full((b,), i, device=device, dtype=torch_int64)
#             if shorten_cond_schedule:
#                 assert self.model.conditioning_key != 'hybrid'
#                 tc = self.cond_ids[ts].to(cond.device, non_blocking=True)
#                 cond = self.q_sample(x_start=cond, t=tc, noise=torch_randn_like(cond))

#             img = self.p_sample(img, cond, ts,
#                                 clip_denoised=self.clip_denoised,
#                                 quantize_denoised=quantize_denoised)
#             if mask is not None:
#                 img_orig = self.q_sample(x0, ts)
#                 img = img_orig * mask + (1. - mask) * img

#             if i % log_every_t == 0 or i == timesteps - 1:
#                 intermediates.append(img)
#             if callback: callback(i)
#             if img_callback: img_callback(img, i)

#         if return_intermediates:
#             return img, intermediates
#         return img

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
#                verbose=True, timesteps=None, quantize_denoised=False,
#                mask=None, x0=None, shape=None,**kwargs):
#         if shape is None:
#             shape = (batch_size, self.channels, self.image_size, self.image_size)
#         if cond is not None:
#             if isinstance(cond, dict):
#                 cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
#                 list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
#             else:
#                 cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
#         return self.p_sample_loop(cond,
#                                   shape,
#                                   return_intermediates=return_intermediates, x_T=x_T,
#                                   verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
#                                   mask=mask, x0=x0)

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

#         if ddim:
#             # self.ddim_sampler = ddim_sampler = DDIMSampler(self) if not hasattr(self, 'ddim_sampler') or self.ddim_sampler is None else self.ddim_sampler
#             ddim_sampler = DDIMSampler(self)
#             shape = (self.channels, self.image_size, self.image_size)
#             samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
#                                                         shape,cond,verbose=False,**kwargs)

#         else:
#             samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
#                                                  return_intermediates=True,**kwargs)

#         return samples, intermediates

#     @torch_no_grad()
#     def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
#                    quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
#                    plot_diffusion_rows=True, split: str = None, **kwargs):

#         # <<< FIX: HELPER FUNCTION TO NORMALIZE IMAGES FOR LOGGING >>>
#         # Models output in [-1, 1] range, but visualization needs [0, 1]
#         def normalize_for_log(img_tensor):
#             return torch.clamp((img_tensor + 1.0) / 2.0, min=0.0, max=1.0)

#         use_ddim = ddim_steps is not None

#         z, c, x, xrec_raw, xc, mask, reference = self.get_input(
#             batch, self.first_stage_key,
#             return_first_stage_outputs=True,
#             force_c_encode=True,
#             return_original_cond=True,
#             get_mask=True,
#             get_reference=True,
#             bs=N
#         )
#         len_x = x.size(0)
#         N = min(len_x, N)
#         n_row = min(len_x, n_row)

#         # <<< DIAGNOSTICS: CHECK RAW MODEL OUTPUT FOR COLLAPSE >>>
#         # This helps check if the model is outputting all zeros (collapse) vs. a logging issue.
#         if rank_zero_only.rank == 0:
#             print(f"[{split}/raw_reconstruction_stats] min: {xrec_raw.min():.3f}, max: {xrec_raw.max():.3f}, mean: {xrec_raw.mean():.3f}")

#         # Normalize all images that will be logged individually
#         log = {
#             'inputs': normalize_for_log(x),
#             'reconstruction': normalize_for_log(xrec_raw),
#             # Masks are typically [0, 1], but normalizing is safe.
#             'mask': normalize_for_log(mask),
#         }

#         if self.model.conditioning_key is not None:
#             if hasattr(self.cond_stage_model, 'decode'):
#                 xc = self.cond_stage_model.decode(c)
#                 log['conditioning'] = normalize_for_log(xc)
#             # ... other conditioning logic ...

#         # if plot_diffusion_rows:
#     #         # get diffusion row
#     #         diffusion_row = []
#     #         z_start = z[:n_row]
#     #         num_timesteps = self.num_timesteps
#     #         log_every_t = self.log_every_t
#     #         for t in range(num_timesteps):
#     #             if t % log_every_t == 0 or t == num_timesteps - 1:
#     #                 t = repeat(torch_as_tensor([t], device=self.device, dtype=torch_int64), '1 -> b', b=n_row)
#     #                 # t = t.to(self.device, dtype=torch_int64, non_blocking=True)
#     #                 noise = torch_randn_like(z_start)
#     #                 z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
#     #                 diffusion_row.append(self.decode_first_stage(z_noisy))

#     #         # diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
#     #         # diffusion_row = torch_stack(diffusion_row, dim=1)  # n_log_step, n_row, C, H, W
#     #         # breakpoint()
#     #         # diffusion_row_stacked_nb = torch_stack(diffusion_row)  # n_log_step, n_row, C, H, W
#     #         diffusion_row_old = torch_stack(diffusion_row, axis=0)  # n=n_log_step, b=n_row, C, H, W
#     #         diffusion_row = torch_stack(diffusion_row, axis=1)  # b=n_row, n=n_log_step, C, H, W

#     #         # torch.equal(diffusion_row_stacked_bn, rearrange(diffusion_row_stacked_nb, 'n b c h w -> b n c h w'))
#     #         diffusion_row_old = rearrange(diffusion_row_old, 'n b c h w -> b n c h w')
#     #         diffusion_row_old = rearrange(diffusion_row_old, 'b n c h w -> (b n) c h w')

#     #         B, n_log_step, C, H, W = diffusion_row.size()
#     #         # breakpoint()
#     #         diffusion_row = diffusion_row.view(B * n_log_step, C, H, W)  # (b * n) c h w
#     #         # diffusion_row_stacked_bn_rearranged = rearrange(rearrange(diffusion_row_stacked_nb, 'n b c h w -> b n c h w'), 'b n c h w -> (b n) c h w')
#     #         # diffusion_row = torch_vstack(diffusion_row)  # n_log_step, n_row, C, H, W
#     #         # torch.equal(diffusion_row_stacked_bn_view, diffusion_row_stacked_bn_rearranged)
#     #         # diffusion_row = rearrange(diffusion_row, 'b n c h w -> (b n) c h w')
#     #         # diffusion_row_old = make_grid(diffusion_row_old, nrow=n_log_step)
#     #         diffusion_row = make_grid(diffusion_row, nrow=n_log_step)
#     #         log['diffusion_row'] = diffusion_row
#     #         # breakpoint()

#     #         # diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
#     #         # diffusion_row_stacked_bn = rearrange(diffusion_row_stacked, 'n b c h w -> b n c h w')
#     #         # diffusion_row_stacked_bn_flat = rearrange(diffusion_row_stacked_bn, 'b n c h w -> (b n) c h w')
#     #         # diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
#     #         # log["diffusion_row"] = diffusion_grid

#         logger_experiment_log = self.logger.experiment.log
#         if sample:
#             with self.ema_scope('Plotting'):
#                 if self.first_stage_key == 'inpaint':
#                     samples_z, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
#                                                    ddim_steps=ddim_steps, eta=ddim_eta, rest=z[:, 4:, :, :])
#                 else:
#                     samples_z, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
#                                                    ddim_steps=ddim_steps, eta=ddim_eta)

#             samples_raw = self.decode_first_stage(samples_z)

#             if rank_zero_only.rank == 0:
#                 print(f"[{split}/raw_sample_stats] min: {samples_raw.min():.3f}, max: {samples_raw.max():.3f}, mean: {samples_raw.mean():.3f}")

#             # Normalize samples for logging
#             # imgs_after_pred_unnorm = normalize_for_log(samples_raw)
#             imgs_after_pred_unnorm = un_norm(samples_raw).clamp(min=0, max=1)
#             log['samples'] = imgs_after_pred_unnorm

#             # <<< COMBINED SOLUTION: CREATE THE INPAINTING RESULTS GRID >>>
#             # 1. Prepare mask for visualization (ensure 3 channels for RGB)
#             # vis_mask = mask if mask.shape[1] == 3 else mask.repeat(1, 3, 1, 1)

#             # 2. Create masked input using the ORIGINAL [-1, 1] ground truth image
#             # masked_input = x * (1. - mask)

#             # 3. Assemble all 4 images for each sample into a list for the grid

#             # COLUMNS = ['split', 's3path_image_before', 'image_before_masked', 'image_after_gt', 'image_after_pred']
#             table_image = wandb_Table(columns=COLUMNS)

#             ref_imgs = batch['ref_imgs']
#             mask_bool = mask.squeeze(1) < 0.5
#             # image_before_unnorm = un_norm(masked_input).clamp(min=0, max=1)  # Ensure masked input is in [0, 1] range for visualization
#             image_before_unnorm = un_norm(batch['images_inpaint']).clamp(min=0, max=1)  # Ensure GT is in [0, 1] range for visualization
#             image_after_gt_unnorm = un_norm(x).clamp(min=0, max=1)  # Ensure GT is in [0, 1] range for visualization
#             # image_cirep_i = un_norm_clip(ref_imgs[i])
#             # image_cirep_unnorm = resize(un_norm_clip(ref_imgs), image_after_gt_unnorm.size()[-2:])
#             # image_cirep_unnorm = resize(un_norm_clip(ref_imgs), (512, 512))
#             image_cirep_unnorm = un_norm_clip(ref_imgs)
#             # grid_items = []
#             global_step = self.global_step if split == 'train' else self.trainer.global_step
#             for i in range(N):
#                 # Order: [Masked Input, Mask, Ground Truth, Prediction]
#                 # Normalize each component for visualization in the grid
#                 s3path_image_before = 'TODO'
#                 # image_before_masked = normalize_for_log(masked_input[i])
#                 # breakpoint()
#                 mask_bool_i = mask_bool[i]
#                 image_before_unnorm_i_masked = draw_segmentation_masks(image_before_unnorm[i], mask_bool_i, alpha=ALPHA, colors='red')

#                 # image_before_masked = un_norm(masked_input[i])
#                 # mask_i = un_norm(vis_mask[i])
#                 # image_after_gt = normalize_for_log(x[i])
#                 # image_after_gt_i = un_norm(x)
#                 # image_cirep_i = un_norm_clip(ref_imgs[i])
#                 image_cirep_i = image_cirep_unnorm[i]
#                 image_after_gt_i = image_after_gt_unnorm[i]
#                 image_after_pred_i = imgs_after_pred_unnorm[i]
#                 # breakpoint()
#                 # grid_items.extend([image_before_unnorm_i_masked, image_cirep_i, image_after_gt_i, image_after_pred_i])
#                 # grid_items.append(normalize_for_log(masked_input[i]))
#                 # grid_items.append(normalize_for_log(masked_input[i]))
#                 # grid_items.append(normalize_for_log(vis_mask[i]))
#                 # grid_items.append(normalize_for_log(x[i]))
#                 # grid_items.append(imgs_after_pred_unnorm[i]) # Already normalized
#                 # wandb_Image(image_data, masks={"": {"mask_data": mask_data}})

#                 table_image.add_data(split, global_step, s3path_image_before, wandb_Image(image_before_unnorm_i_masked), wandb_Image(image_cirep_i), wandb_Image(image_after_gt_i), wandb_Image(image_after_pred_i))

#                             # for i in range(N):
#                 # # Order: [Masked Input, Mask, Ground Truth, Prediction]
#                 # # Normalize each component for visualization in the grid
#                 # grid_items.append(normalize_for_log(masked_input[i]))
#                 # grid_items.append(normalize_for_log(vis_mask[i]))
#                 # grid_items.append(normalize_for_log(x[i]))
#                 # grid_items.append(imgs_after_pred_unnorm[i]) # Already normalized

#             # generate two slightly overlapping image intensity distributions
#             # imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
#             # imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
#             # imgs_after_pred = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
#             # imgs_after_gt = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)


#             # breakpoint()

#             # 4. Create the grid with 4 columns per row
#             # if grid_items:
#             # grid_tensor = torch_stack(grid_items)
#             # results_grid = make_grid(grid_tensor, nrow=N)
#             # log['inpainting_results'] = results_grid

#             # logger_experiment_log(grid_items)
#             # logger_experiment_log({f'{split}/inpainting_results_grid': wandb_Image(results_grid)})
#             logger_experiment_log({f'{split}/inpainting_results_table': table_image})
#             with torch_inference_mode():
#                 fid = self.fid
#                 fid.reset()
#                 # fid.update(image_after_gt_unnorm.to('cpu', non_blocking=True), real=True)
#                 # fid.update(imgs_after_pred_unnorm.to('cpu', non_blocking=True), real=False)
#                 fid.update(image_after_gt_unnorm, real=True)
#                 fid.update(imgs_after_pred_unnorm, real=False)
#                 fid_value = fid.compute()

#             logger_experiment_log({f'{split}/fid': fid_value})

#         # Logging loop
#         # log['inputs']: torch.Size([4, 3, 512, 512])
#         # log['mask']: torch.Size([4, 1, 512, 512])
#         # log['mask']: torch.Size([4, 1, 512, 512])
#         # log['reconstruction']: torch.Size([4, 3, 512, 512])
#         # log['samples']: torch.Size([4, 3, 512, 512])
#         # logger_experiment_log = self.logger.experiment.log
#         # for image_before_i, image_masked_i, image_after_gt_i, image_after_pred_i in zip(log['inputs'], log['mask'], log['reconstruction'], log['samples']):

#         #     # Create a grid of 4 images: Masked Input, Mask, Ground Truth, Prediction
#         #     grid = torch.stack((image_before_i, image_masked_i, image_after_gt_i, image_after_pred_i))
#         #     grid = make_grid(grid, nrow=4)
#         #     logger_experiment_log(grid)

#         #     # Add to the log
#         #     # log.setdefault('inpainting_results', []).append(grid)
#         # if rank_zero_only.rank == 0:
#         #     try:
#         #         for key, value in log.items():
#         #             # Log single grid images directly
#         #             if key in {'inputs', 'diffusion_row', 'inpainting_results', 'denoise_row', 'progressive_row'}:
#         #                 logger_experiment_log({f'{split}/{key}': [wandb_Image(img) for img in value]})
#         #             # Log lists of images (batches)
#         #             elif isinstance(value, torch_Tensor) and value.ndim == 4:
#         #                 logger_experiment_log({f'{split}/{key}': wandb_Image(value)})
#         #             else:
#         #                 breakpoint()
#         #     except Exception as e:
#         #         print(f"Error during W&B logging: {e}")
#         #         breakpoint()

#         if return_keys:
#             return {key: log[key] for key in return_keys if key in log}
#         return log
#     # @torch_no_grad()
#     # def log_images(self, batch, N=4, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
#     #                quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
#     #                plot_diffusion_rows=True, split: str = None, **kwargs):

#     #     use_ddim = ddim_steps is not None

#     #     z, c, x, xrec, xc, mask, reference = self.get_input(batch, self.first_stage_key,
#     #                                     return_first_stage_outputs=True,
#     #                                     force_c_encode=True,
#     #                                     return_original_cond=True,
#     #                                     get_mask=True,
#     #                                     get_reference=True,
#     #                                     bs=N)
#     #     len_x = x.size(0)
#     #     N = min(len_x, N)
#     #     n_row = min(len_x, n_row)
#     #     log = {
#     #         'inputs': x,
#     #         'reconstruction': xrec,
#     #         'mask': mask,
#     #     }

#     #     # log["reference"]=reference
#     #     if self.model.conditioning_key is not None:
#     #         if hasattr(self.cond_stage_model, 'decode'):
#     #             xc = self.cond_stage_model.decode(c)
#     #             log['conditioning'] = xc
#     #         elif self.cond_stage_key in ('caption', 'txt'):
#     #             xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key])
#     #             log['conditioning'] = xc
#     #         elif self.cond_stage_key == 'class_label':
#     #             xc = log_txt_as_img((x.shape[2], x.shape[3]), batch['human_label'])
#     #             log["conditioning"] = xc
#     #         elif isimage(xc):
#     #             log['conditioning'] = xc
#     #         if ismap(xc):
#     #             log['original_conditioning'] = self.to_rgb(xc)

#     #     if plot_diffusion_rows:
#     #         # get diffusion row
#     #         diffusion_row = []
#     #         z_start = z[:n_row]
#     #         num_timesteps = self.num_timesteps
#     #         log_every_t = self.log_every_t
#     #         for t in range(num_timesteps):
#     #             if t % log_every_t == 0 or t == num_timesteps - 1:
#     #                 t = repeat(torch_as_tensor([t], device=self.device, dtype=torch_int64), '1 -> b', b=n_row)
#     #                 # t = t.to(self.device, dtype=torch_int64, non_blocking=True)
#     #                 noise = torch_randn_like(z_start)
#     #                 z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
#     #                 diffusion_row.append(self.decode_first_stage(z_noisy))

#     #         # diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
#     #         # diffusion_row = torch_stack(diffusion_row, dim=1)  # n_log_step, n_row, C, H, W
#     #         # breakpoint()
#     #         # diffusion_row_stacked_nb = torch_stack(diffusion_row)  # n_log_step, n_row, C, H, W
#     #         diffusion_row_old = torch_stack(diffusion_row, axis=0)  # n=n_log_step, b=n_row, C, H, W
#     #         diffusion_row = torch_stack(diffusion_row, axis=1)  # b=n_row, n=n_log_step, C, H, W

#     #         # torch.equal(diffusion_row_stacked_bn, rearrange(diffusion_row_stacked_nb, 'n b c h w -> b n c h w'))
#     #         diffusion_row_old = rearrange(diffusion_row_old, 'n b c h w -> b n c h w')
#     #         diffusion_row_old = rearrange(diffusion_row_old, 'b n c h w -> (b n) c h w')

#     #         B, n_log_step, C, H, W = diffusion_row.size()
#     #         # breakpoint()
#     #         diffusion_row = diffusion_row.view(B * n_log_step, C, H, W)  # (b * n) c h w
#     #         # diffusion_row_stacked_bn_rearranged = rearrange(rearrange(diffusion_row_stacked_nb, 'n b c h w -> b n c h w'), 'b n c h w -> (b n) c h w')
#     #         # diffusion_row = torch_vstack(diffusion_row)  # n_log_step, n_row, C, H, W
#     #         # torch.equal(diffusion_row_stacked_bn_view, diffusion_row_stacked_bn_rearranged)
#     #         # diffusion_row = rearrange(diffusion_row, 'b n c h w -> (b n) c h w')
#     #         # diffusion_row_old = make_grid(diffusion_row_old, nrow=n_log_step)
#     #         diffusion_row = make_grid(diffusion_row, nrow=n_log_step)
#     #         log['diffusion_row'] = diffusion_row
#     #         # breakpoint()

#     #         # diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
#     #         # diffusion_row_stacked_bn = rearrange(diffusion_row_stacked, 'n b c h w -> b n c h w')
#     #         # diffusion_row_stacked_bn_flat = rearrange(diffusion_row_stacked_bn, 'b n c h w -> (b n) c h w')
#     #         # diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
#     #         # log["diffusion_row"] = diffusion_grid

#     #     if sample:
#     #         # get denoise row
#     #         with self.ema_scope('Plotting'):
#     #             if self.first_stage_key=='inpaint':
#     #                 samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
#     #                                                         ddim_steps=ddim_steps,eta=ddim_eta,rest=z[:,4:,:,:])
#     #             else:
#     #                 samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
#     #                                                         ddim_steps=ddim_steps,eta=ddim_eta)
#     #             # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
#     #         imgs_after_pred_unnorm = self.decode_first_stage(samples)
#     #         log['samples'] = imgs_after_pred_unnorm
#     #         if plot_denoise_rows:
#     #             denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
#     #             log['denoise_row'] = denoise_grid

#     #         if quantize_denoised and not isinstance(self.first_stage_model, (AutoencoderKL, IdentityFirstStage)):
#     #             # also display when quantizing x0 while sampling
#     #             with self.ema_scope('Plotting Quantized Denoised'):
#     #                 samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
#     #                                                          ddim_steps=ddim_steps,eta=ddim_eta,
#     #                                                          quantize_denoised=True)
#     #                 # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
#     #                 #                                      quantize_denoised=True)
#     #             imgs_after_pred_unnorm = self.decode_first_stage(samples.to(self.device, non_blocking=True))
#     #             log["samples_x0_quantized"] = imgs_after_pred_unnorm

#     #         if inpaint:
#     #             # make a simple center square
#     #             b, h, w = z.shape[0], z.shape[2], z.shape[3]
#     #             mask = torch.ones(N, h, w, device=self.device)
#     #             # zeros will be filled in
#     #             mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
#     #             mask = mask[:, None, ...]
#     #             with self.ema_scope('Plotting Inpaint'):

#     #                 samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
#     #                                             ddim_steps=ddim_steps, x0=z[:N,:4], mask=mask)
#     #             imgs_after_pred_unnorm = self.decode_first_stage(samples.to(self.device, non_blocking=True))
#     #             log['samples_inpainting'] = imgs_after_pred_unnorm
#     #             log['mask'] = mask

#     #             # outpaint
#     #             with self.ema_scope('Plotting Outpaint'):
#     #                 samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
#     #                                             ddim_steps=ddim_steps, x0=z[:N], mask=mask)
#     #             imgs_after_pred_unnorm = self.decode_first_stage(samples.to(self.device, non_blocking=True))
#     #             log['samples_outpainting'] = imgs_after_pred_unnorm

#     #     if plot_progressive_rows:
#     #         with self.ema_scope('Plotting Progressives'):
#     #             img, progressives = self.progressive_denoising(c,
#     #                                                            shape=(self.channels, self.image_size, self.image_size),
#     #                                                            batch_size=N)
#     #         prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
#     #         log['progressive_row'] = prog_row


#     #     # def log_image_pair(self, train_valid_test: str, image_1: PILImage, image_2: PILImage, bbox_1: list[int, int, int, int] | tuple[int, int, int, int] = None, bbox_2: list[int, int, int, int] | tuple[int, int, int, int] = None, bbox_caption_1: str = None, bbox_caption_2: str = None) -> None:
#     #     # train_valid_test = 'val'
#     #     # self.log_image_pair(train_valid_test, image_1=log['inputs'], image_2=log['reconstruction'], bbox_1=None, bbox_2=None, bbox_caption_1='Input', bbox_caption_2='Reconstruction')
#     #     # self.log_image(key="samples", images=[img1, img2], caption=["tree", "person"])
#     #     if rank_zero_only.rank == 0:
#     #         try:
#     #             logger_experiment_log = self.logger.experiment.log
#     #             for key, value in log.items():
#     #                 # self.log_image(key=f'{split}/{key}', images=[value])
#     #                 # self.log(f'{split}/{key}', [wandb_Image(image) for image in value])# images=[value])
#     #                 # wandb_log({f'{split}/{key}': [wandb_Image(image) for image in value]})
#     #                 # *** RuntimeError: permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 2 is not equal to len(dims) = 3
#     #                 if key in {'diffusion_row'}:

#     #                     logger_experiment_log({f'{split}/{key}': wandb_Image(value)})
#     #                 else:
#     #                     logger_experiment_log({f'{split}/{key}': [wandb_Image(image) for image in value]})
#     #                 # self.log({f'{split}/{key}': [wandb_Image(image) for image in value]})
#     #         except:
#     #             breakpoint()

#     #     # self.log({})
#     #     if return_keys:
#     #         if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
#     #             return log
#     #         else:
#     #             return {key: log[key] for key in return_keys}
#     #     return log

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         # params = list(self.model.parameters())
#         params = list(param for param in self.model.parameters() if param.requires_grad is True)


#         # params_subbed = list(subtract_generators(self.model.parameters(), self.fid.parameters()))
#         # breakpoint()
#         if self.cond_stage_trainable:
#             print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
#             params.extend(list(self.cond_stage_model.final_ln.parameters())+list(self.cond_stage_model.mapper.parameters())+list(self.proj_out.parameters()))
#         if self.learn_logvar:
#             print('Diffusion model optimizing logvar')
#             params.append(self.logvar)
#         params.append(self.learnable_vector)

#         # for k, p in self.fid.named_parameters():
#         #     breakpoint()
#         #     params.remove(p)

#         opt = torch.optim.AdamW(params, lr=lr)
#         if self.use_scheduler:
#             assert 'target' in self.scheduler_config
#             scheduler = instantiate_from_config(self.scheduler_config)

#             print("Setting up LambdaLR scheduler...")
#             scheduler = [
#                 {
#                     'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
#                     'interval': 'step',
#                     'frequency': 1
#                 }]
#             return [opt], scheduler
#         return opt

#     # @wrap(depth=DEPTH, action=StackPrinter)
#     @torch_no_grad()
#     def to_rgb(self, x):
#         x = x.to(torch_float32, non_blocking=True)
#         if not hasattr(self, 'colorize'):
#             self.colorize = torch.randn(3, x.shape[1], 1, 1, device=x.device, dtype=torch_float32)
#         x = conv2d(x, weight=self.colorize)
#         x_min = x.min()
#         x = 2. * (x - x_min) / (x.max() - x_min) - 1.
#         return x


class DiffusionWrapper(LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # self.diffusion_model = instantiate_from_config(diff_model_config).to(dtype=torch_dtype, non_blocking=True)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        assert conditioning_key in {None, 'concat', 'crossattn', 'hybrid', 'adm'}
        self.conditioning_key = conditioning_key

    # @wrap(depth=DEPTH, action=StackPrinter)
    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        match self.conditioning_key:
            case None:
                out = self.diffusion_model(x, t)
            case 'concat':
                xc = torch_cat([x] + c_concat, dim=1)
                out = self.diffusion_model(xc, t)
            case 'crossattn':
                cc = torch_cat(c_crossattn, 1)
                out = self.diffusion_model(x, t, context=cc)
            case 'hybrid':
                xc = torch_cat([x] + c_concat, dim=1)
                cc = torch_cat(c_crossattn, 1)
                out = self.diffusion_model(xc, t, context=cc)
            case 'adm':
                cc = c_crossattn[0]
                out = self.diffusion_model(x, t, y=cc)
            case _:
                raise NotImplementedError()

        # if self.conditioning_key is None:
        #     out = self.diffusion_model(x, t)
        # elif self.conditioning_key == 'concat':
        #     xc = torch_cat([x] + c_concat, dim=1)
        #     out = self.diffusion_model(xc, t)
        # elif self.conditioning_key == 'crossattn':
        #     cc = torch_cat(c_crossattn, 1)
        #     out = self.diffusion_model(x, t, context=cc)
        # elif self.conditioning_key == 'hybrid':
        #     xc = torch_cat([x] + c_concat, dim=1)
        #     cc = torch_cat(c_crossattn, 1)
        #     out = self.diffusion_model(xc, t, context=cc)
        # elif self.conditioning_key == 'adm':
        #     cc = c_crossattn[0]
        #     out = self.diffusion_model(x, t, y=cc)
        # else:
        #     raise NotImplementedError()

        return out


# class Layout2ImgDiffusion(LatentDiffusion):
#     # TODO: move all layout-specific hacks to this class
#     def __init__(self, cond_stage_key, *args, **kwargs):
#         assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
#         super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

#     def log_images(self, batch, N=8, *args, **kwargs):
#         logs = super().log_images(batch=batch, N=N, *args, **kwargs)

#         key = 'train' if self.training else 'validation'
#         dset = self.trainer.datamodule.datasets[key]
#         mapper = dset.conditional_builders[self.cond_stage_key]

#         bbox_imgs = []
#         map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
#         for tknzd_bbox in batch[self.cond_stage_key][:N]:
#             bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
#             bbox_imgs.append(bboximg)

#         cond_img = torch_stack(bbox_imgs, dim=0)
#         logs['bbox_image'] = cond_img
#         return logs
