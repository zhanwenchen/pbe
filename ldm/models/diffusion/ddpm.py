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

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        return AdamW(params, lr=lr, fused=True)


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
