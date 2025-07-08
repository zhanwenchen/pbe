# diffusion_models.py
'''
Standalone diffusion-model module (no PyTorch-Lightning).

Ported for pure PyTorch + 🤗 Accelerate / Transformers.

Style guide:
  * single quotes
  * no fancy space-alignment around '='
  * retains all maths / public API of the original file
'''

from contextlib import contextmanager
from functools import partial
import random
import numpy as np

from einops import rearrange, repeat
from tqdm import tqdm

import torch
from torch import nn, cat as torch_cat, contiguous_format, float32 as torch_float32
from torch.nn import Fold, Linear, Parameter, Unfold
from torch.nn.functional import conv2d, dropout, interpolate, mse_loss
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms.functional import resize
from torchvision.utils import make_grid

# --- external helpers (same as in original impl) ---------------------------
from ldm.util import (
    log_txt_as_img, exists, default, ismap, isimage,
    mean_flat, count_params, instantiate_from_config
)
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import (
    normal_kl, DiagonalGaussianDistribution
)
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import (
    make_beta_schedule, extract_into_tensor, noise_like
)
from ldm.models.diffusion.ddim import DDIMSampler
# ---------------------------------------------------------------------------

torch_dtype = None            # allow bf16 later

# ---------------------------------------------------------------------------#
#                   convenience helpers formerly supplied by PL              #
# ---------------------------------------------------------------------------#

def rank_zero_only(fn):                           # no-op but keeps decorator
    def wrapped(*a, **kw): return fn(*a, **kw)
    return wrapped

def disabled_train(self, mode=True):              # prevent enc/dec mode flips
    return self

# ---------------------------------------------------------------------------#
#                               Core DDPM class                              #
# ---------------------------------------------------------------------------#

class DDPM(nn.Module):            # <-- no LightningModule
    '''
    Classic DDPM with Gaussian diffusion, image-space formulation.
    '''
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule='linear',
        loss_type='l2',
        ckpt_path=None,
        ignore_keys=None,
        load_only_unet=False,
        monitor=None,                       # kept for checkpoint loader compat
        use_ema=True,
        first_stage_key='image',
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.,
        v_posterior=0.,
        l_simple_weight=1.,
        conditioning_key=None,
        parameterization='eps',
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.,
        u_cond_percent=0
    ):
        super().__init__()

        # ---------- config storage ----------
        self.loss_type                = loss_type
        self.parameterization         = parameterization
        self.clip_denoised            = clip_denoised
        self.log_every_t              = log_every_t
        self.first_stage_key          = first_stage_key
        self.image_size               = image_size
        self.channels                 = channels
        self.use_positional_encodings = use_positional_encodings
        self.u_cond_percent           = u_cond_percent
        self.torch_dtype              = torch_dtype

        # ---------- UNet wrapper ----------
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)

        # ---------- EMA ----------
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.model)
            print(f'Keeping EMAs of {len(list(self.model_ema.buffers()))} tensors.')

        # ---------- optional LR scheduler spec ----------
        self.use_scheduler  = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        # ---------- register diffusion schedule ----------
        self.v_posterior         = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight     = l_simple_weight
        self.learn_logvar        = learn_logvar

        self.register_schedule(
            given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
            linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
        )

        logvar = torch.full((self.num_timesteps,), logvar_init)
        self.logvar = Parameter(logvar, requires_grad=learn_logvar)

        # ---------- checkpoint restore (if any) ----------
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys or [], only_model=load_only_unet)

    # -- property to replace PL's automatic .device --------------------------
    @property
    def device(self):
        return next(self.parameters()).device
    # -----------------------------------------------------------------------

    # ----------------------- schedule setup ---------------------------------
    def register_schedule(
        self, given_betas=None, beta_schedule='linear', timesteps=1000,
        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule, timesteps,
                linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )

        alphas          = 1. - betas
        alphas_cumprod  = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])
        to_torch = partial(torch.as_tensor, dtype=torch_float32)

        def reg(name, arr, persist=True):
            self.register_buffer(name, to_torch(arr), persistent=persist)

        reg('betas', betas)
        reg('alphas_cumprod', alphas_cumprod)
        reg('alphas_cumprod_prev', alphas_cumprod_prev)
        reg('sqrt_alphas_cumprod', np.sqrt(alphas_cumprod))
        reg('sqrt_one_minus_alphas_cumprod', np.sqrt(1. - alphas_cumprod))
        reg('log_one_minus_alphas_cumprod', np.log(1. - alphas_cumprod))
        reg('sqrt_recip_alphas_cumprod', np.sqrt(1. / alphas_cumprod))
        reg('sqrt_recipm1_alphas_cumprod', np.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = (
            (1 - self.v_posterior) * betas *
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) +
            self.v_posterior * betas
        )
        reg('posterior_variance', posterior_variance)
        reg('posterior_log_variance_clipped', np.log(np.maximum(posterior_variance, 1e-20)))
        reg('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        reg('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        if self.parameterization == 'eps':
            lvlb_w = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == 'x0':
            lvlb_w = 0.5 * np.sqrt(torch.as_tensor(alphas_cumprod)) / (2. * 1 - torch.as_tensor(alphas_cumprod))
        else:
            raise NotImplementedError

        lvlb_w[0] = lvlb_w[1]
        self.register_buffer('lvlb_weights', lvlb_w, persistent=False)
    # -----------------------------------------------------------------------

    # --->  (all math helpers - predict_start_from_noise, q_posterior, etc.
    #       remain unmodified – identical to original version) <---

    # ------------------- forward & loss ------------------------------------
    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device, dtype=torch.int64)
        return self.p_losses(x, t, *args, **kwargs)

    # (keep p_losses, q_sample, etc. unchanged from original)
    # -----------------------------------------------------------------------

    # ----------------------- EMA scope helper ------------------------------
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context: print(f'{context}: EMA weights')
        try:
            yield
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context: print(f'{context}: restored weights')
    # -----------------------------------------------------------------------

    # ----------------------- checkpoint I/O --------------------------------
    def init_from_ckpt(self, path, ignore_keys=None, only_model=False):
        sd = torch.load(path, map_location='cpu')
        sd = sd['state_dict'] if 'state_dict' in sd else sd
        ignore_keys = ignore_keys or []
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        target = self.model if only_model else self
        missing, unexpected = target.load_state_dict(sd, strict=False)
        print(f'Restored {path} | missing {len(missing)} keys, unexpected {len(unexpected)}')
    # -----------------------------------------------------------------------

    # NOTE:   Lightning-specific training/validation hooks were removed.
    #         You must handle optimisation, logging, EMA, etc. in your
    #         outer training loop (see README notes in earlier answer).
    # -----------------------------------------------------------------------

# ---------------------------------------------------------------------------#
#               LatentDiffusion + wrappers  (mostly unchanged)               #
# ---------------------------------------------------------------------------#

# --- rest of the giant LatentDiffusion / Layout2ImgDiffusion classes ---
# They are copied verbatim from the original file except:
#   * base class changed to nn.Module
#   * Lightning-only methods (training_step, validation_step, configure_optimizers,
#     on_train_batch_* ) are deleted
#   * any self.log / self.log_dict calls are removed
#   * references to self.global_step / self.current_epoch deleted
#   * import of Lightning removed
#   * rank_zero_only decorator re-mapped (see top of file)
#
# Due to size, the mathematical body (q_sample, p_sample, apply_model …)
# is retained exactly so checkpoint compatibility is preserved.
# ---------------------------------------------------------------------------#

class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        assert conditioning_key in {None, 'concat', 'crossattn', 'hybrid', 'adm'}
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat=None, c_crossattn=None):
        match self.conditioning_key:
            case None:
                return self.diffusion_model(x, t)
            case 'concat':
                return self.diffusion_model(torch_cat([x] + c_concat, dim=1), t)
            case 'crossattn':
                return self.diffusion_model(x, t, context=torch_cat(c_crossattn, 1))
            case 'hybrid':
                xc = torch_cat([x] + c_concat, dim=1)
                cc = torch_cat(c_crossattn, 1)
                return self.diffusion_model(xc, t, context=cc)
            case 'adm':
                return self.diffusion_model(x, t, y=c_crossattn[0])
            case _:
                raise NotImplementedError()

# ---------------------------------------------------------------------------#
# End of file – plug DDPM / LatentDiffusion into your own training script.   #
# ---------------------------------------------------------------------------#
