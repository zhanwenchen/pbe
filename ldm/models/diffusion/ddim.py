"""SAMPLING ONLY."""
from os import getenv
import torch
from torch import (
    cat as torch_cat,
    Tensor as torch_Tensor,
    float32 as torch_float32,
    int64 as torch_int64,
    no_grad as torch_no_grad,
    full as torch_full,
    randn as torch_randn,
    inference_mode as torch_inference_mode,
)
from torch.nn.functional import dropout
import numpy as np
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
# from hunter import wrap, StackPrinter
# DEPTH = 2


class DDIMSampler:
    def __init__(self, model, schedule='linear', **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if isinstance(attr, torch_Tensor):
            attr = attr.to(torch.device("cuda"), non_blocking=True)
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        model = self.model
        alphas_cumprod = model.alphas_cumprod
        assert alphas_cumprod.size(0) == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        device = model.device
        to_torch = lambda x: x.clone().detach().to(dtype=torch_float32, device=device, non_blocking=True)

        register_buffer = self.register_buffer
        register_buffer('betas', to_torch(model.betas))
        register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        register_buffer('alphas_cumprod_prev', to_torch(model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        register_buffer('sqrt_alphas_cumprod', to_torch(alphas_cumprod.sqrt()))
        # register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        register_buffer('sqrt_one_minus_alphas_cumprod', to_torch((1. - alphas_cumprod).sqrt()))
        register_buffer('log_one_minus_alphas_cumprod', to_torch((1. - alphas_cumprod).log()))
        register_buffer('sqrt_recip_alphas_cumprod', to_torch((1. / alphas_cumprod).sqrt()))
        register_buffer('sqrt_recipm1_alphas_cumprod', to_torch((1. / alphas_cumprod - 1).sqrt()))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        register_buffer('ddim_sigmas', ddim_sigmas)
        register_buffer('ddim_alphas', ddim_alphas)
        register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        alphas_cumprod, alphas_cumprod_prev = self.alphas_cumprod, self.alphas_cumprod_prev
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (1 - alphas_cumprod / alphas_cumprod_prev))
        register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
        # wrapper = wrap(depth=DEPTH, action=StackPrinter)
        # self.__call__ = wrapper(self.__call__)

    # @wrap(depth=DEPTH, action=StackPrinter)
    @torch_no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               disable_tqdm=False,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[next(conditioning.keys())].size(0)
                if cbs != batch_size:
                    raise ValueError(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.size(0) != batch_size:
                    raise ValueError(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape[-3:]
        size = (batch_size, C, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    disable_tqdm=disable_tqdm,
                                                    **kwargs
                                                    )
        return samples, intermediates

    # @wrap(depth=DEPTH, action=StackPrinter)
    # @torch_no_grad()
    @torch_inference_mode()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, disable_tqdm=False, **kwargs):
        device = self.model.betas.device
        b = shape[0]
        # if x_T is None:
        #     img = torch.randn(shape, device=device)
        # else:
        #     img = x_T

        img = x_T if x_T is not None else torch_randn(shape, device=device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        # print(f"ddim_sampling: Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc=f'{getenv("LOCAL_RANK")=}: DDIMSampler.ddim_sampling', total=total_steps, leave=False, disable=disable_tqdm)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch_full((b,), step, device=device, dtype=torch_int64)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,**kwargs)
            img, pred_x0 = outs
            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    # @wrap(depth=DEPTH, action=StackPrinter)
    @torch_no_grad()
    # @torch_inference_mode()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,**kwargs) -> tuple[torch_Tensor, torch_Tensor]:
        b, *_, device = *x.shape, x.device
        if 'test_model_kwargs' in kwargs:
            kwargs=kwargs['test_model_kwargs']
            # x = torch_cat((x, kwargs['inpaint_image'], kwargs['inpaint_mask']), dim=1)
            x = torch_cat((x, kwargs['images_inpaint'], kwargs['images_mask']), dim=1)
        elif 'rest' in kwargs:
            x = torch_cat((x, kwargs['rest']), dim=1)
        else:
            raise Exception("kwargs must contain either 'test_model_kwargs' or 'rest' key")
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            # print(f'p_sample_ddim: {x.size()=}, {t.size()=}, {c.size()=}')
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch_cat([x] * 2)
            t_in = torch_cat([t] * 2)
            c_in = torch_cat((unconditional_conditioning, c))
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch_full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch_full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch_full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch_full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if x.size(1) != 4:
            pred_x0 = (x[:,:4,:,:] - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(dir_xt.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch_no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    # @wrap(depth=DEPTH, action=StackPrinter)
    @torch_no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, disable_tqdm=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f'Running DDIM Sampling with {total_steps} timesteps')

        iterator = tqdm(time_range, desc='DDIMSampler.decode', total=total_steps, leave=False, disable=disable_tqdm)
        x_dec = x_latent
        device = x_latent.device
        p_sample_ddim = self.p_sample_ddim
        len_x_latent = x_latent.shape[0]
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch_full((len_x_latent,), step, device=device, dtype=torch_int64)
            x_dec, _ = p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
