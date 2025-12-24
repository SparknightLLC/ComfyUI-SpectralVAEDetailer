# custom_nodes/ComfyUI-SpectralVAE/__init__.py

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

try:
	import comfy.samplers as comfy_samplers
	import comfy.model_management as model_management
	import comfy.sampler_helpers as sampler_helpers
except Exception:
	comfy_samplers = None
	model_management = None
	sampler_helpers = None

# -----------------------------
# Tuned constants (hardcoded)
# -----------------------------

# Noise flat suppression (local smoothstep mode)
_NOISE_SUPPRESS_ENERGY_RADIUS_MULT = 6
_NOISE_SUPPRESS_LO = 0.20
_NOISE_SUPPRESS_HI = 0.80

# Anti-splotch: remove residual LF drift from grain
_NOISE_KILL_LOWFREQ = True
_NOISE_KILL_LOWFREQ_MULT = 4

# Grain chroma handling (reduce chroma blotches)
_GRAIN_CHROMA_MODE_SEPARATE = True
_GRAIN_CHROMA_STRENGTH = 0.35

# Exposure-dependent grain (disabled via constant)
_GRAIN_EXPOSURE_MAP = False
_GRAIN_EXPOSURE_RADIUS = 16
_GRAIN_EXPOSURE_STRENGTH = 0.8

# Adaptive CFG mask thresholds.
# NOTE: mask energy is per-image normalized => mean ~= 1.0.
_CFG_ADAPT_LO = 0.75
_CFG_ADAPT_HI = 1.35

# -----------------------------
# Helpers / plumbing
# -----------------------------


def _get_base_model(model_patcher):
	m = getattr(model_patcher, "model", None)
	return m if m is not None else model_patcher


def _get_model_options(model_patcher, base_model):
	mo = getattr(model_patcher, "model_options", None)
	if isinstance(mo, dict):
		return mo
	mo = getattr(base_model, "model_options", None)
	if isinstance(mo, dict):
		return mo
	return {}


def _get_model_device_dtype(model_patcher, fallback_device, fallback_dtype):
	base_model = _get_base_model(model_patcher)
	dm = getattr(base_model, "diffusion_model", None)
	if dm is not None:
		try:
			p = next(dm.parameters())
			return p.device, p.dtype
		except Exception:
			pass
	if model_management is not None:
		try:
			return model_management.get_torch_device(), fallback_dtype
		except Exception:
			pass
	return fallback_device, fallback_dtype


def _ensure_model_loaded(model_patcher):
	if model_management is None:
		return
	try:
		model_management.load_model_gpu(model_patcher)
	except Exception:
		pass


@contextmanager
def _patcher_ctx(model_patcher):
	try:
		if hasattr(model_patcher, "pre_run"):
			model_patcher.pre_run()
		yield
	finally:
		if hasattr(model_patcher, "cleanup"):
			model_patcher.cleanup()


def _move_tensors(obj: Any, device: torch.device, dtype: torch.dtype | None):
	if torch.is_tensor(obj):
		t = obj.to(device=device)
		if dtype is not None and torch.is_floating_point(t) and t.dtype != dtype:
			t = t.to(dtype=dtype)
		return t
	if isinstance(obj, dict):
		return {k: _move_tensors(v, device, dtype) for k, v in obj.items()}
	if isinstance(obj, list):
		return [_move_tensors(v, device, dtype) for v in obj]
	if isinstance(obj, tuple):
		return tuple(_move_tensors(v, device, dtype) for v in obj)
	return obj


def _strip_timestep_limits(cond_list: list[dict]) -> list[dict]:
	out = []
	for c in cond_list:
		if not isinstance(c, dict):
			out.append(c)
			continue
		c2 = dict(c)
		for k in ("timestep_start", "timestep_end", "start_timestep", "end_timestep"):
			c2.pop(k, None)
		out.append(c2)
	return out


def _maybe_convert_conditioning(cond):
	if cond is None:
		return []
	if isinstance(cond, list) and len(cond) > 0 and isinstance(cond[0], dict):
		return cond
	if sampler_helpers is not None and hasattr(sampler_helpers, "convert_cond"):
		return sampler_helpers.convert_cond(cond)
	if comfy_samplers is not None and hasattr(comfy_samplers, "convert_cond"):
		return comfy_samplers.convert_cond(cond)
	raise RuntimeError("convert_cond not found (expected comfy.sampler_helpers.convert_cond).")


def _encode_model_conds_if_possible(base_model, conds, x_in, prompt_type: str):
	extra_conds = getattr(base_model, "extra_conds", None)
	if extra_conds is None or comfy_samplers is None or not hasattr(comfy_samplers, "encode_model_conds"):
		return conds
	try:
		return comfy_samplers.encode_model_conds(extra_conds, conds, x_in, x_in.device, prompt_type)
	except Exception:
		return conds


def _lowpass_avgpool(x: torch.Tensor, radius: int) -> torch.Tensor:
	# NOTE: original behavior intentionally preserved (avg_pool2d with zero padding).
	r = int(max(0, radius))
	if r <= 0:
		return x
	k = 2 * r + 1
	return F.avg_pool2d(x, kernel_size=k, stride=1, padding=r)


def _lowpass_avgpool_reflect(x: torch.Tensor, radius: int) -> torch.Tensor:
	# Used ONLY for masks/energy maps to avoid border artifacts caused by zero padding.
	r = int(max(0, radius))
	if r <= 0:
		return x
	k = 2 * r + 1
	xp = F.pad(x, (r, r, r, r), mode="reflect")
	return F.avg_pool2d(xp, kernel_size=k, stride=1, padding=0)


def _calculate_denoised(base_model, x_in: torch.Tensor, sigma: float, model_out: torch.Tensor) -> torch.Tensor:
	ms = getattr(base_model, "model_sampling", None)
	if ms is not None and hasattr(ms, "calculate_denoised"):
		try:
			return ms.calculate_denoised(float(sigma), x_in, model_out)
		except Exception:
			sig = x_in.new_full((x_in.shape[0], ), float(sigma))
			return ms.calculate_denoised(sig, x_in, model_out)
	# fallback (eps-pred assumption)
	return x_in - model_out * float(sigma)


def _randn_like(x: torch.Tensor, seed: int) -> torch.Tensor:
	if seed < 0:
		return torch.randn_like(x)
	try:
		g = torch.Generator(device=x.device)
		g.manual_seed(int(seed))
		return torch.randn_like(x, generator=g)
	except Exception:
		torch.manual_seed(int(seed))
		return torch.randn_like(x)


def _resolve_seed(seed: int) -> int:
	if seed >= 0:
		return int(seed)
	return int(torch.randint(0, 2**31 - 1, (1, )).item())


def _rms_norm_(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	var = x.pow(2).mean(dim=(2, 3), keepdim=True)
	return x.mul_(torch.rsqrt(var + eps))


def _bandpass_grain(noise: torch.Tensor, r: int) -> torch.Tensor:
	r = int(max(0, r))
	if r == 0:
		return _rms_norm_(noise)
	lp1 = _lowpass_avgpool(noise, r)
	lp2 = _lowpass_avgpool(noise, r * 2)
	band = lp1 - lp2
	return _rms_norm_(band)


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
	return x * x * (3.0 - 2.0 * x)


def _local_energy_map(hp: torch.Tensor, r: int) -> torch.Tensor:
	# Per-image normalized (mean~=1) energy map, smoothed locally.
	e = hp.pow(2).mean(dim=1, keepdim=True)
	if r > 0:
		e = _lowpass_avgpool_reflect(e, r)
	e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)
	return e


def _content_detail_mask_from_latent(x: torch.Tensor, r: int) -> torch.Tensor:
	"""
	Mask in [0..1] based on *image content* (not model-injected detail).
	Uses gradient magnitude of the latent structure channel (channel 0) -> local energy -> smoothstep.
	"""
	rr = int(max(0, r))
	l = x[:, :1]

	dx = l[..., 1:] - l[..., :-1]
	dy = l[..., 1:, :] - l[..., :-1, :]

	dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
	dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")

	gm = dx.abs_().add_(dy.abs_())

	e = gm
	if rr > 0:
		e = _lowpass_avgpool_reflect(e, rr)
	e = e / (e.mean(dim=(2, 3), keepdim=True) + 1e-6)

	lo = _CFG_ADAPT_LO
	hi = _CFG_ADAPT_HI
	t = torch.clamp((e - lo) / (hi - lo), 0.0, 1.0)
	return _smoothstep01(t)


def _soft_clip_tanh(x: torch.Tensor, k: float) -> torch.Tensor:
	kk = float(max(1e-6, k))
	return torch.tanh(x * kk) / kk


# -----------------------------
# Node
# -----------------------------


class SpectralVAEDetailer:
	"""
	SpectralVAEDetailer
	- 1x UNet forward at chosen sigma
	- Base detail projection from den_pos
	- Dedicated CFG injection from (den_pos - den_neg)
	- Optional adaptive CFG split (flat vs detailed) to avoid background splotchiness
	- Latent micrograin injection with local flat suppression + optional LF cleanup
	"""

	def __init__(self):
		self._conv_cache = {}
		self._enc_cache = {}

	@classmethod
	def INPUT_TYPES(cls):
		return {
		    "required": {
		        # --- Top controls
		        "seed": ("INT", {
		            "default": -1,
		            "min": -1,
		            "max": 2**31 - 1,
		            "step": 1,
		            "tooltip": "Random seed for grain. Use -1 for random each run."
		        }),

		        # --- Main inputs
		        "model": ("MODEL", ),
		        "latent": ("LATENT", ),
		        "positive": ("CONDITIONING", ),
		        "negative": ("CONDITIONING", ),

		        # --- CFG group
		        "cfg": ("FLOAT", {
		            "default": 7.0,
		            "min": 0.0,
		            "max": 10.0,
		            "step": 0.05,
		            "tooltip": "Base CFG. This node applies additional HF/LF shaping when cfg > 1."
		        }),
		        "cfg_hf_boost": ("FLOAT", {
		            "default": 5.0,
		            "min": 0.0,
		            "max": 5.0,
		            "step": 0.05,
		            "tooltip": "How strongly to inject high-frequency CFG detail (from den_pos - den_neg)."
		        }),
		        "cfg_lf_boost": ("FLOAT", {
		            "default": 0.0,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.02,
		            "tooltip": "How strongly to inject low-frequency CFG contrast (usually keep low)."
		        }),
		        "cfg_radius": ("INT", {
		            "default": 5,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "CFG split radius. In adaptive mode, this is the DETAIL radius."
		        }),
		        "cfg_radius_flat": ("INT", {
		            "default": 0,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "CFG split radius used in flat/low-detail regions when adaptive mode is ON."
		        }),
		        "cfg_radius_adaptive": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "If ON, blends between cfg_radius_flat (flat) and cfg_radius (detail)."
		        }),
		        # NEW: adaptive mask smoothing controls
		        "cfg_adapt_feather": ("INT", {
		            "default": 2,
		            "min": 0,
		            "max": 32,
		            "step": 1,
		            "tooltip": "Blur radius applied to the adaptive mask. Higher reduces halos but can soften detail reach."
		        }),
		        "cfg_adapt_gamma": ("FLOAT", {
		            "default": 2.0,
		            "min": 0.5,
		            "max": 3.0,
		            "step": 0.05,
		            "tooltip": "Mask curve. >1 shrinks 'detailed' regions (less spill/halo). <1 expands them."
		        }),

		        # --- Core look
		        "sigma": ("FLOAT", {
		            "default": 0.4,
		            "min": 0.001,
		            "max": 50.0,
		            "step": 0.001,
		            "tooltip": "Sigma at which to run the single UNet forward for detail projection."
		        }),
		        "detail_strength": ("FLOAT", {
		            "default": 0.65,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.01,
		            "tooltip": "Strength of injected high-frequency detail from denoised estimate."
		        }),
		        "hf_radius": ("INT", {
		            "default": 4,
		            "min": 0,
		            "max": 64,
		            "step": 1,
		            "tooltip": "Detail split radius for base projection (larger = coarser separation)."
		        }),
		        "mid_strength": ("FLOAT", {
		            "default": 0.05,
		            "min": 0.0,
		            "max": 0.5,
		            "step": 0.01,
		            "tooltip": "Adds some mid/low component of the base projection (contrast/shape)."
		        }),
		        "chroma_strength": ("FLOAT", {
		            "default": 0.1,
		            "min": 0.0,
		            "max": 2.0,
		            "step": 0.01,
		            "tooltip": "Scales chroma channels (1..3) for detail/CFG injections. Grain chroma is limited internally."
		        }),
		        "protect_lows": ("FLOAT", {
		            "default": 0.9,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Prevents HF detail from over-applying in low-frequency regions (reduces harshness)."
		        }),

		        # --- Detail soft-clip
		        "soft_clip_detail": ("BOOLEAN", {
		            "default": False,
		            "tooltip": "Soft-limits HF detail to reduce halos/zipper edges."
		        }),
		        "soft_clip_detail_k": ("FLOAT", {
		            "default": 2.2,
		            "min": 0.5,
		            "max": 8.0,
		            "step": 0.05,
		            "tooltip": "Detail soft-clip amount. Higher = weaker limiting."
		        }),
		        "soft_clip_cfg": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "Soft-limits HF CFG injection to reduce harsh edges and background speckle."
		        }),
		        "soft_clip_cfg_k": ("FLOAT", {
		            "default": 2.0,
		            "min": 0.5,
		            "max": 8.0,
		            "step": 0.05,
		            "tooltip": "CFG soft-clip amount. Higher = weaker limiting."
		        }),

		        # --- Grain
		        "noise_scale": ("FLOAT", {
		            "default": 0.2,
		            "min": 0.0,
		            "max": 0.5,
		            "step": 0.01,
		            "tooltip": "Micrograin intensity in latent space."
		        }),
		        "noise_radius": ("INT", {
		            "default": 1,
		            "min": 0,
		            "max": 16,
		            "step": 1,
		            "tooltip": "Grain correlation radius. 0=white, 1..3 often looks most photographic."
		        }),
		        "noise_flat_suppress": ("FLOAT", {
		            "default": 1.0,
		            "min": 0.0,
		            "max": 1.0,
		            "step": 0.01,
		            "tooltip": "Suppresses grain in flat regions (stronger in local smoothstep mode)."
		        }),
		        "noise_suppress_mode": (["global", "local_smoothstep"], {
		            "default": "local_smoothstep",
		            "tooltip": "global = older behavior. local_smoothstep = per-pixel local contrast gating (recommended)."
		        }),

		        # --- Bottom toggles
		        "ignore_cond_timestep_range": ("BOOLEAN", {
		            "default": True,
		            "tooltip": "If ON, strips timestep limits from conditioning ranges (more consistent behavior)."
		        }),
		        "debug_print": ("BOOLEAN", {
		            "default": False,
		            "tooltip": "Prints some diagnostics to console."
		        }),
		    }
		}

	RETURN_TYPES = ("LATENT", )
	FUNCTION = "apply"
	CATEGORY = "latent/postprocess"

	def _convert_cached(self, cond_obj, ignore_range: bool) -> list[dict]:
		key = (id(cond_obj), bool(ignore_range))
		if key in self._conv_cache:
			return self._conv_cache[key]
		c = _maybe_convert_conditioning(cond_obj)
		if ignore_range:
			c = _strip_timestep_limits(c)
		self._conv_cache[key] = c
		if len(self._conv_cache) > 6:
			self._conv_cache.pop(next(iter(self._conv_cache)))
		return c

	def _encode_cached(self, base_model, conds: list[dict], x_in: torch.Tensor, prompt_type: str) -> list[dict]:
		key = (id(conds), str(x_in.device), str(x_in.dtype), tuple(x_in.shape), prompt_type)
		if key in self._enc_cache:
			return self._enc_cache[key]
		enc = _encode_model_conds_if_possible(base_model, conds, x_in, prompt_type)
		self._enc_cache[key] = enc
		if len(self._enc_cache) > 12:
			self._enc_cache.pop(next(iter(self._enc_cache)))
		return enc

	def _cond_uncond_outs(self, model_patcher, x_in, sigma_value: float, positive, negative, ignore_range: bool):
		if comfy_samplers is None or not hasattr(comfy_samplers, "calc_cond_batch"):
			raise RuntimeError("comfy.samplers.calc_cond_batch is unavailable. Update ComfyUI.")

		base_model = _get_base_model(model_patcher)
		model_options = _get_model_options(model_patcher, base_model)

		pos = self._convert_cached(positive, ignore_range)
		neg = self._convert_cached(negative, ignore_range)

		pos = self._encode_cached(base_model, pos, x_in, "positive")
		neg = self._encode_cached(base_model, neg, x_in, "negative")

		pos = _move_tensors(pos, x_in.device, x_in.dtype)
		neg = _move_tensors(neg, x_in.device, x_in.dtype)

		sig = x_in.new_full((x_in.shape[0], ), float(sigma_value))
		outs = comfy_samplers.calc_cond_batch(base_model, [pos, neg], x_in, sig, model_options)
		if len(outs) < 2:
			raise RuntimeError("calc_cond_batch did not return [cond, uncond].")
		return base_model, outs[0], outs[1]

	@torch.no_grad()
	def apply(
	    self,
	    seed: int,
	    model,
	    latent,
	    positive,
	    negative,
	    cfg: float,
	    cfg_hf_boost: float,
	    cfg_lf_boost: float,
	    cfg_radius: int,
	    cfg_radius_flat: int,
	    cfg_radius_adaptive: bool,
	    cfg_adapt_feather: int,
	    cfg_adapt_gamma: float,
	    sigma: float,
	    detail_strength: float,
	    hf_radius: int,
	    mid_strength: float,
	    chroma_strength: float,
	    protect_lows: float,
	    soft_clip_detail: bool,
	    soft_clip_detail_k: float,
	    soft_clip_cfg: bool,
	    soft_clip_cfg_k: float,
	    noise_scale: float,
	    noise_radius: int,
	    noise_flat_suppress: float,
	    noise_suppress_mode: str,
	    ignore_cond_timestep_range: bool,
	    debug_print: bool,
	):
		_ensure_model_loaded(model)

		x_orig = latent["samples"]
		if not torch.is_tensor(x_orig):
			raise RuntimeError("LATENT['samples'] was not a tensor.")

		orig_dev = x_orig.device
		orig_dtype = x_orig.dtype

		model_dev, model_dtype = _get_model_device_dtype(model, orig_dev, orig_dtype)

		x_in = x_orig.to(device=model_dev)
		if torch.is_floating_point(x_in) and x_in.dtype != model_dtype:
			x_in = x_in.to(dtype=model_dtype)

		sig = float(max(1e-6, sigma))
		used_seed = _resolve_seed(int(seed))

		with _patcher_ctx(model):
			base_model, out_pos, out_neg = self._cond_uncond_outs(model, x_in, sig, positive, negative, bool(ignore_cond_timestep_range))

		den_pos = _calculate_denoised(base_model, x_in, sig, out_pos)
		den_neg = _calculate_denoised(base_model, x_in, sig, out_neg)

		# Base detail projection uses den_pos
		base_delta = den_pos - x_in
		base_low = _lowpass_avgpool(base_delta, int(hf_radius))
		base_hp = base_delta - base_low

		# Protect low-frequency regions from HF over-application
		pl = float(max(0.0, min(1.0, protect_lows)))
		if pl > 0.0:
			hp_e = base_hp.abs().mean(dim=1, keepdim=True).add_(1e-6)
			d_e = base_delta.abs().mean(dim=1, keepdim=True).add_(1e-6)
			gate = hp_e / (hp_e + d_e)
			factor = gate.mul(pl).add_(1.0 - pl)  # (1-pl) + pl*gate
			base_hp.mul_(factor)

		# Optional soft clip to reduce halos / zipper edges
		if bool(soft_clip_detail):
			base_hp = _soft_clip_tanh(base_hp, float(soft_clip_detail_k))

		# Chroma scaling for detail (in-place)
		cs = float(chroma_strength)
		if base_hp.shape[1] >= 4 and cs != 1.0:
			base_hp[:, 1:4].mul_(cs)

		# Build output (in-place accumulation)
		out = x_in.clone()

		ds = float(detail_strength)
		if ds != 0.0:
			out.add_(base_hp, alpha=ds)

		ms = float(max(0.0, mid_strength))
		if ms != 0.0:
			out.add_(base_low, alpha=ms)

		# Dedicated CFG injection
		c = float(cfg)
		cfg_scale = max(0.0, c - 1.0)
		if cfg_scale > 0.0 and (cfg_hf_boost > 0.0 or cfg_lf_boost > 0.0):
			cfg_delta = den_pos - den_neg

			if bool(cfg_radius_adaptive):
				r_flat = int(max(0, cfg_radius_flat))
				r_det = int(max(0, cfg_radius))  # cfg_radius is detail radius

				# Content-derived mask (0..1)
				mask_r = max(1, int(hf_radius))
				m = _content_detail_mask_from_latent(x_in, mask_r)  # (B,1,H,W)

				# Feather the mask to prevent boundary halos/bloom
				fr = int(max(0, cfg_adapt_feather))
				if fr > 0:
					m = _lowpass_avgpool_reflect(m, fr)

				# Curve the mask: gamma > 1 reduces spill into flat regions
				gam = float(max(1e-3, cfg_adapt_gamma))
				if abs(gam - 1.0) > 1e-6:
					m = m.clamp(0.0, 1.0).pow(gam)

				m = m.clamp(0.0, 1.0)

				low_flat = _lowpass_avgpool(cfg_delta, r_flat)
				low_det = _lowpass_avgpool(cfg_delta, r_det)

				# Blend low, then derive hp from blended low (numerically consistent)
				cfg_low = low_flat.mul(1.0 - m).add_(low_det.mul(m))
				cfg_hp = cfg_delta - cfg_low
			else:
				cfg_low = _lowpass_avgpool(cfg_delta, int(cfg_radius))
				cfg_hp = cfg_delta - cfg_low

			if bool(soft_clip_cfg):
				cfg_hp = _soft_clip_tanh(cfg_hp, float(soft_clip_cfg_k))

			# Chroma scaling for CFG injections (in-place)
			if cfg_hp.shape[1] >= 4 and cs != 1.0:
				cfg_hp[:, 1:4].mul_(cs)

			hf_a = float(cfg_hf_boost) * cfg_scale
			lf_a = float(cfg_lf_boost) * cfg_scale
			if hf_a != 0.0:
				out.add_(cfg_hp, alpha=hf_a)
			if lf_a != 0.0:
				out.add_(cfg_low, alpha=lf_a)

		# Micrograin
		ns = float(max(0.0, noise_scale))
		if ns > 0.0:
			n = _randn_like(out, used_seed)
			g = _bandpass_grain(n, int(noise_radius))

			fs = float(max(0.0, min(1.0, noise_flat_suppress)))
			if fs > 0.0:
				mode = str(noise_suppress_mode)
				if mode == "global":
					e = base_hp.abs().mean(dim=1, keepdim=True)
					e = e / (e.mean() + 1e-6)
					allow = torch.clamp(e, 0.0, 1.0).pow(2.0)
					allow = (1.0 - fs) + fs * allow
					g = g * allow
				else:
					er = max(1, int(noise_radius) * _NOISE_SUPPRESS_ENERGY_RADIUS_MULT)
					e = _local_energy_map(base_hp, er)
					t = torch.clamp((e - _NOISE_SUPPRESS_LO) / (_NOISE_SUPPRESS_HI - _NOISE_SUPPRESS_LO), 0.0, 1.0)
					allow = _smoothstep01(t)
					allow = (1.0 - fs) + fs * allow
					g = g * allow

			if _NOISE_KILL_LOWFREQ and int(noise_radius) > 0:
				rr = int(noise_radius) * _NOISE_KILL_LOWFREQ_MULT
				g = g - _lowpass_avgpool(g, rr)

			if _GRAIN_EXPOSURE_MAP:
				r = int(max(0, _GRAIN_EXPOSURE_RADIUS))
				lum = den_pos[:, :1]
				if r > 0:
					lum = _lowpass_avgpool(lum, r)
				lum = (lum - lum.mean(dim=(2, 3), keepdim=True)) / (lum.std(dim=(2, 3), keepdim=True) + 1e-6)
				grain_map = torch.sigmoid(-float(_GRAIN_EXPOSURE_STRENGTH) * lum)
				g = g * grain_map

			if g.shape[1] >= 4 and _GRAIN_CHROMA_MODE_SEPARATE:
				g[:, 1:4].mul_(float(_GRAIN_CHROMA_STRENGTH))

			out.add_(g, alpha=ns)

		if debug_print:
			d_unet = (out_pos - out_neg).abs().mean().item()
			d_den = (den_pos - den_neg).abs().mean().item()
			print(f"[SpectralVAEDetailer] used_seed={used_seed} cfg={c:.3f} sigma={sig:.4f} noise_scale={ns:.3f} "
			      f"| mean|pos-neg|={d_unet:.6g} mean|den_pos-den_neg|={d_den:.6g}")

		# Move back to original device/dtype
		out = out.to(device=orig_dev)
		if torch.is_floating_point(out) and out.dtype != orig_dtype:
			out = out.to(dtype=orig_dtype)

		out_latent = dict(latent)
		out_latent["samples"] = out
		return (out_latent, )


NODE_CLASS_MAPPINGS = {
    "SpectralVAEDetailer": SpectralVAEDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpectralVAEDetailer": "SpectralVAEDetailer",
}
