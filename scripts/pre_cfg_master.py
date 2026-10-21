import math

import gradio as gr
import torch

from modules import scripts

LOG_PREFIX = "[Pre-CFG Master]"
EPS = 1e-8

TARGETS = ["both", "cond", "uncond"]
SHARP_MODES = ["Disabled", "Difference", "Consensus", "Exponentiate"]
VAR_MODES = ["sigma", "steps progression"]

CONSENSUS_HIST_MAX = 8
CONSENSUS_MIN_HIST = 4

# Standard SDXL latent RGB factors for colour steering (4-channel latents only)
SDXL_LATENT_RGB_FACTORS = [
    [0.298, 0.207, 0.208],
    [0.187, 0.286, 0.173],
    [-0.158, 0.189, 0.264],
    [-0.184, -0.271, -0.473],
]
SDXL_SCALE_FACTOR = 0.13025


# ============================================================
# TENSOR MATHS
# Everything works on whole batches at once: (B, C, *spatial).
# The original looped over [b][c] in Python; these are the same
# formulas, vectorised over the sample / channel axes.
# ============================================================
def _keep(x, reduced_dims):
    """Shape that broadcasts a reduction over the first `reduced_dims` axes back onto x."""
    return x.shape[:reduced_dims] + (1,) * (x.dim() - reduced_dims)


def norm_per_channel(x):
    return x.flatten(2).norm(dim=2).view(_keep(x, 2))


def norm_per_sample(x):
    return x.flatten(1).norm(dim=1).view(_keep(x, 1))


def normalize_adjust(a, b, strength):
    """Pull every (sample, channel) slice of `a` away from the part of `b`
    orthogonal to it, keeping the slice's norm."""
    norm_a = norm_per_channel(a)
    a_n = a / (norm_a + EPS)
    b_n = b / (norm_per_channel(b) + EPS)
    dot = (a_n * b_n).flatten(2).sum(dim=2).view(_keep(a, 2))
    res = torch.nan_to_num(b_n - a_n * dot, nan=0.0)
    return (a_n - res * strength) * norm_a


def normalized_pow(t, p):
    t_norm = norm_per_sample(t)
    t_pow = (t / (t_norm + EPS)).abs().pow(p)
    return t_pow / (norm_per_sample(t_pow) + EPS) * t_norm * t.sign()


def subtract_mean(x, per_channel):
    dims = tuple(range(2 if per_channel else 1, x.dim()))
    return x - x.mean(dim=dims, keepdim=True)


def color_shift(x, r, g, b):
    shift = [
        (r * f[0] + g * f[1] + b * f[2]) / SDXL_SCALE_FACTOR
        for f in SDXL_LATENT_RGB_FACTORS
    ]
    shift = torch.tensor(shift, dtype=x.dtype, device=x.device).view((1, 4) + (1,) * (x.dim() - 2))
    shifted = x + shift
    return shifted / (norm_per_sample(shifted) + EPS) * norm_per_sample(x)


def euclidean_weights(tensors, exponent=2, proximity_exponent=1, min_score_into_zeros=0):
    """Weighted consensus of a stack of tensors: elements that agree with the
    rest of the stack get more weight. Returns (consensus, mean_score)."""
    n = tensors.shape[0]
    if exponent == 0:
        exponent = 6.55

    distances = []
    for i in range(n):
        d = (tensors - tensors[i]).abs() / n  # the i == i term is zero
        if proximity_exponent > 1:
            d = d ** proximity_exponent
        distances.append(d.sum(dim=0))
    distances = torch.stack(distances)

    min_d = distances.min(dim=0).values
    span = distances.max(dim=0).values - min_d
    # Where every entry agrees, span is 0. The original divided by it and
    # turned those elements into NaN, then zero; equal weights is the intent.
    span = torch.where(span == 0, torch.ones_like(span), span)

    weights = (1.0 - (distances - min_d) / span).clamp(min=0.0)
    if min_score_into_zeros > 0:
        weights = torch.where(weights < min_score_into_zeros, torch.zeros_like(weights), weights)
    if exponent > 1:
        weights = weights ** exponent

    total = weights.sum(dim=0)
    mean_score = (total.mean() / n) ** exponent
    total = torch.where(total == 0, torch.full_like(total, 1.0 / n), total)
    weights = torch.nan_to_num(weights / total, nan=0.0, posinf=0.0, neginf=0.0)

    return (tensors * weights).sum(dim=0), mean_score


# ============================================================
# SIGMA SCHEDULE
# ============================================================
def resolve_sampling(unet):
    """The sigma <-> timestep object: `model_sampling` on reForge (comfy's
    ldm_patched), `predictor` on Forge / Forge Neo (backend/)."""
    model = getattr(unet, "model", None)
    for name in ("model_sampling", "predictor"):
        sampling = getattr(model, name, None)
        if sampling is not None:
            return sampling
    return None


class SigmaSchedule:
    """Maps the current sigma to sampling progress in [0, 1]."""

    def __init__(self, sampling):
        self.sampling = sampling
        self.sigma_max = self._read(sampling, "sigma_max", 14.6146)
        self.sigma_min = self._read(sampling, "sigma_min", 0.0292)
        self.t_max = self.t_min = None
        if sampling is not None:
            try:
                self.t_max = self._timestep(self.sigma_max)
                self.t_min = self._timestep(self.sigma_min)
                if not math.isfinite(self.t_max - self.t_min) or self.t_max == self.t_min:
                    self.t_max = self.t_min = None
            except Exception:
                self.t_max = self.t_min = None

    @staticmethod
    def _read(sampling, name, default):
        try:
            value = float(getattr(sampling, name))
            return value if math.isfinite(value) and value > 0 else default
        except Exception:
            return default

    def _timestep(self, sigma):
        t = self.sampling.timestep(torch.tensor([sigma], dtype=torch.float32))
        return float(t.reshape(-1)[0])

    def by_sigma(self, sigma):
        return min(max(1.0 - sigma / self.sigma_max, 0.0), 1.0)

    def by_steps(self, sigma):
        """Progress through the model's own timestep axis. On SD1.x / SDXL this
        is 1 - t/999, as before; on EDM and flow models (where timestep() is
        not an index into a 1000-entry table) it still runs 0 -> 1 instead of
        producing nonsense."""
        if self.t_max is None or sigma <= 0.0:
            return self.by_sigma(sigma)
        try:
            t = self._timestep(sigma)
        except Exception:
            return self.by_sigma(sigma)
        return min(max((self.t_max - t) / (self.t_max - self.t_min), 0.0), 1.0)


def sigma_window(start, end):
    """Active while end < sigma <= start. A reversed pair is read the obvious way."""
    hi, lo = (start, end) if start >= end else (end, start)
    return lambda sigma: hi >= sigma > lo


# ============================================================
# ENGINE - one per sampling pass, so history never leaks between
# batches or into the Hires. fix pass
# ============================================================
class PreCFGEngine:
    def __init__(self, cfg, schedule):
        self.cfg = cfg
        self.schedule = schedule
        self.calls = 0
        self.prev = [None, None]  # Difference mode
        self.history = [[], []]  # Consensus mode
        self.color_warned = False

        self.in_mean = sigma_window(cfg["mean_start"], cfg["mean_end"])
        self.in_sharp = sigma_window(cfg["sharp_start"], cfg["sharp_end"])
        self.in_color = sigma_window(cfg["col_start"], cfg["col_end"])

    @staticmethod
    def targets(target, has_uncond):
        idx = []
        if target in ("both", "cond"):
            idx.append(0)
        if has_uncond and target in ("both", "uncond"):
            idx.append(1)
        return idx

    # -------- pre-CFG: edits [cond_pred, uncond_pred] --------
    @torch.no_grad()
    def pre_cfg(self, cond, uncond, sigma_t):
        """Returns new (cond, uncond); uncond may be None (CFG 1 optimisation)."""
        c = self.cfg
        sigma = float(sigma_t.reshape(-1)[0])
        first_call = self.calls == 0
        self.calls += 1

        has_uncond = uncond is not None and bool(torch.any(uncond))
        dtype = cond.dtype
        out = [cond.float(), uncond.float() if has_uncond else None]

        # ---------- 1. Colour ----------
        if c["en_color"] and self.in_color(sigma) and any((c["col_r"], c["col_g"], c["col_b"])):
            if out[0].shape[1] == 4:
                for i in self.targets(c["color_target"], has_uncond):
                    out[i] = color_shift(out[i], c["col_r"], c["col_g"], c["col_b"])
            elif not self.color_warned:
                self.color_warned = True
                print(f"{LOG_PREFIX} Latent Color needs a 4-channel latent (SD1.x / SDXL), "
                      f"this model has {out[0].shape[1]} - skipped.")

        # ---------- 2. Variable scale ----------
        if c["en_var"] and has_uncond:
            if c["var_prop"] == "steps progression":
                prog = self.schedule.by_steps(sigma)
            else:
                prog = self.schedule.by_sigma(sigma)
            mult = c["var_start"] * (1.0 - prog) + c["var_end"] * prog
            out[0] = out[0] * mult
            out[1] = out[1] * mult

        # ---------- 3. Sharpening ----------
        mode = c["sharp_mode"]
        if mode == "Difference":
            self._difference(out, sigma, first_call, has_uncond)
        elif mode == "Consensus":
            self._consensus(out, sigma, first_call, has_uncond)
        elif mode == "Exponentiate" and self.in_sharp(sigma):
            for i in self.targets(c["sharp_target"], has_uncond):
                out[i] = normalized_pow(out[i], c["exp_scale"])

        # ---------- 4. Mean (pre) ----------
        if c["en_mean_pre"] and self.in_mean(sigma):
            for i in range(2 if has_uncond else 1):
                out[i] = subtract_mean(out[i], c["mean_per_channel"])

        new_cond = out[0].to(dtype)
        new_uncond = out[1].to(dtype) if has_uncond else uncond
        return new_cond, new_uncond

    def _difference(self, out, sigma, first_call, has_uncond):
        c = self.cfg
        if first_call:
            self.prev = [None, None]
        if not first_call and self.in_sharp(sigma):
            for i in self.targets(c["sharp_target"], has_uncond):
                prev = self.prev[i]
                if prev is not None and prev.shape == out[i].shape:
                    out[i] = normalize_adjust(out[i], prev, c["sharp_scale"])
        self.prev[0] = out[0].clone()
        if has_uncond:
            self.prev[1] = out[1].clone()

    def _consensus(self, out, sigma, first_call, has_uncond):
        c = self.cfg
        if first_call:
            self.history = [[], []]
        for i in range(2 if has_uncond else 1):
            hist = self.history[i]
            if hist and hist[-1].shape != out[i].shape:
                hist.clear()
            hist.append(out[i] / (out[i].norm() + EPS))
            if len(hist) > CONSENSUS_HIST_MAX:
                hist.pop(0)

        if not self.in_sharp(sigma):
            return
        for i in self.targets(c["sharp_target"], has_uncond):
            hist = self.history[i]
            if len(hist) < CONSENSUS_MIN_HIST:
                continue
            consensus, score = euclidean_weights(torch.stack(hist))
            consensus = consensus * out[i].norm()
            out[i] = normalize_adjust(out[i], consensus, score * c["sharp_scale"])

    # -------- post-CFG --------
    @torch.no_grad()
    def post_cfg(self, args):
        denoised = args["denoised"]
        sigma = float(args["sigma"].reshape(-1)[0])
        if self.cfg["en_mean_post"] and self.in_mean(sigma):
            return subtract_mean(denoised, self.cfg["mean_per_channel"])
        return denoised

    # -------- backend glue --------
    def wrap_cfg(self, previous, edit_strength):
        """Forge / Forge Neo hand the predictions to sampler_cfg_function, which
        must return the CFG'd noise (the caller does cfg_result = x - fn(args)).
        Whatever cfg function is installed at that moment is chained, not lost."""

        @torch.no_grad()
        def cfg_function(args):
            x = args["input"]
            cond, uncond = self.pre_cfg(args["cond_denoised"], args["uncond_denoised"], args["sigma"])
            if uncond is None:
                uncond = torch.zeros_like(cond)
            if previous is not None:
                forwarded = dict(args)
                forwarded.update(cond_denoised=cond, uncond_denoised=uncond, cond=x - cond, uncond=x - uncond)
                return previous(forwarded)
            scale = args["cond_scale"] * edit_strength
            return x - (uncond + (cond - uncond) * scale)

        cfg_function._pre_cfg_master_owner = self
        return cfg_function

    def hook(self, *args):
        """One pre-CFG hook for both contracts, picked by how it is called:

        reForge (comfy ldm_patched): fn(args: dict) after prediction, with
            args["conds_out"] = [cond_pred, uncond_pred or None].
        Forge / Forge Neo (backend/): fn(model, cond, uncond, x, timestep,
            model_options) BEFORE prediction. The predictions are not there yet,
            so this returns a model_options copy whose sampler_cfg_function is
            wrapped. Doing it here, per step, means a cfg function installed
            after this script (Neo's Rescale CFG, Dynamic Thresholding...) is
            chained instead of silently overwriting this one.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            a = args[0]
            conds_out = list(a["conds_out"])
            uncond = conds_out[1] if len(conds_out) > 1 else None
            cond, uncond = self.pre_cfg(conds_out[0], uncond, a["sigma"])
            conds_out[0] = cond
            if len(conds_out) > 1:
                conds_out[1] = uncond
            return conds_out

        model, cond, uncond, x, timestep, model_options = args
        current = model_options.get("sampler_cfg_function")
        if getattr(current, "_pre_cfg_master_owner", None) is self:
            return args
        edit_strength = sum(item.get("strength", 1) for item in cond) if cond else 1.0
        options = dict(model_options)
        options["sampler_cfg_function"] = self.wrap_cfg(current, edit_strength)
        return model, cond, uncond, x, timestep, options


# ============================================================
# SCRIPT
# ============================================================
# Order of ui() return values == order of process() arguments.
ARG_NAMES = [
    "apply_hr",
    "en_color", "color_target", "col_r", "col_g", "col_b", "col_start", "col_end",
    "en_var", "var_prop", "var_start", "var_end",
    "sharp_mode", "sharp_target", "sharp_scale", "exp_scale", "sharp_start", "sharp_end",
    "en_mean_pre", "en_mean_post", "mean_per_channel", "mean_start", "mean_end",
]

# Infotext key per argument, grouped by the switch that makes them meaningful.
INFOTEXT_GROUPS = {
    "en_color": {
        "en_color": "PreCFG color",
        "color_target": "PreCFG color target",
        "col_r": "PreCFG color R", "col_g": "PreCFG color G", "col_b": "PreCFG color B",
        "col_start": "PreCFG color start", "col_end": "PreCFG color end",
    },
    "en_var": {
        "en_var": "PreCFG var scale",
        "var_prop": "PreCFG var scale mode",
        "var_start": "PreCFG var scale start", "var_end": "PreCFG var scale end",
    },
    "sharp_mode": {
        "sharp_mode": "PreCFG sharp mode",
        "sharp_target": "PreCFG sharp target",
        "sharp_scale": "PreCFG sharp scale", "exp_scale": "PreCFG sharp exponent",
        "sharp_start": "PreCFG sharp start", "sharp_end": "PreCFG sharp end",
    },
    "en_mean": {
        "en_mean_pre": "PreCFG mean pre", "en_mean_post": "PreCFG mean post",
        "mean_per_channel": "PreCFG mean per-channel",
        "mean_start": "PreCFG mean start", "mean_end": "PreCFG mean end",
    },
}
SWITCH_DEFAULTS = {
    "en_color": False, "en_var": False, "en_mean_pre": False, "en_mean_post": False,
    "sharp_mode": "Disabled",
}


def _sigma_range(elem_id, prefix, start, end):
    # Labels must be unique per script: ui-config.json keys components by label.
    with gr.Row():
        lo = gr.Slider(0.0, 100.0, step=0.1, value=start, label=f"{prefix} σ start",
                       elem_id=elem_id("start"))
        hi = gr.Slider(0.0, 100.0, step=0.1, value=end, label=f"{prefix} σ end",
                       elem_id=elem_id("end"))
    return lo, hi


def _block_title(text, hint):
    # Plain Markdown, not a Label/Button: nothing for Lobe Theme's replaceIcon
    # to grab, and it renders the same on gradio 3.41 and 4.x. The hint lives
    # here rather than in a component's `info`, which makes one column of a
    # row taller than the other and misaligns it in narrow layouts.
    gr.Markdown(f"**{text}** - {hint}")


class MasterPreCFGScript(scripts.Script):
    sorting_priority = 8888

    def title(self):
        return "Advanced Pre-CFG Master Controller"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        def eid(section):
            return lambda name: self.elem_id(f"pcm_{section}_{name}")

        # Flat layout: one accordion, one gr.Group per module, in the order they
        # run each step. No emoji in the accordion title: Lobe Theme's replaceIcon
        # swaps the whole innerHTML of any <button> whose text contains one of its
        # listed emoji, and gradio 4 renders accordion headers as buttons.
        with gr.Accordion("Advanced Model Patches (Pre-CFG)", open=False,
                          elem_id=self.elem_id("pcm_root")):

            with gr.Group(elem_id=self.elem_id("pcm_color")):
                e = eid("color")
                _block_title("Latent Color", "steers RGB inside a 4-channel latent (SDXL / SD1.x)")
                with gr.Row():
                    en_color = gr.Checkbox(label="Enable color", value=False, elem_id=e("enable"))
                    color_target = gr.Dropdown(TARGETS, value="both", label="Color apply to", elem_id=e("target"))
                # One row even in narrow columns: style.css shrinks these sliders' number boxes.
                with gr.Row():
                    col_r = gr.Slider(-2.0, 2.0, step=0.01, value=0.0, label="Red", elem_id=e("r"), elem_classes=["pcm-rgb"])
                    col_g = gr.Slider(-2.0, 2.0, step=0.01, value=0.0, label="Green", elem_id=e("g"), elem_classes=["pcm-rgb"])
                    col_b = gr.Slider(-2.0, 2.0, step=0.01, value=0.0, label="Blue", elem_id=e("b"), elem_classes=["pcm-rgb"])
                col_start, col_end = _sigma_range(e, "Color", 15.0, 0.0)

            with gr.Group(elem_id=self.elem_id("pcm_var")):
                e = eid("var")
                _block_title("Variable Scale", "multiplies cond and uncond as sampling progresses (CFG > 1)")
                with gr.Row():
                    en_var = gr.Checkbox(label="Enable var. scale", value=False, elem_id=e("enable"))
                    var_prop = gr.Dropdown(VAR_MODES, value="sigma", label="Progress measured by", elem_id=e("mode"))
                with gr.Row():
                    var_start = gr.Slider(0.0, 10.0, step=0.01, value=1.50, label="Start mult.", elem_id=e("start"))
                    var_end = gr.Slider(0.0, 10.0, step=0.01, value=1.00, label="End mult.", elem_id=e("end"))

            with gr.Group(elem_id=self.elem_id("pcm_sharp")):
                e = eid("sharp")
                _block_title("Cond Sharpening", "Difference: away from the last step, Consensus: toward recent steps, "
                                                "Exponentiate: power on magnitudes")
                with gr.Row():
                    sharp_mode = gr.Dropdown(SHARP_MODES, value="Disabled", label="Sharpening mode", elem_id=e("mode"))
                    sharp_target = gr.Dropdown(TARGETS, value="both", label="Sharpen apply to", elem_id=e("target"))
                with gr.Row():
                    sharp_scale = gr.Slider(-10.0, 10.0, step=0.01, value=0.15, label="Sharpen scale",
                                            visible=False, elem_id=e("scale"))
                    exp_scale = gr.Slider(0.0, 10.0, step=0.01, value=1.05, label="Sharpen exponent",
                                          visible=False, elem_id=e("exp"))
                sharp_start, sharp_end = _sigma_range(e, "Sharp", 15.0, 1.0)

                sharp_mode.change(
                    fn=lambda m: (gr.update(visible=m in ("Difference", "Consensus")),
                                  gr.update(visible=m == "Exponentiate")),
                    inputs=[sharp_mode],
                    outputs=[sharp_scale, exp_scale],
                    show_progress=False,
                )

            with gr.Group(elem_id=self.elem_id("pcm_mean")):
                e = eid("mean")
                _block_title("Mean Normalization", "subtracts the mean of cond / uncond (pre) or of the CFG result (post)")
                with gr.Row():
                    en_mean_pre = gr.Checkbox(label="Pre-CFG", value=False, elem_id=e("pre"), min_width=100)
                    en_mean_post = gr.Checkbox(label="Post-CFG", value=False, elem_id=e("post"), min_width=100)
                    mean_per_channel = gr.Checkbox(label="Per channel", value=False, elem_id=e("per_channel"), min_width=100)
                mean_start, mean_end = _sigma_range(e, "Mean", 15.0, 0.0)

            apply_hr = gr.Checkbox(label="Also apply during Hires. fix pass", value=True,
                                   elem_id=self.elem_id("pcm_apply_hr"))

        components = [
            apply_hr,
            en_color, color_target, col_r, col_g, col_b, col_start, col_end,
            en_var, var_prop, var_start, var_end,
            sharp_mode, sharp_target, sharp_scale, exp_scale, sharp_start, sharp_end,
            en_mean_pre, en_mean_post, mean_per_channel, mean_start, mean_end,
        ]
        by_name = dict(zip(ARG_NAMES, components))

        # Pasting a prompt that has no Pre-CFG keys switches every module off
        # instead of leaving whatever was enabled before.
        self.infotext_fields = []
        for group in INFOTEXT_GROUPS.values():
            for name, key in group.items():
                if name in SWITCH_DEFAULTS:
                    default = SWITCH_DEFAULTS[name]
                    if isinstance(default, bool):
                        value = (lambda d, k=key: str(d.get(k, "False")) == "True")
                    else:
                        value = (lambda d, k=key, dv=default: d.get(k, dv))
                    self.infotext_fields.append((by_name[name], value))
                else:
                    self.infotext_fields.append((by_name[name], key))
        # Seen by sharp_mode.change, so the Scale/Exponent sliders follow a paste.
        self.paste_field_names = [key for group in INFOTEXT_GROUPS.values() for key in group.values()]

        return components

    @staticmethod
    def _config(args):
        if len(args) != len(ARG_NAMES):
            return None
        cfg = dict(zip(ARG_NAMES, args))
        for name in ("col_r", "col_g", "col_b", "col_start", "col_end", "var_start", "var_end",
                     "sharp_scale", "exp_scale", "sharp_start", "sharp_end", "mean_start", "mean_end"):
            cfg[name] = float(cfg[name])
        for name in ("apply_hr", "en_color", "en_var", "en_mean_pre", "en_mean_post", "mean_per_channel"):
            cfg[name] = bool(cfg[name])
        if cfg["sharp_mode"] not in SHARP_MODES:
            cfg["sharp_mode"] = "Disabled"
        return cfg

    @staticmethod
    def _active(cfg):
        return (
            (cfg["en_color"] and any((cfg["col_r"], cfg["col_g"], cfg["col_b"])))
            or cfg["en_var"]
            or cfg["sharp_mode"] != "Disabled"
            or cfg["en_mean_pre"]
            or cfg["en_mean_post"]
        )

    def process(self, p, *args):
        cfg = self._config(args)
        if cfg is None or not self._active(cfg):
            return

        params = {}
        if cfg["en_color"] and any((cfg["col_r"], cfg["col_g"], cfg["col_b"])):
            params.update({k: cfg[n] for n, k in INFOTEXT_GROUPS["en_color"].items()})
        if cfg["en_var"]:
            params.update({k: cfg[n] for n, k in INFOTEXT_GROUPS["en_var"].items()})
        if cfg["sharp_mode"] != "Disabled":
            params.update({k: cfg[n] for n, k in INFOTEXT_GROUPS["sharp_mode"].items()})
        if cfg["en_mean_pre"] or cfg["en_mean_post"]:
            params.update({k: cfg[n] for n, k in INFOTEXT_GROUPS["en_mean"].items()})
        p.extra_generation_params.update(params)

    def process_before_every_sampling(self, p, *args, **kwargs):
        # This, not process(): all three forks reset sd_model.forge_objects from
        # forge_objects_after_applying_lora right before sampling, which throws
        # away a unet patched any earlier. It also runs again for Hires. fix.
        cfg = self._config(args)
        if cfg is None or not self._active(cfg):
            return
        if getattr(p, "is_hr_pass", False) and not cfg["apply_hr"]:
            return

        forge_objects = getattr(p.sd_model, "forge_objects", None)
        unet = getattr(forge_objects, "unet", None)
        if unet is None or not hasattr(unet, "set_model_sampler_pre_cfg_function"):
            print(f"{LOG_PREFIX} not a Forge / reForge / Forge Neo model - skipped.")
            return

        unet = unet.clone()
        engine = PreCFGEngine(cfg, SigmaSchedule(resolve_sampling(unet)))

        needs_pre = (
            cfg["en_color"] or cfg["en_var"] or cfg["sharp_mode"] != "Disabled" or cfg["en_mean_pre"]
        )
        if needs_pre:
            unet.set_model_sampler_pre_cfg_function(engine.hook)
        if cfg["en_mean_post"]:
            # Same dict contract on all three backends.
            unet.set_model_sampler_post_cfg_function(engine.post_cfg)

        forge_objects.unet = unet
