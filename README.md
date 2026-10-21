# Advanced Pre-CFG Master Controller

An experimental sampling-control extension for **Forge, reForge and Forge Neo**. This project adapts pre-CFG prediction-adjustment concepts from [Extraltodeus' `pre_cfg_comfy_nodes_for_ComfyUI`](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI) for Forge-family WebUIs, bringing selected controls into one panel.

This is an independent adaptation, not an official release of or a project affiliated with Extraltodeus. Please refer to the [original ComfyUI project](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI) for its upstream implementation and documentation.

The controls operate around classifier-free guidance (CFG) and can be limited to parts of the sampling schedule. They are intended for users who are comfortable comparing outputs and tuning settings carefully; even small changes can affect colour, contrast, detail or composition.

## Features

| Module | What it does | Notes |
|---|---|---|
| **Latent Color** | Steers colour through the RGB directions represented by a four-channel latent. | The built-in mapping is for SD 1.x and SDXL-style latents. Other latent formats are not adjusted by this module. |
| **Variable Scale** | Changes the relative scale of conditional and unconditional predictions as sampling progresses. | Intended for CFG values above 1. Choose whether progress is measured by sigma or by the model's timestep progression. |
| **Cond Sharpening** | Adjusts prediction detail using **Difference**, **Consensus** or **Exponentiate** mode. | Difference uses the previous step; Consensus uses recent steps; Exponentiate reshapes prediction magnitudes. |
| **Mean Normalization** | Removes the mean from conditional/unconditional predictions before CFG, the CFG result after guidance, or both. | Can operate per channel and within a selected sigma range. |

The modules have independent controls, targets and sampling ranges where applicable. The **Also apply during Hires. fix pass** option controls whether the adjustments are applied during the second pass.

## Install

1. Copy this folder into your WebUI's `extensions/` directory, keeping the folder name `sd-pre_cfg_master`.
2. Restart the WebUI.
3. Find **Advanced Pre-CFG Master Controller** in the Scripts area.

No separate package installation is defined by this extension; it uses the WebUI's existing PyTorch and Gradio environment.

## Getting started

1. Leave the modules disabled and establish a baseline image with a fixed prompt and seed.
2. Enable **one module at a time**.
3. Make small adjustments, compare against the baseline, and only then combine modules.
4. For Latent Color, start with SD 1.x or SDXL. Do not assume its colour mapping applies to other model families.

There is no universal preset that works across checkpoints. The same setting can behave differently with different models, samplers and prompts.

## Compatibility and limitations

The extension uses sampling hooks exposed by Forge-family backends. If the loaded backend does not provide a required hook, the extension skips that patch and prints a note to the console. Compatibility can vary by model family and WebUI fork; treat the settings as experimental and inspect the result rather than assuming every model responds the same way.

When active, the relevant settings are written into the generation information so they can be reviewed with the image. Inactive modules do not alter sampling.

## Credits and licensing

Credit for the upstream pre-CFG node project goes to [Extraltodeus](https://github.com/Extraltodeus), creator of [`pre_cfg_comfy_nodes_for_ComfyUI`](https://github.com/Extraltodeus/pre_cfg_comfy_nodes_for_ComfyUI). This repository adapts that work for Forge-family WebUIs.

The supplied extension package did not include a `LICENSE` file, and no explicit license file was visible in the upstream repository when checked on **21 October 2026**. Public availability on GitHub does not, by itself, grant permission to redistribute or relicense adapted source code. Before publishing this adaptation, confirm the applicable license or obtain permission from the upstream author where needed. Do not assume that adding a new license to this repository would automatically cover upstream-derived code.
