# MOFA configuration

Shared `output_dir` and `random_seed` fields are documented in [Library configuration](shared.md). Model-specific fields for `type: mofa`:

| Key | Type | Default | Description |
|---|---|---|---|
| `n_factors` | `int` | `50` | Number of factors to infer. Must be positive. |
| `ard_factors` | `bool` | `true` | Applies automatic relevance determination to the factors. |
| `drop_r2` | `float` or `null` | `0.001` | Drops factors that explain no more variance than this threshold in every view and group. `null` keeps every factor. |
| `scale_views` | `bool` | `false` | Scales views to unit variance before fitting. |
| `scale_groups` | `bool` | `false` | Scales groups to unit variance before fitting. |
| `convergence_mode` | `str` | `fast` | Convergence tolerance. Options are `fast`, `medium`, or `slow`. |
| `iterations` | `int` | `1000` | Maximum number of training updates. Must be positive. [`train`](../commands/train.md) warns when a run stops at this cap without converging. |
| `gpu_mode` | `bool` | `false` | Enables GPU training through [CuPy](https://cupy.dev/). Requires an NVIDIA GPU and a [CuPy](https://cupy.dev/) installation compatible with the server's [CUDA](https://developer.nvidia.com/cuda-toolkit) version. [mofapy2](https://biofam.github.io/MOFA2/) falls back to CPU if [CuPy](https://cupy.dev/) cannot be imported. |
| `gpu_device` | `int` or `null` | `null` | [CUDA](https://developer.nvidia.com/cuda-toolkit) device index (non-negative), used when `gpu_mode: true`. `null` uses [CuPy](https://cupy.dev/)'s current device. |

