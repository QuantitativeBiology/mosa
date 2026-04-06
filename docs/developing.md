# Developer Guide

How to extend MOSA with new fusion methods, loss functions, and models. Assumes familiarity with the [Architecture Guide](architecture.md).

## Adding a new fusion method

Fusion methods determine how per-view embeddings are combined into the joint latent space. MOSA uses a registry pattern: write a class, decorate it, and it becomes available in the config.

Write the class in `src/mosa/model/latent.py` (or a new file that gets imported):

```python
from mosa.model.latent import BaseLatentSpace, register_latent

@register_latent("moe")  # this name goes in the YAML config
class MoELatentSpace(BaseLatentSpace):

    def _build(self):
        """Called once during __init__. Set up any layers here.

        Available attributes:
          self.view_dims  — dict mapping view name -> embedding dim
          self.latent_dim — target joint latent dimensionality
        """
        total_dim = sum(self.view_dims.values())
        self.gate = nn.Linear(total_dim, len(self.view_dims))
        self.fc_mu = nn.Linear(total_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(total_dim, self.latent_dim)

    def forward(self, view_embeddings, view_order, sample_masks=None):
        """Fuse per-view embeddings into (mu, logvar, z).

        Parameters
        ----------
        view_embeddings : dict[str, Tensor]
            Per-view embeddings, each [batch, view_dim]. Missing views are zero tensors.
        view_order : list[str]
            Deterministic ordering of view names.
        sample_masks : dict[str, Tensor] or None
            Per-view boolean masks [batch], True where the view is present.

        Returns
        -------
        mu, logvar, z : Tensor
            All [batch, latent_dim].
        """
        concat = torch.cat([view_embeddings[name] for name in view_order], dim=1)
        mu = self.fc_mu(concat)
        logvar = self.fc_logvar(concat)
        z = self.reparameterize(mu, logvar)  # inherited from BaseLatentSpace
        return mu, logvar, z
```

Then add your method name to the valid options in `src/mosa/config.py`:

```python
_VALID_FUSION_METHODS = ("concat", "poe", "moe")
```

Add any cross-field validation in `MOSAConfig.__post_init__()` if needed (e.g., PoE validates that all views share the same last hidden dim).

Use it:

```yaml
fusion_method: moe
```

No changes needed in the model, CLI, or training loop — `BaseLatentSpace.create()` handles instantiation automatically.

## Adding a new loss function

Loss functions live in `src/mosa/losses.py`.

Write the function:

```python
def my_custom_loss(mu: Tensor, labels: Tensor) -> Tensor:
    """Compute my custom loss on the latent space."""
    return loss_scalar
```

Integrate it into `_compute_losses()` in `src/mosa/model/mosavae.py`:

```python
from mosa.losses import my_custom_loss

def _compute_losses(self, batch, out):
    # ... existing losses ...

    custom = torch.tensor(0.0, device=self.device)
    if self.config.custom_weight > 0:
        custom = my_custom_loss(out["mu"], batch.tissue_labels)

    return {
        "recon": recon_loss,
        "kl": kl_loss,
        "contrastive": c_loss,
        "custom": custom,
        "recon_metrics": recon_metrics,
    }
```

Add it to the total loss in `training_step()`:

```python
total = (
    losses["recon"]
    + current_kl_weight * losses["kl"]
    + self.config.custom_weight * losses["custom"]
    - self.config.adv_weight * adv_loss_val
)
```

Add the config field in `src/mosa/config.py`:

```python
custom_weight: float = 0.0
```

Log it in `training_step()`:

```python
if self.config.custom_weight > 0:
    self.log("train/custom", losses["custom"])
```

## Building a new model on MOSA's architecture

MOSA's data pipeline and training infrastructure are decoupled from the model. You can reuse data loading, preprocessing, and config while replacing the model entirely.

### Subclassing MOSAVAE

For variations of the VAE (different encoder architecture, different training loop):

```python
from mosa.model.mosavae import MOSAVAE

class MyModel(MOSAVAE):

    def __init__(self, config):
        super().__init__(config)
        self.my_extra_module = nn.Linear(config.joint_latent_dim, 10)

    def forward(self, batch):
        out = super().forward(batch)
        out["my_output"] = self.my_extra_module(out["z"])
        return out

    def training_step(self, batch, batch_idx):
        # Custom training logic
        ...
```

### Writing a new LightningModule

For a fundamentally different model, write a new `LightningModule` that uses the same data pipeline:

```python
import pytorch_lightning as pl
from mosa.config import MOSAConfig
from mosa.data.batch import MOSABatch

class MyNewModel(pl.LightningModule):

    def __init__(self, config: MOSAConfig):
        super().__init__()
        self.config = config
        # Build your model using config.views, config.joint_latent_dim, etc.

    def forward(self, batch: MOSABatch) -> dict:
        # batch.encoder_inputs is dict[str, Tensor] with one entry per view
        # batch.conditionals is Tensor [batch, cond_dim]
        ...

    def training_step(self, batch: MOSABatch, batch_idx: int):
        ...

    def configure_optimizers(self):
        ...
```

Plug it into the existing CLI or use it directly:

```python
from mosa.data.datamodule import MOSADataModule
from mosa.utils import load_config

config = load_config("configs/my_experiment.yaml")
config.validate_paths()

datamodule = MOSADataModule(config)
datamodule.setup()

model = MyNewModel(config)

trainer = pl.Trainer(max_epochs=config.num_epochs)
trainer.fit(model, datamodule)
```

`MOSADataModule` handles all data loading, splitting, preprocessing, and batching. Your model receives `MOSABatch` objects with these fields:

| Field | Type | Shape |
|-------|------|-------|
| `encoder_inputs` | `dict[str, Tensor]` | `[B, D_view]` |
| `decoder_targets` | `dict[str, Tensor]` | `[B, D_view]` |
| `missing_masks` | `dict[str, Tensor]` | `[B, D_view]` (bool) |
| `conditionals` | `Tensor` | `[B, cond_dim]` |
| `tissue_labels` | `Tensor` | `[B, n_tissues]` |
| `source_ids` | `Tensor` | `[B]` |
| `sample_weights` | `Tensor` | `[B]` |
| `sample_names` | `list[str]` | `[B]` |
