# validate

[Documentation](../../index.md)

Checks a configuration against the input data structure without training.

```bash
mosa validate --config <your-config>.yaml
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `-c`, `--config` | path | yes | Path to [YAML](https://yaml.org/) config file |
| `-d`, `--debug` | flag | no | Identify compatibility checks and add failure tracebacks |

## Outputs

```
Config OK
  data:    <data>.h5mu
  views:   ['gexp', 'meth'] (discrete: [])
  model:   MOSAConfig
  output:  <output-dir>
  seed:    42
  eval:    test_size=0.1, n_folds=5, strategy=stratified, shuffle=True
  arch:    fusion=concat, latent=8
  train:   epochs=2, batch_size=8
```

Checks run cheapest first: [YAML](https://yaml.org/) parsing and key names, then path existence,
then a structural summary of the data, then model-specific requirements.

Shared failures exit with status 1: a missing path or view, a missing mask layer,
or no `model_type` column. Model-specific checks add further failures, such
as a VAE `target_batch` absent from the data when conditional reconstruction is enabled.

Model-specific warnings still exit with status 0. For the MOSA VAE, these include no `tissue` column while tissue conditioning is on, no `mutation_*` columns
while mutation conditioning is on, or adversarial correction configured with
fewer than two `model_type` categories.

Progress and warnings are written to stderr; the configuration summary stays
on stdout. For example, a missing optional mutation input produces
`[WARNING] No 'mutation_*' columns in data; mutation conditioning will be disabled.`

## Limitations

Validation reads structural metadata without loading matrices and cannot check numerical values. Use [inspect](inspect.md) for value ranges and observed-entry counts.

