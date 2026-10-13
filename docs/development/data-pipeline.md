# Extending data handling

The conversion layer aligns source tables and records observed values. Models receive `MultiOmicDataset` objects and fit any model-specific preprocessing themselves. Keeping these responsibilities separate allows cross-validation to fit preprocessing on each training fold without using held-out values.

## Add a table format

Table detection and reading live in [data/io.py](../../src/mosa/data/io.py). For example, adding [Feather](https://arrow.apache.org/docs/python/feather.html) support requires recognizing `.feather` in `_table_format()` and reading it through `pandas.read_feather()` in `_read_table()`.

Keep the existing indexing convention: an omic table stores feature identifiers in its first column and samples in the remaining columns; a metadata table retains its named columns. For the new reader, set the first column as the index only when `index_col` requests it. Do not transpose or align samples in the reader, since the converter already performs those operations.

A format addition is complete when it has equivalent conversion results for the same CSV and [Feather](https://arrow.apache.org/docs/python/feather.html) inputs, including feature names, sample order, masks, and metadata. Also test missing values, repeated identifiers, and invalid numeric entries. Update the accepted-format error, CLI help, and [preparation guide](../guides/preparing-data.md) together. [Feather](https://arrow.apache.org/docs/python/feather.html) is an example of a possible extension, not a currently supported input format.

## Add model preprocessing

VAE preprocessing is fitted in [the data module](../../src/mosa/models/mosa/datamodule.py). A new scaling method must cover fitting, application to validation and new samples, inverse transformation of reconstructions, and persistence.

For example, a median/IQR scaler would calculate each feature's median and interquartile range from observed training entries only. Replace a zero scale with a defined fallback such as one, reject features without usable training observations, and impute missing entries after scaling. Validation and projection must reuse these stored statistics without refitting them.

The implementation points are `_setup_inmemory()`, `_apply_fitted_preprocessing()`, and `inverse_transform_view()`, together with `state_dict()` and `load_state_dict()`. If the new method also supports the lazy path, update and test that path explicitly. Add the configuration value and its validation rather than relying on an unrecognized value falling through to an existing branch.

Verify that an observed training value survives the forward/inverse transform, that changing validation values cannot alter fitted statistics, and that a loaded checkpoint produces the same reconstructions. Keep discrete-view handling explicit: a continuous scaler should not silently change the scale of views explicitly marked as discrete.

## Add a dataset loader

A loader for another stored representation should produce the same [dataset contract](../reference/model-interface.md): one sample axis, ordered feature names, numeric view matrices, Boolean observed-entry masks, and metadata indexed by sample ID. Validate that container before handing it to a model.

A missing measurement must have a false mask entry. A measured zero must remain observed. Test sample alignment with deliberately different source orders, because a shape-correct container can still associate measurements with the wrong samples.

Run conversion and data tests for reader changes, and model/checkpoint tests for fitted preprocessing changes. Keep small format fixtures in tests so future reader changes can be checked without access to a study dataset.
