# Data and checkpoint formats

## Dataset repository

```text
data/
  training_data.nc
  expert_annotations.nc
metadata/
  samples.parquet
  axes.parquet
  dataset_manifest.json
  INTERFACE.md
```

`training_data.nc` contains 800 task rows on a 41 × 71 regular 1° grid:
600 trough rows and 200 ridge rows. Its `float32[sample, latitude, longitude]`
variables are:

- inputs: `z500`, `u500`, and `v500`;
- targets: `axis_target`, `supervision_mask_r5`, `side_label_r5`, and
  `signed_split_r5`.

Coordinates include `sample_id`, `task`, `valid_time`, `latitude`, and
`longitude`. Z500 is geopotential height in metres; U500 and V500 are in
m s⁻¹. `metadata/samples.parquet` is an ordinary tabular Parquet catalog that
provides task labels, timestamps, provenance, and deterministic fold
assignments. `metadata/axes.parquet` is GeoParquet 1.1: its `geometry` column
contains expert axes as WKB LineStrings in OGC:CRS84, with a GeoParquet `bbox`
covering for spatial filtering. The retained `curve_x` and `curve_y` spline
coordinates are the authoritative inputs used for the paper's curve metrics.

`expert_annotations.nc` provides thin raster versions of the expert axes for
visualization and interoperability. The Parquet spline catalog remains the
authoritative representation for benchmark metrics.

## Model repository

```text
production/
  trough/{model.safetensors,config.json,provenance.json}
  ridge/{model.safetensors,config.json,provenance.json}
cross_validation/
  trough/fold_0 ... fold_9/
  ridge/fold_0 ... fold_9/
manifest.json
```

Use `production/<task>` for ordinary inference. The 20 cross-validation
directories contain the held-out-fold models used to reproduce the reported
metrics.

Every `model.safetensors` file contains only model tensors. Its adjacent
`config.json` specifies the complete architecture and preprocessing contract;
`provenance.json` records artifact checksums and conversion metadata. Lightning
trainer state, optimizer state, and executable pickle payloads are not part of
the release.
