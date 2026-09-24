# NIST iPSC phase-image feature export

This importer connects the existing feature extractor to a real, publicly
available dataset. It predicts a measurement: the fraction of tile area
occupied by the provided nuclear mask. It supplies no labels for senescence,
differentiation, viability or the four synthetic morphology classes.

## Export

```bash
python -m pip install -e '.[images]'
OPENBLAS_NUM_THREADS=1 python -m colonyqc.nist_ipsc \
  --cache artifacts/nist-cache --download --out artifacts/nist-features \
  --tile-size 512 --tiles-per-well 64 --seed 0
```

This downloads three archives totaling about 501 MB. Allow a further 250 MB
of temporary disk space and about 1 GB RAM. Omit `--download` to work from an
existing cache. Existing files must match their pinned size and SHA-256 hash;
corrupt files are rejected, not silently reused. Outputs must be new directories.

`features.csv` contains the nine existing fixed-threshold image features,
`reference_nuclear_fraction`, source well, density condition, pixel coordinates,
tile size and a phase-pixel checksum. `provenance.json` records source archive
and extracted TIFF hashes, sampled coordinates, settings and software versions.
Intensity scaling is fixed at uint8 / 255; nothing is learned across wells.
The reference mask is used only to compute the target, never image features.

Sampling chooses 64 nonoverlapping 512-square tiles from each source image,
seed 0, without using image content or annotation values. Only incomplete edge
tiles are excluded. All selected background tiles remain in the analysis.

## Evaluate with Regen Benchmark Kit

After installing the sibling package:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m regenbench.cli regress \
  artifacts/nist-features/features.csv --target reference_nuclear_fraction \
  --group-by source_well --seed 0 --out artifacts/nist-results
```

The companion repository includes the
[derived table, analysis plan, full results and rerun instructions](https://github.com/dylanstechmann/regen-benchmark-kit/tree/main/examples/nist_ipsc).
Its fixed Ridge baseline has a pooled MAE of 2.14 percentage points of tile
area, rising to 4.17 on the high-density well. This small within-study result
supports further evaluation of the descriptors; it is not a validated QC model.

## Annotation and holdout meaning

The three source archives are `training_low.zip`, `training_medium.zip` and
`training_high.zip`. Each contains a large phase image, a broad foreground mask,
and an automatically derived nuclear mask. **The target uses
`img_segmented.tif`, not `img_fg.tif`.** Nuclear area is neither nucleus count
nor total colony coverage. The masks are generated from fluorescence and may
contain errors; they are not manual ground truth.

Asmar et al. describe three different source wells with different seeding
densities. Holding out a whole well prevents tile mixing across training and
test. It cannot establish donor, acquisition-day or laboratory generalization.
Well and density are confounded, with one well per condition. No original
pretrained model is used: these archives were training data in the source paper,
but each baseline here is fitted from scratch on its own training wells.

The source workbook had a checksum discrepancy and is not an input to this
analysis; the [companion data card](https://github.com/dylanstechmann/regen-benchmark-kit/blob/main/examples/nist_ipsc/README.md)
records it. All three image archive checksums matched the NIST catalog.

## Sources and reuse

- [NIST dataset, doi:10.18434/mds2-2960](https://doi.org/10.18434/mds2-2960),
  version 1.1.0; accessed 2026-09-24.
- [Asmar et al., PLOS ONE (2024)](https://doi.org/10.1371/journal.pone.0298446).
- [NIST data terms](https://www.nist.gov/open/license) and
  [attribution/modification notice for the derived example](https://github.com/dylanstechmann/regen-benchmark-kit/blob/main/examples/nist_ipsc/SOURCE_NOTICE.md).

NIST did not create or endorse the derived benchmark. Raw TIFFs, archives,
source software and original model weights are not redistributed. Retain the
source notice and record the date and nature of modifications when sharing a
new derived table. The original importer code is MIT licensed.
