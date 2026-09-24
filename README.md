# Brightfield colony QC

A retrainable morphology triage for cheap transmitted-light images of colonies and organoids.

The model that ships here is trained on a **synthetic** brightfield generator, so the repository runs without licensed cell images. On that generator a multinomial logistic regression is essentially perfect (holdout accuracy 1.0 versus a ~0.2 majority baseline, seed 0). Two passes of a 3×3 box blur drop it to about 0.77. That gap is the point: clean synthetic accuracy is a software ceiling, not an iPSC result.

## Non-goals

This repository does **not**:

- Diagnose mycoplasma, fungus, or bacteria. A contamination flag means quarantine the culture and run a validated assay.
- Replace karyotype, STR identity, residual reprogramming factors, or a potency assay.
- Score clinical release, or anything meant to go into a person.
- Claim that a convolutional net would have been more honest. The features are area, circularity, halo, interior texture, and a thin-trace score, so a collaborator can argue with them.

## Why it is shaped this way

Imaging QC is one of the few regenerative-medicine jobs that is native to a CS workflow: fewer plates wasted, earlier flags, a model a core facility can retrain on its own annotated fields. The sterile room is still someone else's until it isn't.

Related work in this account: [anagen](https://github.com/dylanstechmann/anagen) (hair and tooth atlas) and [geroscience-compound-atlas](https://github.com/dylanstechmann/geroscience-compound-atlas) (compounds and evidence grades, not images).

## Run

```bash
make test
PYTHONPATH=src python3 -m colonyqc.cli demo --seed 0
PYTHONPATH=src python3 -m colonyqc.cli train --out artifacts/model.json
```

`predict` reads a binary PGM (`P5`), or 8-bit grayscale PNG/TIFF with the optional images extra. The generator can write one:

```bash
PYTHONPATH=src python3 - << 'PY'
from colonyqc.synthetic import render, write_pgm
import numpy as np
write_pgm("artifacts/field.pgm", render(np.random.default_rng(0), "undifferentiated"))
PY
PYTHONPATH=src python3 -m colonyqc.cli predict --model artifacts/model.json --image artifacts/field.pgm
```

Needs Python 3.10+ and numpy. No GPU.

## Labels

`undifferentiated`, `differentiating`, `debris`, `contamination_suspect`.

The last label is a filament-like texture class. It is not a microbe ID.

## License

MIT. Cite the staining papers and the assay vendor, not this model, if you write about a real culture.

## Annotated-image import (v0.2)

Install with `python -m pip install -e '.[images]'` for PNG/TIFF, or `-e .`
for PGM only. PNG/TIFF inputs must be single-frame 8-bit grayscale. There is
no automatic per-image normalization or RGB conversion; record any conversion
and image scale upstream. These fixed-threshold features are scale-sensitive.

A manifest has `sample_id,image_path,label,group_id` columns, with optional
`donor_id,batch_id,plate_id`. Paths are relative to the manifest. The labels
are supplied annotations. Use the four labels listed above; no label is inferred
by the importer. Exact duplicate image bytes and duplicate IDs are rejected.

```bash
python -m pip install -e .
python examples/make_synthetic_manifest.py
colonyqc export-features artifacts/manifest-demo/manifest.csv --out artifacts/features.csv
```

The helper generates only **synthetic** fields. Export writes a numerical
feature CSV and a provenance sidecar with image/manifest/output SHA-256 hashes.
It preserves grouping metadata for
[regen-benchmark-kit](https://github.com/dylanstechmann/regen-benchmark-kit):

```bash
# After installing regen-benchmark-kit into this environment:
regenbench run artifacts/features.csv --group-by donor_id,batch_id \
  --folds 4 --out artifacts/grouped-benchmark
```

Keep related fields, wells and donors together. Splitting image rows randomly
can leak experimental identity. Exact hashes cannot detect near-duplicate crops.
The importer makes real-data evaluation possible; it does not validate a real
cell classifier. `train` still fits the synthetic model.
