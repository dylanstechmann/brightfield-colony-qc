# Brightfield colony QC

A retrainable morphology triage for cheap transmitted-light images of colonies and organoids.

The model that ships here is trained on a **synthetic** brightfield generator, so the repository runs without licensed cell images. On that generator a multinomial logistic regression is essentially perfect (holdout accuracy 1.0 versus a ~0.2 majority baseline, seed 0). Two passes of a 3×3 box blur drop it to about 0.77. That gap is the point: clean synthetic accuracy is a software ceiling, not an iPSC result.

Version 0.3 adds a **real iPSC phase-image importer** for the public NIST
mds2-2960 dataset. A companion
[whole-well regression benchmark](https://github.com/dylanstechmann/regen-benchmark-kit/blob/main/examples/nist_ipsc/results/STUDY_REPORT.md)
uses these nine features to predict nuclear mask area on 192 tiles from three
wells. The fixed Ridge baseline achieved 2.14 percentage-point MAE versus
9.72 for a training-fold mean. Only three wells were evaluated; this does not
validate the synthetic morphology classes. See the [data workflow](docs/NIST_IPSC.md)
for sources, hashes, annotation meaning and limitations.

## Non-goals

This repository does **not**:

- Diagnose mycoplasma, fungus, or bacteria. A morphology flag requires manual review and validated assays; it does not by itself justify quarantine or release.
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

`predict` reads a binary PGM (`P5`), or 8-bit grayscale PNG/TIFF with the optional images extra. It defaults to `unscorable` for synthetic-trained models. The explicit `--input-domain synthetic-demo` option is only for generator fixtures: it can show synthetic probabilities but returns `synthetic_demo_only`, never a biological class call or lab action. Reports record the input/model hashes, model and feature schema versions, training status, and image-only training-domain summary; blank, saturated, blurred, intensity-shifted, contrast-shifted, or wrong-size fields are rejected. Microscope acquisition metadata and real annotated validation data are still absent.

Corrupt model arrays, invalid scales, duplicate class labels, and a recorded
feature order that differs from the extractor are rejected before prediction.
Malformed JSON artifacts produce an `unscorable` report retaining their source
hash, so an invalid handoff remains inspectable.

The generator can write one:

```bash
PYTHONPATH=src python3 - << 'PY'
from colonyqc.synthetic import render, write_pgm
import numpy as np
write_pgm("artifacts/field.pgm", render(np.random.default_rng(0), "undifferentiated"))
PY
PYTHONPATH=src python3 -m colonyqc.cli predict --model artifacts/model.json --image artifacts/field.pgm --input-domain synthetic-demo --html artifacts/triage_report.html
```

### Selective prediction and abstention

A classifier forced to answer every field spends its errors on the fields it
understands least. `predict` now withholds its demonstration class when
top-label confidence falls below `--abstain-threshold` (default 0.60), flags the
field `abstained_low_confidence`, and asks for human review. The withheld class
and the confidence are still recorded so the decision is inspectable, and the
probabilities are unchanged by abstaining. **An abstention is a request for
review, not a statement that a culture is abnormal.**

`demo` reports coverage alongside accuracy for both the clean and blurred
holdouts, plus a risk–coverage curve over fixed thresholds. Selective accuracy
is accuracy among *answered* fields only: a model can push it to 1.0 by
answering almost nothing, so coverage is always reported with it.

The curve carries a `confidence_diagnostic` that states whether abstaining
actually helps, and for this model it reports an honest negative result:

| Holdout | Full-coverage accuracy | Diagnostic |
|---|---:|---|
| Clean generator fields | 1.00 | `no_headroom` — nothing to improve; a generator ceiling, not an iPSC result |
| Two 3×3 box blurs | 0.767 | `does_not_improve` — selective accuracy *falls* as coverage falls |

On blurred fields this model's confidence does not rank correctness, so
thresholding it buys nothing there. That is a property worth knowing before
trusting any confidence score under acquisition shift, and it is reported rather
than left for a reader to notice. Thresholds on the curve are fixed, not chosen
to look good; picking one by inspecting the curve would make its coverage and
accuracy development estimates rather than predictions for new fields.

### Visual HTML Triage Report

Passing `--html <path>` to `predict` generates a self-contained, standalone visual HTML report containing:
- **Triage Call & Status Banner:** Shows `unscorable`, `out_of_domain`, or `synthetic_demo_only` for the shipped model.
- **Input Image Preview:** Embedded base64 preview of the scanned field with resolution details.
- **Probabilities Breakdown:** Visual confidence bars for each morphology class.
- **Extracted Feature Table:** Complete numerical dump and interpretations for all 9 morphology metrics (`fg_fraction`, `largest_circularity`, `thin_fraction`, etc.).
- **Next Step:** Acquisition review for rejected fields; synthetic-only reports explicitly prohibit culture decisions.
- **Embedded Research & Regulatory Disclaimer:** Preserved in the report header and footer.

Needs Python 3.10+ and numpy. No GPU.

## Labels

`undifferentiated`, `differentiating`, `debris`, `contamination_suspect`.

The last label is a filament-like texture class. It is not a microbe ID.

## License

Original code: MIT. The NIST source data and derived tables retain their
[source terms and attribution](https://github.com/dylanstechmann/regen-benchmark-kit/blob/main/examples/nist_ipsc/SOURCE_NOTICE.md).
Cite the dataset and paper separately from this software.

## Annotated-image import (v0.2)

Install with `python -m pip install -e '.[images]'` for PNG/TIFF, or `-e .`
for PGM only. PNG/TIFF inputs must be single-frame 8-bit grayscale. There is
no automatic per-image normalization or RGB conversion; record any conversion
and image scale upstream. These fixed-threshold features are scale-sensitive.

A manifest has `sample_id,image_path,label,group_id` columns, with optional
`donor_id,batch_id,plate_id` plus acquisition and annotation metadata:
`imaging_lab,microscope_id,objective_magnification,pixel_size_um,exposure_ms`,
`illumination_mode,contrast_method,annotation_protocol,annotation_version`,
`annotator_id,source_uri,license`. Positive finite numbers are required when
magnification, pixel size, or exposure is supplied. Values are carried into the
feature CSV unchanged; pixel size is not used to normalize scale-sensitive
features. Paths are relative to the manifest. Labels are supplied annotations;
no label is inferred by the importer. Exact duplicate image bytes and duplicate
IDs are rejected.

```bash
python -m pip install -e .
python examples/make_synthetic_manifest.py
colonyqc export-features artifacts/manifest-demo/manifest.csv --out artifacts/features.csv
```

The helper generates only **synthetic** fields. Export writes a numerical
feature CSV and a provenance sidecar with image/manifest/output SHA-256 hashes.
The manifest and each image are decoded from the same bytes used for their
hashes, so a source file changing during export cannot silently mismatch the
features and recorded source hash.
It preserves grouping metadata for
[regen-benchmark-kit](https://github.com/dylanstechmann/regen-benchmark-kit):

```bash
# After installing regen-benchmark-kit into this environment:
regenbench run artifacts/features.csv --group-by donor_id,batch_id \
  --folds 4 --out artifacts/grouped-benchmark
```

Keep related fields, wells and donors together. Splitting image rows randomly
can leak experimental identity. Exact hashes cannot detect near-duplicate crops.
The importer makes acquisition-aware, grouped real-data evaluation possible;
it does not validate a real cell classifier. `train` still fits the synthetic
model, and these class names still need a separately reviewed real annotation
protocol before they can be interpreted on organoid or iPSC cultures.
