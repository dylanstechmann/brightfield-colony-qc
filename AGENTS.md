# Agent instructions — brightfield-colony-qc

Work only in this repository. The shipped classifier is trained on a synthetic
brightfield generator. Holdout accuracy 1.0 on that generator is a software
ceiling. The NIST importer is a real image path; it does not validate the four
synthetic class names.

## Do not

- Call `contamination_suspect` a microbe identification.
- Add clinical-release, karyotype, or “ready for a person” language.
- Train on synthetic fields and report the score as an iPSC result.
- Normalize images inside the feature extractor without recording the scale. Thresholds are scale-sensitive.
- Commit large image dumps.

## First commands

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
PYTHONPATH=src python3 -m colonyqc.cli demo --seed 0
```

## Improve, in this order

1. Keep the blur ablation (about 0.77 after two 3×3 box blurs). If you change features, re-measure clean vs blurred accuracy and put both numbers in the README.
2. `export-features` must keep grouping columns compatible with `regen-benchmark-kit` (`f_*` features, `donor_id` / `batch_id` when present).
3. Allowed: a test that duplicate image bytes are rejected, if one is missing.
4. Not allowed: a new CNN “because it is more advanced,” unless the same synthetic protocol reports clean accuracy, blurred accuracy, and an explicit “not iPSC cells” line.

## Done when

`make test` passes and the README still says the synthetic model is not a cell result.
