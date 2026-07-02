# Configuration layout

Experiment YAML files are separated first by entry point and then by experiment series.

```
config/
  train/
    A/       # direct-image prior ablations
    R/       # residual-output ablations
    G/       # single-task sanity checks
    legacy/  # older sample/resume configurations
  test/
    A/
    R/
    G/
    batch.yml
```

Series:

- `A`: direct restored-image prediction and prior ablations.
- `R`: residual prediction counterparts of A0--A3.
- `G`: single-task baseline checks.
- `S`: reserved for the later spatial-expert experiments.

R mappings:

- `R0`: A0 plain UNet with residual prediction.
- `R1`: A1 oracle-type prior with residual prediction.
- `R2`: A2 calibrated Qwen probabilities with residual prediction.
- `R3`: A3 calibrated probabilities plus severity with residual prediction.

Train example:

```
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 python script/train_assess_tpgd.py --config config/train/R/R3.yml
```

Test example:

```
CUDA_VISIBLE_DEVICES=5 PYTHONUNBUFFERED=1 python script/test_assess_tpgd.py --config config/test/R/R3.yml --checkpoint /path/to/best.pt --split val
```

`prediction_target` records the checkpoint output semantics explicitly:

- `image`: the network output is the restored image.
- `residual`: the restored image is `LQ + network_output`.

Checkpoint files do not infer this setting automatically; always test with the matching YAML series.
