# Data

## Synthetic experiments

No external files are required. The four synthetic data-generating processes are implemented in `Code/src/dgp.py` and are generated from recorded random seeds.

## IHDP semi-synthetic benchmark

Place the IHDP archive at:

```text
data/ihdp/ihdp_npci_1-1000.train.npz
```

Expected SHA256:

```text
b7dbb5e26324b3b23c90ac177e1f1c411ab8562b3fc9b78d9a4a308819f54cce
```

Verify it on macOS or Linux with:

```bash
shasum -a 256 data/ihdp/ihdp_npci_1-1000.train.npz
```

The file is deliberately excluded from Git. Confirm the original IHDP benchmark source and its distribution terms before publishing a download link or redistributing the data.

