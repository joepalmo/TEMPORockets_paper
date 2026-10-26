# Plume detection outputs

This directory contains the detection run used in the paper:

- `detections.csv`
- `*.nc` (one file per detected plume hour)
- `figs/detections/`, `figs/nondetections/`

Re-running the detection step overwrites these files:

```bash
python compute_emissions.py
# or: sbatch compute_emissions.sh
```
