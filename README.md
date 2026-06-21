[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D1tTEX1K)

# Wheel-Level Brake Fault Isolation on Circle-Driving Scenarios

A 1-D CNN that performs **5-class wheel-level brake fault isolation** from vehicle
state-estimator residual signals, on **circle (constant-radius cornering)** driving
scenarios generated in CarMaker.

## Links

| Item | Link |
|------|------|
| Demo video | https://www.youtube.com/watch?v=CxOpFXpXizk |
| Pre-recorded presentation video | https://www.youtube.com/watch?v=kjgbUn0a3xE |
| Presentation slides | https://drive.google.com/drive/folders/1542vV_baLzaNtgAZN_sRC0cqAw0fKb6n?usp=drive_link |
| Report | https://drive.google.com/drive/folders/1542vV_baLzaNtgAZN_sRC0cqAw0fKb6n?usp=drive_link |
| Dataset | https://drive.google.com/drive/folders/1oGLW7-zTQwz_8fNuTltY-ex2-pyChc2D?usp=drive_link |

## Project Overview

Given the 9 residual signals from a vehicle state observer, the model classifies which
brake actuator (if any) is producing an over-output fault:

- **0** — Normal
- **1–4** — Brake over-output at **FL / FR / RL / RR**

The model consumes a raw **`(9, 500)`** window — the 9 residual signals over a 0.5 s
sliding window (500 samples at 1000 Hz, advanced with a 0.1 s stride / 80 % overlap). No
hand-crafted statistics are used; the CNN learns features directly from the raw signals.

**Residual signals (9):** `res_vx`, `res_vy`, `res_r`, `res_slip_fl/fr/rl/rr`, `res_ax`,
`res_ay` (longitudinal/lateral velocity, yaw rate, per-wheel slip ×4, longitudinal/lateral
acceleration).

> **Scope note:** Motor fault localization was investigated but found infeasible with the
> current observer design (it does not model per-wheel motor torques, producing identical
> residuals regardless of which motor wheel is faulty). The project scope is therefore
> brake-only (5 classes).

## Model Architecture

```
input (9, 500)
 → Conv1d(9→64,  k16) → BN → ReLU → MaxPool(4)
 → Conv1d(64→128, k8) → BN → ReLU → MaxPool(4)
 → Conv1d(128→256,k4) → BN → ReLU → AdaptiveAvgPool(1)
 → Flatten → Linear(256→128) → ReLU → Dropout(0.1) → Linear(128→5)
```

The final layer outputs raw logits (apply `torch.softmax` externally for probabilities).
Loss: focal loss (γ=2.0). Optimizer: Adam (lr=3e-3, weight_decay=1e-4). Training uses
full-dataset lazy chunk streaming with per-chunk class balancing, signal augmentation
(noise 0.02, scale ±0.10), early stopping, and `ReduceLROnPlateau`.

## Results

Production checkpoint: `outputs/cnn_run_20260618_125218_seed0/`.

| Set | Accuracy | Macro F1 |
|-----|----------|----------|
| Test | **0.9524** | **0.918** |

Per-class F1 (full test): Normal 1.000 · bc_FL 0.902 · bc_FR 0.890 · bc_RL 0.911 · bc_RR 0.887.

## How to Run

The full walkthrough — checkpoint loading, window extraction, evaluation, and an inference
demo with executed cells — is in **[`final-project.ipynb`](final-project.ipynb)**.

```bash
pip install -r requirements.txt
```

Then open and run `final-project.ipynb` top to bottom. With `USE_FULL_DATA_IF_AVAILABLE = False`
it loads the bundled checkpoint and evaluates the 8 smoke-test samples in
`final_project_assets/sample_data/`, so the notebook runs end-to-end straight from this
repository without the full dataset.

To evaluate on the full dataset, download it from the Dataset link above, place it at
`data/brake_fault_dataset_circle_10s/`, and set `USE_FULL_DATA_IF_AVAILABLE = True`.

To retrain the production model from raw data:

```bash
python train_circle_cnn.py
```

## Dataset

Source: `data/brake_fault_dataset_circle_10s/` — 44,652 circle-driving brake scenarios.
Each `.mat` is a 10 s CarMaker run sampled at 1000 Hz (HDF5 / MATLAB v7.3), with the fault
injected at **t = 6.0 s**. Labels come from the filename prefix (`bc_FL/FR/RL/RR → 1/2/3/4`)
and the time boundary (`t < 6.0 s → Normal`). Because the dataset is large (~75 GB), it is
not bundled in this repo; download it from the Dataset link above.

## Repository Structure

```
final-project.ipynb              # main deliverable — runnable notebook with executed cells
model_cnn.py                     # production 1-D CNN (CnnMlpFaultIsolation)
train_circle_cnn.py              # training pipeline (focal loss, lazy streaming, augmentation)
dataset.py                       # HDF5 read, windowing, class balancing
split.py                         # stratified train/val/test split by run_id
requirements.txt                 # Python dependencies
tests/                           # unit tests for dataset.py and split.py
final_project_assets/            # lightweight bundle that keeps the notebook runnable
  ├─ checkpoint/                 #   released CNN weights + metadata.json (metrics)
  └─ sample_data/                #   8 smoke-test .mat files (2 per fault class)
outputs/cnn_run_20260618_125218_seed0/   # production run checkpoint (weights only)
```
