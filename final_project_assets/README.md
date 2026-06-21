# Final Project Assets

This directory keeps `final-project.ipynb` runnable from a lightweight submission.

- `checkpoint/`: released CNN checkpoint files used by the notebook.
- `sample_data/`: small held-out smoke-test sample copied from the seed-0 test split, with two files per wheel-fault class.

The bundled sample is for proving that the notebook, checkpoint loading, window extraction, plots, and inference path execute from the submitted repository. It is not used as the official performance estimate; the official metrics are the full-test values in `checkpoint/metadata.json`.

The full circle-driving dataset is not bundled here because it is about 75 GB. To run notebook evaluation on the full dataset, place it at `data/brake_fault_dataset_circle_10s/` and set `USE_FULL_DATA_IF_AVAILABLE = True` in the notebook.
