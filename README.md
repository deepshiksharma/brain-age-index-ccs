# Brain Age Index

## Dataset
Directory: **`tuh_dataset_scripts/`**

All EDF files from the entire TUH-EEG (Healthy) dataset [[1]] are kept in a fixed directory. CSV files containing the paths to individual EDF files serve as input to all scripts. This is to avoid unnecessary copying or moving of files.

**`generate_c22_features.py`** <br>
- For each EDF file in the input csv:
    - Only select channels present in `TARGET_ELECTRODES` list.
    - Crop the start and end of each file (determined by `START_CROP` and `END_CROP`).
    - Apply a bandpass filter (0.5 - 45 Hz). If required, downsample to match `TARGET_SFREQ`.
    - Split into 30s epochs. For each epoch:
        - Extract *catch22* [[2]] features.
    - Average *catch22* features across all epochs.
    - Save extracted features for this file as a single row in the output csv.

[1]: https://doi.org/10.3389/fnins.2016.00196
[2]: https://doi.org/10.1007/s10618-019-00647-x

---

Deepshik Sharma, Siddharth Rajesh <br>
Research Trainees <br>
Center for Consciousness Studies <br>
Department of Neurophysiology, <br>
NIMHANS, Bangalore
