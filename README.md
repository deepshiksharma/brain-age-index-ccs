# Brain Age Index

## Dataset
### Directory: `tuh_dataset_scripts/`
All EDF files from the entire TUH-EEG (Healthy) dataset [[1]] are kept in a fixed directory. CSV files containing the paths to individual EDF files serve as input to all scripts. This is to avoid unnecessary copying or moving of files.

- `generate_c22_features.py` <br>
For each EDF file in the input csv: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Only select channels present in `TARGET_ELECTRODES` list. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Crop the start and end of each file (determined by `START_CROP` and `END_CROP`). <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Apply a bandpass filter (0.5 - 45 Hz). If required, downsample to match `TARGET_SFREQ`. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Split into 30s epochs. For each epoch: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Extract *catch22* [[2]] features. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Average *catch22* features across all epochs. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Save extracted features for this file as a single row in the output csv.

[1]: https://doi.org/10.3389/fnins.2016.00196
[2]: https://doi.org/10.1007/s10618-019-00647-x

---

Deepshik Sharma <br>
Research Trainee *(Jul - Nov 2025)* <br>
Center for Consciousness Studies <br>
Department of Neurophysiology, <br>
NIMHANS, Bangalore, India
