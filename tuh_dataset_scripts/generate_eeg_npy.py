import os, csv
import mne
import numpy as np
import pandas as pd
from utils import make_subdir


# INPUT CSV SHOULD HAVE THESE COLUMNS: [filepath,age,gender]
INPUT_CSV = "train.csv"

OUTPUT_DIR = "output_dir"
EPOCHS_PER_SUBDIR = 10_000
PAD_DIGITS = 8 # padding digits for filename (ex: 00000001)

START_CROP = 180
END_CROP = 60
EPOCH_DURATION = 30.0
TARGET_SFREQ = 250.0
TARGET_ELECTRODES = [
    "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
    "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", 
    "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
    "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF"
]


os.mkdir(OUTPUT_DIR)

file_count, epoch_count = 0, 0
subdir_idx, subdir_file_idx = 1, 1
current_subdir = make_subdir(subdir_idx)

metadata_csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
metadata_fp = open(metadata_csv_path, "w", newline="")
metadata_writer = csv.writer(metadata_fp)
metadata_writer.writerow(["filepath", "age", "gender"])

filelist = pd.read_csv(INPUT_CSV)
total_files = len(filelist)


for _, row in filelist.iterrows():
    filepath, age, gender = row['filepath'], row['age'], row['gender']

    raw_edf = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    # Get duration
    duration = raw_edf.times[-1]
    if duration < (START_CROP + END_CROP + EPOCH_DURATION):
        print(f"\nWARNING: Insufficient seconds: {duration}\nSkipping file: {filepath}\n")
        continue
    
    # Get sampling rate
    sfreq = raw_edf.info.get('sfreq')

    # Select all target electrodes and reorder
    present = [ch for ch in TARGET_ELECTRODES if ch in raw_edf.ch_names]
    missing = [ch for ch in TARGET_ELECTRODES if ch not in raw_edf.ch_names]
    if missing:
        print(f"\nWARNING: Target channel(s) not found: {missing}\nSkipping file: {filepath}\n")
        continue
    edf_selective = raw_edf.copy().pick(picks=TARGET_ELECTRODES, verbose=False)
    edf_selective.reorder_channels(present)

    # Crop
    edf_selective.crop(tmin=START_CROP, tmax=(edf_selective.times[-1] - END_CROP))

    # Band-pass filter
    edf_selective.filter(l_freq=0.5, h_freq=45.0, picks='all', verbose=False)

    # If required, downsample to match TARGET_SFREQ
    if sfreq < TARGET_SFREQ:
        print(f"\nWARNING: Sampling rate of file is below minimum: {sfreq}\nSkipping file: {filepath}\n")
        continue
    elif  sfreq != TARGET_SFREQ:
        edf_selective.resample(TARGET_SFREQ, npad="auto")

    # Create 30 second epochs
    epochs = mne.make_fixed_length_epochs(
        edf_selective, duration=EPOCH_DURATION,
        preload=True, verbose=False
    )
    n_epochs = len(epochs)
    epoch_count += n_epochs  # update epoch count

    epoch_data = epochs.get_data()
    for ei, arr in enumerate(epoch_data):
        if subdir_file_idx > EPOCHS_PER_SUBDIR:
            subdir_idx += 1
            subdir_file_idx = 1
            current_subdir = make_subdir(subdir_idx)

        out_filename = f"{subdir_file_idx:0{PAD_DIGITS}d}.npy"
        out_path = os.path.join(current_subdir, out_filename)

        np.save(out_path, arr.astype(np.float32), allow_pickle=False)

        metadata_writer.writerow([os.path.abspath(out_path), age, gender])

        subdir_file_idx += 1
        file_count += 1


metadata_fp.close()
print(f"Done. {file_count} files processed. {subdir_idx} subdirs created.")
print(f"Total npy files written: {file_count}; total epochs encountered: {epoch_count}")
