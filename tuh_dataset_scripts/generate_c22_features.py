import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from  utils import compute_catch22, clean_channel_name


# INPUT CSV SHOULD HAVE THESE COLUMNS: [filepath,age,gender]
INPUT_CSV = "../io/inputs/TUH-EEG_selected_data.csv"

OUTPUT_CSV = "../io/outputs/c22_features.csv"
START_CROP = 180
END_CROP = 60
TARGET_SFREQ = 250.0
TARGET_ELECTRODES = [
    "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF",
    "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", 
    "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF",
    "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF"
]


# channel names without leading "EEG " and training "-REF"
clean_channels_names = [clean_channel_name(c) for c in TARGET_ELECTRODES]

# Catch22 fixed feature names
feature_names = compute_catch22(np.random.randn(2000), get_only_names=True)

# Build feature column names
final_feature_names = []
for ch in clean_channels_names:
    for fn in feature_names:
        final_feature_names.append(f"{ch}_{fn}")

# Initialize master list to hold features computed from each EDF file
master_list = []


filelist = pd.read_csv(INPUT_CSV)
total_files = len(filelist)


file_count, epoch_count = 0, 0

for _, row in filelist.iterrows():
    filepath, age, gender = row['filepath'], row['age'], row['gender']

    raw_edf = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    
    # Get duration
    duration = raw_edf.times[-1]
    if duration < 270:
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
        edf_selective, duration=30.0,
        preload=True, verbose=False
    )
    
    # Catch-22
    # get epoch data shape (n_epochs, n_channels, n_times)
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    epoch_count += n_epochs  # update epoch count

    ts_list = [data[e, c, :] for e in range(n_epochs) for c in range(n_channels)]
    results = Parallel(n_jobs=-1, backend="loky")(delayed(compute_catch22)(ts) for ts in ts_list)
    catch22_all = np.asarray(results, dtype=np.float32).reshape(n_epochs, n_channels, 22)

    # average across epochs to get shape (n_channels, 22)
    catch22_averaged = np.nanmean(catch22_all, axis=0)

    # flatten features in channel-major order
    flat_feats = catch22_averaged.reshape(-1)  # (16*22 = 352)

    # Dictionary to hold features for this iteration
    features_this_iter = {'filepath': filepath, 'age': age}
    
    # one-hot gender encoding
    features_this_iter['gender_male'] = 1 if gender == 'Male' else 0
    features_this_iter['gender_female'] = 1 if gender == 'Female' else 0

    # add the 352 features from catch22 (16*22)
    for name, value in zip(final_feature_names, flat_feats):
        features_this_iter[name] = float(value) if np.isfinite(value) else np.nan

    master_list.append(features_this_iter)
    
    # update file count and print current status
    file_count += 1
    print(f"{file_count}/{total_files} files processed.", end="\r", flush=True)


# Save master_list as csv
master_list = pd.DataFrame(master_list)
master_list.to_csv(OUTPUT_CSV, index=False)

print(f"{len(master_list)} rows written to {OUTPUT_CSV}")
print(f"{epoch_count} epochs processed in total.")
