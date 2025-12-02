import pandas as pd
from pathlib import Path

# Create a new CSV that keeps only rows whose filepath points to an existing file under a local base directory.
# And the files have a valid age and gender.

INPUT_CSV = "../../outputs/01_selective_16.csv"
OUTPUT_CSV = "../../outputs/02_valid_files.csv"
BASE_PATH = Path("/NAS_all/CCS_Common/OpenSourceDatasets/TUH_Healthy")
MIN_DURATION = 270


def make_local_path(orig_fp: str) -> Path:
    s = str(orig_fp).strip()
    # remove all leading "../"
    while s.startswith("../") or s.startswith("./"):
        s = s[3:]
    return BASE_PATH.joinpath(s).resolve()


def main():
    df = pd.read_csv(INPUT_CSV)
    # Build local absolute paths
    df["_local_path"] = df["filepath"].apply(make_local_path)

    # Keep only rows where the file exists on disk
    mask = df["_local_path"].apply(lambda p: p.exists())
    df_keep = df.loc[mask].copy()
    
    # Replace filepath column with the full local path (string)
    df_keep["filepath"] = df_keep["_local_path"].astype(str)

    # Drop helper column and write CSV with the original column order
    df_keep = df_keep.drop(columns=["_local_path"])

    # Filter files by minumum duration
    df_keep = df_keep[df_keep["duration"] > MIN_DURATION]

    # Exclude rows where age is missing/empty/'nan' or numeric 999
    age_series = df_keep["age"]
    age_str = age_series.fillna("").astype(str).str.strip()
    age_not_blank = age_str.ne("") & ~age_str.str.lower().eq("nan")
    age_num = pd.to_numeric(age_str, errors="coerce")
    age_not_999 = ~(age_num == 999)
    age_mask = age_not_blank & age_not_999

    # Exclude rows where gender is missing/empty/'nan'
    gender_series = df_keep["gender"]
    gender_str = gender_series.fillna("").astype(str).str.strip()
    gender_mask = gender_str.ne("") & ~gender_str.str.lower().eq("nan")

    # combined mask: keep only rows having both age & gender
    combined_mask = age_mask & gender_mask
    df_keep = df_keep.loc[combined_mask].copy()

    df_keep.to_csv(OUTPUT_CSV, index=False)
    print(f"{len(df_keep)} rows written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
