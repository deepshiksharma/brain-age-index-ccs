import numpy as np
import pycatch22


def compute_catch22(ts, get_only_names=False):
    if get_only_names:
        return pycatch22.catch22_all(ts)["names"]
    try:
        out = pycatch22.catch22_all(ts)
        return np.asarray(out["values"], dtype=np.float32)
    except Exception:
        return np.full(22, np.nan, dtype=np.float32)


def clean_channel_name(ch):
    return ch.replace("EEG ", "").replace("-REF", "")
