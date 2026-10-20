import numpy as np
import pandas as pd

def make_features(dbm, mask):
    d = dbm.to_numpy(dtype=float)
    m = mask.to_numpy().astype(bool)
    obs = np.where(m, d, np.nan)
    return pd.DataFrame({
        "n_heard": m.sum(axis=1),
        "max_rssi": np.nanmax(obs, axis=1),
        "mean_rssi": np.nanmean(obs, axis=1),
        "std_rssi": np.nanstd(obs, axis=1),
        "n_strong": (obs > -70).sum(axis=1),
        "top1_ap": d.argmax(axis=1),
    }, index=dbm.index)
