# %% LIBRARIES

import h5py
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils import calculate_fft

# %%

with h5py.File('master.hdf5', 'a') as f:
    for cmpgn in tqdm(list(f)):
        f[cmpgn].create_group('FFT')
        for sgn in tqdm(f[cmpgn]['Signals']):
            temp_signal = np.array(f[cmpgn]['Signals'][sgn])
            pxx = calculate_fft(temp_signal)
            try:
                f[cmpgn]['FFT'].create_dataset(sgn, data=pxx)
            except Exception as e:
                print(e)
