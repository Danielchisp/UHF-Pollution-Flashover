# %% LIBRARIES

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.signal import welch
from scipy.signal import butter, sosfiltfilt


# %% FUNCTIONS

def morphological_gradient(signal, size=3):
    """Gradiente morfológico: dilatación - erosión."""
    dilated = grey_dilation(signal, size=size)
    eroded = grey_erosion(signal, size=size)
    return dilated - eroded

def apply_highpass(signal: np.ndarray, fs: float, fc: float, order: int = 3) -> np.ndarray:
    fn = fc / (fs / 2)
    sos = butter(order, fn, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

# %% INPUT DATA

archivo_hdf5 = 'Main Databases/new_master.hdf5'
fs = 3e9

signals = []

with h5py.File(archivo_hdf5, 'r') as f:
    groups = list(f)
    num_sel = -1
    sgn_names = list(f[groups[num_sel]]['Signals'])
    for sgn in sgn_names:
        signals.append(f[groups[num_sel]]['Signals'][sgn][:])
    
# %%
from tqdm import tqdm
vpp_orig = []
vpp_filt = []

fig, ax = plt.subplots(dpi = 300)

for signal_num in tqdm(range(len(signals))):
    vpp_orig.append(np.ptp(signals[signal_num]))
    sgn_filtered = apply_highpass(signals[signal_num], fs = 3e9, fc = 1000e6)
    vpp_filt.append(np.ptp(sgn_filtered))
    #
    # plt.plot(signals[signal_num]/max(signals[signal_num]))
# %%
    
# plt.plot([vpp_orig[i]/vpp_filt[i] for i in range(len(vpp_orig))])
plt.plot(vpp_orig)
plt.plot(vpp_filt)