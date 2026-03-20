# %% LIBRARIES

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion
from scipy.signal import welch
# %% FUNCTIONS

def morphological_gradient(signal, size=3):
    """Gradiente morfológico: dilatación - erosión."""
    dilated = grey_dilation(signal, size=size)
    eroded = grey_erosion(signal, size=size)
    return dilated - eroded

# %% INPUT DATA

archivo_hdf5 = 'Main Databases/new_master.hdf5'
fs = 3e9

signals = []

with h5py.File(archivo_hdf5, 'r') as f:
    groups = list(f)
    num_sel = -4
    sgn_names = list(f[groups[num_sel]]['Signals'])
    for sgn in sgn_names:
        signals.append(f[groups[num_sel]]['Signals'][sgn][:])
    
# %%
cummulative_energy = []
vpp = []
energy = []
Tw_list = []
At_list = []
mg_temporal_peak = []
mg_fft_peak = []
Fw_list = []
Af_list = []

for sgn in signals:
    # --- Dominio temporal ---
    
    # Energía acumulativa normalizada
    cumm_energy_temp = np.cumsum(sgn ** 2)
    total_energy = cumm_energy_temp[-1]
    cum_energy_norm = cumm_energy_temp / total_energy
    cummulative_energy.append(cum_energy_norm)
    
    # Tw: tiempo entre 20% y 80% de energía (en muestras)
    idx_20 = np.searchsorted(cum_energy_norm, 0.2)
    idx_80 = np.searchsorted(cum_energy_norm, 0.8)
    Tw_list.append(idx_80 - idx_20)
    
    # At: área bajo la curva de energía acumulativa entre 20% y 80%
    At_list.append(np.trapz(cum_energy_norm[idx_20:idx_80]))
    
    # Vpp y energía total
    vpp.append(np.ptp(sgn))
    energy.append(total_energy)
    
    # Gradiente morfológico temporal (peak)
    mg_temporal_peak.append(np.max(morphological_gradient(sgn)))
    
    # --- Dominio frecuencial ---
    
    # PSD con Welch
    freqs, psd = welch(sgn, fs=fs)
    
    # Energía acumulativa de la PSD
    cum_energy_fft = np.cumsum(psd)
    total_energy_fft = cum_energy_fft[-1]
    
    # Fw: frecuencia donde se alcanza el 80% de la energía
    idx_80_fft = np.searchsorted(cum_energy_fft, 0.8 * total_energy_fft)
    Fw_list.append(freqs[idx_80_fft])
    
    # Af: área total bajo la curva de energía acumulativa en frecuencia
    Af_list.append(np.trapz(cum_energy_fft, freqs))
    
    # Gradiente morfológico frecuencial (peak)
    mg_fft_peak.append(np.max(morphological_gradient(psd)))

# Diccionario de features
UHF_features = {
    'T_w': Tw_list,
    'F_w': Fw_list,
    'A_t': At_list,
    'A_f': Af_list,
    'xi_t': mg_temporal_peak,
    'xi_f': mg_fft_peak,
    'vpp': vpp,
    'E': energy
}

# %%

plt.scatter(vpp, mg_fft_peak, s=1)