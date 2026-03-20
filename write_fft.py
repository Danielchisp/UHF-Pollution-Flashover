import h5py
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils import calculate_fft

with h5py.File('Main Databases/new_master.hdf5', 'a') as f:
    for cmpgn in tqdm(list(f)):
        signals = list(f[cmpgn]['Signals'])

        # Verificar si FFT existe y tiene todos los datasets
        if 'FFT' in f[cmpgn] and set(signals).issubset(set(f[cmpgn]['FFT'].keys())):
            tqdm.write(f"[SKIP] {cmpgn} ya tiene FFT completo.")
            continue

        # Si FFT existe pero incompleto, calcular solo los faltantes
        if 'FFT' not in f[cmpgn]:
            f[cmpgn].create_group('FFT')
            tqdm.write(f"[NUEVO] {cmpgn} sin FFT, calculando...")
        else:
            tqdm.write(f"[PARCIAL] {cmpgn} con FFT incompleto, completando...")

        existing_ffts = set(f[cmpgn]['FFT'].keys())
        pending = [sgn for sgn in signals if sgn not in existing_ffts]

        for sgn in tqdm(pending, desc=f"{cmpgn} - FFTs", leave=False):
            temp_signal = np.array(f[cmpgn]['Signals'][sgn])
            pxx = calculate_fft(temp_signal)
            try:
                f[cmpgn]['FFT'].create_dataset(sgn, data=pxx)
            except Exception as e:
                print(f"Error en {cmpgn}/{sgn}: {e}")