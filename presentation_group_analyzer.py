# %%

import h5py
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import SparsePCA
from matplotlib.patches import Patch
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d


# %%

names_sgns= []
for i in range(4):
    names_sgns.append(pd.read_csv(str(i+1) + '.csv'))
    
# %%

db_name = 'Main Databases/new_master.hdf5'
signals_filtered = []
times_filtered = []

with h5py.File(db_name, 'r') as f:
    groups = list(f)[-4:]
    for group in tqdm(range(len(groups))):
        group_temp_signals = []
        group_temp_times = []
        for sgn_name in tqdm(names_sgns[group]['signal_id']):
            group_temp_signals.append(f[groups[group]]['Signals'][sgn_name][:])
            group_temp_times.append(f[groups[group]]['Signals'][sgn_name].attrs['timestamp_s'])
        signals_filtered.append(group_temp_signals)
        times_filtered.append(group_temp_times)
        
# %%

# Constantes
FS = 3e9          # frecuencia de muestreo [Hz]
F_L, F_H = 0.75e9, 1.49e9
FC_HPF = 10e6
WIN = 30         # ventana de 180 segundos
TIME_REF = np.linspace(0, 1, 3000)

# ── Leer humedad y temperatura desde el HDF5 ────────────────────────────────
# Relative Humidity está dentro de cada grupo (mismo nivel que Signals)
with h5py.File(db_name, 'r') as _f:
    _groups = list(_f)[-4:]
    _ts, _hum, _temp = [], [], []
    for _g in _groups:
        _ts.append(_f[_g]['Relative Humidity']['timestamps'][:])
        _hum.append(_f[_g]['Relative Humidity']['humidity'][:])
    rh_timestamps  = np.concatenate(_ts)
    rh_humidity    = np.concatenate(_hum)
    _ord = np.argsort(rh_timestamps)
    rh_timestamps  = rh_timestamps[_ord]
    rh_humidity    = rh_humidity[_ord]

# ── Interpoladores de humedad y temperatura (construidos una sola vez) ───────
_interp_humidity    = interp1d(rh_timestamps, rh_humidity,
                               kind='linear', bounds_error=False,
                               fill_value=(rh_humidity[0], rh_humidity[-1]))


def apply_highpass(signal, fs=FS, fc=FC_HPF, order=3):
    """Aplica filtro high-pass a la señal"""
    fn = fc / (fs / 2)
    sos = butter(order, fn, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

def lambda_W(t, T):
    """Tasa de eventos por segundo"""
    return len(t) / T if T > 0 else 0.0

def D50(t):
    """Mediana de los intervalos entre eventos"""
    return np.median(np.diff(np.sort(t))) if len(t) >= 2 else 0.0

def burst_index(t, tau_b):
    """Índice de ráfaga"""
    if len(t) < 2 or tau_b is None:
        return 0.0
    d = np.diff(np.sort(t))
    return np.sum(d < tau_b) / len(d)

def kE(e, e_base):
    """Energía relativa"""
    return np.median(e) / e_base if len(e) > 0 and e_base else 0.0

def crest_factor(s):
    """Factor de cresta"""
    r = np.sqrt(np.mean(s**2))
    return np.max(np.abs(s)) / r if r > 0 else 0.0

def gW(sigs):
    """Factor de cresta mediano"""
    gs = [crest_factor(s) for s in sigs]
    gs = [g for g in gs if g > 0]
    return np.median(gs) if gs else 0.0

def rhoW(sigs):
    """Relación espectral"""
    rhos = []
    for s in sigs:
        if len(s) < 4:
            continue
        f, P = welch(s, fs=FS, nperseg=min(256, len(s)))
        fM = (F_L + F_H) / 2
        lo = np.sum(P[(f >= F_L) & (f < fM)])
        hi = np.sum(P[(f >= fM) & (f <= F_H)])
        if lo > 0:
            rhos.append(hi / lo)
    return np.median(rhos) if rhos else 0.0

def eqTime(time_ref, sig):
    """Tiempo equivalente"""
    energy = sig ** 2
    total = np.sum(energy)
    return np.sum(time_ref[:len(sig)] * energy) / total if total > 0 else 0.0

def eqFreq(sig, fs=FS):
    """Frecuencia equivalente"""
    f, P = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    total = np.sum(P)
    return np.sum(f * P) / total if total > 0 else 0.0

def process_group(signals_filtered, times_filtered):
    """
    Procesa un grupo de señales y calcula métricas por ventanas de 180 segundos.
    Incluye humedad y temperatura interpoladas por ventana.
    """
    t     = np.array(times_filtered)   # timestamps absolutos originales
    t_rel = t - t[0]                   # tiempo relativo en segundos

    energies = [np.sqrt(np.mean(sig**2)) for sig in signals_filtered]
    e = np.array(energies)

    n_b   = max(2, int(len(t_rel) * 0.1))
    idx   = np.argsort(t_rel)
    tau_b = np.percentile(np.diff(np.sort(t_rel[idx[:n_b]])), 10) if n_b >= 2 else None
    e_base = np.median(e[idx[:n_b]])

    edges = np.arange(0, t_rel[-1] + WIN, WIN)

    tc          = []
    lam         = []
    d50         = []
    bw          = []
    ke          = []
    gw          = []
    rw          = []
    kurt_vals   = []
    skew_vals   = []
    eqt_vals    = []
    eqf_vals    = []
    vpp         = []
    hum_vals    = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask   = (t_rel >= lo) & (t_rel < hi)
        idx_w  = np.where(mask)[0]
        tw     = t_rel[mask]
        tw_abs = t[mask]               # timestamps absolutos para interpolar RH
        ew     = e[mask]
        sw     = [signals_filtered[i] for i in idx_w]

        tc.append((lo + hi) / 2 / 60)
        lam.append(lambda_W(tw, hi - lo))
        vpp.append(np.ptp(sw) if len(sw) > 0 else 0.0)
        d50.append(D50(tw))
        bw.append(burst_index(tw, tau_b))
        ke.append(kE(ew, e_base))
        gw.append(gW(sw))
        rw.append(rhoW(sw))
        kurt_vals.append(kurtosis_W(sw))
        skew_vals.append(skewness_W(sw))
        eqt_vals.append(eqTime_W(sw))
        eqf_vals.append(eqFreq_W(sw))

        # Humedad y temperatura: interpolar en los timestamps absolutos de la ventana
        if len(tw_abs) > 0:
            hum_vals.append(float(np.median(_interp_humidity(tw_abs))))
        else:
            # Ventana vacía: interpolar en el centro temporal absoluto
            t_center = t[0] + (lo + hi) / 2
            hum_vals.append(float(_interp_humidity(t_center)))

    return {
        'tc'          : np.array(tc),
        'lambda'      : np.array(lam),
        'D50'         : np.array(d50),
        'B'           : np.array(bw),
        'kE'          : np.array(ke),
        'g'           : np.array(gw),
        'rho'         : np.array(rw),
        'kurtosis'    : np.array(kurt_vals),
        'skewness'    : np.array(skew_vals),
        'eqTime'      : np.array(eqt_vals),
        'eqFreq'      : np.array(eqf_vals),
        'humidity'    : np.array(hum_vals),
        'raw_times'   : t_rel,
        'raw_energies': e,
        'vpp'         : vpp,
        'n_windows'   : len(tc)
    }

def kurtosis_W(sigs):
    vals = [kurtosis(s, fisher=True, bias=True) for s in sigs if len(s) >= 4]
    return np.median(vals) if vals else 0.0

def skewness_W(sigs):
    vals = [skew(s, bias=True) for s in sigs if len(s) >= 4]
    return np.median(vals) if vals else 0.0

def eqTime_W(sigs):
    vals = [eqTime(TIME_REF[:len(s)], s) for s in sigs if len(s) >= 2]
    return np.median(vals) if vals else 0.0

def eqFreq_W(sigs):
    vals = [eqFreq(s) for s in sigs if len(s) >= 4]
    return np.median(vals) if vals else 0.0

def process_all_groups(signals_filtered_list, times_filtered_list, group_names=None):
    results = {}
    for i, (signals, times) in enumerate(zip(signals_filtered_list, times_filtered_list)):
        group_name = group_names[i] if group_names else f'group_{i}'
        print(f"Procesando {group_name}...")
        metrics = process_group(signals, times)
        results[group_name] = metrics
        print(f"  - Ventanas   : {metrics['n_windows']}")
        print(f"  - Rango temp.: {metrics['raw_times'][-1]/60:.1f} minutos")
        print(f"  - Humedad    : {metrics['humidity'].mean():.1f} % (media)")
    return results


results = process_all_groups(signals_filtered, times_filtered, groups)

# %%




# %%

output_file = 'cuatro_mediciones_25kv.hdf5'
with h5py.File(output_file, 'w') as f, h5py.File(db_name, 'r') as g:
    for i, group in enumerate(groups):
        grp = f.create_group(group)
        signals_grp = grp.create_group('Signals')

        for j, sig_id in enumerate(names_sgns[i]['signal_id']):
            g.copy(g[group]['Signals'][sig_id], signals_grp, name=sig_id)
            signals_grp[sig_id].attrs['timestamp_s'] = times_filtered[i][j]
