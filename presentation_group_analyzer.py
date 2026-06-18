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
db_name = 'Main Databases/new_master.hdf5'
signals_filtered = []
times_filtered = []

with h5py.File(db_name, 'r') as f:
    groups = list(f)[-5:]  # <-- últimos 5 grupos
    for group in tqdm(range(len(groups))):
        group_temp_signals = []
        group_temp_times = []
        names_sgns = list(f[groups[group]]['Signals'])
        for sgn_name in tqdm(names_sgns):
            group_temp_signals.append(f[groups[group]]['Signals'][sgn_name][:])
            group_temp_times.append(f[groups[group]]['Signals'][sgn_name].attrs['timestamp_s'])
        signals_filtered.append(group_temp_signals)
        times_filtered.append(group_temp_times)
        
# %%

from tqdm import tqdm

# Constantes
FS = 3e9          # frecuencia de muestreo [Hz]
F_L, F_H = 0.75e9, 1.49e9
FC_HPF = 10e6
WIN = 60         # ventana de 180 segundos
TIME_REF = np.linspace(0, 1, 3000)

# ── Leer humedad y temperatura desde el HDF5 ────────────────────────────────
with h5py.File(db_name, 'r') as _f:
    _groups = list(_f)[-5:]  # <-- últimos 5 grupos
    _ts, _hum, _temp = [], [], []
    for _g in _groups:
        _ts.append(_f[_g]['Relative Humidity']['timestamps'][:])
        _hum.append(_f[_g]['Relative Humidity']['humidity'][:])
    rh_timestamps  = np.concatenate(_ts)
    rh_humidity    = np.concatenate(_hum)
    _ord = np.argsort(rh_timestamps)
    rh_timestamps  = rh_timestamps[_ord]
    rh_humidity    = rh_humidity[_ord]

_interp_humidity    = interp1d(rh_timestamps, rh_humidity,
                               kind='linear', bounds_error=False,
                               fill_value=(rh_humidity[0], rh_humidity[-1]))


def apply_highpass(signal, fs=FS, fc=FC_HPF, order=3):
    fn = fc / (fs / 2)
    sos = butter(order, fn, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

def lambda_W(t, T):
    return len(t) / T if T > 0 else 0.0

def D50(t):
    return np.median(np.diff(np.sort(t))) if len(t) >= 2 else 0.0

def burst_index(t, tau_b):
    if len(t) < 2 or tau_b is None:
        return 0.0
    d = np.diff(np.sort(t))
    return np.sum(d < tau_b) / len(d)

def kE(e, e_base):
    return np.median(e) / e_base if len(e) > 0 and e_base else 0.0

def crest_factor(s):
    r = np.sqrt(np.mean(s**2))
    return np.max(np.abs(s)) / r if r > 0 else 0.0

def gW(sigs):
    gs = [crest_factor(s) for s in sigs]
    gs = [g for g in gs if g > 0]
    return np.median(gs) if gs else 0.0

def rhoW(sigs):
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
    energy = sig ** 2
    total = np.sum(energy)
    return np.sum(time_ref[:len(sig)] * energy) / total if total > 0 else 0.0

def eqFreq(sig, fs=FS):
    f, P = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    total = np.sum(P)
    return np.sum(f * P) / total if total > 0 else 0.0

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

def process_group(signals_filtered, times_filtered, progress_bar=None):
    t = np.array(times_filtered)
    t_rel = t - t[0]

    energies = [np.sqrt(np.mean(sig**2)) for sig in signals_filtered]
    e = np.array(energies)

    n_b = max(2, int(len(t_rel) * 0.1))
    idx = np.argsort(t_rel)
    tau_b = np.percentile(np.diff(np.sort(t_rel[idx[:n_b]])), 10) if n_b >= 2 else None
    e_base = np.median(e[idx[:n_b]])

    edges = np.arange(0, t_rel[-1] + WIN, WIN)

    tc, lam, d50, bw, ke, gw, rw, kurt_vals, skew_vals, eqt_vals, eqf_vals, vpp, hum_vals = [], [], [], [], [], [], [], [], [], [], [], [], []

    if progress_bar is None:
        window_iter = tqdm(zip(edges[:-1], edges[1:]), 
                           desc="Procesando ventanas", 
                           total=len(edges)-1,
                           unit="ventana")
    else:
        window_iter = zip(edges[:-1], edges[1:])
        if hasattr(progress_bar, 'set_description'):
            progress_bar.set_description(f"Procesando {len(edges)-1} ventanas")

    for lo, hi in window_iter:
        mask = (t_rel >= lo) & (t_rel < hi)
        idx_w = np.where(mask)[0]
        tw = t_rel[mask]
        tw_abs = t[mask]
        ew = e[mask]
        sw = [signals_filtered[i] for i in idx_w]

        tc.append((lo + hi) / 2 / 60)
        lam.append(lambda_W(tw, hi - lo))
        vpp.append(np.ptp(np.concatenate(sw)) if len(sw) > 0 else 0.0)
        d50.append(D50(tw))
        bw.append(burst_index(tw, tau_b))
        ke.append(kE(ew, e_base))
        gw.append(gW(sw))
        rw.append(rhoW(sw))
        kurt_vals.append(kurtosis_W(sw))
        skew_vals.append(skewness_W(sw))
        eqt_vals.append(eqTime_W(sw))
        eqf_vals.append(eqFreq_W(sw))

        if len(tw_abs) > 0:
            hum_vals.append(float(np.median(_interp_humidity(tw_abs))))
        else:
            t_center = t[0] + (lo + hi) / 2
            hum_vals.append(float(_interp_humidity(t_center)))

        if progress_bar is not None:
            progress_bar.update(1)

    return {
        'tc': np.array(tc),
        'lambda': np.array(lam),
        'D50': np.array(d50),
        'B': np.array(bw),
        'kE': np.array(ke),
        'g': np.array(gw),
        'rho': np.array(rw),
        'kurtosis': np.array(kurt_vals),
        'skewness': np.array(skew_vals),
        'eqTime': np.array(eqt_vals),
        'eqFreq': np.array(eqf_vals),
        'humidity': np.array(hum_vals),
        'raw_times': t_rel,
        'raw_energies': e,
        'vpp': vpp,
        'n_windows': len(tc)
    }

def process_all_groups(signals_filtered_list, times_filtered_list, group_names=None):
    results = {}
    
    with tqdm(total=len(signals_filtered_list), desc="Procesando grupos", unit="grupo") as pbar:
        for i, (signals, times) in enumerate(zip(signals_filtered_list, times_filtered_list)):
            group_name = group_names[i] if group_names else f'group_{i}'
            pbar.set_description(f"Procesando {group_name}")
            
            metrics = process_group(signals, times, progress_bar=pbar)
            results[group_name] = metrics
            
            pbar.set_postfix({
                'ventanas': metrics['n_windows'],
                'humedad': f"{metrics['humidity'].mean():.1f}%"
            })
    
    return results

# Ejecutar procesamiento
print("Iniciando procesamiento de grupos...")
results = process_all_groups(signals_filtered, times_filtered, groups)
print("\n✅ Procesamiento completado!")

# %%
def plot_metrics_with_vpp(results, group_name):
    """
    Creates a plot with peak-to-peak time series at the top (point by point)
    and all other metrics below, sharing the same x-axis (time in minutes)
    """
    metrics = results[group_name]
    
    metric_names = {
        'lambda': 'λ (event rate) [Hz]',
        'D50': 'D50 [s]',
        'B': 'Burst Index',
        'kE': 'Relative Energy (kE)',
        'g': 'Crest Factor (g)',
        'rho': 'Spectral Ratio (ρ)',
        'kurtosis': 'Kurtosis',
        'skewness': 'Skewness',
        'eqTime': 'Equivalent Time',
        'eqFreq': 'Equivalent Frequency [Hz]',
        'humidity': 'Relative Humidity [%]'
    }
    
    n_metrics = len(metric_names)
    
    fig = plt.figure(figsize=(14, 3 * (n_metrics + 1)))
    gs = gridspec.GridSpec(n_metrics + 1, 1, height_ratios=[2] + [1] * n_metrics, hspace=0.05)
    
    # Top subplot - energies as proxy for peak-to-peak
    ax_top = plt.subplot(gs[0])
    times_min = metrics['raw_times'] / 60
    vpp_values = metrics['vpp_signals']
    
    ax_top.plot(times_min, vpp_values, color='green', alpha=1)
    ax_top.set_ylabel('Peak-to-Peak [dB]', fontsize=12)
    ax_top.set_xlim([0, times_min[-1]])
    ax_top.grid(False)
    ax_top.tick_params(labelbottom=False, labelsize=10)
    ax_top.set_title(f'Group: {group_name} - Time Series Analysis', fontsize=14, pad=10)
    
    # Subplots for each metric
    for idx, (metric_key, metric_label) in enumerate(metric_names.items()):
        ax = plt.subplot(gs[idx + 1])
        metric_data = metrics[metric_key]
        
        ax.plot(metrics['tc'], metric_data, color='blue', linewidth=1.5)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_xlim([0, metrics['tc'][-1]])
        ax.grid(False)
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.tick_params(labelsize=9)
        
        if idx < n_metrics - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('Time [minutes]', fontsize=12)
            ax.xaxis.set_label_coords(0.5, -0.05)
    
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.05)
    return fig

# %%
# Plot for each group
for group_name in results.keys():   
    fig = plot_metrics_with_vpp(results, group_name)
    plt.show()
    
# %%
    
# Calcular vpp_signals punto a punto por señal y agregarlo a results

for i, group_name in enumerate(results.keys()):
    vpp_per_signal = [np.ptp(sig) for sig in signals_filtered[i]]
    results[group_name]['vpp_signals'] = np.array(vpp_per_signal)

print("✅ vpp_signals agregado a todos los grupos!")