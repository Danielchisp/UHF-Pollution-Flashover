# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import welch, butter, sosfiltfilt
from scipy.stats import kurtosis, skew

DATA = 'Main Databases/new_master.hdf5'
WIN  = 30       # segundos por ventana
FS   = 3e9      # frecuencia de muestreo [Hz]
F_L, F_H = 0.75e9, 1.49e9
FC_HPF   = 10e6
TIME_REF = np.linspace(0, 1, 3000)

# ── Preprocesamiento ─────────────────────────────────────────────────────────

def apply_highpass(signal, fs=FS, fc=FC_HPF, order=3):
    fn  = fc / (fs / 2)
    sos = butter(order, fn, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

# ── Métricas por ventana ─────────────────────────────────────────────────────

def lambda_W(t, T):
    return len(t) / T if T > 0 else 0.0

def D50(t):
    return np.median(np.diff(np.sort(t))) if len(t) >= 2 else 0.0

def burst_index(t, tau_b):
    if len(t) < 2 or tau_b is None: return 0.0
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
        if len(s) < 4: continue
        f, P = welch(s, fs=FS, nperseg=min(256, len(s)))
        fM   = (F_L + F_H) / 2
        lo   = np.sum(P[(f >= F_L) & (f <  fM)])
        hi   = np.sum(P[(f >= fM)  & (f <= F_H)])
        if lo > 0: rhos.append(hi / lo)
    return np.median(rhos) if rhos else 0.0

def eqTime(time_ref, sig):
    """Tiempo equivalente: centroide temporal de la energía."""
    energy = sig ** 2
    total  = np.sum(energy)
    return np.sum(time_ref * energy) / total if total > 0 else 0.0

def eqFreq(sig, fs=FS):
    """Frecuencia equivalente: centroide espectral."""
    f, P  = welch(sig, fs=fs, nperseg=min(256, len(sig)))
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

# ── Procesar un grupo ────────────────────────────────────────────────────────

def process(f, group):
    sgns = list(f[group]['Signals'])
    times, energies, signals = [], [], []

    for sgn in tqdm(sgns, desc=group, leave=False):
        times.append(float(f[group]['Signals'][sgn].attrs['timestamp_s']))
        sig = f[group]['Signals'][sgn][:]
        sig = apply_highpass(sig)
        energies.append(np.sqrt(np.mean(sig**2)))
        signals.append(sig)

    t = np.array(times) - times[0]
    e = np.array(energies)

    # baseline: 10% inicial
    n_b   = max(2, int(len(t) * 0.1))
    idx   = np.argsort(t)
    tau_b = np.percentile(np.diff(np.sort(t[idx[:n_b]])), 10) if n_b >= 2 else None
    e_base = np.median(e[idx[:n_b]])

    edges = np.arange(0, t[-1] + WIN, WIN)
    tc, lam, d50, bw, ke, gw, rw = [], [], [], [], [], [], []
    kurt, skewn, eqt, eqf        = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask  = (t >= lo) & (t < hi)
        idx_w = np.where(mask)[0]
        tw    = t[mask]
        ew    = e[mask]
        sw    = [signals[i] for i in idx_w]

        tc.append((lo + hi) / 2 / 60)
        lam.append(lambda_W(tw, hi - lo))
        d50.append(D50(tw))
        bw.append(burst_index(tw, tau_b))
        ke.append(kE(ew, e_base))
        gw.append(gW(sw))
        rw.append(rhoW(sw))
        kurt.append(kurtosis_W(sw))
        skewn.append(skewness_W(sw))
        eqt.append(eqTime_W(sw))
        eqf.append(eqFreq_W(sw))

    return {
        'group': group,
        'fo':    f[group].attrs.get('flashover_status', 'N/A'),
        'kv':    f[group].attrs.get('rated_voltage',    'N/A'),
        'tc':    np.array(tc),
        'lambda': np.array(lam),  'D50':  np.array(d50),
        'B':     np.array(bw),    'kE':   np.array(ke),
        'g':     np.array(gw),    'rho':  np.array(rw),
        'kurt':  np.array(kurt),  'skew': np.array(skewn),
        'eqT':   np.array(eqt),   'eqF':  np.array(eqf),
        'raw_times':   t,
        'raw_signals': signals,
    }

# ── Serie de tiempo de voltaje ───────────────────────────────────────────────

def build_voltage_timeseries(raw_times, raw_signals):
    if len(raw_times) < 2:
        return np.array([0.0]), np.array([0.0])

    idx_sort   = np.argsort(raw_times)
    t_sorted   = raw_times[idx_sort]
    sigs_sorted = [raw_signals[i] for i in idx_sort]

    diffs = np.diff(t_sorted)
    dt    = np.min(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0

    t_out, v_out = [], []

    for k, (tk, sig) in enumerate(zip(t_sorted, sigs_sorted)):
        vmax = float(np.max(sig))
        vmin = float(np.min(sig))

        # Relleno con 0s entre pulso anterior y este
        if k > 0:
            t_prev_end = t_sorted[k-1] + 2 * dt
            t_fill_end = tk - dt
            if t_fill_end > t_prev_end + dt:
                t_fill = np.arange(t_prev_end + dt, t_fill_end + dt, dt)
                t_out.extend(t_fill.tolist())
                v_out.extend([0.0] * len(t_fill))

        # 4 puntos: 0, vmax, vmin, 0
        t_out.extend([tk - dt,  tk,   tk + dt,  tk + 2*dt])
        v_out.extend([0.0,      vmax, vmin,     0.0      ])

    return np.array(t_out) / 60.0, np.array(v_out)

# ── Graficar ─────────────────────────────────────────────────────────────────

LABELS = [r'$\lambda_W$ [p/s]', r'$D_{50}$ [s]',  r'$B_W$',      r'$k_E$',
          r'$g_W$',             r'$\rho_W$',        r'Kurtosis',   r'Skewness',
          r'$t_{eq}$ [s]',      r'$f_{eq}$ [Hz]']
KEYS   = ['lambda', 'D50', 'B', 'kE', 'g', 'rho', 'kurt', 'skew', 'eqT', 'eqF']

def clean(arr):
    a = np.array(arr, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def plot_group(res):
    tag = res['group'].replace('/', '_')

    t_ts, v_ts = build_voltage_timeseries(res['raw_times'], res['raw_signals'])

    n_rows = 1 + len(KEYS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2 * n_rows), sharex=False)
    fig.suptitle(f"{res['group']}  [{res['kv']}]  —  {res['fo']}", fontsize=10)

    # ── Serie de tiempo de voltaje (panel superior) ────────────────────────
    axes[0].plot(t_ts, v_ts, color='tab:blue', lw=0.6)
    axes[0].axhline(0, color='k', lw=0.4, ls='--')
    axes[0].set_ylabel('Amplitud [u.a.]', fontsize=7)

    # ── Métricas (paneles inferiores) ──────────────────────────────────────
    for i, (key, lbl) in enumerate(zip(KEYS, LABELS)):
        axes[i + 1].plot(res['tc'], clean(res[key]))
        axes[i + 1].set_ylabel(lbl, fontsize=7)

    # Solo el último panel lleva etiqueta en X
    axes[-1].set_xlabel('Tiempo [min]', fontsize=8)

    # plt.tight_layout()
    plt.savefig(f"{tag}_full.png", dpi=300)
    plt.close()

# ── Main ─────────────────────────────────────────────────────────────────────

# %%
with h5py.File(DATA, 'r') as f:
    all_groups = list(f.keys())        # todos los ensayos
    results = [process(f, g) for g in all_groups]

# Guardar métricas
with h5py.File('pd_metrics_all.hdf5', 'w') as f:
    for res in results:
        grp = f.create_group(res['group'])
        grp.attrs['fo_status']     = res['fo']
        grp.attrs['rated_voltage'] = res['kv']
        for k in KEYS + ['tc']:
            grp.create_dataset(k, data=res[k])

# %%
# Graficar
for res in results:
    plot_group(res)

print("Listo.")