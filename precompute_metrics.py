# precompute_metrics.py
# Ejecuta ESTO ANTES de correr main_app.py
# Calcula todas las métricas por señal y las guarda dentro del archivo HDF5.
# Así Dash carga instantáneamente sin quedarse en "updating".

import h5py
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from tqdm import tqdm
import argparse

SAMPLE_RATE = 3e9
B0_MAX_HZ = 100e6
B1_MIN_HZ = 100e6
B1_MAX_HZ = 600e6


def _compute_metrics_from_signals(signals: np.ndarray, timestamps: np.ndarray):
    """Copia exacta de la lógica que usa callbacks.py"""
    n = len(signals)
    vpp_list = []
    kurt_list = []
    skew_list = []
    crest_list = []
    energy_list = []
    eqFreq_list = []
    eqTime_list = list(timestamps[:n])
    B0_list = []
    B1_list = []

    for i, sig in enumerate(signals):
        sig = sig.astype(np.float64)
        sig_len = len(sig)

        vpp = float(sig.max() - sig.min())
        rms = float(np.sqrt(np.mean(sig ** 2)))
        peak = float(np.max(np.abs(sig)))
        crest = float(peak / rms) if rms > 0 else 0.0
        energy = float(np.sum(sig ** 2))
        kurt = float(sp_kurtosis(sig, fisher=False))
        skew = float(sp_skew(sig))

        freqs = rfftfreq(sig_len, d=1.0 / SAMPLE_RATE)
        fft_mag = np.abs(rfft(sig))

        peak_idx = int(np.argmax(fft_mag))
        eq_freq = float(freqs[peak_idx]) / 1e6

        total_energy = float(np.sum(fft_mag ** 2)) or 1.0
        mask_b0 = freqs <= B0_MAX_HZ
        mask_b1 = (freqs >= B1_MIN_HZ) & (freqs <= B1_MAX_HZ)
        b0 = float(np.sum(fft_mag[mask_b0] ** 2)) / total_energy
        b1 = float(np.sum(fft_mag[mask_b1] ** 2)) / total_energy

        vpp_list.append(vpp)
        kurt_list.append(kurt)
        skew_list.append(skew)
        crest_list.append(crest)
        energy_list.append(energy)
        eqFreq_list.append(eq_freq)
        B0_list.append(b0)
        B1_list.append(b1)

    return {
        'vpp': vpp_list,
        'kurtosis': kurt_list,
        'skewness': skew_list,
        'crest_factor': crest_list,
        'energy': energy_list,
        'eqFreq': eqFreq_list,
        'eqTime': eqTime_list,
        'B0': B0_list,
        'B1': B1_list,
        'timestamp': list(timestamps[:n]),
    }


def detect_format(group):
    keys = list(group.keys())
    if 'Metrics' in keys and 'Signals' in keys:
        return 'fmt1'
    for k in keys:
        try:
            sub = group[k]
            if 'signals' in sub and 'data' in sub['signals']:
                return 'fmt2'
        except Exception:
            continue
    return 'fmt1'


def precompute_file(hdf5_path: str, force: bool = False):
    with h5py.File(hdf5_path, 'a') as f:
        groups = list(f.keys())
        for gname in tqdm(groups, desc="Grupos"):
            grp = f[gname]
            fmt = detect_format(grp)

            if 'Metrics' in grp and not force:
                m = grp['Metrics']
                required = {'vpp', 'eqTime', 'eqFreq', 'energy', 'B0', 'B1',
                            'kurtosis', 'skewness', 'crest_factor', 'timestamp'}
                if required.issubset(set(m.keys())):
                    tqdm.write(f"[SKIP] {gname} ya tiene métricas completas.")
                    continue
                else:
                    tqdm.write(f"[UPDATE] {gname} incompleto, recalculando...")
                    del grp['Metrics']

            tqdm.write(f"[CALC] {gname} ({fmt})")

            if fmt == 'fmt1':
                signals_group = grp['Signals']
                signal_names = list(signals_group.keys())
                signals = [signals_group[s][:] for s in signal_names]
                timestamps = np.array([
                    signals_group[s].attrs.get('timestamp_s', 0.0)
                    for s in signal_names
                ])
                metrics = _compute_metrics_from_signals(np.array(signals), timestamps)
                if 'Metrics' in grp:
                    del grp['Metrics']
                mg = grp.create_group('Metrics')
                for k, v in metrics.items():
                    mg.create_dataset(k, data=np.array(v))
                mg.create_dataset('signal_ids', data=np.array(signal_names, dtype='S'))
            else:
                # fmt2 (el que causa el lag)
                chunk_names = sorted([
                    k for k in grp.keys()
                    if 'signals' in grp[k] and 'data' in grp[k]['signals']
                ])
                all_sigs = []
                all_ts = []
                all_ids = []
                for cn in chunk_names:
                    ch = grp[cn]
                    data = ch['signals']['data'][:]
                    ts = ch['signals']['timestamps'][:]
                    for i in range(len(data)):
                        all_sigs.append(data[i])
                        all_ts.append(ts[i])
                        all_ids.append(f"{cn}::{i}")

                if not all_sigs:
                    tqdm.write(f"[EMPTY] {gname}")
                    continue

                metrics = _compute_metrics_from_signals(np.array(all_sigs), np.array(all_ts))
                if 'Metrics' in grp:
                    del grp['Metrics']
                mg = grp.create_group('Metrics')
                for k, v in metrics.items():
                    mg.create_dataset(k, data=np.array(v))
                mg.create_dataset('signal_ids', data=np.array(all_ids, dtype='S'))

            tqdm.write(f"[DONE] {gname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precomputa métricas en HDF5 para que Dash cargue rápido.")
    parser.add_argument("hdf5_file", help="Ruta al archivo .hdf5")
    parser.add_argument("--force", action="store_true", help="Forzar recálculo aunque ya existan métricas")
    args = parser.parse_args()

    precompute_file(args.hdf5_file, force=args.force)
    print("\n✓ Precomputación terminada. Ahora ejecuta main_app.py sin lag.")
