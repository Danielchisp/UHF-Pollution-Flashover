import h5py
import numpy as np
from tqdm import tqdm
from utils import all_metrics
from scipy.stats import kurtosis, skew
from scipy.signal import butter, sosfiltfilt

# Configuración
file = 'Main Databases/selected_25_kv_signals.hdf5'
time_ref = np.linspace(0, 1, 3000)
band_ranges = [(0, 200e6), (200e6, 1499e6)]
band_names = ['B0', 'B1']
mtrcs_keys = ['energy', 'eqTime', 'eqFreq', 'vpp']
FS = 3e9
FC_HPF = 10e6

# ── Si True, borra y recalcula Metrics aunque ya existan ──────────────────
FORCE_RECALC = True
# ─────────────────────────────────────────────────────────────────────────

EXPECTED_DATASETS = set(mtrcs_keys + band_names + ['timestamp', 'kurtosis', 'skewness', 'crest_factor'])

def apply_highpass(signal: np.ndarray, fs: float, fc: float, order: int = 3) -> np.ndarray:
    fn = fc / (fs / 2)
    sos = butter(order, fn, btype='highpass', output='sos')
    return sosfiltfilt(sos, signal)

def compute_crest_factor(signal: np.ndarray) -> float:
    """Compute crest factor as peak value divided by RMS value."""
    rms = np.sqrt(np.mean(signal**2))
    if rms == 0:
        return 0.0
    peak = np.max(np.abs(signal))
    return peak / rms

def metrics_are_complete(group) -> bool:
    """Retorna True si el grupo ya tiene Metrics con todos los datasets esperados."""
    if 'Metrics' not in group:
        return False
    existing = set(group['Metrics'].keys())
    return EXPECTED_DATASETS.issubset(existing)

def get_missing_datasets(group) -> set:
    """Retorna el conjunto de datasets faltantes en Metrics."""
    if 'Metrics' not in group:
        return EXPECTED_DATASETS.copy()
    existing = set(group['Metrics'].keys())
    return EXPECTED_DATASETS - existing

with h5py.File(file, 'a') as f:
    for group_name in tqdm(f.keys(), desc="Grupos"):
        group = f[group_name]

        if FORCE_RECALC:
            # Borrar Metrics existente para recalcular desde cero
            if 'Metrics' in group:
                del group['Metrics']
                tqdm.write(f"[FORCE] {group_name}: Metrics eliminado para recalcular.")
            group.create_group('Metrics')
            missing_datasets = EXPECTED_DATASETS.copy()

        else:
            # Comportamiento normal: saltar si está completo, completar si falta algo
            if metrics_are_complete(group):
                tqdm.write(f"[SKIP] {group_name} ya tiene métricas completas.")
                continue

            if 'Metrics' in group:
                missing_datasets = get_missing_datasets(group)
                tqdm.write(f"[UPDATE] {group_name} tiene Metrics incompleto, faltan: {missing_datasets}")
            else:
                tqdm.write(f"[NUEVO] {group_name} sin métricas, calculando...")
                group.create_group('Metrics')
                missing_datasets = EXPECTED_DATASETS.copy()

        # Preparar listas para datasets faltantes
        temp_dict_metrics = []
        timestamp_list = []
        kurtosis_list = []
        skewness_list = []
        crest_factor_list = []

        need_signal_processing = any([
            'timestamp' in missing_datasets,
            any(m in missing_datasets for m in mtrcs_keys),
            any(b in missing_datasets for b in band_names),
            'kurtosis' in missing_datasets,
            'skewness' in missing_datasets,
            'crest_factor' in missing_datasets,
        ])

        if need_signal_processing:
            for signall in tqdm(group['Signals'], desc="Señales", leave=False):
                signal_data = group['Signals'][signall][:]
                signal_filtered = apply_highpass(signal_data, FS, FC_HPF, order=3)

                if any(m in missing_datasets for m in mtrcs_keys + band_names):
                    metrics = all_metrics(time_ref, signal_filtered, FS, band_ranges=band_ranges)
                    temp_dict_metrics.append(metrics)

                if 'timestamp' in missing_datasets:
                    timestamp_list.append(
                        group['Signals'][signall].attrs.get('timestamp_s', 0.0)
                    )

                if 'kurtosis' in missing_datasets:
                    kurtosis_list.append(kurtosis(signal_filtered, fisher=True, bias=True))

                if 'skewness' in missing_datasets:
                    skewness_list.append(skew(signal_filtered, bias=True))

                if 'crest_factor' in missing_datasets:
                    crest_factor_list.append(compute_crest_factor(signal_filtered))

        # Guardar datasets faltantes
        metrics_group = group['Metrics']

        if 'timestamp' in missing_datasets and timestamp_list:
            try:
                metrics_group.create_dataset(
                    'timestamp',
                    data=np.array(timestamp_list) - timestamp_list[0],
                )
            except Exception as e:
                print(f'Error timestamp en {group_name}: {e}')

        for metric in mtrcs_keys:
            if metric in missing_datasets and temp_dict_metrics:
                values = [temp_dict_metrics[i][metric] for i in range(len(temp_dict_metrics))]
                metrics_group.create_dataset(metric, data=values)

        for band_idx, band_name in enumerate(band_names):
            if band_name in missing_datasets and temp_dict_metrics:
                metrics_group.create_dataset(
                    band_name,
                    data=[temp_dict_metrics[i]['bandEnergy'][band_idx]
                          for i in range(len(temp_dict_metrics))],
                )

        if 'kurtosis' in missing_datasets and kurtosis_list:
            metrics_group.create_dataset('kurtosis', data=np.array(kurtosis_list))

        if 'skewness' in missing_datasets and skewness_list:
            metrics_group.create_dataset('skewness', data=np.array(skewness_list))

        if 'crest_factor' in missing_datasets and crest_factor_list:
            metrics_group.create_dataset('crest_factor', data=np.array(crest_factor_list))

        tqdm.write(f"[COMPLETED] {group_name}: datasets añadidos: {missing_datasets}")