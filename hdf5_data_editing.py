import h5py
import numpy as np
from tqdm import tqdm
from utils import all_metrics
from scipy.stats import kurtosis, skew
from scipy.signal import butter, sosfiltfilt

# Configuración
file = 'Main Databases/new_master.hdf5'
time_ref = np.linspace(0, 1, 3000)
band_ranges = [(0, 100e6), (100e6, 600e6)]
band_names = ['B0', 'B1']
mtrcs_keys = ['energy', 'eqTime', 'eqFreq', 'vpp']
FS = 3e9
FC_HPF = 300e6

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
        
        # Verificar si ya tiene todos los datasets
        if metrics_are_complete(group):
            tqdm.write(f"[SKIP] {group_name} ya tiene métricas completas.")
            continue
        
        # Si existe Metrics pero incompleto, ver qué datasets faltan
        if 'Metrics' in group:
            missing_datasets = get_missing_datasets(group)
            tqdm.write(f"[UPDATE] {group_name} tiene Metrics incompleto, faltan: {missing_datasets}")
            recalc_all = False  # No eliminamos el grupo, solo calculamos lo faltante
        else:
            tqdm.write(f"[NUEVO] {group_name} sin métricas, calculando...")
            group.create_group('Metrics')
            missing_datasets = EXPECTED_DATASETS.copy()
            recalc_all = True
        
        # Preparar listas para datasets faltantes
        temp_dict_metrics = []
        timestamp_list = []
        kurtosis_list = []
        skewness_list = []
        crest_factor_list = []
        
        # Procesar señales solo si es necesario
        need_signal_processing = any([
            'timestamp' in missing_datasets,
            any(m in missing_datasets for m in mtrcs_keys),
            any(b in missing_datasets for b in band_names),
            'kurtosis' in missing_datasets,
            'skewness' in missing_datasets,
            'crest_factor' in missing_datasets
        ])
        
        if need_signal_processing:
            for signall in tqdm(group['Signals'], desc="Señales", leave=False):
                signal_data = group['Signals'][signall][:]
                signal_filtered = apply_highpass(signal_data, FS, FC_HPF, order=3)
                
                # Calcular métricas solo si algún dataset relacionado falta
                if any([m in missing_datasets for m in mtrcs_keys + band_names]):
                    metrics = all_metrics(time_ref, signal_filtered, FS, band_ranges=band_ranges)
                    temp_dict_metrics.append(metrics)
                
                # Timestamp
                if 'timestamp' in missing_datasets:
                    timestamp_list.append(
                        group['Signals'][signall].attrs.get('timestamp_s', 0.0)
                    )
                
                # Kurtosis
                if 'kurtosis' in missing_datasets:
                    kurtosis_list.append(kurtosis(signal_filtered, fisher=True, bias=True))
                
                # Skewness
                if 'skewness' in missing_datasets:
                    skewness_list.append(skew(signal_filtered, bias=True))
                
                # Crest Factor
                if 'crest_factor' in missing_datasets:
                    crest_factor_list.append(compute_crest_factor(signal_filtered))
        
        # Guardar datasets faltantes
        metrics_group = group['Metrics']
        
        if 'timestamp' in missing_datasets and timestamp_list:
            try:
                metrics_group.create_dataset(
                    'timestamp',
                    data=np.array(timestamp_list) - timestamp_list[0]
                )
            except Exception as e:
                print(f'Error timestamp en {group_name}: {e}')
        
        # Guardar métricas principales
        for metric in mtrcs_keys:
            if metric in missing_datasets and temp_dict_metrics:
                values = [temp_dict_metrics[i][metric] for i in range(len(temp_dict_metrics))]
                metrics_group.create_dataset(metric, data=values)
        
        # Guardar band energies
        for band_idx, band_name in enumerate(band_names):
            if band_name in missing_datasets and temp_dict_metrics:
                metrics_group.create_dataset(
                    band_name,
                    data=[temp_dict_metrics[i]['bandEnergy'][band_idx] for i in range(len(temp_dict_metrics))]
                )
        
        if 'kurtosis' in missing_datasets and kurtosis_list:
            metrics_group.create_dataset('kurtosis', data=np.array(kurtosis_list))
        
        if 'skewness' in missing_datasets and skewness_list:
            metrics_group.create_dataset('skewness', data=np.array(skewness_list))
        
        if 'crest_factor' in missing_datasets and crest_factor_list:
            metrics_group.create_dataset('crest_factor', data=np.array(crest_factor_list))
        
        tqdm.write(f"[COMPLETED] {group_name}: datasets añadidos: {missing_datasets}")