import h5py
import numpy as np
from tqdm import tqdm
from utils import all_metrics
from scipy.stats import kurtosis, skew

# Configuración
file = 'Main Databases/master.hdf5'
time_ref = np.linspace(0, 1, 3000)
band_ranges = [(0, 100e6), (100e6, 600e6)]
band_names = ['B0', 'B1']
mtrcs_keys = ['energy', 'eqTime', 'eqFreq', 'vpp']

with h5py.File(file, 'a') as f:
    for group_name in tqdm(f.keys(), desc="Grupos"):
        if 'Metrics' in f[group_name]:
            del f[group_name]['Metrics']
        f[group_name].create_group('Metrics')
        temp_dict_metrics = []
        timestamp_list = []
        kurtosis_list = []
        skewness_list = []

        for signall in tqdm(f[group_name]['Signals'], desc="Señales"):
            signal_data = f[group_name]['Signals'][signall][:]
            metrics = all_metrics(time_ref, f[group_name]['Signals'][signall], 3e9,
                                  band_ranges=band_ranges)
            temp_dict_metrics.append(metrics)
            timestamp_list.append(f[group_name]['Signals'][signall].attrs.get('timestamp_s', 0.0))

            kurtosis_list.append(kurtosis(signal_data, fisher=True, bias=True))
            skewness_list.append(skew(signal_data, bias=True))

        try:
            f[group_name]['Metrics'].create_dataset(
                'timestamp', data=np.array(timestamp_list) - timestamp_list[0])
        except:
            print('Unknown Error')

        for metric in mtrcs_keys:
            values = [temp_dict_metrics[i][metric] for i in range(len(temp_dict_metrics))]
            f[group_name]['Metrics'].create_dataset(metric, data=values)

        for band_idx, band_name in enumerate(band_names):
            f[group_name]['Metrics'].create_dataset(
                band_name,
                data=[temp_dict_metrics[i]['bandEnergy'][band_idx] for i in range(len(temp_dict_metrics))]
            )

        f[group_name]['Metrics'].create_dataset('kurtosis', data=np.array(kurtosis_list))
        f[group_name]['Metrics'].create_dataset('skewness', data=np.array(skewness_list))