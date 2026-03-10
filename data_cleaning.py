# %% LIBRARIES

import h5py
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# %% DIR DEF

DB_DIR = 'Respaldo Datos Experimentos UHF Pollution/PD Time Series Output Folder'
HR_DIR = 'Respaldo Datos Experimentos UHF Pollution/humidity logs'

DB_FILES = os.listdir(DB_DIR)
HR_FILES = os.listdir(HR_DIR)

# %% HR_DICT
HR_DICT = []

for hr in HR_FILES:
    temp_hr_file = pd.read_csv(os.path.join(HR_DIR, hr))
    duration = round(temp_hr_file['Tiempo (s)'].iloc[-1] - temp_hr_file['Tiempo (s)'].iloc[0],0)
    HR_DICT.append({'date': datetime.strptime(hr[-19:-4], "%Y%m%d_%H%M%S") - timedelta(seconds=duration) + timedelta(minutes=180),
                    'duration': duration,
                    'name': hr})                   
    
# %% 
DB_DICT = []
for db in DB_FILES:
    with h5py.File(db, 'r') as f:
        for i in list(f):
            sgn_names = list(f[i]['Signals'])
            if len(sgn_names) == 0:
                continue            
            total_time_temp = f[i]['Signals'][sgn_names[-1]].attrs.get('timestamp_s') - f[i]['Signals'][sgn_names[0]].attrs.get('timestamp_s')
            DB_DICT.append({'name': db,
                            'date': datetime.strptime(i[-20:], "%Y-%m-%dT%H-%M-%SZ"),
                            'num_signals': len(f[i]['Signals']),
                            'duration': total_time_temp,
                            'real_name': i})
            
DB_DICT = [DB_DICT[i] for i in range(len(DB_DICT)) if DB_DICT[i]['duration'] > 1000]

# %%

index_pair = []

for db in DB_DICT:
    for hr in HR_DICT:
        if abs((db['date'] - hr['date']) // timedelta(seconds=1)) < 180:
            index_pair.append((DB_DICT.index(db), HR_DICT.index(hr)))
            print(f'Database {db['date']} matches {hr['date']}')
            print(f'Database duration: {db['duration']} | HR log duration {hr['duration']}')
            print(f'{'-'*100}')
            break
        

# %%
with h5py.File('Main DB.hdf5', 'a') as f:
    for cmpgn in list(f):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Graficar Vpp
        timestamps_vpp = np.array(f[cmpgn]['Metrics']['timestamp'])
        vpp_values = np.array(f[cmpgn]['Metrics']['vpp'])
        ax.plot(timestamps_vpp, vpp_values, color='blue', label='Vpp')
        ax.set_xlabel('Timestamp (s)')
        ax.set_ylabel('Vpp', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Verificar si hay datos de humedad relativa
        if list(f[cmpgn]['Relative Humidity']) == []:
            # No hay datos de HR en el HDF5, buscar en los archivos CSV
            ax2 = ax.twinx()
            
            # Encontrar el índice de esta campaña en DB_DICT
            db_idx = None
            for i, db in enumerate(DB_DICT):
                if db['real_name'] == cmpgn:
                    db_idx = i
                    break
            
            if db_idx is not None:
                # Buscar el archivo HR correspondiente
                hr_idx = None
                for pair in index_pair:
                    if pair[0] == db_idx:
                        hr_idx = pair[1]
                        break
                
                if hr_idx is not None:
                    # Cargar el archivo CSV de humedad
                    hr_file_path = os.path.join(HR_DIR, HR_DICT[hr_idx]['name'])
                    hr_data = pd.read_csv(hr_file_path)
                    
                    # Extraer datos (ajusta los nombres de columnas según tu CSV)
                    time_hr = hr_data['Tiempo (s)'].values
                    humidity = hr_data['Humedad (%)'].values  # Ajusta el nombre si es diferente
                    
                    # ESCRIBIR en el HDF5
                    f[cmpgn]['Relative Humidity'].create_dataset('timestamp', data=time_hr)
                    f[cmpgn]['Relative Humidity'].create_dataset('HR', data=humidity)
                    
                    # Graficar humedad relativa
                    ax2.plot(time_hr, humidity, color='red', label='HR (%)', alpha=0.7)
                    ax2.set_ylabel('Humedad Relativa (%)', color='red')
                    ax2.tick_params(axis='y', labelcolor='red')
                    
                    print(f'Campaña: {cmpgn}')
                    print(f'  - HR cargada desde CSV: {HR_DICT[hr_idx]["name"]}')
                    print(f'  - Datos escritos en HDF5: {len(time_hr)} puntos')
                    print(f'  - Rango HR: {humidity.min():.2f}% - {humidity.max():.2f}%')
                else:
                    print(f'Campaña: {cmpgn} - No se encontró archivo HR correspondiente')
                    ax2 = ax.twinx()
                    ax2.set_ylabel('Humedad Relativa (%) - No disponible', color='red')
            else:
                print(f'Campaña: {cmpgn} - No se encontró en DB_DICT')
                ax2 = ax.twinx()
                ax2.set_ylabel('Humedad Relativa (%) - No disponible', color='red')
        else:
            # Hay datos de HR en el HDF5, graficarlos directamente
            ax2 = ax.twinx()
            timestamps_hr = np.array(f[cmpgn]['Relative Humidity']['timestamp'])
            humidity = np.array(f[cmpgn]['Relative Humidity']['HR'])
            
            ax2.plot(timestamps_hr, humidity, color='red', label='HR (%)', alpha=0.7)
            ax2.set_ylabel('Humedad Relativa (%)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            print(f'Campaña: {cmpgn}')
            print(f'  - HR ya existe en HDF5: {len(humidity)} puntos')
            print(f'  - Rango HR: {humidity.min():.2f}% - {humidity.max():.2f}%')
        
        # Título y grid
        ax.set_title(f'Campaña: {cmpgn}')
        ax.grid(True, alpha=0.3)
        
        # Leyendas
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Mostrar la figura
        plt.tight_layout()
        plt.show()
        
        print(f'{"-"*100}\n')