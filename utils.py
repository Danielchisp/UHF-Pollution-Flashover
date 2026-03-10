import pickle
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_scatter(x, y, output='scatter_video.gif', fps=30, batch_size=100):

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Configurar ejes
    ax.set_xlim(min(x)*1.1, max(x)*1.1)
    ax.set_ylim(min(y)*1.1, max(y)*1.1)
    # ax.grid(True, alpha=0.3)
    
    # Pre-crear scatter vacío
    scatter = ax.scatter([], [], s=10, alpha=0.6)
    
    def update(frame):
        fin = min(frame * batch_size, len(x))
        scatter.set_offsets(np.c_[x[:fin], y[:fin]])
        return scatter,
    
    # Calcular frames necesarios
    n_frames = len(x) // batch_size + 10
    
    # Crear animación
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=True)
    
    # Guardar como GIF
    anim.save(output, writer=PillowWriter(fps=fps))
    print(f"Animación guardada: {output}")
    plt.close()
    

def save_data_safe(data, filename="data.pkl"):
    """Save data with error handling"""
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Successfully saved {len(data)} records to {filename}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_data_safe(filename="data.pkl"):
    """Load data with error handling"""
    try:
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return []
        
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        print(f"Successfully loaded {len(data)} records from {filename}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def time_metrics(t: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    E = np.sum(s**2)
    t0 = np.sum(t * s**2) / E
    eqTime = np.sqrt(np.sum((t - t0) ** 2 * s**2) / E)
    return E, eqTime

def freq_metrics(f: np.ndarray, Pxx: np.ndarray) -> float:
    eqFreq = np.sqrt(np.sum(f**2 * Pxx) / np.sum(Pxx))
    return eqFreq

def all_metrics(t, s, fs=3e9, num_bands=2, band_ranges=None):
    s = np.array(s)
    f, pxx = signal.welch(s, fs, nperseg=len(s))
    E, eqTime = time_metrics(t, s)
    eqFreq = freq_metrics(f, pxx)
    vpp = np.ptp(s)

    bandEnergy = []

    if band_ranges is not None:
        for (f_low, f_high) in band_ranges:
            mask = (f >= f_low) & (f < f_high)
            bandEnergy.append(np.sum(pxx[mask]))  # sin normalizar
    else:
        segment_length = int(len(pxx) / num_bands)
        for i in range(num_bands):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < num_bands - 1 else len(pxx)
            bandEnergy.append(np.sum(pxx[start_idx:end_idx]))  # sin normalizar

    return {'energy': E,
            'eqTime': eqTime,
            'eqFreq': eqFreq,
            'vpp': vpp,
            'bandEnergy': bandEnergy
            }

def calculate_fft(s, fs=3e9):
    f, pxx = signal.welch(s, fs, nperseg=len(s))
    
    return pxx