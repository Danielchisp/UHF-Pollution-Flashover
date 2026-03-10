import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ── Configuración ──
file = 'master.hdf5'
fps = 10
dpi = 150
output_dir = '.'
batch_size = 50
omit_last_seconds = 60

with h5py.File(file, 'r') as f:
    for group_name in tqdm(f.keys(), desc="Experimentos"):
        try:
            metrics = f[group_name]['Metrics']
            timestamps = metrics['timestamp'][:]
            vpp = metrics['vpp'][:]
            energy = metrics['energy'][:]

            # ── Omitir últimos 60 segundos ──
            t_cutoff = timestamps[-1] - omit_last_seconds
            mask = timestamps <= t_cutoff
            timestamps = timestamps[mask]
            vpp = vpp[mask]
            energy = energy[mask]

            n_points = len(timestamps)
            n_frames = (n_points + batch_size - 1) // batch_size
            indices = np.arange(n_points)

            # ── Crear figura ──
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'Experimento: {group_name}', fontsize=14, fontweight='bold')

            # Gráfico 1: Vpp y Timestamp vs Índice (doble eje Y)
            ax1.set_xlim(0, n_points)
            ax1.set_ylim(0, np.max(vpp) * 1.1)
            ax1.set_xlabel('Índice de señal')
            ax1.set_ylabel('Vpp', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_title('Vpp y Timestamp vs Índice')
            line_vpp, = ax1.plot([], [], 'b-', lw=0.8, label='Vpp')
            dot_vpp,  = ax1.plot([], [], 'ro', ms=4)

            # Segundo eje Y: Timestamp
            ax1b = ax1.twinx()
            ax1b.set_ylim(timestamps[0], timestamps[-1])
            ax1b.set_ylabel('Timestamp [s]', color='tab:red')
            ax1b.tick_params(axis='y', labelcolor='tab:red')
            line_ts, = ax1b.plot([], [], '-', color='tab:red', lw=2.0, alpha=0.7, label='Timestamp')

            # Gráfico 2: Energía vs VPP (scatter color fijo)
            ax2.set_xlim(0, np.max(vpp) * 1.1)
            ax2.set_ylim(0, np.max(energy) * 1.1)
            ax2.set_xlabel('Vpp')
            ax2.set_ylabel('Energía')
            ax2.set_title('Energía vs Vpp')
            scat = ax2.scatter([], [], color='steelblue', s=15, alpha=0.6)

            fig.tight_layout(rect=[0, 0, 1, 0.93])

            # ── Funciones de animación ──
            def init():
                line_vpp.set_data([], [])
                dot_vpp.set_data([], [])
                line_ts.set_data([], [])
                scat.set_offsets(np.empty((0, 2)))
                return line_vpp, dot_vpp, line_ts, scat

            def update(frame):
                i = min((frame + 1) * batch_size, n_points)

                line_vpp.set_data(indices[:i], vpp[:i])
                dot_vpp.set_data([indices[i - 1]], [vpp[i - 1]])

                line_ts.set_data(indices[:i], timestamps[:i])

                offsets = np.column_stack((vpp[:i], energy[:i]))
                scat.set_offsets(offsets)

                return line_vpp, dot_vpp, line_ts, scat

            # ── Generar GIF ──
            anim = FuncAnimation(
                fig, update, frames=n_frames,
                init_func=init, blit=True, interval=1000 // fps
            )

            writer = PillowWriter(fps=fps)
            output_path = f'{output_dir}/{group_name}.gif'
            anim.save(output_path, writer=writer, dpi=dpi)
            plt.close(fig)
            print(f'  ✔ Guardado: {output_path}')

        except Exception as e:
            print(f'  ✘ Error en {group_name}: {e}')
            plt.close('all')
            continue

print('¡Listo!')