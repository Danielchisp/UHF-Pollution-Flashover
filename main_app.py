# main.py
import dash
from callbacks import app_callbacks
from layout import app_layout
from precompute_metrics import precompute_file

app = dash.Dash(__name__, title="UHF Pollution Flashover Monitor")
archivo_hdf5 = r"E:\Chunks\data\nuevo_exp_ciclos_12.hdf5"

app_layout(app)
app_callbacks(app, archivo_hdf5)

if __name__ == "__main__":
    # Precomputar y guardar las metricas dentro del HDF5 ANTES de iniciar Dash.
    # Asi los callbacks solo leen las metricas y la app no se queda en "updating".
    print("Precomputando metricas en el archivo HDF5...")
    precompute_file(archivo_hdf5)
    print("Metricas listas. Iniciando Dash en http://127.0.0.1:8050")

    app.run(debug=False)
