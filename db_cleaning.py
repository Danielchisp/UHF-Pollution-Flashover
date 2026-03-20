# %%
import h5py

filename = 'Main Databases/master_2.hdf5'
dest_filename = 'Main Databases/new_master.hdf5'

# %% ELIMINATE GROUPS

with h5py.File(filename, 'r') as src, h5py.File(dest_filename, 'a') as dst:
    grupos = list(src)
    grupo = grupos[-1]
    num_signals = len(src[grupo]['Signals'])
    if num_signals > 1000:
        src.copy(grupo, dst)
        print(grupo)
        print(num_signals)
        print('--------')
        
# %%

flashover = [1]

tension = [1]

flash_dict = {0: 'No hay Flashover',
              1: 'Hay Flashover'}

tension_dict = {0: '15 kV', 
           1: '25 kV'}

# %%

with h5py.File(dest_filename, 'a') as f:
    grupos = list(f)
    grupo = grupos[-1]
    # for grupo in range(len(grupos)):
    f[grupo].attrs['flashover_status'] = flash_dict[flashover[0]]
    f[grupo].attrs['rated_voltage'] = tension_dict[tension[0]]