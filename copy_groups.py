import h5py

archivo_hdf5 = 'Main Databases/new_master.hdf5'
archivo_nuevo = 'Main Databases/last_two_groups.hdf5'

with h5py.File(archivo_hdf5, 'r') as f_src:
    grupos = list(f_src.keys())
    ultimos_dos = grupos[-2:]
    
    with h5py.File(archivo_nuevo, 'w') as f_dst:
        for grupo in ultimos_dos:
            f_src.copy(grupo, f_dst)
            print(f"Grupo '{grupo}' copiado exitosamente.")

print(f"\nArchivo creado: {archivo_nuevo}")
print(f"Grupos copiados: {ultimos_dos}")