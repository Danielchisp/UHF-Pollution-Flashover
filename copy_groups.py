import h5py

src_file = 'Main Databases/single_group.hdf5'
dst_file = 'Main Databases/new_master.hdf5'

with h5py.File(src_file, 'r') as src, h5py.File(dst_file, 'a') as dst:
    group = list(src)[0]
    src.copy(src[group], dst)
