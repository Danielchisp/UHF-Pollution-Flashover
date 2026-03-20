import h5py

db_original = 'Main Databases/new_master.hdf5'
db_new = 'Main Databases/master_2.hdf5'

with h5py.File(db_original, 'r') as f:
    orig_db = (list(f))
    
with h5py.File(db_new, 'r') as f:
    new_db = (list(f))