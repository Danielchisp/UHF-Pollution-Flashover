import h5py
from utils import animate_scatter
from tqdm import tqdm

# %%


db = 'Main DB.hdf5'
eqFreq = []
eqTime = []

with h5py.File(db, 'r') as f:
    groups = list(f)
    for group in groups:
        eqTime.append(list(f[group]['Metrics']['eqTime'][:]))
        eqFreq.append(list(f[group]['Metrics']['eqFreq'][:]))
        
# %%

for i in tqdm(range(10)):

    animate_scatter(eqTime[i], eqFreq[i], output=str(i) + 'scatter_video.gif', fps=30, batch_size=50)
