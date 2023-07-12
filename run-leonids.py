import pickle
import numpy as np

from astropy.time import Time
from astropy.coordinates import get_sun

from model import run_model

import os
from pathlib import Path

with open('pipeline-events-07.pkl', 'rb') as f:
    bdf = pickle.load(f)
bdf = bdf.reset_index()

coords = get_sun(Time(bdf.datetime))
bdf['sollon'] = [c.ra.deg for c in coords]
bdf = bdf[bdf.sollon.between(232, 238)]

print(f'will run for {len(bdf)} rows')
for i, (idx, row) in enumerate(bdf.iterrows()):
    if str(i)+'.pkl' in os.listdir('exports'):
        continue
    Path(f'exports/{i}.pkl').touch()

    print(i)
    data1 = row
    diffs = np.abs(data1.datetime-bdf.datetime)
    data2_idx = np.argpartition(diffs, 1)[1]
    data2 = bdf.iloc[data2_idx, :]
    if np.abs(data1.datetime-data2.datetime).total_seconds() > 0.5:
        datas = [data1]
    else:
        datas = [data1, data2]
    # datas, idata, reg = run_model(datas, mcmc_kwargs=dict(tune=5000,chains=3,draws=10000,target_accept=0.8), lc=False)
    datas, idata, reg = run_model(datas, mcmc_kwargs=dict(tune=1000, chains=3, draws=1000, target_accept=0.8), lc=False)
    export = {'datas': datas, 'pos': idata.posterior, 'reg': reg}

    with open(f'exports/{i}.pkl', 'wb') as f:
        pickle.dump(export, f)
